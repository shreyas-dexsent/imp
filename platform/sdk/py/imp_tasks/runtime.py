"""Task Runtime: drive a :class:`CompiledTask` on the live Zenoh fabric.

What it does today (v1):

* Spins every compiled module in its own daemon ``ModuleNode`` thread.
* Emits ``run.started`` / ``run.stage_changed`` / ``run.succeeded`` /
  ``run.failed`` events on ``imp/<station>/ctrl/runs/<run_id>/events``
  (matches spec §11 "Lifecycle events"). Events are JSON-over-bytes on
  the wire for v1; the typed ``RunEvent`` protobuf message lands with
  the services/jobs schemas in P8.
* For each ``SequenceStage`` with an ``until`` keyexpr, blocks until a
  message lands there (or the configured ``stage_timeout_s`` elapses),
  then advances to the next stage.
* Returns a :class:`RunResult` summarising the final state + elapsed
  time so callers (and the ``jobs/run-task`` wrapper) can serialise it
  into the run store.

What it doesn't do (deferred):

* Per-stage branching / loops -- those need the richer FSM in P7.
* Bag capture of every produced topic -- P7 wires that into the
  supervisor / ``imp bag record``.

The runtime owns a single ``Bus`` session for events + terminator
subscriptions; every module owns its own bus for the actual data flow.
"""

from __future__ import annotations

import json
import queue
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, List, Optional

from imp_sdk.bus import Bus, QosClass
from imp_sdk.keyexpr import ctrl
from imp_sdk.module import ModuleNode

from .compiler import CompiledTask
from .spec import SequenceStage


class RunStatus(str, Enum):
    """Terminal status of a :class:`TaskRun`."""

    SUCCEEDED = "succeeded"
    FAILED = "failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


@dataclass
class RunResult:
    """Summary returned by :meth:`TaskRun.run`."""

    run_id: str
    task_id: str
    status: RunStatus
    elapsed_s: float
    stages_completed: List[str] = field(default_factory=list)
    reject_reason: Optional[str] = None
    stage_durations: List[tuple[str, float]] = field(default_factory=list)


@dataclass
class _StageOutcome:
    name: str
    ok: bool
    elapsed_s: float
    reject_reason: Optional[str] = None


class TaskRun:
    """Drive a compiled task to a terminal state.

    Pass a default ``stage_timeout_s`` to cap how long the runtime waits
    on a stage's ``until`` topic (``None`` = wait forever). Callers can
    invoke :meth:`stop` from another thread to request a clean shutdown.
    """

    def __init__(
        self,
        compiled: CompiledTask,
        *,
        run_id: Optional[str] = None,
        stage_timeout_s: Optional[float] = 30.0,
        settle_s: float = 0.5,
    ):
        self.compiled = compiled
        self.run_id = run_id or _new_run_id(compiled.spec.id)
        self.stage_timeout_s = stage_timeout_s
        self.settle_s = settle_s

        self.events_key = ctrl(
            compiled.spec.station, f"runs/{self.run_id}", "events"
        )

        self._stop = threading.Event()
        self._cancelled = False
        self._nodes: List[ModuleNode] = []
        self._threads: List[threading.Thread] = []
        self._event_bus: Optional[Bus] = None

    # ------------------------------------------------------------------
    # Public lifecycle
    # ------------------------------------------------------------------

    def run(self) -> RunResult:
        """Spin everything, drive the sequence, return when terminal."""
        t0 = time.monotonic()
        self._event_bus = Bus.open()
        try:
            self._spin_up()
            self._emit("run.started",
                       task_id=self.compiled.spec.id,
                       station=self.compiled.spec.station,
                       nodes=[n.spec.id for n in self.compiled.nodes])

            outcomes = self._drive_sequence()
            ok = all(o.ok for o in outcomes)

            elapsed = time.monotonic() - t0
            if self._cancelled:
                status = RunStatus.CANCELLED
            elif ok:
                status = RunStatus.SUCCEEDED
            else:
                bad = next(o for o in outcomes if not o.ok)
                status = (RunStatus.TIMEOUT if bad.reject_reason == "timeout"
                          else RunStatus.FAILED)

            reject = next((o.reject_reason for o in outcomes if o.reject_reason), None)
            self._emit(
                "run.succeeded" if status is RunStatus.SUCCEEDED else "run.failed",
                status=status.value,
                elapsed_s=elapsed,
                reject_reason=reject,
            )
            return RunResult(
                run_id=self.run_id,
                task_id=self.compiled.spec.id,
                status=status,
                elapsed_s=elapsed,
                stages_completed=[o.name for o in outcomes if o.ok],
                stage_durations=[(o.name, o.elapsed_s) for o in outcomes],
                reject_reason=reject,
            )
        finally:
            self._spin_down()
            if self._event_bus is not None:
                self._event_bus.close()
                self._event_bus = None

    def stop(self) -> None:
        """Signal cancellation; the next stage check observes it."""
        self._cancelled = True
        self._stop.set()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _spin_up(self) -> None:
        for cn in self.compiled.nodes:
            node = ModuleNode(cn.module)
            self._nodes.append(node)
            t = threading.Thread(target=node.run, daemon=True,
                                 name=f"node:{cn.spec.id}")
            t.start()
            self._threads.append(t)
        # Let Zenoh discovery wire subscribes before the first publish.
        time.sleep(self.settle_s)

    def _spin_down(self) -> None:
        for n in self._nodes:
            n.stop()
        for t in self._threads:
            t.join(timeout=2.0)

    def _drive_sequence(self) -> List[_StageOutcome]:
        stages = self.compiled.spec.sequence or [SequenceStage(stage="run")]
        outcomes: List[_StageOutcome] = []
        for stage in stages:
            if self._cancelled:
                outcomes.append(_StageOutcome(stage.stage, False, 0.0, "cancelled"))
                break
            outcome = self._run_stage(stage)
            outcomes.append(outcome)
            if not outcome.ok:
                break
        return outcomes

    def _run_stage(self, stage: SequenceStage) -> _StageOutcome:
        self._emit("run.stage_changed", stage=stage.stage, until=stage.until)
        started = time.monotonic()

        # Stages without a terminator are declarative checkpoints -- the
        # modules already keep streaming; the runtime just records the
        # transition.
        if stage.until is None:
            return _StageOutcome(stage.stage, True, 0.0)

        msg_type = _guess_terminator_type(stage.until, self.compiled)
        if msg_type is None:
            # Best-effort default; the bus rejects on schema mismatch so a
            # bogus default just means we never get a hit -> timeout.
            from imp_sdk.schemas import imp_pb2
            msg_type = imp_pb2.Scalar

        sub = self._event_bus.subscribe(stage.until, msg_type)

        # Run recv() in a worker so the main loop can honour timeouts and
        # cancel requests without losing the message that arrives "just
        # after" the deadline.
        q: queue.Queue = queue.Queue(maxsize=1)

        def _worker() -> None:
            try:
                q.put(("ok", sub.recv()))
            except Exception as e:  # pragma: no cover - bus-side failure
                q.put(("err", e))

        threading.Thread(target=_worker, daemon=True).start()

        deadline = (started + self.stage_timeout_s
                    if self.stage_timeout_s is not None else None)
        while True:
            if self._stop.is_set():
                return _StageOutcome(stage.stage, False,
                                     time.monotonic() - started, "cancelled")
            try:
                kind, payload = q.get(timeout=0.2)
            except queue.Empty:
                if deadline is not None and time.monotonic() > deadline:
                    return _StageOutcome(stage.stage, False,
                                         time.monotonic() - started, "timeout")
                continue
            if kind == "err":
                return _StageOutcome(stage.stage, False,
                                     time.monotonic() - started, repr(payload))
            return _StageOutcome(stage.stage, True,
                                 time.monotonic() - started)

    def _emit(self, kind: str, **fields: Any) -> None:
        """Publish a JSON event on the run's events keyexpr.

        Uses Zenoh's raw ``session.put`` so the payload stays JSON without
        a protobuf wrapper. The typed ``RunEvent`` message lands in P8.
        """
        payload = {"kind": kind, "run_id": self.run_id, "ts_ns": time.time_ns(), **fields}
        body = json.dumps(payload).encode("utf-8")
        if self._event_bus is None:
            return
        try:
            # Bus.session exposes the underlying zenoh session; raw put
            # accepts a bytes/bytearray/str payload.
            self._event_bus.session.put(self.events_key, body)
        except Exception:
            # Event publish failures are non-fatal; the run continues.
            pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _new_run_id(task_id: str) -> str:
    return f"{task_id}-{uuid.uuid4().hex[:8]}"


def _guess_terminator_type(key: str, compiled: CompiledTask) -> Optional[Any]:
    """Find the protobuf type one of the compiled modules publishes on ``key``."""
    for cn in compiled.nodes:
        for _, out_key, _ in cn.outputs:
            if out_key == key:
                for o in cn.module.outputs():
                    if o.key == key:
                        return o.msg_type
    return None
