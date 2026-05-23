"""Implementation for `orchestrator.core.executor`."""

from dataclasses import dataclass
from threading import Event, Lock, Thread
from typing import Callable, Dict, List, Optional

from orchestrator.core.context import StationContext
from orchestrator.core.runs import RunState
from orchestrator.tasks.bin_picking import run_bin_picking
from orchestrator.tasks.dummy_testing import run_dummy_testing
from orchestrator.tasks.follow_object import run_follow_object
from orchestrator.tasks.pallatizing import run_pallatizing
from orchestrator.tasks.pick_place import run_pick_place_demo


@dataclass
class RunHandle:
    run_id: str
    stop_event: Event
    thread: Optional[Thread] = None
    vision_request_id: Optional[str] = None
    last_vision_request_id: Optional[str] = None
    pause_requested: bool = False


class RunExecutor:
    def __init__(self, ctx: StationContext):
        self.ctx = ctx
        self._handles: Dict[str, RunHandle] = {}
        self._lock = Lock()

        self._handlers: Dict[
            str, Callable[[StationContext, RunState, RunHandle], None]
        ] = {
            "pick_place_demo": run_pick_place_demo,
            "bin_picking": run_bin_picking,
            "vim303_pick_place": run_pick_place_demo,
            "pallatizing": run_pallatizing,
            "palletizing": run_pallatizing,
            "follow_object": run_follow_object,
            "dummy_testing": run_dummy_testing,
        }

    def start(self, state: RunState) -> bool:
        handler = self._handlers.get(state.task_type)
        if not handler:
            self.ctx.run_manager.set_state(
                state.run_id, "failed", {"reason": "unknown_task"}
            )
            return False

        handle = RunHandle(run_id=state.run_id, stop_event=Event(), thread=None)

        def _runner():
            try:
                handler(self.ctx, state, handle)
                if handle.stop_event.is_set():
                    status = "paused" if handle.pause_requested else "aborted"
                    self.ctx.run_manager.set_state(state.run_id, status)
                else:
                    self.ctx.run_manager.set_state(state.run_id, "completed")
            except Exception as exc:
                if str(exc) == "run_stopped":
                    self.ctx.run_manager.set_state(state.run_id, "aborted")
                else:
                    self.ctx.run_manager.set_state(
                        state.run_id, "failed", {"error": str(exc)}
                    )
            finally:
                with self._lock:
                    self._handles.pop(state.run_id, None)

        # Daemon worker allows fast process termination if a task thread is stuck.
        thread = Thread(target=_runner, daemon=True)
        handle.thread = thread
        with self._lock:
            self._handles[state.run_id] = handle
        self.ctx.run_manager.set_state(state.run_id, "running")
        thread.start()
        return True

    def list_task_types(self) -> List[str]:
        return sorted(self._handlers.keys())

    def stop(self, run_id: str) -> bool:
        with self._lock:
            handle = self._handles.get(run_id)
        if not handle:
            return False
        self.ctx.run_manager.set_state(run_id, "stopping", {"reason": "stop_requested"})
        handle.stop_event.set()
        handle.pause_requested = False
        if handle.vision_request_id:
            try:
                self.ctx.vision.stop_session(handle.vision_request_id)
            except Exception:
                pass
        try:
            self.ctx.robot.stop()
        except Exception:
            pass
        return True

    def pause(self, run_id: str) -> bool:
        with self._lock:
            handle = self._handles.get(run_id)
        if not handle:
            return False
        handle.pause_requested = True
        handle.stop_event.set()
        self.ctx.run_manager.set_state(run_id, "paused", {"reason": "pause_requested"})
        if handle.vision_request_id:
            try:
                self.ctx.vision.stop_session(handle.vision_request_id)
            except Exception:
                pass
        try:
            self.ctx.robot.stop()
        except Exception:
            pass
        return True

    def get_handle(self, run_id: str) -> Optional[RunHandle]:
        with self._lock:
            return self._handles.get(run_id)

    def close(self, join_timeout_s: float = 2.0) -> None:
        with self._lock:
            handles = list(self._handles.values())

        for handle in handles:
            handle.pause_requested = False
            handle.stop_event.set()
            if handle.vision_request_id:
                try:
                    self.ctx.vision.stop_session(handle.vision_request_id)
                except Exception:
                    pass

        if handles:
            try:
                self.ctx.robot.stop()
            except Exception:
                pass

        for handle in handles:
            thread = handle.thread
            if thread and thread.is_alive():
                thread.join(timeout=join_timeout_s)

        # Final safeguard for active runs only; offline engines can otherwise add
        # needless shutdown delay when there is nothing to cancel.
        if handles:
            try:
                stop_all = getattr(self.ctx.vision, "stop_all_sessions", None)
                if callable(stop_all):
                    stop_all()
            except Exception:
                pass
