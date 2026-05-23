"""Implementation for `orchestrator.core.runs`."""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from orchestrator.runs.store import RunStore


@dataclass
class RunState:
    run_id: str
    station_id: str
    process_id: str
    task_id: str
    task_type: str
    task: Dict[str, Any] = field(default_factory=dict)
    status: str = "created"
    params: Dict[str, Any] = field(default_factory=dict)


class RunManager:
    def __init__(self, store: RunStore):
        self._runs: Dict[str, RunState] = {}
        self._store = store

    def create(self, meta: Dict[str, Any], task_payload: Dict[str, Any]) -> RunState:
        meta = dict(meta)
        meta.setdefault("state", "created")
        stored = self._store.create(meta)
        run_id = stored["run_id"]
        state = RunState(
            run_id=run_id,
            station_id=stored.get("station_id"),
            process_id=stored.get("asset_id") or stored.get("process_id"),
            task_id=stored.get("task_id"),
            task_type=stored.get("task_type"),
            task=task_payload or {},
            status=stored.get("state", "created"),
            params=meta.get("params") or {},
        )
        self._runs[run_id] = state
        self._store.append_event(run_id, {"event": "RUN_CREATED", "run_id": run_id})
        return state

    def get(self, run_id: str) -> Optional[RunState]:
        return self._runs.get(run_id)

    def set_state(
        self, run_id: str, state: str, detail: Optional[Dict[str, Any]] = None
    ) -> None:
        if run_id in self._runs:
            self._runs[run_id].status = state
        event_map = {
            "running": "RUN_START",
            "paused": "RUN_PAUSED",
            "stopping": "RUN_STOPPING",
            "completed": "RUN_COMPLETED",
            "failed": "RUN_FAILED",
            "aborted": "RUN_ABORTED",
            "created": "RUN_CREATED",
        }
        payload = {
            "event": event_map.get(state, f"RUN_{state.upper()}"),
            "run_id": run_id,
        }
        if detail:
            payload.update(detail)
        updates = {"state": state}
        if state in {"created", "completed", "failed", "aborted", "paused", "stopping"}:
            updates["phase"] = state
        self._store.update(run_id, updates)
        self._store.append_event(run_id, payload)

    def set_phase(
        self, run_id: str, phase: str, detail: Optional[Dict[str, Any]] = None
    ) -> None:
        payload = {
            "event": "RUN_PHASE",
            "run_id": run_id,
            "phase": phase,
        }
        if detail:
            payload.update(detail)
        update = {"phase": phase}
        if detail:
            update["phase_detail"] = dict(detail)
        self._store.update(run_id, update)
        self._store.append_event(run_id, payload)

    def set_last_vision(
        self,
        run_id: str,
        request_id: Optional[str] = None,
        frame_id: Optional[str] = None,
        timestamp_ns: Optional[int] = None,
    ) -> None:
        self._store.set_last_vision(
            run_id, request_id=request_id, frame_id=frame_id, timestamp_ns=timestamp_ns
        )

    def delete(self, run_id: str) -> bool:
        self._runs.pop(run_id, None)
        return self._store.delete(run_id)
