"""Implementation for `orchestrator.core.tasks`."""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional
from uuid import uuid4

from orchestrator.runs.logger import RunLogger


@dataclass
class TaskState:
    task_id: str
    task_type: str
    recipe_id: Optional[str]
    params: Dict[str, Any] = field(default_factory=dict)
    status: str = "idle"
    run_id: Optional[str] = None


class TaskManager:
    def __init__(self, run_logger: RunLogger):
        self._tasks: Dict[str, TaskState] = {}
        self._runs = run_logger

    def start(
        self, task_type: str, recipe_id: Optional[str], params: Dict[str, Any]
    ) -> TaskState:
        task_id = f"task-{uuid4().hex[:8]}"
        state = TaskState(
            task_id=task_id,
            task_type=task_type,
            recipe_id=recipe_id,
            params=params,
            status="running",
        )
        state.run_id = self._runs.start_run(
            {
                "task_id": task_id,
                "task_type": task_type,
                "recipe_id": recipe_id,
            }
        )
        self._runs.append_event(
            state.run_id, {"event": "TASK_STARTED", "task_id": task_id}
        )
        self._tasks[task_id] = state
        return state

    def stop(self, task_id: str) -> Optional[TaskState]:
        state = self._tasks.get(task_id)
        if not state:
            return None
        state.status = "stopped"
        if state.run_id:
            self._runs.append_event(
                state.run_id, {"event": "TASK_STOPPED", "task_id": task_id}
            )
        return state

    def pause(self, task_id: str) -> Optional[TaskState]:
        state = self._tasks.get(task_id)
        if not state:
            return None
        state.status = "pausing"
        if state.run_id:
            self._runs.append_event(
                state.run_id, {"event": "TASK_PAUSING", "task_id": task_id}
            )
        return state

    def get(self, task_id: str) -> Optional[TaskState]:
        return self._tasks.get(task_id)

    def set_status(
        self, task_id: str, status: str, detail: Optional[Dict[str, Any]] = None
    ) -> Optional[TaskState]:
        state = self._tasks.get(task_id)
        if not state:
            return None
        state.status = status
        if state.run_id:
            event = {"event": f"TASK_{status.upper()}", "task_id": task_id}
            if detail:
                event.update(detail)
            self._runs.append_event(state.run_id, event)
        return state
