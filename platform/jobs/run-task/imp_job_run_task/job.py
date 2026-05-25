"""The actual run-task entrypoint."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

from imp_tasks import RunResult, RunStatus, TaskRun, TaskSpec, compile_task


@dataclass
class RunTaskRequest:
    """Inputs to a single task run."""

    task_path: str
    """Path to the workspace task.yaml. Resolved against the current dir."""

    workspace_root: Optional[str] = None
    """Optional override for the workspace root. Defaults to ``IMP_WORKSPACE``
    (or ``~/.imp/workspace``) -- inherited from the workspace loader."""

    run_id: Optional[str] = None
    """Pin a run id; otherwise the runtime generates one."""

    stage_timeout_s: Optional[float] = 30.0
    """Per-stage timeout passed to :class:`imp_tasks.TaskRun`."""


@dataclass
class RunTaskResult:
    """Output: a JSON-serialisable summary of the run."""

    run_id: str
    task_id: str
    status: str
    elapsed_s: float
    reject_reason: Optional[str]
    stages_completed: list
    stage_durations: list
    meta_path: Optional[str]
    """``workspace/runs/<run_id>/meta.json`` (if the run-store was reachable)."""


def run_task(req: RunTaskRequest) -> RunTaskResult:
    """Execute a task.yaml end-to-end and persist the result."""
    spec = TaskSpec.from_yaml(req.task_path)
    compiled = compile_task(spec)
    run = TaskRun(
        compiled,
        run_id=req.run_id,
        stage_timeout_s=req.stage_timeout_s,
    )
    result: RunResult = run.run()
    meta_path = _persist(result, workspace_root=req.workspace_root)
    return RunTaskResult(
        run_id=result.run_id,
        task_id=result.task_id,
        status=result.status.value,
        elapsed_s=result.elapsed_s,
        reject_reason=result.reject_reason,
        stages_completed=result.stages_completed,
        stage_durations=result.stage_durations,
        meta_path=str(meta_path) if meta_path is not None else None,
    )


def _persist(result: RunResult, *, workspace_root: Optional[str]) -> Optional[Path]:
    """Best-effort write of meta.json. Returns None if run-store can't load."""
    try:
        from imp_service_run_store import write_run
    except Exception:
        # run-store not installed; fall back to a local runs/<id>/meta.json
        # next to the cwd so the run isn't lost.
        meta = Path("runs") / result.run_id / "meta.json"
        meta.parent.mkdir(parents=True, exist_ok=True)
        _dump_meta(meta, result)
        return meta
    return write_run(result, workspace_root=workspace_root)


def _dump_meta(path: Path, result: RunResult) -> None:
    import json

    payload = {
        "run_id": result.run_id,
        "task_id": result.task_id,
        "status": result.status.value,
        "elapsed_s": result.elapsed_s,
        "stages_completed": result.stages_completed,
        "stage_durations": list(result.stage_durations),
        "reject_reason": result.reject_reason,
    }
    path.write_text(json.dumps(payload, indent=2))
