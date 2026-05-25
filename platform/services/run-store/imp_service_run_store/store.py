"""``workspace/runs/<id>/`` reader + writer."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, List, Optional


def _workspace_root(explicit: Optional[str]) -> Path:
    """Resolve the workspace root: explicit arg > IMP_WORKSPACE > ~/.imp/workspace."""
    if explicit:
        return Path(explicit).expanduser().resolve()
    env = os.environ.get("IMP_WORKSPACE")
    if env:
        return Path(env).expanduser().resolve()
    return (Path.home() / ".imp" / "workspace").resolve()


@dataclass(frozen=True)
class RunRecord:
    """One row in ``workspace/runs/<run_id>/meta.json``."""

    run_id: str
    task_id: str
    status: str
    elapsed_s: float
    stages_completed: List[str]
    stage_durations: List[List[Any]]  # [(name, elapsed_s), ...]
    reject_reason: Optional[str]
    meta_path: Path


def _result_to_payload(result: Any) -> dict:
    """Serialise a :class:`imp_tasks.RunResult` (or compatible) to JSON."""
    return {
        "run_id": result.run_id,
        "task_id": result.task_id,
        "status": getattr(result.status, "value", str(result.status)),
        "elapsed_s": result.elapsed_s,
        "stages_completed": list(result.stages_completed),
        "stage_durations": [list(pair) for pair in result.stage_durations],
        "reject_reason": result.reject_reason,
    }


def write_run(result: Any, *, workspace_root: Optional[str] = None) -> Path:
    """Persist a ``RunResult`` to ``workspace/runs/<run_id>/meta.json``.

    Creates the run directory if missing. Returns the meta.json path so
    callers can surface the location to the operator.
    """
    root = _workspace_root(workspace_root)
    run_dir = root / "runs" / result.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    meta = run_dir / "meta.json"
    meta.write_text(json.dumps(_result_to_payload(result), indent=2))
    return meta


def read_run(run_id: str, *, workspace_root: Optional[str] = None) -> RunRecord:
    """Return the ``RunRecord`` for ``run_id`` from the workspace store."""
    root = _workspace_root(workspace_root)
    meta = root / "runs" / run_id / "meta.json"
    if not meta.is_file():
        raise FileNotFoundError(f"no run {run_id!r} (looked at {meta})")
    payload = json.loads(meta.read_text())
    return RunRecord(
        run_id=payload["run_id"],
        task_id=payload["task_id"],
        status=payload["status"],
        elapsed_s=payload["elapsed_s"],
        stages_completed=payload.get("stages_completed", []),
        stage_durations=payload.get("stage_durations", []),
        reject_reason=payload.get("reject_reason"),
        meta_path=meta,
    )


def list_runs(*, workspace_root: Optional[str] = None,
              task_id: Optional[str] = None) -> List[RunRecord]:
    """All runs in the workspace store, optionally filtered by ``task_id``."""
    root = _workspace_root(workspace_root) / "runs"
    if not root.is_dir():
        return []
    out: List[RunRecord] = []
    for sub in sorted(root.iterdir()):
        meta = sub / "meta.json"
        if not meta.is_file():
            continue
        try:
            rec = read_run(sub.name, workspace_root=str(root.parent))
        except Exception:
            continue
        if task_id is not None and rec.task_id != task_id:
            continue
        out.append(rec)
    return out


@dataclass
class RunStore:
    """Thin OO wrapper over the module-level helpers."""

    workspace_root: Optional[str] = None

    def write(self, result: Any) -> Path:
        return write_run(result, workspace_root=self.workspace_root)

    def read(self, run_id: str) -> RunRecord:
        return read_run(run_id, workspace_root=self.workspace_root)

    def list(self, task_id: Optional[str] = None) -> Iterable[RunRecord]:
        return list_runs(workspace_root=self.workspace_root, task_id=task_id)
