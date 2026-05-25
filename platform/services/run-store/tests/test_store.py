"""Pure-stdlib tests for the workspace-backed run store."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import pytest

from imp_service_run_store import list_runs, read_run, write_run
from imp_service_run_store.store import RunStore


class _Status(str, Enum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"


@dataclass
class _FakeResult:
    """Stand-in for imp_tasks.RunResult that satisfies write_run."""

    run_id: str
    task_id: str
    status: _Status
    elapsed_s: float
    stages_completed: list
    stage_durations: list
    reject_reason: str | None = None


def test_write_then_read_round_trips(tmp_path: Path):
    result = _FakeResult(
        run_id="demo-12345678",
        task_id="pick_place",
        status=_Status.SUCCEEDED,
        elapsed_s=1.234,
        stages_completed=["acquire", "solve"],
        stage_durations=[("acquire", 0.4), ("solve", 0.8)],
    )
    meta_path = write_run(result, workspace_root=str(tmp_path))
    assert meta_path.exists()
    assert meta_path.parent == tmp_path / "runs" / "demo-12345678"

    rec = read_run("demo-12345678", workspace_root=str(tmp_path))
    assert rec.run_id == "demo-12345678"
    assert rec.task_id == "pick_place"
    assert rec.status == "succeeded"
    assert rec.elapsed_s == pytest.approx(1.234)
    assert rec.stages_completed == ["acquire", "solve"]


def test_read_missing_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        read_run("nope", workspace_root=str(tmp_path))


def test_list_filters_by_task_id(tmp_path: Path):
    write_run(_FakeResult("a-001", "task_a", _Status.SUCCEEDED, 1.0, [], []),
              workspace_root=str(tmp_path))
    write_run(_FakeResult("b-001", "task_b", _Status.FAILED, 0.5, [], [],
                          reject_reason="timeout"),
              workspace_root=str(tmp_path))
    write_run(_FakeResult("a-002", "task_a", _Status.SUCCEEDED, 0.9, [], []),
              workspace_root=str(tmp_path))

    everything = list_runs(workspace_root=str(tmp_path))
    assert {r.run_id for r in everything} == {"a-001", "a-002", "b-001"}

    only_a = list_runs(workspace_root=str(tmp_path), task_id="task_a")
    assert {r.run_id for r in only_a} == {"a-001", "a-002"}

    fail = [r for r in everything if r.status == "failed"]
    assert len(fail) == 1
    assert fail[0].reject_reason == "timeout"


def test_run_store_class_wrapper(tmp_path: Path):
    store = RunStore(workspace_root=str(tmp_path))
    result = _FakeResult("x-1", "tx", _Status.SUCCEEDED, 0.1, [], [])
    store.write(result)
    got = store.read("x-1")
    assert got.task_id == "tx"
    assert [r.run_id for r in store.list()] == ["x-1"]


def test_empty_workspace_lists_zero(tmp_path: Path):
    assert list_runs(workspace_root=str(tmp_path)) == []
