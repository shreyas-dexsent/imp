"""Implementation for `orchestrator.runs.logger`."""

import json
import time
from pathlib import Path
from typing import Any, Dict


def _timestamp_run_base() -> str:
    return time.strftime("run-%Y%m%d_%H%M%S", time.localtime())


def _allocate_run_id(root: Path, requested_run_id: str) -> str:
    run_id = str(requested_run_id or "").strip()
    if run_id:
        return run_id
    base = _timestamp_run_base()
    candidate = base
    counter = 1
    while (root / candidate).exists():
        candidate = f"{base}_{counter:02d}"
        counter += 1
    return candidate


class RunLogger:
    def __init__(self, runs_root: Path):
        self.runs_root = runs_root

    def start_run(self, meta: Dict[str, Any]) -> str:
        run_id = _allocate_run_id(self.runs_root, meta.get("run_id"))
        run_dir = self.runs_root / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        meta_path = run_dir / "run_meta.json"
        meta = dict(meta)
        meta["run_id"] = run_id
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return run_id

    def append_event(self, run_id: str, event: Dict[str, Any]) -> None:
        run_dir = self.runs_root / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        log_path = run_dir / "timeline.jsonl"
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, separators=(",", ":"), ensure_ascii=False) + "\n")
