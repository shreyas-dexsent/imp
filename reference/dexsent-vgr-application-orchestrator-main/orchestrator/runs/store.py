import json
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _timestamp_run_base() -> str:
    return time.strftime("run-%Y%m%d_%H%M%S", time.localtime())


def _allocate_run_id(root: Path, requested_run_id: Optional[str] = None) -> str:
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


class RunStore:
    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def _run_dir(self, run_id: str) -> Path:
        return self.root / run_id

    def _meta_path(self, run_id: str) -> Path:
        return self._run_dir(run_id) / "run_meta.json"

    def _timeline_path(self, run_id: str) -> Path:
        return self._run_dir(run_id) / "timeline.jsonl"

    def create(self, meta: Dict[str, Any]) -> Dict[str, Any]:
        now = _now_iso()
        run_id = _allocate_run_id(self.root, meta.get("run_id"))
        run_dir = self._run_dir(run_id)
        run_dir.mkdir(parents=True, exist_ok=True)
        asset_id = meta.get("asset_id") or meta.get("process_id")
        data = {
            "run_id": run_id,
            "station_id": meta.get("station_id"),
            "asset_id": asset_id,
            "process_id": asset_id,
            "task_id": meta.get("task_id"),
            "task_type": meta.get("task_type"),
            "state": meta.get("state", "created"),
            "params": meta.get("params") or {},
            "created_at": meta.get("created_at", now),
            "updated_at": now,
            "last_vision_request_id": meta.get("last_vision_request_id"),
            "last_vision_frame_id": meta.get("last_vision_frame_id"),
            "last_vision_timestamp_ns": meta.get("last_vision_timestamp_ns"),
        }
        self._meta_path(run_id).write_text(json.dumps(data, indent=2), encoding="utf-8")
        return data

    def get(self, run_id: str) -> Optional[Dict[str, Any]]:
        path = self._meta_path(run_id)
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None

    def update(self, run_id: str, patch: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        data = self.get(run_id)
        if not data:
            return None
        for key, value in patch.items():
            if value is not None:
                data[key] = value
        data["updated_at"] = _now_iso()
        self._meta_path(run_id).write_text(json.dumps(data, indent=2), encoding="utf-8")
        return data

    def append_event(self, run_id: str, event: Dict[str, Any]) -> None:
        run_dir = self._run_dir(run_id)
        run_dir.mkdir(parents=True, exist_ok=True)
        path = self._timeline_path(run_id)
        payload = dict(event)
        payload.setdefault("timestamp_ns", time.time_ns())
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, separators=(",", ":"), ensure_ascii=False) + "\n")

    def set_last_vision(
        self,
        run_id: str,
        request_id: Optional[str] = None,
        frame_id: Optional[str] = None,
        timestamp_ns: Optional[int] = None,
    ) -> Optional[Dict[str, Any]]:
        return self.update(
            run_id,
            {
                "last_vision_request_id": request_id,
                "last_vision_frame_id": frame_id,
                "last_vision_timestamp_ns": timestamp_ns,
            },
        )

    def list_by_task(self, task_id: str) -> List[Dict[str, Any]]:
        runs: List[Dict[str, Any]] = []
        if not self.root.exists():
            return runs
        for entry in self.root.iterdir():
            if not entry.is_dir():
                continue
            data = self.get(entry.name)
            if not data:
                continue
            if data.get("task_id") == task_id:
                runs.append(data)
        runs.sort(key=lambda item: item.get("created_at", ""), reverse=True)
        return runs

    def get_timeline(self, run_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        path = self._timeline_path(run_id)
        if not path.exists():
            return []
        safe_limit = max(1, int(limit))
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except Exception:
            return []
        events: List[Dict[str, Any]] = []
        for raw in lines[-safe_limit:]:
            try:
                events.append(json.loads(raw))
            except Exception:
                continue
        return events

    def delete(self, run_id: str) -> bool:
        run_dir = self._run_dir(run_id)
        if not run_dir.exists():
            return False
        if not run_dir.is_dir():
            raise ValueError("run_storage_corrupt")
        shutil.rmtree(run_dir)
        return True

