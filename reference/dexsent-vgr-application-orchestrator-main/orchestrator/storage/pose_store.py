"""Implementation for `orchestrator.storage.pose_store`."""

import json
import shutil
from pathlib import Path
from typing import Any, Dict, List

from orchestrator.storage.paths import DataPaths
from orchestrator.storage.process_store import ProcessStore


def _safe_name(raw: str) -> str:
    cleaned = []
    for ch in (raw or "").strip():
        if ch.isalnum() or ch in ("-", "_"):
            cleaned.append(ch)
    return "".join(cleaned)


class PoseStore:
    def __init__(self, paths: DataPaths, processes: ProcessStore):
        self.paths = paths
        self.processes = processes

    def _poses_dir(self, station_id: str, process_id: str) -> Path:
        return self.paths.process_poses_dir(station_id, process_id)

    def _import_legacy_poses(self, station_id: str, process_id: str) -> None:
        poses_dir = self._poses_dir(station_id, process_id)
        marker = poses_dir / ".legacy_imported"
        if marker.exists():
            return
        legacy_dir = self.paths.legacy_station_poses_dir()
        poses_dir.mkdir(parents=True, exist_ok=True)
        if legacy_dir.exists():
            for path in legacy_dir.glob("*.json"):
                dest = poses_dir / path.name
                if dest.exists():
                    continue
                shutil.copy2(path, dest)
        marker.write_text("ok", encoding="utf-8")

    def list(self, process_id: str) -> List[Dict[str, Any]]:
        process = self.processes.get(process_id)
        if not process:
            return []
        station_id = process.get("station_id")
        if not station_id:
            return []
        poses_dir = self._poses_dir(station_id, process_id)
        if not poses_dir.exists() or not any(poses_dir.glob("*.json")):
            self._import_legacy_poses(station_id, process_id)
        poses = []
        if not poses_dir.exists():
            return poses
        for path in poses_dir.glob("*.json"):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                payload["name"] = path.stem
                poses.append(payload)
            except Exception:
                continue
        poses.sort(key=lambda p: p.get("name", ""))
        return poses

    def save(self, process_id: str, name: str, payload: Dict[str, Any]) -> Path:
        process = self.processes.get(process_id)
        if not process:
            raise ValueError("process_not_found")
        station_id = process.get("station_id")
        if not station_id:
            raise ValueError("process_missing_station")
        clean = _safe_name(name)
        if not clean:
            raise ValueError("pose_name_required")
        poses_dir = self._poses_dir(station_id, process_id)
        poses_dir.mkdir(parents=True, exist_ok=True)
        path = poses_dir / f"{clean}.json"
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path

    def delete(self, process_id: str, name: str) -> str:
        process = self.processes.get(process_id)
        if not process:
            raise ValueError("process_not_found")
        station_id = process.get("station_id")
        if not station_id:
            raise ValueError("process_missing_station")
        clean = _safe_name(name)
        if not clean:
            raise ValueError("pose_name_required")
        poses_dir = self._poses_dir(station_id, process_id)
        path = poses_dir / f"{clean}.json"
        if not path.exists():
            raise FileNotFoundError("pose_not_found")
        path.unlink()
        return clean

    def rename(self, process_id: str, name: str, new_name: str) -> str:
        process = self.processes.get(process_id)
        if not process:
            raise ValueError("process_not_found")
        station_id = process.get("station_id")
        if not station_id:
            raise ValueError("process_missing_station")

        clean = _safe_name(name)
        clean_new = _safe_name(new_name)
        if not clean:
            raise ValueError("pose_name_required")
        if not clean_new:
            raise ValueError("new_pose_name_required")
        if clean == clean_new:
            return clean_new

        poses_dir = self._poses_dir(station_id, process_id)
        src = poses_dir / f"{clean}.json"
        dst = poses_dir / f"{clean_new}.json"
        if not src.exists():
            raise FileNotFoundError("pose_not_found")
        if dst.exists():
            raise FileExistsError("pose_already_exists")
        src.rename(dst)
        return clean_new
