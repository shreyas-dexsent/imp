"""Implementation for `orchestrator.storage.process_store`."""

import json
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

from orchestrator.storage.paths import DataPaths


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


class ProcessStore:
    def __init__(self, paths: DataPaths):
        self.paths = paths

    def _process_dir(self, station_id: str, process_id: str) -> Path:
        return self.paths.process_dir(station_id, process_id)

    def _process_path(self, station_id: str, process_id: str) -> Path:
        return self._process_dir(station_id, process_id) / "process.json"

    def _canonical_asset_id(self, process_id: str) -> str:
        raw = str(process_id or "").strip()
        if raw.startswith("process-"):
            return "asset-" + raw[len("process-") :]
        return raw

    def _legacy_process_id(self, process_id: str) -> str:
        raw = str(process_id or "").strip()
        if raw.startswith("asset-"):
            return "process-" + raw[len("asset-") :]
        return raw

    def _rewrite_value(self, value: Any, old: str, new: str) -> Any:
        if isinstance(value, str):
            return value.replace(old, new)
        if isinstance(value, list):
            return [self._rewrite_value(item, old, new) for item in value]
        if isinstance(value, dict):
            return {k: self._rewrite_value(v, old, new) for k, v in value.items()}
        return value

    def _rewrite_process_tree(
        self, station_id: str, process_dir: Path, old_id: str, new_id: str
    ) -> None:
        process_path = process_dir / "process.json"
        if process_path.exists():
            try:
                data = json.loads(process_path.read_text(encoding="utf-8"))
                changed = False
                current_id = data.get("asset_id") or data.get("process_id")
                if current_id != new_id:
                    data["asset_id"] = new_id
                    changed = True
                elif "asset_id" not in data:
                    data["asset_id"] = new_id
                    changed = True
                if "process_id" in data:
                    data.pop("process_id", None)
                    changed = True
                if data.get("station_id") != station_id:
                    data["station_id"] = station_id
                    changed = True
                if data.get("name") == "Default Process":
                    data["name"] = "Default Asset"
                    changed = True
                if changed:
                    process_path.write_text(
                        json.dumps(data, indent=2),
                        encoding="utf-8",
                    )
            except Exception:
                pass

        tasks_dir = process_dir / "tasks"
        if not tasks_dir.exists():
            return
        old_rel = f"stations/{station_id}/processes/{old_id}/"
        new_rel = f"stations/{station_id}/assets/{new_id}/"
        old_rel_asset_id = f"stations/{station_id}/assets/{old_id}/"
        for task_path in tasks_dir.glob("*.json"):
            try:
                payload = json.loads(task_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            changed = False
            if isinstance(payload, dict):
                current_id = payload.get("asset_id") or payload.get("process_id")
                if current_id != new_id:
                    payload["asset_id"] = new_id
                    changed = True
                elif "asset_id" not in payload:
                    payload["asset_id"] = new_id
                    changed = True
                if "process_id" in payload:
                    payload.pop("process_id", None)
                    changed = True
            rewritten = self._rewrite_value(payload, old_rel, new_rel)
            rewritten = self._rewrite_value(rewritten, old_rel_asset_id, new_rel)
            if rewritten != payload:
                payload = rewritten
                changed = True
            if changed:
                task_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _merge_or_move(self, src: Path, dst: Path) -> None:
        if src == dst:
            return
        if dst.exists():
            shutil.copytree(src, dst, dirs_exist_ok=True)
            shutil.rmtree(src, ignore_errors=True)
            return
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))

    def _rewrite_run_meta_process_ids(self, id_map: Dict[str, str]) -> None:
        if not id_map:
            return
        for run_dir in self.paths.runs.glob("run-*"):
            meta_path = run_dir / "run_meta.json"
            if not meta_path.exists():
                continue
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            current_id = meta.get("asset_id") or meta.get("process_id")
            if not current_id:
                continue
            rewritten_id = id_map.get(current_id, current_id)
            changed = False
            if meta.get("asset_id") != rewritten_id:
                meta["asset_id"] = rewritten_id
                changed = True
            if "process_id" in meta:
                meta.pop("process_id", None)
                changed = True
            if changed:
                meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    def _normalize_process_doc(
        self, station_id: str, data: Dict[str, Any]
    ) -> Dict[str, Any]:
        normalized = dict(data or {})
        asset_id = self._canonical_asset_id(
            str(normalized.get("asset_id") or normalized.get("process_id") or "")
        )
        if not asset_id:
            return normalized
        normalized["asset_id"] = asset_id
        normalized.pop("process_id", None)
        if station_id:
            normalized["station_id"] = station_id
        if normalized.get("name") == "Default Process":
            normalized["name"] = "Default Asset"
        return normalized

    def _migrate_station_processes(self, station_id: str) -> None:
        station_dir = self.paths.station_dir(station_id)
        assets_root = self.paths.station_processes_dir(station_id)
        legacy_root = self.paths.station_legacy_processes_dir(station_id)
        if not assets_root.exists() and not legacy_root.exists():
            return
        assets_root.mkdir(parents=True, exist_ok=True)
        id_map: Dict[str, str] = {}

        # Rename old id prefix inside assets root if present.
        for entry in list(assets_root.iterdir()) if assets_root.exists() else []:
            if not entry.is_dir():
                continue
            old_id = entry.name
            new_id = self._canonical_asset_id(old_id)
            if new_id == old_id:
                self._rewrite_process_tree(station_id, entry, old_id, new_id)
                continue
            dst_dir = assets_root / new_id
            self._merge_or_move(entry, dst_dir)
            self._rewrite_process_tree(station_id, dst_dir, old_id, new_id)
            id_map[old_id] = new_id

        # Move legacy `processes/` -> `assets/` and rewrite ids/paths.
        if legacy_root.exists():
            for entry in list(legacy_root.iterdir()):
                if not entry.is_dir():
                    continue
                old_id = entry.name
                new_id = self._canonical_asset_id(old_id)
                dst_dir = assets_root / new_id
                self._merge_or_move(entry, dst_dir)
                self._rewrite_process_tree(station_id, dst_dir, old_id, new_id)
                if old_id != new_id:
                    id_map[old_id] = new_id
            try:
                if not any(legacy_root.iterdir()):
                    legacy_root.rmdir()
            except Exception:
                pass

        self._rewrite_run_meta_process_ids(id_map)

        # Ensure the station folder still exists after migrations.
        station_dir.mkdir(parents=True, exist_ok=True)

    def list(self, station_id: str) -> List[Dict[str, Any]]:
        self._migrate_station_processes(station_id)
        root = self.paths.station_processes_dir(station_id)
        processes: List[Dict[str, Any]] = []
        if not root.exists():
            return processes
        for entry in root.iterdir():
            if not entry.is_dir():
                continue
            path = entry / "process.json"
            if not path.exists():
                continue
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
                normalized = self._normalize_process_doc(station_id, raw)
                if normalized != raw:
                    path.write_text(json.dumps(normalized, indent=2), encoding="utf-8")
                processes.append(normalized)
            except Exception:
                continue
        processes.sort(key=lambda p: p.get("asset_id", ""))
        return processes

    def _find_process_dir(self, process_id: str) -> Optional[Path]:
        if not self.paths.stations.exists():
            return None
        asset_id = self._canonical_asset_id(process_id)
        legacy_id = self._legacy_process_id(process_id)
        for station_dir in self.paths.stations.iterdir():
            if not station_dir.is_dir():
                continue
            station_id = station_dir.name
            self._migrate_station_processes(station_id)
            roots = (
                self.paths.station_processes_dir(station_id),
                self.paths.station_legacy_processes_dir(station_id),
            )
            for root in roots:
                if not root.exists():
                    continue
                for candidate_id in (process_id, asset_id, legacy_id):
                    candidate = root / candidate_id
                    if candidate.exists():
                        return candidate
        return None

    def get(self, process_id: str) -> Optional[Dict[str, Any]]:
        proc_dir = self._find_process_dir(process_id)
        if not proc_dir:
            return None
        path = proc_dir / "process.json"
        if not path.exists():
            return None
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            station_id = str(raw.get("station_id") or proc_dir.parent.parent.name)
            normalized = self._normalize_process_doc(station_id, raw)
            if normalized != raw:
                path.write_text(json.dumps(normalized, indent=2), encoding="utf-8")
            return normalized
        except Exception:
            return None

    def create(self, station_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        requested_id = (
            payload.get("asset_id")
            or payload.get("process_id")
            or f"asset-{uuid4().hex[:6]}"
        )
        process_id = self._canonical_asset_id(str(requested_id))
        process_dir = self._process_dir(station_id, process_id)
        process_dir.mkdir(parents=True, exist_ok=True)
        self.paths.process_tasks_dir(station_id, process_id).mkdir(
            parents=True, exist_ok=True
        )
        self.paths.process_objects_dir(station_id, process_id).mkdir(
            parents=True, exist_ok=True
        )
        self.paths.process_poses_dir(station_id, process_id).mkdir(
            parents=True, exist_ok=True
        )
        now = _now_iso()
        data = {
            "asset_id": process_id,
            "station_id": station_id,
            "name": payload.get("name") or process_id,
            "description": payload.get("description", ""),
            "camera_ids": payload.get("camera_ids", []),
            "robot_ids": payload.get("robot_ids", []),
            "created_at": payload.get("created_at", now),
            "updated_at": now,
        }
        if payload.get("task_type") is not None:
            data["task_type"] = payload.get("task_type")
        self._process_path(station_id, process_id).write_text(
            json.dumps(data, indent=2),
            encoding="utf-8",
        )
        return data

    def patch(self, process_id: str, patch: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        data = self.get(process_id)
        if not data:
            return None
        station_id = data.get("station_id")
        if not station_id:
            return None
        data.update({k: v for k, v in patch.items() if v is not None})
        data["updated_at"] = _now_iso()
        asset_id = self._canonical_asset_id(
            str(data.get("asset_id") or data.get("process_id") or process_id)
        )
        data["asset_id"] = asset_id
        data.pop("process_id", None)
        self._process_path(station_id, asset_id).write_text(
            json.dumps(data, indent=2),
            encoding="utf-8",
        )
        return data

    def delete(self, process_id: str) -> bool:
        proc_dir = self._find_process_dir(process_id)
        if not proc_dir or not proc_dir.exists():
            return False
        shutil.rmtree(proc_dir, ignore_errors=True)
        return True

    def ensure_default(
        self, station_id: str, task_type: Optional[str] = None
    ) -> Dict[str, Any]:
        processes = self.list(station_id)
        if processes:
            return processes[0]
        return self.create(
            station_id,
            {
                "asset_id": "asset-1",
                "name": "Default Asset",
                "task_type": task_type,
            },
        )
