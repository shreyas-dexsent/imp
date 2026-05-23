"""Implementation for `orchestrator.storage.task_store`."""

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

import yaml
from orchestrator.storage.paths import DataPaths
from orchestrator.storage.process_store import ProcessStore


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _safe_name(raw: str) -> str:
    cleaned = []
    for ch in (raw or "").strip():
        if ch.isalnum() or ch in ("-", "_"):
            cleaned.append(ch)
    return "".join(cleaned)


def _deep_merge_dict(base: Dict[str, Any], patch: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base or {})
    for key, value in (patch or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge_dict(merged[key], value)
        else:
            merged[key] = value
    return merged


def _sanitize_pose_reference(value: Any) -> Any:
    if not isinstance(value, dict):
        return value
    name = str(value.get("name") or value.get("pose_name") or "").strip()
    pose_path = str(value.get("pose_path") or value.get("path") or "").strip()
    if not pose_path and name:
        pose_path = f"poses/{name}.json"
    if not name and pose_path:
        name = Path(pose_path).stem
    ref: Dict[str, Any] = {}
    if name:
        ref["name"] = name
    if pose_path:
        ref["pose_path"] = pose_path
    return ref if ref else {}


def _sanitize_task_payload(task_payload: Any) -> Any:
    # Hand-eye is station-level calibration and should not be persisted in task JSON.
    if not isinstance(task_payload, dict):
        return task_payload
    task = dict(task_payload)
    robot = task.get("robot")
    if isinstance(robot, dict):
        robot = dict(robot)
        robot.pop("hand_eye", None)
        robot.pop("hand_eye_source", None)
        for key in ("capture_pose", "place_pose"):
            if key in robot:
                robot[key] = _sanitize_pose_reference(robot.get(key))
        if isinstance(robot.get("capture_pose"), dict) and not robot["capture_pose"]:
            robot.pop("capture_pose", None)
        if isinstance(robot.get("place_pose"), dict) and not robot["place_pose"]:
            robot.pop("place_pose", None)
        task["robot"] = robot
    return task


class TaskStore:
    def __init__(self, paths: DataPaths, processes: ProcessStore):
        self.paths = paths
        self.processes = processes

    def _task_dir(self, station_id: str, process_id: str) -> Path:
        return self.paths.process_tasks_dir(station_id, process_id)

    def _task_path(self, station_id: str, process_id: str, task_id: str) -> Path:
        return self._task_dir(station_id, process_id) / f"{task_id}.json"

    def _find_task_path(self, task_id: str) -> Optional[Path]:
        if not self.paths.stations.exists():
            return None
        for station_dir in self.paths.stations.iterdir():
            if not station_dir.is_dir():
                continue
            roots = (
                self.paths.station_processes_dir(station_dir.name),
                self.paths.station_legacy_processes_dir(station_dir.name),
            )
            for proc_root in roots:
                if not proc_root.exists():
                    continue
                for proc_dir in proc_root.iterdir():
                    if not proc_dir.is_dir():
                        continue
                    candidate = proc_dir / "tasks" / f"{task_id}.json"
                    if candidate.exists():
                        return candidate
        return None

    def _task_stub_from_path(self, path: Path) -> Dict[str, Any]:
        task_id = path.stem
        station_id = ""
        asset_id = ""
        parts = list(path.parts)
        try:
            idx = parts.index("stations")
            if idx + 1 < len(parts):
                station_id = parts[idx + 1]
        except ValueError:
            pass
        for key in ("assets", "processes"):
            try:
                idx = parts.index(key)
                if idx + 1 < len(parts):
                    asset_id = parts[idx + 1]
                    break
            except ValueError:
                continue
        return {
            "task_id": task_id,
            "asset_id": asset_id,
            "station_id": station_id,
            "name": task_id,
            "description": "invalid_task_json",
            "task_type": "pick_place_demo",
            "task": {},
            "created_at": _now_iso(),
            "updated_at": _now_iso(),
            "_parse_error": True,
        }

    def _normalize_task_doc(
        self, payload: Dict[str, Any], station_id: str = "", asset_id: str = ""
    ) -> Dict[str, Any]:
        data = dict(payload or {})
        resolved_asset_id = str(
            asset_id or data.get("asset_id") or data.get("process_id") or ""
        ).strip()
        if resolved_asset_id:
            data["asset_id"] = resolved_asset_id
        data.pop("process_id", None)
        if station_id:
            data["station_id"] = station_id
        data["task"] = _sanitize_task_payload(data.get("task"))
        return data

    def _read_legacy_recipe(self, path: Path) -> Optional[Dict[str, Any]]:
        try:
            if path.suffix.lower() in (".yaml", ".yml"):
                return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            if path.suffix.lower() == ".json":
                return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None
        return None

    def _import_legacy_recipes(self, station_id: str, process_id: str) -> None:
        tasks_dir = self._task_dir(station_id, process_id)
        marker = tasks_dir / ".legacy_imported"
        if marker.exists():
            return
        legacy_root = self.paths.legacy_recipes
        if not legacy_root.exists():
            marker.write_text("no-legacy", encoding="utf-8")
            return
        tasks_dir.mkdir(parents=True, exist_ok=True)
        for path in legacy_root.iterdir():
            if not path.is_file():
                continue
            if path.suffix.lower() not in (".yaml", ".yml", ".json"):
                continue
            data = self._read_legacy_recipe(path)
            if not isinstance(data, dict):
                continue
            recipe_id = data.get("recipe_id") or path.stem
            task_id = _safe_name(str(recipe_id))
            if not task_id:
                continue
            task_path = self._task_path(station_id, process_id, task_id)
            if task_path.exists():
                continue
            now = _now_iso()
            payload = {
                "task_id": task_id,
                "asset_id": process_id,
                "station_id": station_id,
                "name": data.get("recipe_id") or task_id,
                "description": data.get("description", ""),
                "task": data,
                "created_at": now,
                "updated_at": now,
            }
            task_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        marker.write_text("ok", encoding="utf-8")

    def list(self, process_id: str) -> List[Dict[str, Any]]:
        process = self.processes.get(process_id)
        if not process:
            return []
        asset_id = str(process.get("asset_id") or process.get("process_id") or process_id)
        station_id = process.get("station_id")
        if not station_id:
            return []
        tasks_dir = self._task_dir(station_id, asset_id)
        if not tasks_dir.exists() or not any(tasks_dir.glob("*.json")):
            self._import_legacy_recipes(station_id, asset_id)
        tasks: List[Dict[str, Any]] = []
        if not tasks_dir.exists():
            return tasks
        for path in tasks_dir.glob("*.json"):
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
                normalized = self._normalize_task_doc(raw, station_id=station_id, asset_id=asset_id)
                if normalized != raw:
                    path.write_text(json.dumps(normalized, indent=2), encoding="utf-8")
                tasks.append(normalized)
            except Exception:
                tasks.append(self._task_stub_from_path(path))
                continue
        tasks.sort(key=lambda t: t.get("task_id", ""))
        return tasks

    def get(self, task_id: str) -> Optional[Dict[str, Any]]:
        path = self._find_task_path(task_id)
        if not path:
            return None
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            parts = list(path.parts)
            station_id = ""
            asset_id = ""
            for key in ("stations",):
                try:
                    idx = parts.index(key)
                    if idx + 1 < len(parts):
                        station_id = parts[idx + 1]
                except ValueError:
                    pass
            for key in ("assets", "processes"):
                try:
                    idx = parts.index(key)
                    if idx + 1 < len(parts):
                        asset_id = parts[idx + 1]
                        break
                except ValueError:
                    continue
            normalized = self._normalize_task_doc(raw, station_id=station_id, asset_id=asset_id)
            if normalized != raw:
                path.write_text(json.dumps(normalized, indent=2), encoding="utf-8")
            return normalized
        except Exception:
            return self._task_stub_from_path(path)

    def create(self, process_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        process = self.processes.get(process_id)
        if not process:
            raise ValueError("process_not_found")
        asset_id = str(process.get("asset_id") or process.get("process_id") or process_id)
        station_id = process.get("station_id")
        if not station_id:
            raise ValueError("process_missing_station")
        task_id = payload.get("task_id") or f"task-{uuid4().hex[:6]}"
        task_id = _safe_name(str(task_id))
        if not task_id:
            raise ValueError("invalid_task_id")
        now = _now_iso()
        task_payload = payload.get("task")
        if task_payload is None:
            task_payload = payload.get("recipe")
        if task_payload is None:
            task_payload = payload.get("params")
        if task_payload is None:
            task_payload = {}
        task_payload = _sanitize_task_payload(task_payload)
        task_type = (
            payload.get("task_type") or process.get("task_type") or "pick_place_demo"
        )
        data = {
            "task_id": task_id,
            "asset_id": asset_id,
            "station_id": station_id,
            "name": payload.get("name") or task_id,
            "description": payload.get("description", ""),
            "task_type": task_type,
            "task": task_payload,
            "created_at": payload.get("created_at", now),
            "updated_at": now,
        }
        self._task_dir(station_id, asset_id).mkdir(parents=True, exist_ok=True)
        self._task_path(station_id, asset_id, task_id).write_text(
            json.dumps(data, indent=2),
            encoding="utf-8",
        )
        return data

    def patch(self, task_id: str, patch: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        data = self.get(task_id)
        if not data:
            return None
        station_id = data.get("station_id")
        asset_id = data.get("asset_id") or data.get("process_id")
        if not station_id or not asset_id:
            return None
        if "task" in patch or "recipe" in patch or "params" in patch:
            incoming_task = patch.get("task")
            if incoming_task is None:
                incoming_task = patch.get("recipe")
            if incoming_task is None:
                incoming_task = patch.get("params")
            if incoming_task is None:
                incoming_task = {}
            existing_task = (
                data.get("task") if isinstance(data.get("task"), dict) else {}
            )
            if isinstance(existing_task, dict) and isinstance(incoming_task, dict):
                data["task"] = _deep_merge_dict(existing_task, incoming_task)
            else:
                data["task"] = incoming_task
            data["task"] = _sanitize_task_payload(data.get("task"))
        for key in ("name", "description"):
            if key in patch and patch[key] is not None:
                data[key] = patch[key]
        if "task_type" in patch and patch.get("task_type") is not None:
            data["task_type"] = patch.get("task_type")
        data["asset_id"] = str(asset_id)
        data.pop("process_id", None)
        data["task"] = _sanitize_task_payload(data.get("task"))
        data["updated_at"] = _now_iso()
        self._task_path(station_id, asset_id, data["task_id"]).write_text(
            json.dumps(data, indent=2),
            encoding="utf-8",
        )
        return data

    def delete(self, task_id: str) -> bool:
        path = self._find_task_path(task_id)
        if not path or not path.exists():
            return False
        try:
            path.unlink()
        except Exception:
            return False
        return True
