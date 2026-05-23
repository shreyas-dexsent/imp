"""Implementation for `orchestrator.storage.station_store`."""

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


class StationStore:
    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def _station_dir(self, station_id: str) -> Path:
        return self.root / station_id

    def _station_path(self, station_id: str) -> Path:
        return self._station_dir(station_id) / "station.json"

    def list(self) -> List[Dict[str, Any]]:
        stations: List[Dict[str, Any]] = []
        if not self.root.exists():
            return stations
        for entry in self.root.iterdir():
            if not entry.is_dir():
                continue
            path = entry / "station.json"
            if not path.exists():
                continue
            try:
                stations.append(json.loads(path.read_text(encoding="utf-8")))
            except Exception:
                continue
        stations.sort(key=lambda s: s.get("station_id", ""))
        return stations

    def get(self, station_id: str) -> Optional[Dict[str, Any]]:
        path = self._station_path(station_id)
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None

    def create(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        station_id = payload.get("station_id") or f"station-{uuid4().hex[:6]}"
        name = payload.get("name") or station_id
        station_dir = self._station_dir(station_id)
        station_dir.mkdir(parents=True, exist_ok=True)
        now = _now_iso()
        data = {
            "station_id": station_id,
            "name": name,
            "description": payload.get("description", ""),
            "camera_ids": payload.get("camera_ids", []),
            "robot_ids": payload.get("robot_ids", []),
            "created_at": payload.get("created_at", now),
            "updated_at": now,
        }
        self._station_path(station_id).write_text(
            json.dumps(data, indent=2), encoding="utf-8"
        )
        return data

    def patch(self, station_id: str, patch: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        data = self.get(station_id)
        if not data:
            return None
        data.update({k: v for k, v in patch.items() if v is not None})
        data["updated_at"] = _now_iso()
        self._station_path(station_id).write_text(
            json.dumps(data, indent=2), encoding="utf-8"
        )
        return data

    def ensure_default(self) -> Dict[str, Any]:
        stations = self.list()
        if stations:
            return stations[0]
        return self.create({"station_id": "station-1", "name": "Default Station"})
