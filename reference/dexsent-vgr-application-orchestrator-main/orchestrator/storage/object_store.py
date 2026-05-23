"""Implementation for `orchestrator.storage.object_store`."""

import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

from orchestrator.storage.paths import DataPaths
from orchestrator.storage.process_store import ProcessStore


def _safe_name(raw: str) -> str:
    cleaned = []
    for ch in (raw or "").strip():
        if ch.isalnum() or ch in ("-", "_"):
            cleaned.append(ch)
    return "".join(cleaned)


def _parse_template_name(raw: str) -> Optional[Dict[str, str]]:
    if not raw:
        return None
    value = raw.strip()
    if not value:
        return None
    if "." in value:
        base, ext = value.rsplit(".", 1)
    else:
        base, ext = value, "png"
    base = _safe_name(base)
    ext = ext.lower().strip(".")
    if not base or ext not in ("png", "jpg", "jpeg"):
        return None
    return {"base": base, "ext": ext}


class ObjectLibraryStore:
    def __init__(self, paths: DataPaths, processes: ProcessStore):
        self.paths = paths
        self.processes = processes

    def _objects_dir(self, station_id: str, process_id: str) -> Path:
        return self.paths.process_objects_dir(station_id, process_id)

    def _templates_dir(self, station_id: str, process_id: str, object_id: str) -> Path:
        return self._objects_dir(station_id, process_id) / object_id / "templates"

    def _metadata_file(self, station_id: str, process_id: str, object_id: str) -> Path:
        return self._objects_dir(station_id, process_id) / object_id / "metadata.json"

    def _import_legacy_objects(self, station_id: str, process_id: str) -> None:
        objects_dir = self._objects_dir(station_id, process_id)
        marker = objects_dir / ".legacy_imported"
        if marker.exists():
            return
        legacy_root = self.paths.legacy_objects
        objects_dir.mkdir(parents=True, exist_ok=True)
        if legacy_root.exists():
            for obj_dir in legacy_root.iterdir():
                if not obj_dir.is_dir():
                    continue
                dest = objects_dir / obj_dir.name
                if dest.exists():
                    continue
                shutil.copytree(obj_dir, dest)
        marker.write_text("ok", encoding="utf-8")

    def list(self, process_id: str) -> List[Dict[str, Any]]:
        process = self.processes.get(process_id)
        if not process:
            return []
        station_id = process.get("station_id")
        if not station_id:
            return []
        objects_dir = self._objects_dir(station_id, process_id)
        objects = []
        if not objects_dir.exists():
            return objects
        for obj_dir in objects_dir.iterdir():
            if not obj_dir.is_dir():
                continue
            templates = []
            tmpl_dir = obj_dir / "templates"
            if tmpl_dir.exists():
                for path in tmpl_dir.iterdir():
                    if not path.is_file():
                        continue
                    ext = path.suffix.lower().strip(".")
                    if ext not in ("png", "jpg", "jpeg"):
                        continue
                    templates.append(
                        {
                            "name": path.stem,
                            "ext": ext,
                            "filename": path.name,
                            "bytes": path.stat().st_size,
                        }
                    )
            templates.sort(key=lambda t: t.get("name", ""))

            # Load metadata if available
            metadata = self._load_metadata(station_id, process_id, obj_dir.name)

            obj_entry = {"object_id": obj_dir.name, "templates": templates}
            if metadata:
                obj_entry["metadata"] = metadata

            objects.append(obj_entry)
        objects.sort(key=lambda o: o.get("object_id", ""))
        return objects

    def get_metadata(self, process_id: str, object_id: str) -> Optional[Dict[str, Any]]:
        """Get object metadata (geometry, etc)."""
        process = self.processes.get(process_id)
        if not process:
            return None
        station_id = process.get("station_id")
        if not station_id:
            return None
        return self._load_metadata(station_id, process_id, object_id)

    def _load_metadata(
        self, station_id: str, process_id: str, object_id: str
    ) -> Optional[Dict[str, Any]]:
        """Load metadata from object's metadata.json file."""
        metadata_file = self._metadata_file(station_id, process_id, object_id)
        if not metadata_file.exists():
            return None
        try:
            with open(metadata_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    def save_metadata(
        self, process_id: str, object_id: str, metadata: Dict[str, Any]
    ) -> Path:
        """Save object metadata."""
        process = self.processes.get(process_id)
        if not process:
            raise ValueError("process_not_found")
        station_id = process.get("station_id")
        if not station_id:
            raise ValueError("process_missing_station")

        obj_id = _safe_name(object_id)
        if not obj_id:
            raise ValueError("invalid_object_id")

        # Ensure object directory exists
        obj_dir = self._objects_dir(station_id, process_id) / obj_id
        obj_dir.mkdir(parents=True, exist_ok=True)

        # Write metadata
        metadata_file = self._metadata_file(station_id, process_id, obj_id)
        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        return metadata_file

    def save_template(
        self, process_id: str, object_id: str, template_name: str, ext: str, data: bytes
    ) -> Path:
        process = self.processes.get(process_id)
        if not process:
            raise ValueError("process_not_found")
        station_id = process.get("station_id")
        if not station_id:
            raise ValueError("process_missing_station")
        obj_id = _safe_name(object_id)
        tmpl_name = _safe_name(template_name)
        if not obj_id or not tmpl_name:
            raise ValueError("invalid_object_or_template")
        ext = ext.lower().strip(".") or "png"
        if ext not in ("png", "jpg", "jpeg"):
            raise ValueError("unsupported_extension")
        tmpl_dir = self._templates_dir(station_id, process_id, obj_id)
        tmpl_dir.mkdir(parents=True, exist_ok=True)
        path = tmpl_dir / f"{tmpl_name}.{ext}"
        path.write_bytes(data)
        return path

    def rename_object(self, process_id: str, object_id: str, new_id: str) -> str:
        process = self.processes.get(process_id)
        if not process:
            raise ValueError("process_not_found")
        station_id = process.get("station_id")
        if not station_id:
            raise ValueError("process_missing_station")
        src = _safe_name(object_id)
        dst = _safe_name(new_id)
        if not src or not dst:
            raise ValueError("invalid_object_id")
        src_path = self._objects_dir(station_id, process_id) / src
        if not src_path.exists():
            raise FileNotFoundError("object_not_found")
        dst_path = self._objects_dir(station_id, process_id) / dst
        if dst_path.exists():
            raise FileExistsError("object_already_exists")
        src_path.rename(dst_path)
        return dst

    def delete_object(self, process_id: str, object_id: str) -> str:
        process = self.processes.get(process_id)
        if not process:
            raise ValueError("process_not_found")
        station_id = process.get("station_id")
        if not station_id:
            raise ValueError("process_missing_station")
        name = _safe_name(object_id)
        if not name:
            raise ValueError("invalid_object_id")
        path = self._objects_dir(station_id, process_id) / name
        if not path.exists():
            raise FileNotFoundError("object_not_found")
        shutil.rmtree(path)
        return name

    def rename_template(
        self, process_id: str, object_id: str, template_name: str, new_name: str
    ) -> str:
        process = self.processes.get(process_id)
        if not process:
            raise ValueError("process_not_found")
        station_id = process.get("station_id")
        if not station_id:
            raise ValueError("process_missing_station")
        obj_id = _safe_name(object_id)
        if not obj_id:
            raise ValueError("invalid_object_id")
        src_info = _parse_template_name(template_name)
        if not src_info:
            raise ValueError("invalid_template_name")
        dst_info = _parse_template_name(new_name)
        if not dst_info:
            raise ValueError("invalid_new_template")
        tmpl_dir = self._templates_dir(station_id, process_id, obj_id)
        src = tmpl_dir / f"{src_info['base']}.{src_info['ext']}"
        if not src.exists():
            raise FileNotFoundError("template_not_found")
        dst = tmpl_dir / f"{dst_info['base']}.{dst_info['ext']}"
        if dst.exists():
            raise FileExistsError("template_already_exists")
        src.rename(dst)
        return dst.name

    def delete_template(
        self, process_id: str, object_id: str, template_name: str
    ) -> str:
        process = self.processes.get(process_id)
        if not process:
            raise ValueError("process_not_found")
        station_id = process.get("station_id")
        if not station_id:
            raise ValueError("process_missing_station")
        obj_id = _safe_name(object_id)
        if not obj_id:
            raise ValueError("invalid_object_id")
        info = _parse_template_name(template_name)
        if not info:
            raise ValueError("invalid_template_name")
        tmpl_dir = self._templates_dir(station_id, process_id, obj_id)
        path = tmpl_dir / f"{info['base']}.{info['ext']}"
        if not path.exists():
            raise FileNotFoundError("template_not_found")
        path.unlink()
        return path.name
