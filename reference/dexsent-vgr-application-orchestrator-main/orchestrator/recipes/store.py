"""Implementation for `orchestrator.recipes.store`."""

import json
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


class RecipeStore:
    def __init__(self, root: Path):
        self.root = root
        self._cache: Dict[str, Dict[str, Any]] = {}

    def _read_file(self, path: Path) -> Dict[str, Any]:
        if path.suffix.lower() in (".yaml", ".yml"):
            with path.open("r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        if path.suffix.lower() == ".json":
            with path.open("r", encoding="utf-8") as f:
                return json.load(f)
        raise ValueError(f"Unsupported recipe format: {path.suffix}")

    def load_from_path(
        self, path: Path, recipe_id: Optional[str] = None
    ) -> Dict[str, Any]:
        data = self._read_file(path)
        rid = recipe_id or data.get("recipe_id") or path.stem
        if not rid:
            raise ValueError("recipe_id is required")
        data["recipe_id"] = rid
        self._cache[rid] = data
        return data

    def load_inline(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        rid = payload.get("recipe_id")
        if not rid:
            raise ValueError("recipe_id is required")
        self._cache[rid] = payload
        return payload

    def get(self, recipe_id: str) -> Optional[Dict[str, Any]]:
        return self._cache.get(recipe_id)
