"""Implementation for `vision_engine.core.config`."""

import json
from pathlib import Path
from typing import Any, Dict

from vision_engine.core.logging import get_logger

log = get_logger("config")


def load_config(path: str) -> Dict[str, Any]:
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(cfg_path)

    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)

    log.info(f"Loaded config: {cfg_path}")
    return cfg
