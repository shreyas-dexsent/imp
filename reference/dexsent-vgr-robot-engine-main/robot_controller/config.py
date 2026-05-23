"""Implementation for `robot_controller.config`."""

import json
from pathlib import Path
from typing import Any, Dict

from robot_controller.logging import get_logger

log = get_logger("config")


def load_config(path: str) -> Dict[str, Any]:
    cfg_path = Path(path).resolve()
    if not cfg_path.exists():
        raise FileNotFoundError(cfg_path)

    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)

    log.info(f"Loaded config: {cfg_path}")
    return cfg
