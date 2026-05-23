"""Implementation for `orchestrator.config`."""

from pathlib import Path
from typing import Any, Dict

import yaml
from orchestrator.logging import get_logger

log = get_logger("config")


def load_config(path: str) -> Dict[str, Any]:
    cfg_path = Path(path).resolve()
    if not cfg_path.exists():
        raise FileNotFoundError(cfg_path)

    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    log.info(f"Loaded config: {cfg_path}")
    return cfg
