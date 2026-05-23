"""Implementation for `camera_core.health.server`."""

import time
from typing import Any, Dict

from fastapi import FastAPI


def create_health_app(get_status_fn):
    app = FastAPI()

    @app.get("/health")
    def health() -> Dict[str, Any]:
        st = get_status_fn()
        st["ts_ns"] = time.time_ns()
        return st

    return app
