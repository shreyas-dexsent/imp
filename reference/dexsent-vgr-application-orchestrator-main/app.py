"""Application entrypoint for `vgr-orchestrator-2.5d`."""

import argparse
import logging
import time
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from orchestrator.api.routes import create_router
from orchestrator.config import load_config
from orchestrator.core.runtime import build_context
from orchestrator.logging import setup_logging


def build_app(config_path: str) -> FastAPI:
    cfg = load_config(config_path)
    setup_logging(cfg.get("runtime", {}).get("log_level", "INFO"))

    ctx = build_context(cfg)
    app = FastAPI(title="DexSent VGR Orchestrator")
    app.state.ctx = ctx
    app.include_router(create_router(ctx))

    ui_dir = Path(__file__).parent / "orchestrator" / "ui"
    assets_dir = ui_dir / "assets"
    ui_path = ui_dir / "index.html"
    calib_path = ui_dir / "calibration.html"
    perception_path = ui_dir / "perception_debug.html"
    _NO_CACHE = {"Cache-Control": "no-store, no-cache, must-revalidate", "Pragma": "no-cache"}
    _STATIC_EXTENSIONS = {".js", ".css", ".html"}

    if assets_dir.exists():
        static_files = StaticFiles(directory=str(assets_dir))

        @app.get("/ui/assets/{path:path}")
        async def serve_static(path: str, request: Request) -> Response:
            scope = dict(request.scope)
            scope["path"] = "/" + path
            response = await static_files.get_response(path, scope)
            if Path(path).suffix.lower() in _STATIC_EXTENSIONS:
                for k, v in _NO_CACHE.items():
                    response.headers[k] = v
            return response

    @app.get("/ui")
    def ui() -> FileResponse:
        return FileResponse(ui_path, headers=_NO_CACHE)

    @app.get("/ui/calibration")
    def calibration_ui() -> FileResponse:
        return FileResponse(calib_path, headers=_NO_CACHE)

    @app.get("/ui/perception_debug")
    @app.get("/ui/perception_debug.html")
    def perception_ui() -> FileResponse:
        return FileResponse(perception_path, headers=_NO_CACHE)

    @app.on_event("shutdown")
    def shutdown_resources() -> None:
        shutdown_log = logging.getLogger("orchestrator.shutdown")

        def _timed(label: str, fn) -> None:
            start = time.perf_counter()
            try:
                fn()
            except Exception:
                pass
            elapsed = time.perf_counter() - start
            if elapsed > 0.25:
                shutdown_log.info("%s took %.2fs", label, elapsed)

        # Stop noisy background subscribers first so Ctrl+C does not keep printing
        # telemetry while shutdown is waiting for worker threads to drain.
        for attr in ("vision_cache", "vision_results", "camera_cache"):
            obj = getattr(ctx, attr, None)
            close_fn = getattr(obj, "close", None)
            if callable(close_fn):
                _timed(f"{attr}.close", close_fn)

        # Stop active task threads; keep timeout short for responsive shutdown.
        if getattr(ctx, "executor", None) is not None:
            _timed("executor.close", lambda: ctx.executor.close(join_timeout_s=0.2))

        # Finally close control clients/sockets.
        for attr in ("vision", "robot"):
            obj = getattr(ctx, attr, None)
            close_fn = getattr(obj, "close", None)
            if callable(close_fn):
                _timed(f"{attr}.close", close_fn)

    return app


app = None if __name__ == "__main__" else build_app("configs/orchestrator.local.yaml")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/orchestrator.local.yaml")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8100)
    args = ap.parse_args()

    runtime_app = build_app(args.config)
    uvicorn.run(runtime_app, host=args.host, port=args.port)
