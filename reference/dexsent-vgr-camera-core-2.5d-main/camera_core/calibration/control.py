"""Implementation for `camera_core.calibration.control`."""

from typing import Any, Dict, List, Optional

import uvicorn
from camera_core.calibration.server import CalibrationServerManager
from camera_core.fusion.control_service import FusionControlService
from fastapi import FastAPI, HTTPException


class CalibrationControl:
    def __init__(
        self,
        host: str,
        port: int,
        server_host: str,
        server_port: int,
        pub_endpoint: str,
        topic: str,
        log_level: str,
        robot_command_endpoint: str,
        robot_timeout_ms: int = 20000,
        cors_origins: Optional[List[str]] = None,
        push_endpoint: Optional[str] = None,
    ) -> None:
        self.manager = CalibrationServerManager(
            host=server_host,
            port=server_port,
            pub_endpoint=pub_endpoint,
            topic=topic,
            log_level=log_level,
            cors_origins=cors_origins,
        )
        # Fusion publishes FRAME_READY events into the broker's PULL socket
        # (the same one camera pipelines use), NOT the PUB endpoint that
        # subscribers read from. Connecting a PUSH socket to the PUB side
        # never completes a ZMTP handshake, so `IMMEDIATE=1` refuses every
        # send and the fusion call dies with `event_bus_publish_timeout`.
        # If the caller didn't supply a push endpoint, fall back to
        # `pub_endpoint` so behaviour matches the old (broken) default,
        # making it obvious that the wiring is wrong.
        fusion_push_endpoint = str(push_endpoint or pub_endpoint)
        self.fusion = FusionControlService(
            push_addr=fusion_push_endpoint,
            topic=topic,
            robot_command_endpoint=robot_command_endpoint,
            robot_timeout_ms=robot_timeout_ms,
        )
        self.host = host
        self.port = port
        self.log_level = log_level

    def create_app(self) -> FastAPI:
        app = FastAPI(title="Camera Core Calibration Control")

        @app.get("/health")
        def health() -> Dict[str, Any]:
            return {"status": "alive"}

        @app.get("/calibration/status")
        def calibration_status() -> Dict[str, Any]:
            return self.manager.status()

        @app.post("/calibration/start")
        def calibration_start() -> Dict[str, Any]:
            return self.manager.start()

        @app.post("/calibration/stop")
        def calibration_stop() -> Dict[str, Any]:
            return self.manager.stop()

        @app.post("/fusion/capture_pose")
        def fusion_capture_pose(payload: Dict[str, Any]) -> Dict[str, Any]:
            try:
                return self.fusion.capture_pose_fusion(payload)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            except Exception as exc:
                raise HTTPException(status_code=500, detail=str(exc)) from exc

        return app

    def run(self) -> None:
        app = self.create_app()
        uvicorn.run(
            app, host=self.host, port=self.port, log_level=self.log_level.lower()
        )


def run_control_server(
    host: str,
    port: int,
    server_host: str,
    server_port: int,
    pub_endpoint: str,
    topic: str,
    log_level: str,
    robot_command_endpoint: str,
    robot_timeout_ms: int = 20000,
    cors_origins: Optional[List[str]] = None,
    push_endpoint: Optional[str] = None,
) -> None:
    control = CalibrationControl(
        host=host,
        port=port,
        server_host=server_host,
        server_port=server_port,
        pub_endpoint=pub_endpoint,
        topic=topic,
        log_level=log_level,
        robot_command_endpoint=robot_command_endpoint,
        robot_timeout_ms=robot_timeout_ms,
        cors_origins=cors_origins,
        push_endpoint=push_endpoint,
    )
    control.run()
