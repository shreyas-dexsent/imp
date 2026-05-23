"""Implementation for `robot_controller.io.command_plane.server`."""

import json

import zmq
from robot_controller.core.controller import RobotController
from robot_controller.logging import get_logger

log = get_logger("command_plane")


class CommandServer:
    def __init__(self, endpoint: str, controller: RobotController):
        self.ctx = zmq.Context.instance()
        self.sock = self.ctx.socket(zmq.REP)
        try:
            self.sock.bind(endpoint)
        except zmq.ZMQError as e:
            if getattr(e, "errno", None) == zmq.EADDRINUSE:
                self.sock.close()
                raise RuntimeError(f"command_plane_addr_in_use: {endpoint}")
            raise
        self.sock.setsockopt(zmq.RCVTIMEO, 500)
        self.controller = controller
        self._running = True

    def serve_forever(self) -> None:
        log.info("Command plane listening")
        while self._running:
            try:
                raw = self.sock.recv_string()
            except zmq.Again:
                continue
            except KeyboardInterrupt:
                break

            try:
                cmd = json.loads(raw)
            except Exception:
                self.sock.send_string(
                    json.dumps({"ok": False, "message": "reject", "reason": "bad_json"})
                )
                continue

            try:
                resp = self.controller.handle_command(cmd)
            except Exception as exc:
                resp = {"ok": False, "message": "reject", "reason": str(exc)}
            self.sock.send_string(
                json.dumps(resp, separators=(",", ":"), ensure_ascii=False)
            )

    def stop(self) -> None:
        self._running = False

    def close(self) -> None:
        self.sock.close()
