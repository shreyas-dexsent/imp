"""Implementation for `robot_controller.io.state_plane.publisher`."""

import json
import time

import zmq
from robot_controller.core.controller import RobotController


class StatePublisher:
    def __init__(
        self, endpoint: str, controller: RobotController, rate_hz: float = 20.0
    ):
        self.ctx = zmq.Context.instance()
        self.sock = self.ctx.socket(zmq.PUB)
        self.sock.bind(endpoint)
        self.controller = controller
        self.rate_hz = max(1.0, float(rate_hz))

    def loop(self) -> None:
        interval = 1.0 / self.rate_hz
        time.sleep(0.5)
        while True:
            state = self.controller.get_state()
            payload = json.dumps(
                {"event": "ROBOT_STATE", **state},
                separators=(",", ":"),
                ensure_ascii=False,
            )
            self.sock.send_string(f"robot {payload}")
            time.sleep(interval)

    def close(self) -> None:
        self.sock.close()
