"""Implementation for `orchestrator.vision.results`."""

import json
import logging
import time
from typing import Any, Dict, Optional

import zmq

log = logging.getLogger("vision_subscriber")


class VisionResultSubscriber:
    def __init__(self, endpoint: str, topic: str = "vision"):
        log.debug(f"[vision_subscriber] Connecting to {endpoint}, topic={topic}")
        self.ctx = zmq.Context.instance()
        self.sock = self.ctx.socket(zmq.SUB)
        self.sock.connect(endpoint)
        self.sock.setsockopt_string(zmq.SUBSCRIBE, topic)
        log.debug("[vision_subscriber] Connected and subscribed")

        self.poller = zmq.Poller()
        self.poller.register(self.sock, zmq.POLLIN)

        self._recv_count = 0
        self._parse_count = 0
        self._error_count = 0

    def recv(self, timeout_ms: int = 100) -> Optional[Dict[str, Any]]:
        try:
            events = dict(self.poller.poll(timeout_ms))
        except zmq.ZMQError:
            return None
        if self.sock not in events:
            return None

        self._recv_count += 1
        try:
            raw = self.sock.recv_string()
            self._parse_count += 1
            if " " not in raw:
                self._error_count += 1
                if self._error_count <= 5:
                    log.warning(
                        f"[vision_subscriber] Invalid message format (no space): {raw[:100]}"
                    )
                return None
            _, payload = raw.split(" ", 1)
            try:
                parsed = json.loads(payload)
                # Log first few and then every 100 messages
                if self._parse_count <= 5 or self._parse_count % 100 == 0:
                    log.debug(
                        f"[vision_subscriber] Parsed message #{self._parse_count}: event={parsed.get('event')}"
                    )
                return parsed
            except json.JSONDecodeError as e:
                self._error_count += 1
                if self._error_count <= 5:
                    log.warning(f"[vision_subscriber] JSON parse error: {str(e)[:100]}")
                return None
        except Exception as e:
            self._error_count += 1
            if self._error_count <= 5:
                log.warning(f"[vision_subscriber] Recv error: {str(e)}")
            return None

    def wait_for_result(
        self,
        request_id: str,
        timeout_s: float,
        stop_event=None,
    ) -> Optional[Dict[str, Any]]:
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            if stop_event is not None and stop_event.is_set():
                return None
            evt = self.recv(timeout_ms=200)
            if not evt:
                continue
            if evt.get("event") != "VISION_RESULT":
                continue
            if request_id and evt.get("request_id") != request_id:
                continue
            return evt
        return None

    def close(self) -> None:
        try:
            self.sock.close(0)
        except Exception:
            pass
