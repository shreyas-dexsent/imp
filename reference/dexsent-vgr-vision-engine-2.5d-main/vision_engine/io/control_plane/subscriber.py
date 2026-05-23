"""Implementation for `vision_engine.io.control_plane.subscriber`."""

import json

import zmq


class ZmqSubscriber:
    def __init__(self, endpoint: str, topic: str):
        self.ctx = zmq.Context.instance()
        self.sock = self.ctx.socket(zmq.SUB)
        self.sock.connect(endpoint)
        self.sock.setsockopt_string(zmq.SUBSCRIBE, topic)

        self.poller = zmq.Poller()
        self.poller.register(self.sock, zmq.POLLIN)
        self._closed = False

    def recv(self, timeout_ms=100):
        if self._closed:
            return None, None
        try:
            events = dict(self.poller.poll(timeout_ms))
        except zmq.ZMQError:
            # Normal during shutdown if the socket is closed while poll() is waiting.
            return None, None
        if self.sock not in events:
            return None, None

        try:
            raw = self.sock.recv_string(zmq.NOBLOCK)

            # Check if message has topic and payload
            if " " not in raw:
                print(
                    f"[subscriber] Warning: Received message without topic separator: {raw[:100]}"
                )
                return None, None

            topic, payload = raw.split(" ", 1)

            # Parse JSON payload
            return topic, json.loads(payload)
        except (json.JSONDecodeError, ValueError) as e:
            print(f"[subscriber] Warning: Failed to parse message - {e}")
            print(
                f"[subscriber] Raw message: {raw[:200] if 'raw' in locals() else 'N/A'}"
            )
            return None, None

    def close(self):
        if self._closed:
            return
        self._closed = True
        try:
            self.poller.unregister(self.sock)
        except Exception:
            pass
        try:
            self.sock.close(0)
        except Exception:
            pass
