"""Implementation for `vision_engine.io.control_plane.publisher`."""

import json

import zmq


class ZmqPublisher:
    """
    PUSH socket that sends raw JSON to an event-bus PULL socket.
    The event bus will add the topic prefix when re-publishing via PUB socket.
    """

    def __init__(self, endpoint: str, topic: str = "vision"):
        self.ctx = zmq.Context.instance()
        self.sock = self.ctx.socket(zmq.PUSH)
        self.sock.connect(endpoint)
        self.topic = topic  # Not used anymore, but kept for backwards compatibility

    def publish(self, msg: dict):
        # Send raw JSON without topic prefix.
        # The event bus can optionally override the topic using __topic__.
        payload_obj = msg
        if "__topic__" not in msg and self.topic:
            payload_obj = dict(msg)
            payload_obj["__topic__"] = self.topic
        payload = json.dumps(payload_obj, separators=(",", ":"), ensure_ascii=False)
        self.sock.send_string(payload)

    def close(self):
        self.sock.close()
