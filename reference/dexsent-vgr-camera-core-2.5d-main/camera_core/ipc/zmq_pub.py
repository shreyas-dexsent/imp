"""Implementation for `camera_core.ipc.zmq_pub`."""

# import zmq
# import json

# class ZmqPublisher:
#     def __init__(self, bind_addr: str, topic: str):
#         self.ctx = zmq.Context.instance()
#         self.sock = self.ctx.socket(zmq.PUB)
#         self.sock.bind(bind_addr)
#         self.topic = topic

#     def publish(self, msg: dict):
#         payload = json.dumps(msg, separators=(",", ":"), ensure_ascii=False)
#         # topic framing: "topic {json}"
#         self.sock.send_string(f"{self.topic} {payload}")

#     def close(self):
#         self.sock.close(0)

############################
# import zmq
# import json

# class ZmqPublisher:
#     def __init__(self, bind_addr: str, topic: str):
#         self.ctx = zmq.Context.instance()
#         self.sock = self.ctx.socket(zmq.PUB)
#         self.sock.bind(bind_addr)
#         self.topic = topic

#     def publish(self, payload: dict):
#         msg = f"{self.topic} {json.dumps(payload)}"
#         self.sock.send_string(msg)

#     def close(self):
#         self.sock.close(0)
#         self.ctx.term()

###########

# camera_core/ipc/zmq_pub.py

import json
import time

import zmq


class ZmqPublisher:
    """
    Per-thread publisher using PUSH socket (connect to event bus).
    Do NOT share this across threads (ZMQ sockets are not thread-safe).

    The event bus will receive messages via PULL socket and forward to subscribers via PUB.

    Notes on the send path:
    - `zmq.IMMEDIATE=1` refuses to queue messages while no peer has completed
      its TCP + ZMTP handshake. Combined with `SNDTIMEO`, a send that races
      the handshake (or arrives when all peers are gone) raises `zmq.Again`.
      That is the correct behaviour for camera drivers at full frame rate
      (we don't want to build up a huge in-process backlog when the broker
      is dead), but it is brittle for low-rate callers like fusion that may
      send their very first message within milliseconds of `connect()`.
    - `wait_for_connect_s` lets callers give libzmq time to finish the
      handshake before the first send. `settle_first_send` retries the
      first `publish()` a few times on `Again`, which covers the residual
      race window without permanently relaxing `IMMEDIATE`.
    """

    def __init__(
        self,
        push_addr: str,
        send_timeout_ms: int = 1000,
        *,
        wait_for_connect_s: float = 0.0,
        settle_first_send: bool = False,
        send_hwm: int = 1000,
        immediate: bool = True,
    ):
        self.ctx = zmq.Context.instance()
        self.sock = self.ctx.socket(zmq.PUSH)
        self.sock.setsockopt(zmq.LINGER, 0)
        self.sock.setsockopt(zmq.IMMEDIATE, 1 if immediate else 0)
        self.sock.setsockopt(zmq.SNDHWM, int(send_hwm))
        self.sock.setsockopt(zmq.SNDTIMEO, max(1, int(send_timeout_ms)))
        self.sock.connect(push_addr)
        if wait_for_connect_s and wait_for_connect_s > 0:
            time.sleep(float(wait_for_connect_s))
        self._first_send = True
        self._settle_first_send = bool(settle_first_send)

    def publish(self, msg: dict):
        payload = json.dumps(msg, separators=(",", ":"), ensure_ascii=False)
        if self._first_send and self._settle_first_send:
            # Retry the very first send a handful of times to absorb any
            # residual PUSH<->PULL handshake race. Each attempt waits up to
            # SNDTIMEO, so a 5s budget is already plenty; this just adds
            # retries in case SNDTIMEO is short.
            deadline = time.monotonic() + 2.0
            last_exc = None
            while True:
                try:
                    self.sock.send_string(payload)
                    self._first_send = False
                    return
                except zmq.Again as exc:
                    last_exc = exc
                    if time.monotonic() >= deadline:
                        break
                    time.sleep(0.05)
            raise RuntimeError("event_bus_publish_timeout") from last_exc
        try:
            self.sock.send_string(payload)
        except zmq.Again as exc:
            raise RuntimeError("event_bus_publish_timeout") from exc

    def close(self):
        try:
            self.sock.close(0)
        except Exception:
            pass
