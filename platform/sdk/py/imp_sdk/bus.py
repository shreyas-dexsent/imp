"""Zenoh transport for Python imp nodes.

Mirrors crates/bus: applies the same QoS mapping, tags every publication with
its schema, and rejects mismatched schemas on receive (spec §6). Python and
Rust nodes interoperate on the wire because both encode the same protobuf and
agree on key conventions + schema tags.
"""

from __future__ import annotations

import os
from enum import Enum
from typing import Optional

import zenoh

from .schemas import schema_tag, schema_name, SCHEMA_VERSION


def default_config() -> zenoh.Config:
    """imp's default Zenoh config, matching crates/bus. Forces IPv4 TCP listening
    (the zenoh default binds ``tcp/[::]:0``, which fails on IPv4-only hosts).
    Honors ``IMP_ZENOH_CONFIG`` (a json5 file path) to override entirely."""
    path = os.environ.get("IMP_ZENOH_CONFIG")
    if path:
        return zenoh.Config.from_file(path)
    config = zenoh.Config()
    config.insert_json5("listen/endpoints", '["tcp/0.0.0.0:0"]')
    return config


class QosClass(Enum):
    COMMAND = "command"
    FRAME = "frame"
    STATE = "state"
    TELEMETRY = "telemetry"


_QOS = {
    QosClass.COMMAND: (zenoh.Reliability.RELIABLE, zenoh.CongestionControl.BLOCK, zenoh.Priority.REAL_TIME),
    QosClass.STATE: (zenoh.Reliability.RELIABLE, zenoh.CongestionControl.DROP, zenoh.Priority.DATA),
    QosClass.FRAME: (zenoh.Reliability.BEST_EFFORT, zenoh.CongestionControl.DROP, zenoh.Priority.DATA_LOW),
    QosClass.TELEMETRY: (zenoh.Reliability.BEST_EFFORT, zenoh.CongestionControl.DROP, zenoh.Priority.BACKGROUND),
}


class Bus:
    """A connection to the Zenoh fabric."""

    def __init__(self, session: zenoh.Session):
        self._session = session

    @classmethod
    def open(cls, config: Optional[zenoh.Config] = None) -> "Bus":
        return cls(zenoh.open(config if config is not None else default_config()))

    @property
    def session(self) -> zenoh.Session:
        return self._session

    def close(self) -> None:
        self._session.close()

    def __enter__(self) -> "Bus":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def publisher(self, key: str, qos: QosClass) -> "Publisher":
        rel, cc, prio = _QOS[qos]
        inner = self._session.declare_publisher(
            key, reliability=rel, congestion_control=cc, priority=prio
        )
        return Publisher(inner)

    def put(self, key: str, msg, qos: QosClass) -> None:
        # Session.put() (unlike declare_publisher) does not take `reliability`;
        # reliability is a publisher/transport setting. Use publisher() when it
        # matters for a stream.
        _, cc, prio = _QOS[qos]
        self._session.put(
            key,
            msg.SerializeToString(),
            attachment=schema_tag(msg).encode(),
            congestion_control=cc,
            priority=prio,
        )

    def subscribe(self, key: str, msg_type) -> "TypedSub":
        """Typed subscription that validates the schema tag and decodes to ``msg_type``."""
        return TypedSub(self._session.declare_subscriber(key), msg_type)

    def subscribe_raw(self, key: str) -> zenoh.Subscriber:
        return self._session.declare_subscriber(key)


class Publisher:
    def __init__(self, inner: zenoh.Publisher):
        self._inner = inner
        # The publisher is bound to one key; resolve the tag lazily on first put
        # from the message it is given (a publisher only ever carries one type).

    def put(self, msg) -> None:
        self._inner.put(msg.SerializeToString(), attachment=schema_tag(msg).encode())


class TypedSub:
    def __init__(self, inner: zenoh.Subscriber, msg_type):
        self._inner = inner
        self._type = msg_type
        self._expected = msg_type.DESCRIPTOR.full_name

    def recv(self):
        """Block for the next message matching ``msg_type``; drop mismatches (spec §6)."""
        while True:
            sample = self._inner.recv()
            att = sample.attachment
            if att is None:
                continue
            tag = bytes(att).decode(errors="replace")
            name = schema_name(tag)
            try:
                version = int(tag.rsplit("/", 1)[1])
            except (IndexError, ValueError):
                continue
            if name != self._expected or version > SCHEMA_VERSION:
                continue  # schema mismatch -> dropped (missing topic)
            msg = self._type()
            msg.ParseFromString(bytes(sample.payload))
            return msg
