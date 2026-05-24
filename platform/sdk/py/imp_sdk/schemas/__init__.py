"""Generated imp wire schemas + the self-describing schema-tag layer.

`imp_pb2` is generated from crates/schemas/proto/imp.proto by protoc; the same
proto produces the Rust (prost) types, so the wire format is identical.
"""

from . import imp_pb2  # noqa: F401

# Schema version (bump on any field addition; spec §7).
SCHEMA_VERSION = 1


def schema_tag(msg) -> str:
    """Versioned tag for a protobuf message instance or class, e.g. ``imp.Pose6D/1``."""
    return f"{msg.DESCRIPTOR.full_name}/{SCHEMA_VERSION}"


def schema_name(tag: str) -> str:
    """Strip the version: ``imp.Pose6D/1`` -> ``imp.Pose6D``."""
    return tag.rsplit("/", 1)[0]
