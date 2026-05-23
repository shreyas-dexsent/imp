"""Implementation for `camera_core.shm.header`."""

import struct

# HEADER (64 bytes):
# timestamp (8) + sequence_id (8) + calib_version (4) + status_flags (4) + reserved (40)

HEADER_FMT = "QQII40s"
HEADER_SIZE = 64

FLAG_VALID = 0x01
FLAG_CORRUPTED = 0x02
FLAG_OVERRUN = 0x04
FLAG_CALIB_STALE = 0x08


def pack_header(
    timestamp_ns: int, sequence_id: int, calib_version: int, status_flags: int
) -> bytes:
    return struct.pack(
        HEADER_FMT, timestamp_ns, sequence_id, calib_version, status_flags, b"\x00" * 40
    )


def unpack_header(buf: bytes):
    ts, seq, calib, flags, _ = struct.unpack(HEADER_FMT, buf[:HEADER_SIZE])
    return ts, seq, calib, flags
