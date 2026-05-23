"""
Shared Memory Result Writer

Writes vision processing results to shared memory for IPC communication.
Results include both metadata (JSON) and optional image data.

Memory Layout:
- Header (64 bytes):
  - timestamp_ns (8 bytes, uint64)
  - sequence_id (8 bytes, uint64)
  - result_data_size (4 bytes, uint32) - size of JSON result data
  - status_flags (4 bytes, uint32)
  - reserved (40 bytes)

- Result Data (variable size):
  - JSON string (UTF-8 encoded)

Status Flags:
- FLAG_VALID (1) - result is valid and ready to read
- FLAG_ERROR (2) - processing error occurred
"""

import json
import struct
from multiprocessing.shared_memory import SharedMemory
from typing import Any, Dict, Optional

import numpy as np

# Header structure: timestamp_ns, sequence_id, result_data_size, status_flags
HDR_STRUCT = struct.Struct("<QQII")
HEADER_BYTES = 64

# Status flags
FLAG_VALID = 1
FLAG_ERROR = 2

# Maximum result JSON size (can be adjusted based on needs)
MAX_RESULT_SIZE = 4096  # 4KB for result JSON


class ShmResultWriter:
    """
    Writes vision processing results to shared memory.
    Each request thread can have its own shared memory segment.
    """

    def __init__(self, request_id: str, max_result_size: int = MAX_RESULT_SIZE):
        """
        Initialize shared memory writer for results.

        Args:
            request_id: Unique identifier for this request (used as SHM name)
            max_result_size: Maximum size for result JSON data
        """
        self.request_id = request_id
        self.max_result_size = max_result_size
        self.total_size = HEADER_BYTES + max_result_size
        self.shm: Optional[SharedMemory] = None
        self.shm_name = f"vgr_result_{request_id.replace('-', '_')}"

        # Create shared memory segment
        try:
            self.shm = SharedMemory(
                name=self.shm_name, create=True, size=self.total_size
            )
            # Initialize header with zeros
            self.shm.buf[:HEADER_BYTES] = b"\x00" * HEADER_BYTES
            print(
                f"[ShmResultWriter] Created SHM: {self.shm_name} ({self.total_size} bytes)"
            )
        except FileExistsError:
            # SHM already exists, try to attach to it
            try:
                self.shm = SharedMemory(name=self.shm_name, create=False)
                print(f"[ShmResultWriter] Attached to existing SHM: {self.shm_name}")
            except Exception as e:
                print(f"[ShmResultWriter] Failed to attach to existing SHM: {e}")
                raise
        except Exception as e:
            print(f"[ShmResultWriter] Failed to create SHM: {e}")
            raise

    def write_result(
        self,
        timestamp_ns: int,
        sequence_id: int,
        result: Dict[str, Any],
        request_id: str,
        camera_id: str,
        module: str,
        process_time_ms: float,
        error: Optional[str] = None,
    ) -> bool:
        """
        Write a vision result to shared memory.

        Args:
            timestamp_ns: Frame timestamp in nanoseconds
            sequence_id: Frame sequence ID
            result: Vision processing result dict
            request_id: Request identifier
            camera_id: Camera identifier
            module: Module name
            process_time_ms: Processing time in milliseconds
            error: Optional error message

        Returns:
            True if write successful, False otherwise
        """
        if self.shm is None:
            return False

        try:
            # Prepare result data
            result_data = {
                "request_id": request_id,
                "camera_id": camera_id,
                "module": module,
                "timestamp_ns": timestamp_ns,
                "sequence_id": sequence_id,
                "process_time_ms": process_time_ms,
                "result": result,
            }

            if error:
                result_data["error"] = error

            # Serialize to JSON
            result_json = json.dumps(result_data, separators=(",", ":"))
            result_bytes = result_json.encode("utf-8")
            result_size = len(result_bytes)

            if result_size > self.max_result_size:
                print(
                    f"[ShmResultWriter] Result too large: {result_size} > {self.max_result_size}"
                )
                return False

            # Set status flags
            status_flags = FLAG_VALID
            if error:
                status_flags |= FLAG_ERROR

            # Write header
            buf = self.shm.buf
            HDR_STRUCT.pack_into(
                buf, 0, timestamp_ns, sequence_id, result_size, status_flags
            )

            # Write result data
            buf[HEADER_BYTES : HEADER_BYTES + result_size] = result_bytes

            return True

        except Exception as e:
            print(f"[ShmResultWriter] Error writing result: {e}")
            return False

    def close(self):
        """Close and cleanup shared memory."""
        if self.shm:
            try:
                shm_name = self.shm.name
                self.shm.close()
                self.shm.unlink()  # Remove the shared memory
                print(f"[ShmResultWriter] Closed and unlinked SHM: {shm_name}")
            except Exception as e:
                print(f"[ShmResultWriter] Error closing SHM: {e}")
            finally:
                self.shm = None

    def __del__(self):
        """Cleanup on deletion."""
        self.close()

    def get_shm_name(self) -> str:
        """Get the shared memory segment name."""
        return self.shm_name
