"""Implementation for `vision_engine.io.data_plane.shm_reader`."""

import struct
import sys
from multiprocessing import resource_tracker
from multiprocessing.shared_memory import SharedMemory
from threading import Lock
from typing import Dict

import numpy as np

HDR_STRUCT = struct.Struct("<QQII")
HEADER_BYTES = 64
FLAG_VALID = 1


class ShmReader:
    def __init__(self, header_bytes: int = HEADER_BYTES):
        self.header_bytes = header_bytes
        self._lock = Lock()
        self._segments: Dict[str, SharedMemory] = {}
        self._unregistered_names: set[str] = set()
        self._cache_enabled = not sys.platform.startswith("win")

    def _attach(self, shm_name: str) -> SharedMemory:
        key = str(shm_name)
        if not self._cache_enabled:
            try:
                shm = SharedMemory(name=key, create=False, track=False)
            except TypeError:
                shm = SharedMemory(name=key, create=False)
                try:
                    if shm._name not in self._unregistered_names:
                        resource_tracker.unregister(shm._name, "shared_memory")
                        self._unregistered_names.add(shm._name)
                except Exception:
                    pass
            return shm
        with self._lock:
            shm = self._segments.get(key)
            if shm is not None:
                return shm
            try:
                shm = SharedMemory(name=key, create=False, track=False)
            except TypeError:
                shm = SharedMemory(name=key, create=False)
                try:
                    # Python 3.11 fallback path: unregister once per attached segment.
                    if shm._name not in self._unregistered_names:
                        resource_tracker.unregister(shm._name, "shared_memory")
                        self._unregistered_names.add(shm._name)
                except Exception:
                    pass
            self._segments[key] = shm
            return shm

    def _drop(self, shm_name: str) -> None:
        key = str(shm_name)
        with self._lock:
            shm = self._segments.pop(key, None)
        if shm is None:
            return
        try:
            internal = getattr(shm, "_name", None)
            if internal:
                self._unregistered_names.discard(str(internal))
        except Exception:
            pass
        try:
            shm.close()
        except Exception:
            pass

    def close(self) -> None:
        with self._lock:
            segments = list(self._segments.values())
            self._segments.clear()
        for shm in segments:
            try:
                shm.close()
            except Exception:
                pass

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def read_image(self, shm_name, shape, dtype):
        dtype = np.dtype(dtype)
        img_bytes = int(np.prod(shape) * dtype.itemsize)

        for attempt in range(2):
            shm = self._attach(str(shm_name))
            try:
                buf = shm.buf
                ts, seq, calib, flags = HDR_STRUCT.unpack_from(buf, 0)

                if (flags & FLAG_VALID) == 0:
                    return None

                img_view = buf[self.header_bytes : self.header_bytes + img_bytes]
                img = np.frombuffer(img_view, dtype=dtype).reshape(shape).copy()

                return {
                    "timestamp_ns": ts,
                    "sequence_id": seq,
                    "calib_version": calib,
                    "flags": flags,
                    "image": img,
                }
            except (OSError, BufferError, ValueError):
                # Segment might have been recreated after camera-core restart.
                self._drop(str(shm_name))
                if attempt == 0:
                    continue
                return None
            finally:
                try:
                    del img_view
                except Exception:
                    pass
                try:
                    del buf
                except Exception:
                    pass
                if not self._cache_enabled:
                    try:
                        shm.close()
                    except Exception:
                        pass

        return None

    def read_rgb(self, shm_name, shape, dtype):
        out = self.read_image(shm_name, shape, dtype)
        if out is None:
            return None
        return {
            "timestamp_ns": out["timestamp_ns"],
            "sequence_id": out["sequence_id"],
            "calib_version": out["calib_version"],
            "flags": out["flags"],
            "rgb": out["image"],
        }
