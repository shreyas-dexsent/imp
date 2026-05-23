"""Implementation for `camera_core.shm.shm_buffer`."""

import time
from multiprocessing import shared_memory
from typing import Optional

import numpy as np

from .header import HEADER_SIZE


class ShmImageBuffer:
    """
    Layout: [64B header] + [image bytes]
    """

    @staticmethod
    def _open_shared_memory(
        name: str, *, create: bool, size: Optional[int] = None
    ):
        kwargs = {"name": name, "create": create}
        if size is not None:
            kwargs["size"] = int(size)
        try:
            return shared_memory.SharedMemory(track=False, **kwargs)
        except TypeError:
            return shared_memory.SharedMemory(**kwargs)

    @classmethod
    def _cleanup_existing_segment(
        cls,
        name: str,
        *,
        retries: int = 6,
        delay_s: float = 0.05,
    ) -> None:
        for attempt in range(max(1, int(retries))):
            existing = None
            try:
                existing = cls._open_shared_memory(name, create=False)
            except FileNotFoundError:
                return
            try:
                existing.close()
            finally:
                try:
                    existing.unlink()
                except FileNotFoundError:
                    return
            time.sleep(delay_s * (attempt + 1))

    def __init__(self, name: str, shape, dtype: np.dtype, create: bool):
        self.name = name
        self._creator = bool(create)
        self.shape = tuple(shape)
        self.dtype = np.dtype(dtype)
        self.image_nbytes = int(np.prod(self.shape) * self.dtype.itemsize)
        self.total_size = HEADER_SIZE + self.image_nbytes

        if create:
            self._cleanup_existing_segment(name)
            last_error = None
            for attempt in range(6):
                try:
                    self.shm = self._open_shared_memory(
                        name, create=True, size=self.total_size
                    )
                    break
                except FileExistsError as exc:
                    last_error = exc
                    self._cleanup_existing_segment(name)
                    time.sleep(0.05 * (attempt + 1))
            else:
                raise FileExistsError(
                    f"Shared memory '{name}' already exists and could not be replaced. "
                    "Another camera-core instance may still be running."
                ) from last_error
        else:
            self.shm = self._open_shared_memory(name, create=False)

        self.buf = self.shm.buf

        # Numpy view over image bytes
        self.img = np.ndarray(
            self.shape,
            dtype=self.dtype,
            buffer=self.buf[HEADER_SIZE : HEADER_SIZE + self.image_nbytes],
        )

    def write_header(self, header_bytes: bytes):
        self.buf[:HEADER_SIZE] = header_bytes

    def close(self, unlink: bool = False):
        try:
            self.shm.close()
        except FileNotFoundError:
            pass
        if unlink and self._creator:
            try:
                self.shm.unlink()
            except FileNotFoundError:
                pass

    def unlink(self):
        try:
            self.shm.unlink()
        except FileNotFoundError:
            pass
