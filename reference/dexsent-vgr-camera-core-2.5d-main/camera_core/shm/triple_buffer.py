"""Implementation for `camera_core.shm.triple_buffer`."""

from dataclasses import dataclass
from typing import List

import numpy as np

from .shm_buffer import ShmImageBuffer


@dataclass
class TripleBufferState:
    write_idx: int = 0
    last_complete_idx: int = 1
    standby_idx: int = 2


class TripleBuffer:
    @staticmethod
    def cleanup_named(name_prefix: str, suffixes: List[str]) -> List[str]:
        names = [f"{name_prefix}_{s}" for s in suffixes]
        for name in names:
            ShmImageBuffer._cleanup_existing_segment(name)
        return names

    def __init__(
        self, name_prefix: str, suffixes: List[str], shape, dtype, create: bool
    ):
        self.names = [f"{name_prefix}_{s}" for s in suffixes]
        if create:
            self.cleanup_named(name_prefix, suffixes)
        self.buffers = []
        try:
            for name in self.names:
                self.buffers.append(
                    ShmImageBuffer(name, shape=shape, dtype=np.dtype(dtype), create=create)
                )
        except Exception:
            for buffer in self.buffers:
                try:
                    buffer.close(unlink=create)
                except Exception:
                    pass
            if create:
                for name in self.names:
                    try:
                        ShmImageBuffer._cleanup_existing_segment(name)
                    except Exception:
                        pass
            raise
        self.state = TripleBufferState(0, 1, 2)

    def rotate(self):
        # A -> B -> C -> A
        self.state.last_complete_idx = self.state.write_idx
        self.state.write_idx = self.state.standby_idx
        self.state.standby_idx = (self.state.standby_idx + 1) % 3

    def get_write_buffer(self) -> ShmImageBuffer:
        return self.buffers[self.state.write_idx]

    def get_last_complete_name(self) -> str:
        return self.names[self.state.last_complete_idx]

    def close(self, unlink: bool = False):
        for b in self.buffers:
            b.close(unlink=unlink)
