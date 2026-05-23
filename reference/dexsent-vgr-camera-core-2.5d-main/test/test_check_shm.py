"""Tests for `test.test_check_shm`."""

from multiprocessing import shared_memory

import numpy as np

name = "cam_cam_webcam_rgb_A"  # try A/B/C

shm = shared_memory.SharedMemory(name=name, create=False)
print("SHM found:", name)
print("Size:", shm.size)

# Peek header (first 64 bytes)
header = bytes(shm.buf[:64])
print("Header bytes:", header)

shm.close()
