"""
Camera-agnostic frame source for the fusion pipeline.

Subscribes to a running camera-core process over its ZMQ event bus and reads
RGB + depth out of the shared-memory triple buffer that camera-core publishes.
The fusion module never opens a camera SDK directly - it just consumes
whatever camera-core is currently serving (D405, D435, Basler, FLIR, webcam,
or any future driver), as configured by the YAML you launched camera-core
with, e.g.:

    conda activate vgr-camera
    cd ~/imp/dexsent-vgr-camera-core-2.5d
    python -m camera_core.main --config config/cam_realsense_d405.yaml

Wire format (defined by camera-core):

    Topic frame on tcp://127.0.0.1:5555 (PUB), prefixed by 'camera ':
        {
          "event": "FRAME_READY",
          "camera_id": "...",
          "sequence_id": int,
          "timestamp_ns": int,
          "rgb_shm":   "<posix_shm_name>",
          "rgb_shape":  [H, W, 3],
          "rgb_dtype": "uint8",
          "depth_shm":  "<posix_shm_name>",   # only present in rgbd mode
          "depth_shape":[H, W],
          "depth_dtype":"float32" | "uint16",
          "camera_data": {
             "K": [[fx,0,cx],[0,fy,cy],[0,0,1]],
             "resolution": [H, W],
             "intrinsics": {fx, fy, cx, cy, ...},
             "depth_scale_m_per_unit": float,
             ...
          }
        }

    Each shm segment has a 64-byte header (camera_core.shm.header) followed by
    the raw image bytes. We read past the header and copy out a fresh numpy
    array (camera-core may rotate the buffer at any moment).
"""

from __future__ import annotations

import json
import logging
import time
from multiprocessing import shared_memory
from typing import Optional, Tuple

import numpy as np
import open3d as o3d
import zmq

from ..shm.header import HEADER_SIZE


log = logging.getLogger("multiview_fusion.camera_core_client")


class CameraCoreClient:
    """Subscribe to camera-core and grab one aligned RGBD frame on demand.

    Parameters
    ----------
    bus_addr : str
        ZMQ PUB endpoint exposed by camera-core's event bus. Matches
        `ipc.zmq_pub` in the active camera config (default tcp://127.0.0.1:5555).
    topic : str
        Topic prefix camera-core uses (default 'camera').
    camera_id : str | None
        If set, only accept FRAME_READY events with this camera_id. Useful when
        camera-core is multiplexing multiple cameras.
    require_depth : bool
        If True (default), reject events that have no depth shm (color-only
        cameras or RGB-only modes).
    settle_frames : int
        On grab_rgbd(), drop this many frames before returning one. Lets the
        sensor / auto-exposure / temporal filter catch up after a robot move.
    """

    def __init__(
        self,
        bus_addr: str = "tcp://127.0.0.1:5555",
        topic: str = "camera",
        camera_id: Optional[str] = None,
        require_depth: bool = True,
        settle_frames: int = 3,
        recv_timeout_ms: int = 5000,
    ):
        self.bus_addr = bus_addr
        self.topic = topic
        self.camera_id = camera_id
        self.require_depth = require_depth
        self.settle_frames = max(0, int(settle_frames))
        self._opened_shms: dict = {}

        self.ctx = zmq.Context.instance()
        self.sock = self.ctx.socket(zmq.SUB)
        self.sock.setsockopt(zmq.RCVTIMEO, int(recv_timeout_ms))
        self.sock.setsockopt(zmq.LINGER, 0)
        self.sock.setsockopt_string(zmq.SUBSCRIBE, topic)
        self.sock.connect(bus_addr)
        # PUB/SUB slow-joiner: give the subscription a moment to register
        time.sleep(0.2)
        log.info("CameraCoreClient subscribed to %s topic=%s", bus_addr, topic)

    # ------------------------------------------------------------------ shm

    def _open_shm(self, name: str):
        shm = self._opened_shms.get(name)
        if shm is not None:
            return shm
        try:
            shm = shared_memory.SharedMemory(name=name, create=False, track=False)
        except TypeError:
            shm = shared_memory.SharedMemory(name=name, create=False)
        self._opened_shms[name] = shm
        return shm

    def _read_array(self, shm_name: str, shape, dtype) -> np.ndarray:
        shm = self._open_shm(shm_name)
        nbytes = int(np.prod(shape) * np.dtype(dtype).itemsize)
        view = np.ndarray(
            shape, dtype=np.dtype(dtype),
            buffer=shm.buf[HEADER_SIZE : HEADER_SIZE + nbytes],
        )
        # Copy: camera-core may rotate / overwrite this buffer mid-read.
        return np.array(view, copy=True)

    # ------------------------------------------------------------------ recv

    def _recv_event(self) -> dict:
        """Block until the next FRAME_READY event matching our filters."""
        while True:
            raw = self.sock.recv_string()
            # Wire frame: "<topic> <json>"
            sp = raw.find(" ")
            if sp < 0:
                continue
            try:
                event = json.loads(raw[sp + 1 :])
            except json.JSONDecodeError:
                continue
            if event.get("event") != "FRAME_READY":
                continue
            if self.camera_id and event.get("camera_id") != self.camera_id:
                continue
            if self.require_depth and not event.get("depth_shm"):
                continue
            return event

    # ------------------------------------------------------------------ api

    def grab_rgbd(self) -> Tuple[np.ndarray, np.ndarray, o3d.camera.PinholeCameraIntrinsic]:
        """Return one (rgb_uint8 HxWx3, depth_m_float32 HxW, intrinsics).

        Depth is converted to meters using the `depth_scale_m_per_unit` field
        published by camera-core, regardless of whether the underlying buffer
        is uint16 raw counts or already-scaled float32.
        """
        # Drain a few stale events to let temporal filters / AE settle.
        for _ in range(self.settle_frames):
            self._recv_event()
        event = self._recv_event()

        rgb_shape = tuple(event["rgb_shape"])
        rgb_dtype = np.dtype(event["rgb_dtype"])
        rgb = self._read_array(event["rgb_shm"], rgb_shape, rgb_dtype)
        if rgb.dtype != np.uint8:
            rgb = rgb.astype(np.uint8)

        depth_shape = tuple(event["depth_shape"])
        depth_dtype = np.dtype(event["depth_dtype"])
        depth_raw = self._read_array(event["depth_shm"], depth_shape, depth_dtype)

        cam_data = event.get("camera_data") or {}
        depth_scale = float(cam_data.get("depth_scale_m_per_unit", 1.0))

        if np.issubdtype(depth_raw.dtype, np.integer):
            depth_m = depth_raw.astype(np.float32) * depth_scale
        else:
            # Float buffers from camera-core drivers are already in meters
            # (drivers apply the sensor depth_scale before writing to shm).
            # depth_scale_m_per_unit in camera_data reports the *sensor*
            # hardware scale for metadata purposes and must NOT be reapplied
            # here — doing so collapses the cloud by ~1000x and produces
            # "trail"-like point clouds.
            depth_m = depth_raw.astype(np.float32)

        intr = self._intrinsics_from_camera_data(cam_data, fallback_shape=rgb_shape)

        log.debug("FRAME_READY seq=%s cam=%s rgb=%s depth=%s scale=%g",
                  event.get("sequence_id"), event.get("camera_id"),
                  rgb.shape, depth_m.shape, depth_scale)
        return rgb, depth_m, intr

    @staticmethod
    def _intrinsics_from_camera_data(
        cam_data: dict,
        fallback_shape: tuple,
    ) -> o3d.camera.PinholeCameraIntrinsic:
        K = cam_data.get("K")
        res = cam_data.get("resolution")
        if K and res:
            fx = float(K[0][0]); fy = float(K[1][1])
            cx = float(K[0][2]); cy = float(K[1][2])
            H, W = int(res[0]), int(res[1])
        else:
            intr = (cam_data or {}).get("intrinsics") or {}
            fx = float(intr.get("fx", 0.0)); fy = float(intr.get("fy", 0.0))
            cx = float(intr.get("cx", 0.0)); cy = float(intr.get("cy", 0.0))
            H, W = int(fallback_shape[0]), int(fallback_shape[1])
        if fx <= 0 or fy <= 0:
            raise RuntimeError(
                "camera-core did not publish usable intrinsics in camera_data"
            )
        return o3d.camera.PinholeCameraIntrinsic(
            width=W, height=H, fx=fx, fy=fy, cx=cx, cy=cy,
        )

    # ------------------------------------------------------------------ ctx

    def close(self) -> None:
        for shm in self._opened_shms.values():
            try:
                shm.close()
            except Exception:
                pass
        self._opened_shms.clear()
        try:
            self.sock.close(0)
        except Exception:
            pass

    def __enter__(self) -> "CameraCoreClient":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
