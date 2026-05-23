"""
Publish fused RGBD frames as synthetic camera-core FRAME_READY events.

This lets downstream vision-engine sessions consume a fused capture exactly as
if it had arrived from a live camera driver: RGB/depth land in the same SHM
triple-buffer format, and the event bus sees the same FRAME_READY schema.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional

import numpy as np

from ..ipc.zmq_pub import ZmqPublisher
from ..shm.header import FLAG_VALID, pack_header
from ..shm.triple_buffer import TripleBuffer
from .capture_pose_fusion import fuse_at_capture_pose


def _normalize_camera_data(
    camera_data: Optional[Dict[str, Any]],
    *,
    image_shape: tuple[int, int],
) -> Dict[str, Any]:
    H, W = int(image_shape[0]), int(image_shape[1])
    normalized: Dict[str, Any] = dict(camera_data or {})
    normalized["resolution"] = [H, W]

    intr = normalized.get("intrinsics")
    if isinstance(intr, dict):
        intr_norm = dict(intr)
        intr_norm["resolution"] = {"width": W, "height": H}
        normalized["intrinsics"] = intr_norm

    normalized.setdefault("depth_scale_m_per_unit", 1.0)
    return normalized


class VisionSessionFramePublisher:
    """Own SHM buffers for a virtual camera and publish FRAME_READY events."""

    def __init__(
        self,
        camera_id: str,
        *,
        push_addr: str = "tcp://127.0.0.1:5556",
    ) -> None:
        self.camera_id = str(camera_id)
        self.push_addr = str(push_addr)
        # Fusion publishes a handful of messages per request (not a camera-rate
        # stream), so we pay the 0.5s connect-settle once per virtual camera
        # to avoid the PUSH<->PULL handshake race, and allow a longer send
        # timeout so a transient broker blip doesn't nuke the fusion call.
        self._publisher = ZmqPublisher(
            self.push_addr,
            send_timeout_ms=5000,
            wait_for_connect_s=0.5,
            settle_first_send=True,
        )
        self._sequence_id = 0
        self._rgb_tb: TripleBuffer | None = None
        self._depth_tb: TripleBuffer | None = None
        self._rgb_signature: tuple[tuple[int, ...], str] | None = None
        self._depth_signature: tuple[tuple[int, ...], str] | None = None

    def _ensure_triple_buffer(
        self,
        current: TripleBuffer | None,
        *,
        kind: str,
        shape: tuple[int, ...],
        dtype: np.dtype,
        signature: tuple[tuple[int, ...], str] | None,
    ) -> tuple[TripleBuffer, tuple[tuple[int, ...], str]]:
        next_signature = (tuple(int(v) for v in shape), np.dtype(dtype).str)
        if current is not None and signature == next_signature:
            return current, next_signature
        if current is not None:
            current.close(unlink=True)
        tb = TripleBuffer(
            name_prefix=f"cam_{self.camera_id}_{kind}",
            suffixes=["A", "B", "C"],
            shape=shape,
            dtype=np.dtype(dtype),
            create=True,
        )
        return tb, next_signature

    def publish_rgbd(
        self,
        *,
        bgr: np.ndarray,
        depth_m: np.ndarray | None,
        camera_data: Optional[Dict[str, Any]] = None,
        timestamp_ns: int | None = None,
        calib_version: int = 1,
    ) -> Dict[str, Any]:
        rgb = np.asarray(bgr, dtype=np.uint8)
        depth = None if depth_m is None else np.asarray(depth_m, dtype=np.float32)

        self._rgb_tb, self._rgb_signature = self._ensure_triple_buffer(
            self._rgb_tb,
            kind="rgb",
            shape=rgb.shape,
            dtype=rgb.dtype,
            signature=self._rgb_signature,
        )
        if depth is not None:
            self._depth_tb, self._depth_signature = self._ensure_triple_buffer(
                self._depth_tb,
                kind="depth",
                shape=depth.shape,
                dtype=depth.dtype,
                signature=self._depth_signature,
            )

        ts_ns = int(timestamp_ns or time.time_ns())
        self._sequence_id += 1
        seq = self._sequence_id

        rgb_wb = self._rgb_tb.get_write_buffer()
        rgb_wb.img[:] = rgb
        rgb_wb.write_header(pack_header(ts_ns, seq, int(calib_version), FLAG_VALID))
        self._rgb_tb.rotate()
        ready_rgb = self._rgb_tb.get_last_complete_name()

        event: Dict[str, Any] = {
            "event": "FRAME_READY",
            "camera_id": self.camera_id,
            "sequence_id": seq,
            "timestamp_ns": ts_ns,
            "calib_version": int(calib_version),
            "status_flags": int(FLAG_VALID),
            "rgb_shm": ready_rgb,
            "rgb_shape": list(rgb.shape),
            "rgb_dtype": str(rgb.dtype),
            "camera_data": _normalize_camera_data(
                camera_data,
                image_shape=(rgb.shape[0], rgb.shape[1]),
            ),
        }

        if depth is not None and self._depth_tb is not None:
            depth_wb = self._depth_tb.get_write_buffer()
            depth_wb.img[:] = depth
            depth_wb.write_header(pack_header(ts_ns, seq, int(calib_version), FLAG_VALID))
            self._depth_tb.rotate()
            ready_depth = self._depth_tb.get_last_complete_name()
            event.update(
                {
                    "depth_shm": ready_depth,
                    "depth_shape": list(depth.shape),
                    "depth_dtype": str(depth.dtype),
                }
            )

        self._publisher.publish(event)
        return event

    def close(self) -> None:
        if self._rgb_tb is not None:
            self._rgb_tb.close(unlink=True)
            self._rgb_tb = None
        if self._depth_tb is not None:
            self._depth_tb.close(unlink=True)
            self._depth_tb = None
        self._publisher.close()


def fuse_and_publish_at_capture_pose(
    *,
    robot,
    capture_pose: Dict[str, Any],
    T_grip_cam: np.ndarray,
    publisher: VisionSessionFramePublisher,
    fusion_cfg=None,
    capture_cfg=None,
    cam_kwargs: Optional[Dict[str, Any]] = None,
    camera_data: Optional[Dict[str, Any]] = None,
    calib_version: int = 1,
    publish: bool = True,
) -> Dict[str, Any]:
    fused = fuse_at_capture_pose(
        robot=robot,
        capture_pose=capture_pose,
        T_grip_cam=T_grip_cam,
        fusion_cfg=fusion_cfg,
        capture_cfg=capture_cfg,
        cam_kwargs=cam_kwargs,
    )
    publish_event = None
    if publish:
        publish_event = publisher.publish_rgbd(
            bgr=fused["bgr"],
            depth_m=fused["depth_m"],
            camera_data=camera_data or fused.get("camera"),
            calib_version=calib_version,
        )
    fused["publish_event"] = publish_event
    return fused
