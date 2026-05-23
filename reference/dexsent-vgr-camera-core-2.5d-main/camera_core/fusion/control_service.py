"""
Camera-core-owned fusion trigger service.

This keeps multi-view capture/fusion execution inside the camera-core env while
letting the orchestrator request a fused capture over HTTP.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from typing import Any, Dict, Optional

import numpy as np
import zmq

from .vision_session_publisher import VisionSessionFramePublisher, fuse_and_publish_at_capture_pose

log = logging.getLogger("camera_core.fusion.control")


def _quat_xyzw_to_matrix(quat_xyzw: list[float]) -> list[list[float]]:
    x, y, z, w = [float(v) for v in quat_xyzw[:4]]
    n = x * x + y * y + z * z + w * w
    if n < 1e-12:
        return [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    s = 2.0 / n
    xx, yy, zz = x * x * s, y * y * s, z * z * s
    xy, xz, yz = x * y * s, x * z * s, y * z * s
    wx, wy, wz = w * x * s, w * y * s, w * z * s
    return [
        [1.0 - (yy + zz), xy - wz, xz + wy],
        [xy + wz, 1.0 - (xx + zz), yz - wx],
        [xz - wy, yz + wx, 1.0 - (xx + yy)],
    ]


def _build_dataclass_config(cls: Any, raw: Dict[str, Any]) -> Any:
    cfg = cls()
    if not isinstance(raw, dict):
        return cfg
    for key, value in raw.items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)
    return cfg


class _RobotClient:
    def __init__(self, command_endpoint: str, timeout_ms: int = 20000) -> None:
        self._command_endpoint = str(command_endpoint)
        self._timeout_ms = max(100, int(timeout_ms))
        self._ctx = zmq.Context.instance()
        self._req = self._ctx.socket(zmq.REQ)
        self._req.connect(self._command_endpoint)
        self._req.setsockopt(zmq.RCVTIMEO, self._timeout_ms)
        self._req.setsockopt(zmq.SNDTIMEO, self._timeout_ms)
        self._req.setsockopt(zmq.LINGER, 0)
        self._lock = threading.Lock()
        self._connect()

    def _send(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        with self._lock:
            self._req.send_string(json.dumps(payload, separators=(",", ":"), ensure_ascii=False))
            raw = self._req.recv_string()
        return json.loads(raw)

    def _connect(self) -> None:
        resp = self._send({"cmd": "CONNECT"})
        if not resp.get("ok", False):
            raise RuntimeError(resp.get("reason") or resp.get("message") or "robot_not_connected")

    def move_tcp(
        self,
        position_m,
        quat_xyzw,
        frame: str = "base",
        profile: str = "normal",
    ) -> None:
        resp = self._send(
            {
                "cmd": "MOVE_TCP",
                "target": {
                    "position_m": [float(position_m[0]), float(position_m[1]), float(position_m[2])],
                    "quat_xyzw": [
                        float(quat_xyzw[0]),
                        float(quat_xyzw[1]),
                        float(quat_xyzw[2]),
                        float(quat_xyzw[3]),
                    ],
                    "frame": str(frame or "base"),
                },
                "profile": str(profile or "normal"),
            }
        )
        if not resp.get("ok", False):
            raise RuntimeError(resp.get("reason") or resp.get("message") or "robot_move_tcp_failed")

    def get_tcp(self):
        resp = self._send({"cmd": "GET_STATE"})
        if not resp.get("ok", False):
            raise RuntimeError(resp.get("reason") or resp.get("message") or "robot_get_state_failed")
        state = resp.get("state") or {}
        tcp_pose = state.get("tcp_pose") or {}
        return (
            list(tcp_pose.get("position_m") or [0.0, 0.0, 0.0]),
            list(tcp_pose.get("quat_xyzw") or [0.0, 0.0, 0.0, 1.0]),
        )

    def close(self) -> None:
        try:
            self._req.close(0)
        except Exception:
            pass


class FusionControlService:
    def __init__(
        self,
        *,
        push_addr: str,
        topic: str,
        robot_command_endpoint: str,
        robot_timeout_ms: int = 20000,
    ) -> None:
        self.push_addr = str(push_addr)
        self.topic = str(topic)
        self.robot_command_endpoint = str(robot_command_endpoint)
        self.robot_timeout_ms = int(robot_timeout_ms)
        self._lock = threading.Lock()
        self._publishers: Dict[str, VisionSessionFramePublisher] = {}

    def _publisher(self, camera_id: str) -> VisionSessionFramePublisher:
        key = str(camera_id)
        publisher = self._publishers.get(key)
        if publisher is not None:
            return publisher
        publisher = VisionSessionFramePublisher(camera_id=key, push_addr=self.push_addr)
        self._publishers[key] = publisher
        return publisher

    def capture_pose_fusion(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        from . import CapturePoseFusionConfig, FusionConfig

        request = dict(payload or {})
        capture_pose = request.get("capture_pose")
        if not isinstance(capture_pose, dict):
            raise ValueError("capture_pose_required")

        virtual_camera_id = str(request.get("virtual_camera_id") or "").strip()
        if not virtual_camera_id:
            raise ValueError("virtual_camera_id_required")

        source_camera_id = str(request.get("source_camera_id") or "").strip()
        if not source_camera_id:
            raise ValueError("source_camera_id_required")

        t_grip_cam_raw = request.get("t_grip_cam")
        if isinstance(t_grip_cam_raw, list):
            T_grip_cam = np.asarray(t_grip_cam_raw, dtype=np.float64)
        else:
            hand_eye = request.get("hand_eye") or {}
            translation = hand_eye.get("translation_m") or [0.0, 0.0, 0.0]
            quat = hand_eye.get("rotation_quat_xyzw") or [0.0, 0.0, 0.0, 1.0]
            T_grip_cam = np.eye(4, dtype=np.float64)
            T_grip_cam[:3, :3] = np.asarray(_quat_xyzw_to_matrix(list(quat)), dtype=np.float64)
            T_grip_cam[:3, 3] = np.asarray(
                [float(translation[0]), float(translation[1]), float(translation[2])],
                dtype=np.float64,
            )

        fusion_cfg = _build_dataclass_config(
            FusionConfig,
            request.get("pipeline") if isinstance(request.get("pipeline"), dict) else {},
        )
        capture_cfg = _build_dataclass_config(
            CapturePoseFusionConfig,
            request.get("capture") if isinstance(request.get("capture"), dict) else {},
        )
        publish_repeats = max(1, int(request.get("publish_repeats", 1)))
        publish_interval_s = max(0.0, float(request.get("publish_interval_s", 0.0)))
        camera_data = request.get("camera_data") if isinstance(request.get("camera_data"), dict) else {}
        calib_version = int(request.get("calib_version", 1) or 1)
        cam_kwargs = {
            "bus_addr": str(request.get("bus_addr") or "tcp://127.0.0.1:5555"),
            "topic": str(request.get("topic") or self.topic or "camera"),
            "camera_id": source_camera_id,
            "require_depth": True,
            "settle_frames": max(0, int(request.get("settle_frames", 2))),
            "recv_timeout_ms": max(100, int(request.get("recv_timeout_ms", 5000))),
        }

        with self._lock:
            robot = _RobotClient(
                self.robot_command_endpoint,
                timeout_ms=self.robot_timeout_ms,
            )
            try:
                publisher = self._publisher(virtual_camera_id)
                started_at = time.time()
                log.info(
                    "fusion_capture_start source=%s virtual=%s publish_repeats=%d out_dir=%s",
                    source_camera_id,
                    virtual_camera_id,
                    publish_repeats,
                    getattr(fusion_cfg, "out_dir", ""),
                )
                fused = fuse_and_publish_at_capture_pose(
                    robot=robot,
                    capture_pose=capture_pose,
                    T_grip_cam=T_grip_cam,
                    publisher=publisher,
                    fusion_cfg=fusion_cfg,
                    capture_cfg=capture_cfg,
                    cam_kwargs=cam_kwargs,
                    camera_data=camera_data or None,
                    calib_version=calib_version,
                    publish=False,
                )
                publish_event = None
                for idx in range(publish_repeats):
                    publish_event = publisher.publish_rgbd(
                        bgr=fused["bgr"],
                        depth_m=fused["depth_m"],
                        camera_data=camera_data or fused.get("camera"),
                        calib_version=calib_version,
                    )
                    if publish_interval_s > 0.0 and idx < (publish_repeats - 1):
                        time.sleep(publish_interval_s)
                artifacts = fused.get("artifacts") if isinstance(fused.get("artifacts"), dict) else {}
                out_dir = str(getattr(fusion_cfg, "out_dir", "") or "").strip()
                visualization_3d = {
                    "fusion_point_cloud_ply_path": f"{out_dir}/fused.ply" if out_dir else None,
                    "fusion_point_cloud_pcd_path": f"{out_dir}/fused.pcd" if out_dir else None,
                    "fusion_point_cloud_cam_ply_path": f"{out_dir}/fused_cam.ply" if out_dir else None,
                    "fusion_tsdf_cloud_ply_path": f"{out_dir}/tsdf_cloud.ply" if out_dir and artifacts.get("tsdf_cloud") is not None else None,
                    "fusion_tsdf_cloud_cam_ply_path": f"{out_dir}/tsdf_cloud_cam.ply" if out_dir and artifacts.get("tsdf_cloud") is not None else None,
                    "fusion_tsdf_mesh_ply_path": f"{out_dir}/tsdf_mesh.ply" if out_dir and artifacts.get("tsdf_mesh") is not None else None,
                    "fusion_views_dir": f"{out_dir}/views" if out_dir else None,
                }
                elapsed_s = time.time() - started_at
                log.info(
                    "fusion_capture_done source=%s virtual=%s captures=%d valid_depth=%d elapsed_s=%.3f",
                    source_camera_id,
                    virtual_camera_id,
                    len(fused.get("captures") or []),
                    int((fused["depth_m"] > 0.0).sum()),
                    elapsed_s,
                )
                return {
                    "status": "ok",
                    "source_camera_id": source_camera_id,
                    "virtual_camera_id": virtual_camera_id,
                    "render_source": str(fused.get("render_source") or "fused"),
                    "render_source_points": int(fused.get("render_source_points") or 0),
                    "captures": len(fused.get("captures") or []),
                    "depth_valid_px": int((fused["depth_m"] > 0.0).sum()),
                    "publish_event": publish_event,
                    "elapsed_s": elapsed_s,
                    "visualization_3d": visualization_3d,
                }
            except Exception:
                log.exception(
                    "fusion_capture_failed source=%s virtual=%s",
                    source_camera_id,
                    virtual_camera_id,
                )
                raise
            finally:
                robot.close()
