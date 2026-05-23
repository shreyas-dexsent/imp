"""Implementation for `orchestrator.api.routes`."""

import asyncio
import base64
import json
import math
import shutil
from datetime import datetime, timezone
import logging
import struct
import sys
import time
from multiprocessing import resource_tracker
from multiprocessing import shared_memory
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request as URLRequest
from urllib.request import urlopen

import cv2
import numpy as np
import yaml
from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.responses import FileResponse, StreamingResponse
from orchestrator.core.context import StationContext
from orchestrator.robot_engine_bridge import (
    apply_tcp_offset_to_ee_pose,
    build_chain_for_process,
    build_scene_request as build_robot_engine_scene_request,
    collision_debug_scene as robot_engine_collision_debug_scene,
    evaluate_scene as evaluate_robot_engine_scene,
    manifest_summary as robot_engine_manifest_summary,
    quat_to_matrix,
)
from orchestrator.vision.calibration import (
    build_station_camera_calibration,
    infer_station_id_for_camera,
)
from pydantic import BaseModel, Field

log = logging.getLogger("orchestrator.events")
_SHM_CACHE_LOCK = Lock()
_SHM_CACHE: Dict[str, shared_memory.SharedMemory] = {}
_UNREGISTERED_SHM_NAMES: set[str] = set()
_CACHE_SHM_ATTACHMENTS = not sys.platform.startswith("win")


def _resolve_station_id_for_vision_request(
    ctx: StationContext,
    camera_id: Optional[str],
    station_id: Optional[str] = None,
) -> Optional[str]:
    return infer_station_id_for_camera(ctx.stations, camera_id, station_id)


def _attach_station_vision_calibration(
    ctx: StationContext,
    payload: Dict[str, Any],
    *,
    camera_id: Optional[str],
    station_id: Optional[str] = None,
) -> None:
    resolved_station_id = _resolve_station_id_for_vision_request(
        ctx,
        camera_id,
        station_id,
    )
    calibration = build_station_camera_calibration(ctx.data_paths, resolved_station_id)
    if calibration:
        payload["calibration"] = calibration


def _quat_xyzw_to_rpy_deg(quat_xyzw: List[float]) -> List[float]:
    quat = np.asarray(quat_xyzw[:4], dtype=np.float64).reshape(4)
    norm = float(np.linalg.norm(quat))
    if norm < 1e-12:
        return [0.0, 0.0, 0.0]
    x, y, z, w = (quat / norm).tolist()
    sinr = 2.0 * (w * x + y * z)
    cosr = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr, cosr)
    sinp = 2.0 * (w * y - z * x)
    pitch = np.sign(sinp) * (np.pi / 2.0) if abs(sinp) >= 1.0 else np.arcsin(sinp)
    siny = 2.0 * (w * z + x * y)
    cosy = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny, cosy)
    return [float(np.degrees(roll)), float(np.degrees(pitch)), float(np.degrees(yaw))]


def _normalize_quat_xyzw(quat_xyzw: List[float]) -> List[float]:
    quat = np.asarray(quat_xyzw[:4], dtype=np.float64).reshape(4)
    norm = float(np.linalg.norm(quat))
    if norm < 1e-12:
        return [0.0, 0.0, 0.0, 1.0]
    quat = quat / norm
    return [float(v) for v in quat.tolist()]


def _rpy_deg_to_quat_xyzw(rpy_deg: List[float]) -> List[float]:
    roll = float(rpy_deg[0]) * np.pi / 180.0
    pitch = float(rpy_deg[1]) * np.pi / 180.0
    yaw = float(rpy_deg[2]) * np.pi / 180.0
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)
    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    return _normalize_quat_xyzw([float(qx), float(qy), float(qz), float(qw)])


def _rvec_to_rpy_deg(rvec: List[float]) -> List[float]:
    vec = np.asarray(rvec[:3], dtype=np.float64).reshape(3, 1)
    rot, _ = cv2.Rodrigues(vec)
    sy = float(np.sqrt(rot[0, 0] * rot[0, 0] + rot[1, 0] * rot[1, 0]))
    singular = sy < 1e-9
    if not singular:
        roll = np.arctan2(rot[2, 1], rot[2, 2])
        pitch = np.arctan2(-rot[2, 0], sy)
        yaw = np.arctan2(rot[1, 0], rot[0, 0])
    else:
        roll = np.arctan2(-rot[1, 2], rot[1, 1])
        pitch = np.arctan2(-rot[2, 0], sy)
        yaw = 0.0
    return [float(np.degrees(roll)), float(np.degrees(pitch)), float(np.degrees(yaw))]


def _rotmat_to_quat_xyzw(rot: np.ndarray) -> List[float]:
    rot = np.asarray(rot, dtype=np.float64).reshape(3, 3)
    trace = float(np.trace(rot))
    if trace > 0.0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (rot[2, 1] - rot[1, 2]) * s
        y = (rot[0, 2] - rot[2, 0]) * s
        z = (rot[1, 0] - rot[0, 1]) * s
    else:
        if rot[0, 0] > rot[1, 1] and rot[0, 0] > rot[2, 2]:
            s = 2.0 * np.sqrt(1.0 + rot[0, 0] - rot[1, 1] - rot[2, 2])
            x = 0.25 * s
            y = (rot[0, 1] + rot[1, 0]) / s
            z = (rot[0, 2] + rot[2, 0]) / s
            w = (rot[2, 1] - rot[1, 2]) / s
        elif rot[1, 1] > rot[2, 2]:
            s = 2.0 * np.sqrt(1.0 + rot[1, 1] - rot[0, 0] - rot[2, 2])
            x = (rot[0, 1] + rot[1, 0]) / s
            y = 0.25 * s
            z = (rot[1, 2] + rot[2, 1]) / s
            w = (rot[0, 2] - rot[2, 0]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + rot[2, 2] - rot[0, 0] - rot[1, 1])
            x = (rot[0, 2] + rot[2, 0]) / s
            y = (rot[1, 2] + rot[2, 1]) / s
            z = 0.25 * s
            w = (rot[1, 0] - rot[0, 1]) / s
    quat = np.asarray([x, y, z, w], dtype=np.float64)
    norm = float(np.linalg.norm(quat))
    if norm < 1e-12:
        return [0.0, 0.0, 0.0, 1.0]
    return [float(v) for v in (quat / norm)]


def _rotmat_to_rpy_deg(rot: np.ndarray) -> List[float]:
    rot = np.asarray(rot, dtype=np.float64).reshape(3, 3)
    sy = float(np.sqrt(rot[0, 0] * rot[0, 0] + rot[1, 0] * rot[1, 0]))
    singular = sy < 1e-9
    if not singular:
        roll = np.arctan2(rot[2, 1], rot[2, 2])
        pitch = np.arctan2(-rot[2, 0], sy)
        yaw = np.arctan2(rot[1, 0], rot[0, 0])
    else:
        roll = np.arctan2(-rot[1, 2], rot[1, 1])
        pitch = np.arctan2(-rot[2, 0], sy)
        yaw = 0.0
    return [float(np.degrees(roll)), float(np.degrees(pitch)), float(np.degrees(yaw))]


def _as_float_triplet(value: Any, fallback: List[float]) -> List[float]:
    if isinstance(value, (list, tuple)) and len(value) >= 3:
        source = value[:3]
    else:
        source = fallback[:3]
    out: List[float] = []
    for idx in range(3):
        try:
            out.append(float(source[idx]))
        except Exception:
            out.append(0.0)
    return out


def _normalize_calibration_tcp_payload(
    raw: Dict[str, Any],
    *,
    allow_transform_aliases: bool = False,
    include_status: bool = False,
) -> Dict[str, Any]:
    x_default = raw.get("tcp_x_offset_m", raw.get("gripper_x_offset_m", 0.0))
    y_default = raw.get("tcp_y_offset_m", raw.get("gripper_y_offset_m", 0.0))
    z_default = raw.get("tcp_z_offset_m", raw.get("gripper_z_offset_m", 0.0))
    translation_source = raw.get("tcp_offset_m")
    if translation_source is None and allow_transform_aliases:
        translation_source = raw.get("translation_m")
    translation = _as_float_triplet(
        translation_source,
        [x_default, y_default, z_default],
    )

    roll_default = raw.get("tcp_roll_offset_deg", raw.get("gripper_roll_offset_deg", 0.0))
    pitch_default = raw.get("tcp_pitch_offset_deg", raw.get("gripper_pitch_offset_deg", 0.0))
    yaw_default = raw.get("tcp_yaw_offset_deg", raw.get("gripper_yaw_offset_deg", 0.0))
    rpy_source = raw.get("tcp_offset_rpy_deg")
    if rpy_source is None and allow_transform_aliases:
        rpy_source = raw.get("rotation_rpy_deg")
    rpy = _as_float_triplet(
        rpy_source,
        [roll_default, pitch_default, yaw_default],
    )

    frame = str(raw.get("tcp_offset_frame") or raw.get("frame") or "flange").strip().lower()
    if not frame:
        frame = "flange"
    payload: Dict[str, Any] = {
        "tcp_offset_frame": frame,
        "tcp_offset_m": translation,
        "tcp_offset_rpy_deg": rpy,
        "translation_m": translation,
        "rotation_rpy_deg": rpy,
        "gripper_x_offset_m": translation[0],
        "gripper_y_offset_m": translation[1],
        "gripper_z_offset_m": translation[2],
        "gripper_roll_offset_deg": rpy[0],
        "gripper_pitch_offset_deg": rpy[1],
        "gripper_yaw_offset_deg": rpy[2],
    }
    if include_status:
        payload["status"] = "saved"
    return payload


def _has_tcp_calibration_fields(raw: Dict[str, Any]) -> bool:
    return any(
        key in raw
        for key in (
            "tcp_offset_m",
            "tcp_offset_rpy_deg",
            "tcp_offset_frame",
            "tcp_x_offset_m",
            "tcp_y_offset_m",
            "tcp_z_offset_m",
            "tcp_roll_offset_deg",
            "tcp_pitch_offset_deg",
            "tcp_yaw_offset_deg",
            "gripper_x_offset_m",
            "gripper_y_offset_m",
            "gripper_z_offset_m",
            "gripper_roll_offset_deg",
            "gripper_pitch_offset_deg",
            "gripper_yaw_offset_deg",
        )
    )


def _tcp_fields_for_handeye(payload: Dict[str, Any]) -> Dict[str, Any]:
    excluded = {"status", "saved_at", "translation_m", "rotation_rpy_deg"}
    return {key: value for key, value in payload.items() if key not in excluded}


def _calibration_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _default_tool_mount_payload() -> Dict[str, Any]:
    return {
        "parent_frame": "robot_flange",
        "child_frame": "gripper_base",
        "translation_m": [0.0, 0.0, 0.0],
        "rotation_rpy_deg": [0.0, 0.0, -45.0],
    }


def _custom_tcp_from_raw(raw: Dict[str, Any]) -> Dict[str, Any]:
    source = raw.get("custom_tcp") if isinstance(raw.get("custom_tcp"), dict) else raw
    tcp = _normalize_calibration_tcp_payload(
        source if isinstance(source, dict) else {},
        allow_transform_aliases=True,
    )
    return {
        "parent_frame": str(
            (source or {}).get("parent_frame") or "gripper_base"
        ),
        "child_frame": str((source or {}).get("child_frame") or "tool_tcp"),
        "translation_m": tcp["translation_m"],
        "rotation_rpy_deg": tcp["rotation_rpy_deg"],
    }


def _hand_eye_from_raw(raw: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    source = raw.get("hand_eye") if isinstance(raw.get("hand_eye"), dict) else raw
    if not isinstance(source, dict):
        return None
    if not source.get("translation_m"):
        return None
    hand_eye: Dict[str, Any] = {
        "type": str(source.get("type") or "eye_in_hand"),
        "parent_frame": str(source.get("parent_frame") or "tool_tcp"),
        "child_frame": str(
            source.get("child_frame") or "camera_color_optical_frame"
        ),
        "translation_m": _as_float_triplet(source.get("translation_m"), [0.0, 0.0, 0.0]),
    }
    if isinstance(source.get("rotation_rpy_deg"), (list, tuple)) and len(source["rotation_rpy_deg"]) >= 3:
        hand_eye["rotation_rpy_deg"] = _as_float_triplet(
            source.get("rotation_rpy_deg"),
            [0.0, 0.0, 0.0],
        )
    elif isinstance(source.get("rotation_quat_xyzw"), (list, tuple)) and len(source["rotation_quat_xyzw"]) >= 4:
        hand_eye["rotation_rpy_deg"] = _quat_xyzw_to_rpy_deg(source["rotation_quat_xyzw"])
    else:
        hand_eye["rotation_rpy_deg"] = [0.0, 0.0, 0.0]
    return hand_eye


def _combined_calibration_payload(
    raw: Optional[Dict[str, Any]] = None,
    *,
    tcp_raw: Optional[Dict[str, Any]] = None,
    hand_eye_raw: Optional[Dict[str, Any]] = None,
    saved_at: Optional[str] = None,
) -> Dict[str, Any]:
    base = raw if isinstance(raw, dict) else {}
    tool_mount_source = (
        tcp_raw.get("tool_mount")
        if isinstance(tcp_raw, dict) and isinstance(tcp_raw.get("tool_mount"), dict)
        else None
    )
    tool_mount = (
        dict(tool_mount_source)
        if isinstance(tool_mount_source, dict)
        else dict(base.get("tool_mount"))
        if isinstance(base.get("tool_mount"), dict)
        else _default_tool_mount_payload()
    )
    custom_tcp = _custom_tcp_from_raw(tcp_raw or base)
    hand_eye = _hand_eye_from_raw(hand_eye_raw or base)
    payload: Dict[str, Any] = {
        "tool_mount": tool_mount,
        "custom_tcp": custom_tcp,
    }
    if hand_eye is not None:
        payload["hand_eye"] = hand_eye
    payload["convention"] = {
        "translation_unit": "meters",
        "rotation_unit": "degrees",
        "rotation_order": "XYZ",
    }
    payload["metadata"] = {
        "saved_at": saved_at
        or (base.get("metadata") or {}).get("saved_at")
        or base.get("saved_at")
        or _calibration_timestamp()
    }
    return payload


def _read_station_combined_calibration(calib_dir: Path) -> Dict[str, Any]:
    path = calib_dir / "tcp.json"
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _flatten_custom_tcp(payload: Dict[str, Any], *, include_status: bool = False) -> Dict[str, Any]:
    raw = payload.get("custom_tcp") if isinstance(payload.get("custom_tcp"), dict) else payload
    tcp = _normalize_calibration_tcp_payload(
        raw if isinstance(raw, dict) else {},
        allow_transform_aliases=True,
        include_status=include_status,
    )
    if isinstance(raw, dict):
        tcp["parent_frame"] = raw.get("parent_frame", "gripper_base")
        tcp["child_frame"] = raw.get("child_frame", "tool_tcp")
    return tcp


def _flatten_hand_eye(payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    hand_eye = _hand_eye_from_raw(payload)
    if hand_eye is None:
        return None
    return {
        "translation_m": hand_eye["translation_m"],
        "rotation_rpy_deg": hand_eye["rotation_rpy_deg"],
        "hand_eye_frame": "camera_in_gripper",
        "parent_frame": hand_eye["parent_frame"],
        "child_frame": hand_eye["child_frame"],
        "type": hand_eye["type"],
    }


def _transform_section_matrix(section: Dict[str, Any]) -> np.ndarray:
    translation = _as_float_triplet(section.get("translation_m"), [0.0, 0.0, 0.0])
    quat = _rpy_deg_to_quat_xyzw(
        _as_float_triplet(section.get("rotation_rpy_deg"), [0.0, 0.0, 0.0])
    )
    return quat_to_matrix(translation, quat)


def _pose_to_matrix(pose: Dict[str, Any]) -> np.ndarray:
    return quat_to_matrix(
        pose.get("position_m") or [0.0, 0.0, 0.0],
        pose.get("quat_xyzw") or [0.0, 0.0, 0.0, 1.0],
    )


def _matrix_to_pose(matrix: np.ndarray) -> Dict[str, Any]:
    mat = np.asarray(matrix, dtype=float).reshape(4, 4)
    quat = _rotmat_to_quat_xyzw(mat[:3, :3])
    return {
        "position_m": [float(v) for v in mat[:3, 3].tolist()],
        "quat_xyzw": quat,
        "rotation_rpy_deg": _quat_xyzw_to_rpy_deg(quat),
        "frame": "base",
    }


def _format_calibration_samples_payload(
    station_id: str, req: "CalibrationSamplesRequest"
) -> Dict[str, Any]:
    rows = []
    for idx, sample in enumerate(req.samples, start=1):
        robot = sample.get("robot") or {}
        detection = sample.get("detection") or {}
        robot_quat = robot.get("quat_xyzw") or [0.0, 0.0, 0.0, 1.0]
        det_rvec = detection.get("rvec") or [0.0, 0.0, 0.0]
        det_tvec = detection.get("tvec") or [0.0, 0.0, 0.0]
        r_ct, _ = cv2.Rodrigues(np.asarray(det_rvec[:3], dtype=np.float64).reshape(3, 1))
        t_ct = np.asarray(det_tvec[:3], dtype=np.float64).reshape(3, 1)
        r_tc = r_ct.T
        t_tc = -r_tc @ t_ct
        rows.append(
            {
                "sample_index": idx,
                "sample_id": sample.get("sample_id", idx),
                "camera_id": sample.get("camera_id") or req.camera_id,
                "sequence_id": detection.get("sequence_id"),
                "timestamp_ns": detection.get("timestamp_ns"),
                "robot_pose": {
                    "interpretation": "gripper_in_robot_base_frame",
                    "transform_symbol": "^bT_g",
                    "translation_m": robot.get("position_m") or [0.0, 0.0, 0.0],
                    "rotation_rpy_deg": _quat_xyzw_to_rpy_deg(robot_quat),
                    "rotation_quat_xyzw": robot_quat,
                    "frame": robot.get("frame") or "base",
                },
                "target_pose_in_camera_frame": {
                    "interpretation": "calibration_target_board_origin_in_camera_frame",
                    "transform_symbol": "^cT_t",
                    "translation_m": [float(v) for v in t_ct.reshape(-1)],
                    "rotation_rpy_deg": _rvec_to_rpy_deg(det_rvec),
                    "rotation_rvec": det_rvec,
                    "rotation_quat_xyzw": _rotmat_to_quat_xyzw(r_ct),
                    "origin_note": "This is the Charuco board origin from OpenCV pose solve, not the drawn overlay axis origin/top corner.",
                    "reprojection_rmse_px": detection.get("reprojection_rmse_px"),
                    "charuco_ids": detection.get("charuco_ids"),
                    "charuco_corner_count": len(detection.get("charuco_ids") or []),
                },
                "overlay_axis_origin_in_camera_frame": {
                    "interpretation": "drawn_charuco_axis_origin_in_camera_frame",
                    "translation_m": detection.get("axis_origin_pose_xyz_m"),
                    "image_uv": detection.get("axis_origin_uv"),
                    "depth_mean_xyz_m": detection.get("axis_origin_xyz_m"),
                    "depth_sample_count": detection.get("axis_origin_depth_samples"),
                    "origin_note": "This matches the displayed overlay axis anchor, which may differ from the raw Charuco board origin used by tvec/rvec.",
                },
                "overlay_first_corner_in_camera_frame": {
                    "interpretation": "selected_charuco_corner_in_camera_frame",
                    "translation_m": detection.get("first_corner_pose_xyz_m"),
                    "image_uv": detection.get("first_corner_uv"),
                    "depth_xyz_m": detection.get("first_corner_xyz_m"),
                    "depth_mean_xyz_m": detection.get("first_corner_depth_xyz_m"),
                    "depth_sample_count": detection.get("first_corner_depth_samples"),
                    "depth_radius_px": detection.get("first_corner_depth_radius_px"),
                    "depth_distance_px": detection.get("first_corner_depth_distance_px"),
                    "origin_note": "This is the selected visible Charuco corner/top-corner helper, not the raw Charuco board origin.",
                },
                "camera_pose_in_target_frame": {
                    "interpretation": "camera_in_calibration_target_frame",
                    "transform_symbol": "^tT_c",
                    "translation_m": [float(v) for v in t_tc.reshape(-1)],
                    "rotation_rpy_deg": _rotmat_to_rpy_deg(r_tc),
                    "rotation_quat_xyzw": _rotmat_to_quat_xyzw(r_tc),
                },
                "robot_state_debug": sample.get("robot_state_debug"),
            }
        )
    return {
        "station_id": station_id,
        "saved_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "camera_id": req.camera_id,
        "method": req.method,
        "mode": req.mode,
        "target": req.target,
        "tcp_calibration": req.tcp_calibration,
        "frame_interpretation": {
            "robot_pose": "gripper/flange pose expressed in robot base frame (^bT_g)",
            "target_pose_in_camera_frame": "raw Charuco board pose from OpenCV: board origin expressed in camera frame (^cT_t)",
            "overlay_axis_origin_in_camera_frame": "the axis anchor currently drawn in the detection overlay, expressed in camera frame",
            "overlay_first_corner_in_camera_frame": "the selected visible Charuco corner helper point, expressed in camera frame",
            "camera_pose_in_target_frame": "inverse of the raw detection pose: camera expressed in calibration target frame (^tT_c)",
            "robot_state_debug": "raw robot-side debug fields captured with the sample, including active TCP/world offsets when available",
        },
        "samples_count": len(rows),
        "samples": rows,
    }


class VisionStartRequest(BaseModel):
    session_id: str = Field(..., description="Unique session/request ID")
    station_id: Optional[str] = None
    camera_id: str
    module: str
    fps_limit: float = 0
    process_mode: str = "continuous"
    params: Dict[str, Any] = Field(default_factory=dict)
    enable_shm_output: bool = True


class VisionStopRequest(BaseModel):
    session_id: str


class VisionCaptureRequest(BaseModel):
    station_id: Optional[str] = None
    camera_id: str
    module: str = "object_proposals"
    timeout_s: float = 2.0
    params: Dict[str, Any] = Field(default_factory=dict)


class StationCreateRequest(BaseModel):
    station_id: Optional[str] = None
    name: Optional[str] = None
    description: str = ""
    camera_ids: List[str] = Field(default_factory=list)
    robot_ids: List[str] = Field(default_factory=list)


class StationPatchRequest(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    camera_ids: Optional[List[str]] = None
    robot_ids: Optional[List[str]] = None


class ProcessCreateRequest(BaseModel):
    asset_id: Optional[str] = None
    process_id: Optional[str] = None
    name: Optional[str] = None
    description: str = ""
    task_type: Optional[str] = None
    camera_ids: List[str] = Field(default_factory=list)
    robot_ids: List[str] = Field(default_factory=list)


class ProcessPatchRequest(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    camera_ids: Optional[List[str]] = None
    robot_ids: Optional[List[str]] = None


class TaskCreateRequest(BaseModel):
    task_id: Optional[str] = None
    name: Optional[str] = None
    description: str = ""
    task_type: Optional[str] = None
    task: Dict[str, Any] = Field(default_factory=dict)


class TaskPatchRequest(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    task_type: Optional[str] = None
    task: Optional[Dict[str, Any]] = None
    params: Optional[Dict[str, Any]] = None


class RunStartRequest(BaseModel):
    params: Dict[str, Any] = Field(default_factory=dict)


class RuntimeRobotModeRequest(BaseModel):
    enabled: bool


class VisionTransportRequest(BaseModel):
    transport: str
    websocket_url: Optional[str] = None


class RobotModelRequest(BaseModel):
    model: str


class RecipeLoadRequest(BaseModel):
    path: Optional[str] = None
    recipe: Optional[Dict[str, Any]] = None
    recipe_id: Optional[str] = None


class HandEyeRequest(BaseModel):
    translation_m: List[float] = Field(..., min_length=3, max_length=3)
    rotation_quat_xyzw: Optional[List[float]] = Field(
        default=None, min_length=4, max_length=4
    )
    rotation_rpy_deg: Optional[List[float]] = Field(
        default=None, min_length=3, max_length=3
    )
    gripper_x_offset_m: Optional[float] = None
    gripper_y_offset_m: Optional[float] = None
    gripper_z_offset_m: Optional[float] = None
    gripper_roll_offset_deg: Optional[float] = None
    gripper_pitch_offset_deg: Optional[float] = None
    gripper_yaw_offset_deg: Optional[float] = None
    tcp_offset_m: Optional[List[float]] = Field(default=None, min_length=3, max_length=3)
    tcp_offset_rpy_deg: Optional[List[float]] = Field(
        default=None, min_length=3, max_length=3
    )
    tcp_offset_frame: Optional[str] = None
    hand_eye_frame: Optional[str] = None
    handeye_result: Optional[Dict[str, Any]] = None


class CalibrationTcpRequest(BaseModel):
    tcp_offset_m: Optional[List[float]] = Field(default=None, min_length=3, max_length=3)
    tcp_offset_rpy_deg: Optional[List[float]] = Field(
        default=None, min_length=3, max_length=3
    )
    translation_m: Optional[List[float]] = Field(default=None, min_length=3, max_length=3)
    rotation_rpy_deg: Optional[List[float]] = Field(
        default=None, min_length=3, max_length=3
    )
    tcp_x_offset_m: Optional[float] = None
    tcp_y_offset_m: Optional[float] = None
    tcp_z_offset_m: Optional[float] = None
    tcp_roll_offset_deg: Optional[float] = None
    tcp_pitch_offset_deg: Optional[float] = None
    tcp_yaw_offset_deg: Optional[float] = None
    gripper_x_offset_m: Optional[float] = None
    gripper_y_offset_m: Optional[float] = None
    gripper_z_offset_m: Optional[float] = None
    gripper_roll_offset_deg: Optional[float] = None
    gripper_pitch_offset_deg: Optional[float] = None
    gripper_yaw_offset_deg: Optional[float] = None
    tcp_offset_frame: Optional[str] = None
    frame: Optional[str] = None
    tool_mount: Optional[Dict[str, Any]] = None
    custom_tcp: Optional[Dict[str, Any]] = None


class IntrinsicsRequest(BaseModel):
    fx: float
    fy: float
    cx: float
    cy: float
    dist_coeffs: Optional[List[float]] = None
    resolution: Optional[Dict[str, int]] = None
    intrinsics_result: Optional[Dict[str, Any]] = None


class CalibrationTargetRequest(BaseModel):
    type: str = "charuco"
    rows: int
    cols: int
    square_size_m: float
    marker_size_m: Optional[float] = None
    aruco_dict: Optional[str] = None


class CalibrationSamplesRequest(BaseModel):
    camera_id: Optional[str] = None
    method: Optional[str] = None
    mode: Optional[str] = None
    target: Dict[str, Any] = Field(default_factory=dict)
    tcp_calibration: Optional[Dict[str, Any]] = None
    samples: List[Dict[str, Any]] = Field(default_factory=list)


class CuboidGeometryRequest(BaseModel):
    """Update cuboid geometry (L, B, H in mm)."""

    L_mm: float
    B_mm: float
    H_mm: float
    axis_convention: str = "LBH_XYZ"


class MegaPoseObjectConfigRequest(BaseModel):
    object_folder: str
    model: str = "rgbd"
    device: str = "auto"
    mesh_units: str = "mm"
    mesh_scale: float = 1.0
    yolo_conf: float = 0.25
    selection_top_k: int = 3
    selection_min_confidence: float = 0.7
    refiner_iterations: int = 1
    coarse_grid_size: int = 72


class ObjectFrameRequest(BaseModel):
    position_m: List[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0])
    rotation_rpy_deg: List[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0])
    frame: str = "object"


class BinRoiRequest(BaseModel):
    """2D image-plane oriented bounding-box for the bin region of interest.

    The OBB is defined by 4 (u, v) corner pixels captured in the same camera
    frame that segmentation runs against. Used at runtime to reject segmentation
    masks whose centroid OR bounding-box corners fall outside this region.
    """

    camera_id: str = ""
    frame_source: str = ""
    image_width: int = 0
    image_height: int = 0
    obb_points_uv: List[List[float]] = Field(default_factory=list)
    yaw_deg: float = 0.0
    bbox_xywh: List[float] = Field(default_factory=list)
    notes: str = ""


class BinPickingAssetSelectRequest(BaseModel):
    catalog_path: str


class TemplateUploadRequest(BaseModel):
    object_id: str
    template_name: str
    image_b64: str
    ext: str = "png"


class ObjectRenameRequest(BaseModel):
    object_id: str
    new_id: str


class TemplateRenameRequest(BaseModel):
    object_id: str
    template_name: str
    new_name: str


class PoseRecordRequest(BaseModel):
    name: str
    mode: str = "auto"
    position_m: Optional[List[float]] = Field(default=None, min_length=3, max_length=3)
    quat_xyzw: Optional[List[float]] = Field(default=None, min_length=4, max_length=4)
    rotation_rpy_deg: Optional[List[float]] = Field(default=None, min_length=3, max_length=3)
    frame: Optional[str] = None


class PoseRenameRequest(BaseModel):
    new_name: str


class RobotMoveJRequest(BaseModel):
    joints: List[float]
    profile: str = "normal"


class RobotMoveLRequest(BaseModel):
    position_m: List[float] = Field(..., min_length=3, max_length=3)
    quat_xyzw: List[float] = Field(
        default_factory=lambda: [0.0, 0.0, 0.0, 1.0], min_length=4, max_length=4
    )
    frame: str = "base"
    profile: str = "normal"


class RobotMoveIkRequest(RobotMoveLRequest):
    seed_joints: Optional[List[float]] = Field(default=None, min_length=7, max_length=7)
    preferred_joints: Optional[List[float]] = Field(default=None, min_length=7, max_length=7)
    position_tolerance_m: float = 0.002
    orientation_tolerance_deg: float = 2.0
    approximate_position_tolerance_m: float = 0.015
    approximate_orientation_tolerance_deg: float = 3.0
    max_iterations: int = 120


class RobotFreedriveRequest(BaseModel):
    enable: bool = True


class RecipeSaveRequest(BaseModel):
    recipe_id: str
    recipe: Dict[str, Any]
    filename: Optional[str] = None
    fmt: str = "yaml"


def _vision_http_url(ctx: StationContext) -> str:
    cfg = ctx.config.get("vision_engine", {}) if ctx and ctx.config else {}
    transport = str(getattr(ctx.vision, "transport", cfg.get("transport", "zmq")) or "zmq")
    if transport == "websocket":
        ws_url = str(
            getattr(ctx.vision, "websocket_url", "")
            or cfg.get("websocket_url")
            or ""
        ).strip()
        if ws_url.startswith("ws://"):
            return ("http://" + ws_url[len("ws://") :]).rsplit("/", 1)[0].rstrip("/")
        if ws_url.startswith("wss://"):
            return ("https://" + ws_url[len("wss://") :]).rsplit("/", 1)[0].rstrip("/")
    return str(cfg.get("http_url", "http://127.0.0.1:8000")).rstrip("/")


def _camera_calibration_control_url(ctx: StationContext) -> str:
    cfg = ctx.config.get("camera_core", {}) if ctx and ctx.config else {}
    return str(cfg.get("calibration_control_url", "http://127.0.0.1:8210")).rstrip("/")


def _camera_calibration_frame_url(ctx: StationContext) -> str:
    cfg = ctx.config.get("camera_core", {}) if ctx and ctx.config else {}
    explicit = cfg.get("calibration_frame_url")
    if explicit:
        return str(explicit).rstrip("/")
    control_url = _camera_calibration_control_url(ctx)
    if control_url.endswith(":8210"):
        return f"{control_url[:-5]}:8211"
    return "http://127.0.0.1:8211"


def _camera_calibration_api_url(ctx: StationContext) -> str:
    return _camera_calibration_frame_url(ctx)


def _http_json(
    url: str,
    method: str = "GET",
    payload: Optional[Dict[str, Any]] = None,
    timeout_s: float = 1.5,
) -> Dict[str, Any]:
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = URLRequest(url, data=data, headers=headers, method=method.upper())
    try:
        with urlopen(req, timeout=timeout_s) as resp:
            raw = resp.read().decode("utf-8")
            try:
                body = json.loads(raw) if raw else {}
            except json.JSONDecodeError:
                body = {"raw": raw}
            return {"ok": True, "status": resp.status, "body": body}
    except HTTPError as exc:
        try:
            raw = exc.read().decode("utf-8")
        except Exception:
            raw = str(exc)
        return {"ok": False, "status": exc.code, "error": raw}
    except URLError as exc:
        return {"ok": False, "status": 503, "error": str(exc)}


HEADER_FMT = "QQII40s"
HEADER_SIZE = 64
FLAG_VALID = 0x01


def _safe_name(raw: str) -> str:
    cleaned = []
    for ch in raw.strip():
        if ch.isalnum() or ch in ("-", "_"):
            cleaned.append(ch)
    return "".join(cleaned)


def _robot_asset_catalog_root() -> Path:
    return Path(__file__).resolve().parents[1] / "robot_asset_catalog"


def _safe_relative_path(raw: str) -> Path:
    text = str(raw or "").replace("\\", "/").strip().strip("/")
    if not text:
        raise ValueError("empty_path")
    path = Path(text)
    if path.is_absolute() or any(part in ("", ".", "..") for part in path.parts):
        raise ValueError("invalid_path")
    return path


def _resolve_under(root: Path, raw: str) -> Path:
    rel = _safe_relative_path(raw)
    resolved_root = root.resolve()
    candidate = (resolved_root / rel).resolve()
    if candidate != resolved_root and resolved_root not in candidate.parents:
        raise ValueError("invalid_path")
    return candidate


def _asset_kind_dir(ctx: StationContext, process: Dict[str, Any], process_id: str, kind: str) -> Path:
    station_id = str(process.get("station_id") or "")
    if not station_id:
        raise ValueError("station_not_set")
    if kind == "robot":
        return ctx.data_paths.process_robot_dir(station_id, process_id)
    if kind == "gripper":
        return ctx.data_paths.process_gripper_dir(station_id, process_id)
    raise ValueError("unsupported_asset_kind")


def _read_json_file(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists() or not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _manifest_summary(root: Path, public_base: str = "") -> Optional[Dict[str, Any]]:
    manifest = _read_json_file(root / "manifest.json")
    if manifest is None:
        return None
    frames = _read_json_file(root / "frames.json")
    urdf = str(manifest.get("urdf") or "").strip()
    item = {
        "id": str(manifest.get("name") or root.name),
        "name": str(manifest.get("display_name") or manifest.get("model") or manifest.get("name") or root.name),
        "path": root.name,
        "manifest": manifest,
        "has_frames": frames is not None,
        "frames": frames,
    }
    if urdf:
        urdf_url = f"{public_base}/{urdf}".replace("//", "/") if public_base else urdf
        urdf_path = root / urdf
        if public_base and urdf_path.exists():
            urdf_url = f"{urdf_url}?v={urdf_path.stat().st_mtime_ns}"
        item["urdf_url"] = urdf_url
    return item


def _asset_gripper_summary(
    ctx: StationContext,
    station_id: str,
    process_id: str,
) -> Optional[Dict[str, Any]]:
    root = ctx.data_paths.process_gripper_dir(station_id, process_id)
    if not root.exists() or not root.is_dir():
        return None
    item = _manifest_summary(
        root,
        public_base=f"/processes/{process_id}/bin-picking/files/gripper",
    )
    if item is None:
        urdfs = sorted((root / "urdf").glob("*.urdf")) if (root / "urdf").exists() else []
        if not urdfs:
            urdfs = sorted(root.rglob("*.urdf"))
        if not urdfs:
            return None
        urdf_rel = urdfs[0].relative_to(root).as_posix()
        item = {
            "id": root.name,
            "name": process_id,
            "path": root.name,
            "manifest": {},
            "has_frames": (root / "frames.json").exists(),
            "frames": _read_json_file(root / "frames.json"),
            "urdf_url": f"/processes/{process_id}/bin-picking/files/gripper/{urdf_rel}",
        }
    item["station_id"] = station_id
    item["asset_id"] = process_id
    item["asset_path"] = f"data/stations/{station_id}/assets/{process_id}/gripper"
    return item


def _open_shared_memory_attach(name: str) -> shared_memory.SharedMemory:
    """Open producer-owned SHM in attach-only mode.

    Newer Python versions support `track=False`; for older runtimes we detach from
    resource_tracker as a fallback.
    """
    key = str(name)
    if not _CACHE_SHM_ATTACHMENTS:
        try:
            shm = shared_memory.SharedMemory(name=key, create=False, track=False)
        except TypeError:
            shm = shared_memory.SharedMemory(name=key, create=False)
            try:
                if shm._name not in _UNREGISTERED_SHM_NAMES:
                    resource_tracker.unregister(shm._name, "shared_memory")
                    _UNREGISTERED_SHM_NAMES.add(shm._name)
            except Exception:
                pass
        return shm
    with _SHM_CACHE_LOCK:
        cached = _SHM_CACHE.get(key)
        if cached is not None:
            return cached
        try:
            shm = shared_memory.SharedMemory(name=key, create=False, track=False)
        except TypeError:
            shm = shared_memory.SharedMemory(name=key, create=False)
            try:
                # Python 3.11 fallback path: unregister once per attached segment.
                if shm._name not in _UNREGISTERED_SHM_NAMES:
                    resource_tracker.unregister(shm._name, "shared_memory")
                    _UNREGISTERED_SHM_NAMES.add(shm._name)
            except Exception:
                pass
        _SHM_CACHE[key] = shm
        return shm


def _drop_cached_shared_memory(name: str) -> None:
    key = str(name)
    with _SHM_CACHE_LOCK:
        shm = _SHM_CACHE.pop(key, None)
    if shm is None:
        return
    try:
        internal = getattr(shm, "_name", None)
        if internal:
            _UNREGISTERED_SHM_NAMES.discard(str(internal))
    except Exception:
        pass
    try:
        shm.close()
    except Exception:
        pass


def _read_rgb_frame(evt: Dict[str, Any]) -> Dict[str, Any]:
    name = evt.get("rgb_shm")
    shape = evt.get("rgb_shape")
    dtype = evt.get("rgb_dtype", "uint8")
    if not name or not shape:
        raise ValueError("missing_rgb_shm")
    try:
        dtype_np = np.dtype(dtype)
    except Exception as exc:
        raise ValueError(f"invalid_rgb_dtype:{dtype}") from exc
    if not isinstance(shape, (list, tuple)) or len(shape) < 2:
        raise ValueError("invalid_rgb_shape")
    name_str = str(name)
    for attempt in range(2):
        shm = _open_shared_memory_attach(name_str)
        try:
            expected = int(np.prod(shape)) * dtype_np.itemsize
            if len(shm.buf) < HEADER_SIZE + expected:
                raise ValueError("rgb_shm_size_mismatch")
            header = bytes(shm.buf[:HEADER_SIZE])
            _, _, _, flags, _ = struct.unpack(HEADER_FMT, header)
            if not (flags & FLAG_VALID):
                raise ValueError("rgb_frame_invalid")
            img = np.ndarray(
                shape=tuple(shape), dtype=dtype_np, buffer=shm.buf, offset=HEADER_SIZE
            )
            frame = img.copy()
            break
        except (OSError, BufferError):
            _drop_cached_shared_memory(name_str)
            if attempt == 0:
                continue
            raise
        finally:
            if not _CACHE_SHM_ATTACHMENTS:
                try:
                    shm.close()
                except Exception:
                    pass
    else:
        raise ValueError("rgb_shm_unavailable")
    return {
        "frame": frame,
        "sequence_id": evt.get("sequence_id"),
        "timestamp_ns": evt.get("timestamp_ns"),
        "camera_id": evt.get("camera_id"),
    }


def _decode_image(payload: str) -> bytes:
    data = payload.strip()
    if data.startswith("data:"):
        parts = data.split(",", 1)
        if len(parts) == 2:
            data = parts[1]
    return base64.b64decode(data)


def _parse_template_name(raw: str) -> Optional[Dict[str, str]]:
    if not raw:
        return None
    value = raw.strip()
    if not value:
        return None
    if "." in value:
        base, ext = value.rsplit(".", 1)
    else:
        base, ext = value, "png"
    base = _safe_name(base)
    ext = ext.lower().strip(".")
    if not base or ext not in ("png", "jpg", "jpeg"):
        return None
    return {"base": base, "ext": ext}


def _read_recipe_file(path: Path) -> Optional[Dict[str, Any]]:
    try:
        if path.suffix.lower() in (".yaml", ".yml"):
            return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if path.suffix.lower() == ".json":
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return None


def _find_recipe_path(root: Path, recipe_id: str) -> Optional[Path]:
    if not root.exists():
        return None
    for ext in ("yaml", "yml", "json"):
        candidate = root / f"{recipe_id}.{ext}"
        if candidate.exists():
            return candidate
    return None


def create_router(ctx: StationContext) -> APIRouter:
    router = APIRouter()
    state_lock = Lock()
    startup_token = str(time.time_ns())
    camera_connected: set[str] = set()
    robot_connected: Optional[bool] = None

    @router.get("/health")
    def health() -> Dict[str, str]:
        return {"status": "alive"}

    @router.get("/ready")
    def ready() -> Dict[str, Any]:
        return {
            "data_root": str(ctx.data_root),
            "robot_adapter": ctx.config.get("robot", {}).get("adapter", "standard"),
            "runtime_robot_enabled": bool(ctx.runtime_state.get("robot_enabled", True)),
            "runtime_robot_mode": (
                "enabled" if bool(ctx.runtime_state.get("robot_enabled", True)) else "disabled"
            ),
            "startup_token": startup_token,
        }

    @router.get("/runtime/robot-mode")
    def runtime_robot_mode() -> Dict[str, Any]:
        enabled = bool(ctx.runtime_state.get("robot_enabled", True))
        return {
            "enabled": enabled,
            "mode": "enabled" if enabled else "disabled",
            "station_id": ctx.runtime_state.get("robot_mode_station_id"),
        }

    @router.post("/runtime/robot-mode")
    def runtime_robot_mode_set(req: RuntimeRobotModeRequest) -> Dict[str, Any]:
        enabled = bool(req.enabled)
        ctx.runtime_state["robot_enabled"] = enabled
        station_id = str(ctx.runtime_state.get("robot_mode_station_id") or "").strip()
        persisted = False
        if station_id:
            station = ctx.stations.patch(
                station_id,
                {"robot_execution_enabled": enabled},
            )
            persisted = station is not None
        return {
            "status": "ok",
            "enabled": enabled,
            "mode": "enabled" if enabled else "disabled",
            "station_id": station_id or None,
            "persisted": persisted,
        }

    @router.get("/runtime/vision-transport")
    def runtime_vision_transport() -> Dict[str, Any]:
        cfg = ctx.config.get("vision_engine", {}) if ctx.config else {}
        transport = str(getattr(ctx.vision, "transport", cfg.get("transport", "zmq")) or "zmq")
        websocket_url = str(
            getattr(ctx.vision, "websocket_url", "")
            or cfg.get("websocket_url")
            or "ws://127.0.0.1:8000/ws"
        )
        return {
            "status": "ok",
            "transport": transport,
            "websocket_url": websocket_url,
        }

    @router.post("/runtime/vision-transport")
    def runtime_vision_transport_set(req: VisionTransportRequest) -> Dict[str, Any]:
        transport = str(req.transport or "").strip().lower()
        if transport not in {"zmq", "websocket"}:
            raise HTTPException(status_code=400, detail="unsupported_vision_transport")
        websocket_url = str(
            req.websocket_url
            or getattr(ctx.vision, "websocket_url", "")
            or ctx.config.get("vision_engine", {}).get("websocket_url")
            or "ws://127.0.0.1:8000/ws"
        )
        try:
            ctx.vision.set_transport(transport, websocket_url=websocket_url)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        ctx.config.setdefault("vision_engine", {})["transport"] = transport
        ctx.config.setdefault("vision_engine", {})["websocket_url"] = websocket_url
        return {
            "status": "ok",
            "transport": transport,
            "websocket_url": websocket_url,
        }

    @router.get("/task_types")
    def task_types() -> Dict[str, Any]:
        types = []
        if ctx.executor and hasattr(ctx.executor, "list_task_types"):
            try:
                types = ctx.executor.list_task_types()
            except Exception:
                types = []
        return {"task_types": types}

    def _station_or_404(station_id: str) -> Dict[str, Any]:
        data = ctx.stations.get(station_id)
        if not data:
            raise HTTPException(status_code=404, detail="station_not_found")
        return data

    def _process_or_404(process_id: str) -> Dict[str, Any]:
        data = ctx.processes.get(process_id)
        if not data:
            raise HTTPException(status_code=404, detail="process_not_found")
        return data

    def _task_or_404(task_id: str) -> Dict[str, Any]:
        data = ctx.tasks.get(task_id)
        if not data:
            raise HTTPException(status_code=404, detail="task_not_found")
        return data

    def _asset_id_from(payload: Optional[Dict[str, Any]]) -> str:
        if not isinstance(payload, dict):
            return ""
        return str(payload.get("asset_id") or payload.get("process_id") or "").strip()

    def _with_asset_id(payload: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        data = dict(payload or {})
        asset_id = _asset_id_from(data)
        if asset_id:
            data["asset_id"] = asset_id
        data.pop("process_id", None)
        return data

    def _with_asset_ids(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [_with_asset_id(item) for item in (items or [])]

    def _default_station_id() -> str:
        stations = ctx.stations.list()
        if not stations:
            stations = [ctx.stations.ensure_default()]
        return stations[0]["station_id"]

    def _default_process_id() -> str:
        station_id = _default_station_id()
        processes = ctx.processes.list(station_id)
        if not processes:
            processes = [ctx.processes.ensure_default(station_id)]
        return _asset_id_from(processes[0])

    def _station_tool_tcp_offset(process_id: Optional[str] = None) -> Optional[np.ndarray]:
        try:
            asset_id = process_id or _default_process_id()
            process = ctx.processes.get(asset_id) or {}
            station_id = str(process.get("station_id") or _default_station_id())
            calibration = _read_station_combined_calibration(
                ctx.data_paths.station_calibration_dir(station_id)
            )
            tool_mount = calibration.get("tool_mount")
            custom_tcp = calibration.get("custom_tcp")
            if not isinstance(tool_mount, dict) or not isinstance(custom_tcp, dict):
                return None
            return _transform_section_matrix(tool_mount) @ _transform_section_matrix(custom_tcp)
        except Exception:
            return None

    def _attach_custom_tcp_pose(state: Dict[str, Any], process_id: Optional[str] = None) -> Dict[str, Any]:
        # Keep every caller using the same base->custom_tcp calculation as /robot/state.
        ee_pose = state.get("tcp_pose") or {}
        tool_tcp_offset = _station_tool_tcp_offset(process_id)
        if tool_tcp_offset is not None:
            base_pose = state.get("flange_pose") if isinstance(state.get("flange_pose"), dict) else ee_pose
            custom_pose = _matrix_to_pose(_pose_to_matrix(base_pose) @ tool_tcp_offset)
            if isinstance(ee_pose, dict) and isinstance(ee_pose.get("quat_xyzw"), list):
                ee_quat = _normalize_quat_xyzw(ee_pose.get("quat_xyzw") or [0.0, 0.0, 0.0, 1.0])
                custom_pose["quat_xyzw"] = ee_quat
                custom_pose["rotation_rpy_deg"] = _quat_xyzw_to_rpy_deg(ee_quat)
            custom_pose["tcp_frame"] = "tool_tcp"
            custom_pose["tcp_offset_m"] = [round(float(v), 6) for v in tool_tcp_offset[:3, 3].tolist()]
            custom_pose["source"] = "station_calibration_tcp"
            state["custom_tcp_pose"] = custom_pose
            state["tool_tcp_pose"] = dict(custom_pose)
            state["display_tcp_pose"] = dict(custom_pose)
            state.pop("custom_tcp_error", None)
        else:
            state.pop("custom_tcp_pose", None)
            state.pop("tool_tcp_pose", None)
            state.pop("display_tcp_pose", None)
            state["custom_tcp_error"] = "custom_tcp_transform_missing"
        return state

    def _current_ee_from_target_flange(target_flange: np.ndarray, state: Dict[str, Any]) -> np.ndarray:
        flange_pose = state.get("flange_pose")
        ee_pose = state.get("tcp_pose")
        if not isinstance(flange_pose, dict) or not isinstance(ee_pose, dict):
            return target_flange
        try:
            base_to_flange = _pose_to_matrix(flange_pose)
            base_to_ee = _pose_to_matrix(ee_pose)
            flange_to_ee = np.linalg.inv(base_to_flange) @ base_to_ee
            return target_flange @ flange_to_ee
        except Exception:
            return target_flange

    def _asset_kind_or_404(process_id: str, kind: str) -> Path:
        process = _process_or_404(process_id)
        try:
            return _asset_kind_dir(ctx, process, process_id, kind)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.get("/bin-picking/catalog")
    def bin_picking_catalog() -> Dict[str, Any]:
        root = _robot_asset_catalog_root()
        catalog = _read_json_file(root / "catalog.json") or {}

        def _items(kind: str) -> List[Dict[str, Any]]:
            entries = catalog.get(kind) if isinstance(catalog.get(kind), list) else []
            out: List[Dict[str, Any]] = []
            for raw in entries:
                try:
                    rel = _safe_relative_path(str(raw))
                    item_root = (root / rel).resolve()
                    if root.resolve() not in item_root.parents:
                        continue
                    item = _manifest_summary(
                        item_root,
                        public_base=f"/bin-picking/catalog/files/{rel.as_posix()}",
                    )
                    if item:
                        item["catalog_path"] = rel.as_posix()
                        out.append(item)
                except Exception:
                    continue
            return out

        return {
            "status": "ok",
            "robots": _items("robots"),
            "grippers": _items("grippers"),
            "assemblies": catalog.get("assemblies") or [],
        }

    @router.get("/bin-picking/catalog/files/{asset_path:path}")
    def bin_picking_catalog_file(asset_path: str) -> FileResponse:
        root = _robot_asset_catalog_root()
        try:
            path = _resolve_under(root, asset_path)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if not path.exists() or not path.is_file():
            raise HTTPException(status_code=404, detail="asset_file_not_found")
        return FileResponse(path, headers={"Cache-Control": "no-store"})

    @router.get("/bin-picking/gripper-assets")
    def bin_picking_gripper_assets() -> Dict[str, Any]:
        items: List[Dict[str, Any]] = []
        stations_root = ctx.data_paths.stations
        if stations_root.exists():
            for station_dir in sorted(p for p in stations_root.iterdir() if p.is_dir()):
                assets_root = station_dir / "assets"
                if not assets_root.exists():
                    continue
                for asset_dir in sorted(p for p in assets_root.iterdir() if p.is_dir()):
                    item = _asset_gripper_summary(ctx, station_dir.name, asset_dir.name)
                    if item:
                        items.append(item)
        return {"status": "ok", "grippers": items}

    @router.get("/bin-picking/robot-assets")
    def bin_picking_robot_assets() -> Dict[str, Any]:
        items: List[Dict[str, Any]] = []
        stations_root = ctx.data_paths.stations
        if stations_root.exists():
            for station_dir in sorted(p for p in stations_root.iterdir() if p.is_dir()):
                assets_root = station_dir / "assets"
                if not assets_root.exists():
                    continue
                for asset_dir in sorted(p for p in assets_root.iterdir() if p.is_dir()):
                    root = asset_dir / "robot"
                    if not root.exists() or not root.is_dir():
                        continue
                    item = robot_engine_manifest_summary(
                        root,
                        public_base=f"/processes/{asset_dir.name}/bin-picking/files/robot",
                    )
                    if not item:
                        continue
                    item["station_id"] = station_dir.name
                    item["asset_id"] = asset_dir.name
                    item["asset_path"] = f"data/stations/{station_dir.name}/assets/{asset_dir.name}/robot"
                    items.append(item)
        return {"status": "ok", "robots": items}

    @router.get("/processes/{process_id}/bin-picking/assets")
    def process_bin_picking_assets(process_id: str) -> Dict[str, Any]:
        _process_or_404(process_id)
        result: Dict[str, Any] = {"status": "ok", "asset_id": process_id}
        for kind in ("robot", "gripper"):
            root = _asset_kind_or_404(process_id, kind)
            item = _manifest_summary(
                root,
                public_base=f"/processes/{process_id}/bin-picking/files/{kind}",
            )
            result[kind] = item
        return result

    @router.get("/processes/{process_id}/robot-digital-twin/scene")
    def process_robot_digital_twin_scene(
        process_id: str,
        object_id: Optional[str] = None,
        task_type: Optional[str] = None,
        task_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        _process_or_404(process_id)
        try:
            scene, meta = build_robot_engine_scene_request(ctx, process_id, object_id, task_type=task_type, task_id=task_id)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"status": "ok", "scene": scene.model_dump(), "meta": meta}

    @router.post("/processes/{process_id}/robot-digital-twin/evaluate")
    def process_robot_digital_twin_evaluate(
        process_id: str,
        payload: Dict[str, Any],
        object_id: Optional[str] = None,
        task_type: Optional[str] = None,
        task_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        _process_or_404(process_id)
        try:
            return evaluate_robot_engine_scene(ctx, process_id, object_id, payload, task_type=task_type, task_id=task_id)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.get("/processes/{process_id}/robot-digital-twin/collision-debug")
    def process_robot_digital_twin_collision_debug(
        process_id: str,
        object_id: Optional[str] = None,
        task_type: Optional[str] = None,
        task_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        _process_or_404(process_id)
        try:
            return robot_engine_collision_debug_scene(ctx, process_id, object_id, task_type=task_type, task_id=task_id)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.post("/robot-engine/scene/load")
    def robot_engine_scene_load(payload: Dict[str, Any]) -> Dict[str, Any]:
        from robot_engine.interfaces.schemas import UISceneRequest
        from robot_engine.interfaces.ui_api import RobotEngineContext

        scene = UISceneRequest.model_validate(payload.get("scene") or payload)
        status = RobotEngineContext().load_scene_from_ui(scene)
        return {"status": "ok", "scene_status": status.model_dump()}

    @router.post("/robot-engine/scene/evaluate")
    def robot_engine_scene_evaluate(payload: Dict[str, Any]) -> Dict[str, Any]:
        from robot_engine.interfaces.schemas import UISceneEvaluationRequest
        from robot_engine.interfaces.ui_api import RobotEngineContext

        req = UISceneEvaluationRequest.model_validate(payload)
        result = RobotEngineContext().evaluate_scene_from_ui(req)
        return {"status": "ok", "result": result.model_dump()}

    @router.post("/robot-engine/collision/check")
    def robot_engine_collision_check(payload: Dict[str, Any]) -> Dict[str, Any]:
        from robot_engine.interfaces.schemas import CollisionCheckRequest, UISceneRequest
        from robot_engine.interfaces.ui_api import RobotEngineContext

        context = RobotEngineContext()
        scene_payload = payload.get("scene")
        if scene_payload:
            context.load_scene_from_ui(UISceneRequest.model_validate(scene_payload))
        result = context.check_collisions(CollisionCheckRequest.model_validate(payload.get("request") or {}))
        return {"status": "ok", "result": result.model_dump()}

    @router.post("/robot-engine/distance/query")
    def robot_engine_distance_query(payload: Dict[str, Any]) -> Dict[str, Any]:
        from robot_engine.interfaces.schemas import MinimumDistanceRequest, UISceneRequest
        from robot_engine.interfaces.ui_api import RobotEngineContext

        context = RobotEngineContext()
        scene_payload = payload.get("scene")
        if scene_payload:
            context.load_scene_from_ui(UISceneRequest.model_validate(scene_payload))
        result = context.query_minimum_distances(MinimumDistanceRequest.model_validate(payload.get("request") or {}))
        return {"status": "ok", "result": [item.model_dump() for item in result]}

    @router.post("/robot-engine/fk")
    def robot_engine_fk(payload: Dict[str, Any]) -> Dict[str, Any]:
        from robot_engine.interfaces.schemas import FKRequest
        from robot_engine.interfaces.ui_api import compute_fk

        result = compute_fk(FKRequest.model_validate(payload))
        return {"status": "ok", "result": result.model_dump()}

    @router.post("/robot-engine/jacobian")
    def robot_engine_jacobian(payload: Dict[str, Any]) -> Dict[str, Any]:
        from robot_engine.interfaces.schemas import JacobianRequest
        from robot_engine.interfaces.ui_api import compute_jacobian

        result = compute_jacobian(JacobianRequest.model_validate(payload))
        return {"status": "ok", "result": result.model_dump()}

    @router.post("/robot-engine/ik")
    def robot_engine_ik(payload: Dict[str, Any]) -> Dict[str, Any]:
        from robot_engine.interfaces.schemas import IKRequest
        from robot_engine.interfaces.ui_api import solve_ik

        result = solve_ik(IKRequest.model_validate(payload))
        return {"status": "ok", "result": result.model_dump()}

    @router.post("/robot-engine/grasp/evaluate")
    def robot_engine_grasp_evaluate(payload: Dict[str, Any]) -> Dict[str, Any]:
        from robot_engine.interfaces.schemas import GraspByIdRequest, UISceneRequest
        from robot_engine.interfaces.ui_api import RobotEngineContext

        context = RobotEngineContext()
        scene_payload = payload.get("scene")
        if scene_payload:
            context.load_scene_from_ui(UISceneRequest.model_validate(scene_payload))
        result = context.evaluate_grasp_by_id(GraspByIdRequest.model_validate(payload.get("request") or payload))
        return {"status": "ok", "result": result.model_dump()}

    @router.post("/robot-engine/motion/plan")
    def robot_engine_motion_plan(payload: Dict[str, Any]) -> Dict[str, Any]:
        from robot_engine.motion.motion_request import MotionRequest
        from robot_engine.motion.path_planner import plan_motion

        result = plan_motion(MotionRequest.model_validate(payload.get("request") or payload))
        return {"status": "ok", "result": result.model_dump()}

    def _dummy_testing_root(process_id: str) -> Path:
        process = _process_or_404(process_id)
        station_id = str(process.get("station_id") or "")
        if not station_id:
            raise HTTPException(status_code=400, detail="station_not_set")
        root = ctx.data_paths.process_dummy_testing_dir(station_id, process_id)
        root.mkdir(parents=True, exist_ok=True)
        (root / "obstacles").mkdir(parents=True, exist_ok=True)
        return root

    def _dummy_scene_read(root: Path) -> Dict[str, Any]:
        yaml_data = _dummy_scene_read_yaml(root / "scene.yaml")
        if yaml_data is not None:
            return yaml_data
        data = _read_json_file(root / "scene.json") or {}
        obstacles = data.get("obstacles") if isinstance(data.get("obstacles"), list) else []
        data["obstacles"] = obstacles
        return data

    def _dummy_scene_read_yaml(path: Path) -> Optional[Dict[str, Any]]:
        if not path.exists() or not path.is_file():
            return None
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except Exception:
            return None
        env = data.get("environment") if isinstance(data.get("environment"), dict) else {}
        raw_obstacles = env.get("obstacles") if isinstance(env.get("obstacles"), list) else []
        obstacles: List[Dict[str, Any]] = []
        for raw in raw_obstacles:
            if not isinstance(raw, dict):
                continue
            transform = raw.get("transform") if isinstance(raw.get("transform"), dict) else {}
            translation = transform.get("translation") if isinstance(transform.get("translation"), list) else [0.0, 0.0, 0.0]
            rotation = transform.get("rotation") if isinstance(transform.get("rotation"), list) else None
            item = {
                "id": str(raw.get("name") or raw.get("id") or "obstacle"),
                "name": str(raw.get("name") or raw.get("id") or "obstacle"),
                "mesh": str(raw.get("path") or raw.get("mesh") or ""),
                "pose": {
                    "position_m": [float(translation[i]) if i < len(translation) else 0.0 for i in range(3)],
                },
            }
            if rotation:
                item["pose"]["rotation_matrix"] = rotation
            obstacles.append(item)
        return {"obstacles": obstacles, "scene_yaml": data}

    def _dummy_scene_write(root: Path, scene: Dict[str, Any]) -> None:
        yaml_path = root / "scene.yaml"
        try:
            if yaml_path.exists():
                data = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
            else:
                data = {}
        except Exception:
            data = {}
        env = data.setdefault("environment", {})
        if not isinstance(env, dict):
            env = {}
            data["environment"] = env
        yaml_obstacles: List[Dict[str, Any]] = []
        for raw in scene.get("obstacles") or []:
            if not isinstance(raw, dict):
                continue
            obstacle_id = str(raw.get("id") or raw.get("name") or "obstacle").strip() or "obstacle"
            pose = raw.get("pose") if isinstance(raw.get("pose"), dict) else {}
            position = pose.get("position_m") if isinstance(pose.get("position_m"), list) else [0.0, 0.0, 0.0]
            rotation_matrix = pose.get("rotation_matrix") if isinstance(pose.get("rotation_matrix"), list) else None
            if rotation_matrix is None:
                rotation_matrix = _rpy_deg_to_matrix(pose.get("rotation_rpy_deg") or [0.0, 0.0, 0.0])
            yaml_obstacles.append({
                "name": obstacle_id,
                "type": raw.get("type") or "mesh",
                "path": str(raw.get("mesh") or raw.get("path") or ""),
                "transform": {
                    "translation": [float(position[i]) if i < len(position) else 0.0 for i in range(3)],
                    "rotation": rotation_matrix,
                },
            })
        env["obstacles"] = yaml_obstacles
        yaml_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
        json_path = root / "scene.json"
        if json_path.exists():
            json_path.unlink()

    def _rpy_deg_to_matrix(rpy_deg: Any) -> List[List[float]]:
        values = rpy_deg if isinstance(rpy_deg, list) else [0.0, 0.0, 0.0]
        roll = math.radians(float(values[0]) if len(values) > 0 else 0.0)
        pitch = math.radians(float(values[1]) if len(values) > 1 else 0.0)
        yaw = math.radians(float(values[2]) if len(values) > 2 else 0.0)
        cr, sr = math.cos(roll), math.sin(roll)
        cp, sp = math.cos(pitch), math.sin(pitch)
        cy, sy = math.cos(yaw), math.sin(yaw)
        return [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]

    @router.get("/processes/{process_id}/dummy-testing/scene")
    def dummy_testing_scene_get(process_id: str) -> Dict[str, Any]:
        root = _dummy_testing_root(process_id)
        data = _dummy_scene_read(root)
        return {"status": "ok", "asset_id": process_id, "scene": data}

    @router.put("/processes/{process_id}/dummy-testing/scene")
    def dummy_testing_scene_put(process_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        root = _dummy_testing_root(process_id)
        scene = payload.get("scene") if isinstance(payload.get("scene"), dict) else payload
        if not isinstance(scene, dict):
            raise HTTPException(status_code=400, detail="scene_required")
        obstacles = scene.get("obstacles")
        if obstacles is not None and not isinstance(obstacles, list):
            raise HTTPException(status_code=400, detail="obstacles_must_be_list")
        _dummy_scene_write(root, scene)
        return {"status": "saved", "asset_id": process_id, "scene": scene}

    @router.post("/processes/{process_id}/dummy-testing/obstacles/upload")
    async def dummy_testing_obstacle_upload(
        process_id: str,
        request: Request,
        filename: str = "obstacle.stl",
    ) -> Dict[str, Any]:
        root = _dummy_testing_root(process_id)
        clean_name = _safe_name(Path(filename or "obstacle").stem) or "obstacle"
        suffix = Path(filename or "").suffix.lower()
        if suffix not in {".stl", ".obj", ".dae", ".ply", ".glb", ".gltf"}:
            raise HTTPException(status_code=400, detail="unsupported_obstacle_mesh")
        body = await request.body()
        if not body:
            raise HTTPException(status_code=400, detail="empty_obstacle_mesh")
        dest_rel = Path("obstacles") / f"{clean_name}{suffix}"
        dest = root / dest_rel
        dest.write_bytes(body)
        scene = _dummy_scene_read(root)
        obstacles = [o for o in scene.get("obstacles", []) if isinstance(o, dict)]
        obstacle_id = clean_name
        existing = {str(o.get("id") or "") for o in obstacles}
        if obstacle_id in existing:
            obstacle_id = f"{clean_name}_{len(existing) + 1}"
        item = {
            "id": obstacle_id,
            "mesh": dest_rel.as_posix(),
            "pose": {
                "position_m": [0.0, 0.0, 0.0],
                "rotation_rpy_deg": [0.0, 0.0, 0.0],
            },
        }
        obstacles.append(item)
        scene["obstacles"] = obstacles
        _dummy_scene_write(root, scene)
        return {
            "status": "uploaded",
            "asset_id": process_id,
            "obstacle": item,
            "url": f"/processes/{process_id}/dummy-testing/files/{dest_rel.as_posix()}",
            "scene": scene,
        }

    @router.get("/processes/{process_id}/dummy-testing/files/{asset_path:path}")
    def dummy_testing_file(process_id: str, asset_path: str) -> FileResponse:
        root = _dummy_testing_root(process_id)
        try:
            path = _resolve_under(root, asset_path)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if not path.exists() or not path.is_file():
            raise HTTPException(status_code=404, detail="dummy_testing_file_not_found")
        return FileResponse(path, headers={"Cache-Control": "no-store"})

    def _task_scene_root(process_id: str, task_id: str) -> Path:
        process = _process_or_404(process_id)
        station_id = str(process.get("station_id") or "")
        if not station_id:
            raise HTTPException(status_code=400, detail="station_not_set")
        clean_task_id = _safe_name(task_id)
        if not clean_task_id:
            raise HTTPException(status_code=400, detail="invalid_task_id")
        root = ctx.data_paths.process_task_scene_dir(station_id, process_id, clean_task_id)
        root.mkdir(parents=True, exist_ok=True)
        (root / "obstacles").mkdir(parents=True, exist_ok=True)
        return root

    @router.get("/processes/{process_id}/tasks/{task_id}/scene")
    def task_scene_get(process_id: str, task_id: str) -> Dict[str, Any]:
        root = _task_scene_root(process_id, task_id)
        data = _dummy_scene_read(root)
        return {"status": "ok", "asset_id": process_id, "task_id": task_id, "scene": data}

    @router.put("/processes/{process_id}/tasks/{task_id}/scene")
    def task_scene_put(process_id: str, task_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        root = _task_scene_root(process_id, task_id)
        scene = payload.get("scene") if isinstance(payload.get("scene"), dict) else payload
        if not isinstance(scene, dict):
            raise HTTPException(status_code=400, detail="scene_required")
        obstacles = scene.get("obstacles")
        if obstacles is not None and not isinstance(obstacles, list):
            raise HTTPException(status_code=400, detail="obstacles_must_be_list")
        _dummy_scene_write(root, scene)
        return {"status": "saved", "asset_id": process_id, "task_id": task_id, "scene": scene}

    @router.post("/processes/{process_id}/tasks/{task_id}/obstacles/upload")
    async def task_obstacle_upload(
        process_id: str,
        task_id: str,
        request: Request,
        filename: str = "obstacle.stl",
    ) -> Dict[str, Any]:
        root = _task_scene_root(process_id, task_id)
        clean_name = _safe_name(Path(filename or "obstacle").stem) or "obstacle"
        suffix = Path(filename or "").suffix.lower()
        if suffix not in {".stl", ".obj", ".dae", ".ply", ".glb", ".gltf"}:
            raise HTTPException(status_code=400, detail="unsupported_obstacle_mesh")
        body = await request.body()
        if not body:
            raise HTTPException(status_code=400, detail="empty_obstacle_mesh")
        dest_rel = Path("obstacles") / f"{clean_name}{suffix}"
        (root / dest_rel).write_bytes(body)
        scene = _dummy_scene_read(root)
        obstacles = [o for o in scene.get("obstacles", []) if isinstance(o, dict)]
        obstacle_id = clean_name
        existing = {str(o.get("id") or "") for o in obstacles}
        if obstacle_id in existing:
            obstacle_id = f"{clean_name}_{len(existing) + 1}"
        item = {
            "id": obstacle_id,
            "mesh": dest_rel.as_posix(),
            "pose": {"position_m": [0.0, 0.0, 0.0], "rotation_rpy_deg": [0.0, 0.0, 0.0]},
        }
        obstacles.append(item)
        scene["obstacles"] = obstacles
        _dummy_scene_write(root, scene)
        return {
            "status": "uploaded",
            "asset_id": process_id,
            "task_id": task_id,
            "obstacle": item,
            "url": f"/processes/{process_id}/tasks/{task_id}/scene-files/{dest_rel.as_posix()}",
            "scene": scene,
        }

    @router.get("/processes/{process_id}/bin-picking/scene-files/{asset_path:path}")
    def bin_picking_scene_file(process_id: str, asset_path: str) -> FileResponse:
        process = _process_or_404(process_id)
        station_id = str(process.get("station_id") or "")
        if not station_id:
            raise HTTPException(status_code=400, detail="station_not_set")
        root = ctx.data_paths.process_dir(station_id, process_id) / "bin_picking"
        try:
            path = _resolve_under(root, asset_path)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if not path.exists() or not path.is_file():
            raise HTTPException(status_code=404, detail="bin_picking_scene_file_not_found")
        return FileResponse(path, headers={"Cache-Control": "no-store"})

    @router.get("/processes/{process_id}/tasks/{task_id}/scene-files/{asset_path:path}")
    def task_scene_file(process_id: str, task_id: str, asset_path: str) -> FileResponse:
        process = _process_or_404(process_id)
        station_id = str(process.get("station_id") or "")
        if not station_id:
            raise HTTPException(status_code=400, detail="station_not_set")
        clean_task_id = _safe_name(task_id)
        if not clean_task_id:
            raise HTTPException(status_code=400, detail="invalid_task_id")
        root = ctx.data_paths.process_task_scene_dir(station_id, process_id, clean_task_id)
        try:
            path = _resolve_under(root, asset_path)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if not path.exists() or not path.is_file():
            raise HTTPException(status_code=404, detail="task_scene_file_not_found")
        return FileResponse(path, headers={"Cache-Control": "no-store"})

    @router.post("/processes/{process_id}/bin-picking/assets/{kind}/select")
    def process_bin_picking_asset_select(
        process_id: str,
        kind: str,
        req: BinPickingAssetSelectRequest,
    ) -> Dict[str, Any]:
        if kind not in {"robot", "gripper"}:
            raise HTTPException(status_code=400, detail="unsupported_asset_kind")
        catalog_kind = "robots" if kind == "robot" else "grippers"
        catalog_root = _robot_asset_catalog_root()
        try:
            rel = _safe_relative_path(req.catalog_path)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if not rel.as_posix().startswith(f"{catalog_kind}/"):
            raise HTTPException(status_code=400, detail="catalog_kind_mismatch")
        source = (catalog_root / rel).resolve()
        if catalog_root.resolve() not in source.parents or not source.exists() or not source.is_dir():
            raise HTTPException(status_code=404, detail="catalog_asset_not_found")

        dest = _asset_kind_or_404(process_id, kind)
        if dest.exists() and any(dest.iterdir()):
            process = _process_or_404(process_id)
            station_id = str(process.get("station_id") or "")
            trash = ctx.data_paths.process_trash_dir(station_id, process_id)
            trash.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            shutil.move(str(dest), str(trash / f"{kind}-{stamp}"))
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(source, dest)

        selected = {
            "kind": kind,
            "catalog_path": rel.as_posix(),
            "copied_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        }
        (dest / "selection.json").write_text(json.dumps(selected, indent=2), encoding="utf-8")
        item = _manifest_summary(
            dest,
            public_base=f"/processes/{process_id}/bin-picking/files/{kind}",
        )
        return {"status": "selected", "asset_id": process_id, "kind": kind, kind: item}

    @router.get("/processes/{process_id}/bin-picking/files/{kind}/{asset_path:path}")
    def process_bin_picking_asset_file(
        process_id: str,
        kind: str,
        asset_path: str,
    ) -> FileResponse:
        root = _asset_kind_or_404(process_id, kind)
        try:
            path = _resolve_under(root, asset_path)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if not path.exists() or not path.is_file():
            raise HTTPException(status_code=404, detail="asset_file_not_found")
        return FileResponse(path, headers={"Cache-Control": "no-store"})

    @router.get("/processes/{process_id}/bin-picking/assets/{kind}/frames")
    def process_bin_picking_get_frames(process_id: str, kind: str) -> Dict[str, Any]:
        if kind not in {"robot", "gripper"}:
            raise HTTPException(status_code=400, detail="unsupported_asset_kind")
        root = _asset_kind_or_404(process_id, kind)
        frames = _read_json_file(root / "frames.json")
        if frames is None:
            raise HTTPException(status_code=404, detail="frames_not_found")
        return {"asset_id": process_id, "kind": kind, "frames": frames}

    @router.put("/processes/{process_id}/bin-picking/assets/{kind}/frames")
    def process_bin_picking_put_frames(
        process_id: str,
        kind: str,
        payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        if kind not in {"robot", "gripper"}:
            raise HTTPException(status_code=400, detail="unsupported_asset_kind")
        root = _asset_kind_or_404(process_id, kind)
        if not root.exists():
            raise HTTPException(status_code=404, detail="asset_not_found")
        frames = payload.get("frames") if isinstance(payload, dict) else None
        if not isinstance(frames, dict):
            raise HTTPException(status_code=400, detail="frames_required")
        (root / "frames.json").write_text(json.dumps(frames, indent=2), encoding="utf-8")
        return {"status": "saved", "asset_id": process_id, "kind": kind, "frames": frames}

    @router.get("/stations")
    def stations_list() -> Dict[str, Any]:
        stations = ctx.stations.list()
        if not stations:
            stations = [ctx.stations.ensure_default()]
        return {"stations": stations}

    @router.post("/stations")
    def stations_create(req: StationCreateRequest) -> Dict[str, Any]:
        payload = req.model_dump()
        payload["station_id"] = _safe_name(req.station_id or "") or None
        data = ctx.stations.create(payload)
        return data

    @router.get("/stations/{station_id}")
    def stations_get(station_id: str) -> Dict[str, Any]:
        return _station_or_404(station_id)

    @router.patch("/stations/{station_id}")
    def stations_patch(station_id: str, req: StationPatchRequest) -> Dict[str, Any]:
        patch = req.model_dump(exclude_unset=True)
        data = ctx.stations.patch(station_id, patch)
        if not data:
            raise HTTPException(status_code=404, detail="station_not_found")
        return data

    @router.get("/stations/{station_id}/processes")
    def processes_list(station_id: str) -> Dict[str, Any]:
        _station_or_404(station_id)
        processes = ctx.processes.list(station_id)
        if not processes:
            processes = [ctx.processes.ensure_default(station_id)]
        normalized = _with_asset_ids(processes)
        return {"processes": normalized}

    @router.post("/stations/{station_id}/processes")
    def processes_create(station_id: str, req: ProcessCreateRequest) -> Dict[str, Any]:
        _station_or_404(station_id)
        payload = req.model_dump()
        payload["asset_id"] = _safe_name(req.asset_id or req.process_id or "") or None
        payload.pop("process_id", None)
        try:
            data = ctx.processes.create(station_id, payload)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return _with_asset_id(data)

    @router.get("/processes/{process_id}")
    def processes_get(process_id: str) -> Dict[str, Any]:
        return _with_asset_id(_process_or_404(process_id))

    @router.patch("/processes/{process_id}")
    def processes_patch(process_id: str, req: ProcessPatchRequest) -> Dict[str, Any]:
        patch = req.model_dump(exclude_unset=True)
        if "task_type" in patch:
            patch.pop("task_type")
        data = ctx.processes.patch(process_id, patch)
        if not data:
            raise HTTPException(status_code=404, detail="process_not_found")
        return _with_asset_id(data)

    @router.delete("/processes/{process_id}")
    def processes_delete(process_id: str) -> Dict[str, Any]:
        if ctx.processes.delete(process_id):
            return {"status": "deleted", "asset_id": process_id}
        raise HTTPException(status_code=404, detail="process_not_found")

    @router.get("/processes/{process_id}/tasks")
    def tasks_list(process_id: str) -> Dict[str, Any]:
        _process_or_404(process_id)
        tasks = ctx.tasks.list(process_id)
        return {"tasks": _with_asset_ids(tasks)}

    @router.post("/processes/{process_id}/tasks")
    def tasks_create(process_id: str, req: TaskCreateRequest) -> Dict[str, Any]:
        _process_or_404(process_id)
        payload = req.model_dump()
        payload["task_id"] = _safe_name(req.task_id or "") or None
        payload["task"] = req.task
        try:
            data = ctx.tasks.create(process_id, payload)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return _with_asset_id(data)

    @router.get("/tasks/{task_id}")
    def tasks_get(task_id: str) -> Dict[str, Any]:
        task = _task_or_404(task_id)
        process = ctx.processes.get(_asset_id_from(task))
        if process:
            task["task_type"] = (
                task.get("task_type") or process.get("task_type") or "pick_place_demo"
            )
        return _with_asset_id(task)

    @router.patch("/tasks/{task_id}")
    def tasks_patch(task_id: str, req: TaskPatchRequest) -> Dict[str, Any]:
        patch = req.model_dump(exclude_unset=True)
        if patch.get("params") is not None and patch.get("task") is None:
            patch["task"] = patch.get("params")
        data = ctx.tasks.patch(task_id, patch)
        if not data:
            raise HTTPException(status_code=404, detail="task_not_found")
        process = ctx.processes.get(_asset_id_from(data))
        if process:
            data["task_type"] = (
                data.get("task_type") or process.get("task_type") or "pick_place_demo"
            )
        return _with_asset_id(data)

    @router.delete("/tasks/{task_id}")
    def tasks_delete(task_id: str) -> Dict[str, Any]:
        if ctx.tasks.delete(task_id):
            return {"status": "deleted", "task_id": task_id}
        raise HTTPException(status_code=404, detail="task_not_found")

    @router.post("/tasks/{task_id}/runs/start")
    def runs_start(task_id: str, req: RunStartRequest) -> Dict[str, Any]:
        task = _task_or_404(task_id)
        asset_id = _asset_id_from(task)
        process = _process_or_404(asset_id)
        station_id = task.get("station_id") or process.get("station_id")
        if not station_id:
            raise HTTPException(status_code=400, detail="station_id_missing")
        task_type = (
            task.get("task_type") or process.get("task_type") or "pick_place_demo"
        )
        meta = {
            "station_id": station_id,
            "asset_id": _asset_id_from(process),
            "task_id": task_id,
            "task_type": task_type,
            "state": "created",
            "params": req.params or {},
        }
        run_state = ctx.run_manager.create(meta, task.get("task") or {})
        run_state.params = req.params or {}
        if not ctx.executor.start(run_state):
            ctx.run_manager.set_state(
                run_state.run_id, "failed", {"reason": "task_handler_not_found"}
            )
            raise HTTPException(status_code=400, detail="task_handler_not_found")
        return {
            "run_id": run_state.run_id,
            "task_id": task_id,
            "asset_id": run_state.process_id,
            "station_id": run_state.station_id,
            "state": run_state.status,
        }

    @router.post("/runs/{run_id}/pause")
    def runs_pause(run_id: str) -> Dict[str, Any]:
        if ctx.executor.pause(run_id):
            return {"status": "paused", "run_id": run_id}
        meta = ctx.runs.get(run_id)
        if not meta:
            raise HTTPException(status_code=404, detail="run_not_found")
        state = meta.get("state")
        if state in ("completed", "failed", "aborted"):
            return {"status": state, "run_id": run_id, "note": "already_final"}
        last_req = meta.get("last_vision_request_id")
        if last_req:
            try:
                ctx.vision.stop_session(last_req)
            except Exception:
                pass
        try:
            ctx.robot.stop()
        except Exception:
            pass
        ctx.run_manager.set_state(run_id, "paused", {"reason": "pause_without_handle"})
        return {"status": "paused", "run_id": run_id, "note": "run_not_active"}

    @router.post("/runs/{run_id}/stop")
    def runs_stop(run_id: str) -> Dict[str, Any]:
        if ctx.executor.stop(run_id):
            return {"status": "stopping", "run_id": run_id}
        meta = ctx.runs.get(run_id)
        if not meta:
            raise HTTPException(status_code=404, detail="run_not_found")
        state = meta.get("state")
        if state in ("completed", "failed", "aborted"):
            return {"status": state, "run_id": run_id, "note": "already_final"}
        last_req = meta.get("last_vision_request_id")
        if last_req:
            try:
                ctx.vision.stop_session(last_req)
            except Exception:
                pass
        try:
            ctx.robot.stop()
        except Exception:
            pass
        ctx.run_manager.set_state(run_id, "aborted", {"reason": "stop_without_handle"})
        return {"status": "aborted", "run_id": run_id, "note": "run_not_active"}

    @router.get("/runs/{run_id}")
    def runs_get(run_id: str) -> Dict[str, Any]:
        meta = ctx.runs.get(run_id)
        if not meta:
            raise HTTPException(status_code=404, detail="run_not_found")
        meta = _with_asset_id(meta)
        handle = ctx.executor.get_handle(run_id)
        vision_request_id = handle.vision_request_id if handle else None
        last_vision_request_id = handle.last_vision_request_id if handle else None
        if not last_vision_request_id:
            last_vision_request_id = meta.get("last_vision_request_id")
        return {
            **meta,
            "vision_request_id": vision_request_id or last_vision_request_id,
            "last_vision_request_id": last_vision_request_id,
        }

    @router.delete("/runs/{run_id}")
    def runs_delete(run_id: str) -> Dict[str, Any]:
        handle = ctx.executor.get_handle(run_id)
        if handle:
            raise HTTPException(status_code=409, detail="run_active")
        if ctx.run_manager.delete(run_id):
            return {"status": "deleted", "run_id": run_id}
        raise HTTPException(status_code=404, detail="run_not_found")

    @router.get("/tasks/{task_id}/runs")
    def runs_list(task_id: str) -> Dict[str, Any]:
        _task_or_404(task_id)
        return {"runs": _with_asset_ids(ctx.runs.list_by_task(task_id))}

    @router.post("/vision/sessions/start")
    def vision_start(req: VisionStartRequest) -> Dict[str, Any]:
        payload = {
            "event": "VISION_START",
            "request_id": req.session_id,
            "camera_id": req.camera_id,
            "module": req.module,
            "fps_limit": req.fps_limit,
            "process_mode": req.process_mode,
            "params": req.params,
            "enable_shm_output": req.enable_shm_output,
        }
        _attach_station_vision_calibration(
            ctx,
            payload,
            camera_id=req.camera_id,
            station_id=req.station_id,
        )
        ctx.vision.start_session(payload)
        return {"status": "started", "session_id": req.session_id}

    @router.post("/vision/sessions/stop")
    def vision_stop(req: VisionStopRequest) -> Dict[str, Any]:
        ctx.vision.stop_session(req.session_id)
        return {"status": "stopped", "session_id": req.session_id}

    @router.post("/vision/capture")
    def vision_capture(req: VisionCaptureRequest) -> Dict[str, Any]:
        if not req.camera_id:
            raise HTTPException(status_code=400, detail="camera_id_required")
        request_id = f"capture-{int(time.time() * 1000)}"
        payload = {
            "event": "VISION_START",
            "request_id": request_id,
            "camera_id": req.camera_id,
            "module": req.module,
            "fps_limit": 0,
            "process_mode": "trigger_only",
            "params": req.params,
            "one_shot": True,
            "enable_shm_output": False,
        }
        _attach_station_vision_calibration(
            ctx,
            payload,
            camera_id=req.camera_id,
            station_id=req.station_id,
        )
        ctx.vision.start_session(payload)
        try:
            evt = ctx.vision_results.wait_for_result(request_id, float(req.timeout_s))
        finally:
            try:
                ctx.vision.stop_session(request_id)
            except Exception:
                pass
        if not evt:
            raise HTTPException(status_code=504, detail="vision_timeout")
        return {
            "status": "ok",
            "request_id": request_id,
            "frame_id": evt.get("frame_id"),
            "timestamp_ns": evt.get("timestamp_ns"),
            "result": evt.get("result"),
        }

    @router.get("/vision/preview")
    def vision_preview(session_id: str, timeout_ms: int = 500) -> Dict[str, Any]:
        if not session_id:
            raise HTTPException(status_code=400, detail="session_id_required")
        timeout_s = max(0.0, float(timeout_ms) / 1000.0)
        evt = ctx.vision_cache.wait_latest(session_id, timeout_s=timeout_s)
        if not evt:
            return {"status": "pending", "session_id": session_id}
        result = evt.get("result") or {}
        return {
            "status": "ok",
            "session_id": session_id,
            "frame_id": evt.get("frame_id"),
            "timestamp_ns": evt.get("timestamp_ns"),
            "result": result,
        }

    @router.get("/vision/stream")
    async def vision_stream(
        request: Request,
        request_id: str,
        timeout_ms: int = 1000,
    ) -> StreamingResponse:
        if not request_id:
            raise HTTPException(status_code=400, detail="request_id_required")
        timeout_s = max(0.0, float(timeout_ms) / 1000.0)

        async def event_generator():
            last_frame = None
            while True:
                if await request.is_disconnected():
                    break
                evt = await asyncio.to_thread(
                    ctx.vision_cache.wait_latest, request_id, timeout_s
                )
                if not evt:
                    yield ": keepalive\n\n"
                    continue
                frame_id = evt.get("frame_id")
                if frame_id == last_frame:
                    continue
                last_frame = frame_id
                evt_request_id = evt.get("request_id")
                fps_instant = (
                    ctx.vision_cache.get_fps_instant(evt_request_id)
                    if evt_request_id
                    else None
                )
                fps_estimate = (
                    ctx.vision_cache.get_fps_estimate(evt_request_id)
                    if evt_request_id
                    else None
                )
                result_fps_instant = (
                    ctx.vision_cache.get_result_fps_instant(evt_request_id)
                    if evt_request_id
                    else None
                )
                result_fps_estimate = (
                    ctx.vision_cache.get_result_fps_estimate(evt_request_id)
                    if evt_request_id
                    else None
                )
                publish_fps_instant = (
                    ctx.vision_cache.get_publish_fps_instant(evt_request_id)
                    if evt_request_id
                    else None
                )
                publish_fps_estimate = (
                    ctx.vision_cache.get_publish_fps_estimate(evt_request_id)
                    if evt_request_id
                    else None
                )
                payload = {
                    "frame_id": frame_id,
                    "timestamp_ns": evt.get("timestamp_ns"),
                    "produced_timestamp_ns": evt.get("produced_timestamp_ns"),
                    "produced_latency_ms": evt.get("produced_latency_ms"),
                    "process_time_ms": evt.get("process_time_ms"),
                    "fps_instant": fps_instant,
                    "fps_estimate": fps_estimate,
                    "result_fps_instant": result_fps_instant,
                    "result_fps_estimate": result_fps_estimate,
                    "publish_fps_instant": publish_fps_instant,
                    "publish_fps_estimate": publish_fps_estimate,
                    "result": evt.get("result") or {},
                }
                yield f"data: {json.dumps(payload)}\n\n"

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-store"},
        )

    @router.get("/camera/cameras")
    def camera_cameras() -> Dict[str, Any]:
        cameras = ctx.camera_cache.list_cameras()
        current = set(cameras)
        with state_lock:
            nonlocal camera_connected
            joined = sorted(current - camera_connected)
            left = sorted(camera_connected - current)
            camera_connected = current
        for camera_id in joined:
            log.info("camera_connected camera_id=%s", camera_id)
        for camera_id in left:
            log.warning("camera_disconnected camera_id=%s", camera_id)
        return {"status": "ok", "count": len(cameras), "cameras": cameras}

    @router.get("/camera/fps")
    def camera_fps(camera_id: Optional[str] = None) -> Dict[str, Any]:
        if camera_id:
            fps = ctx.camera_cache.get_camera_fps(camera_id)
            return {"status": "ok", "camera_id": camera_id, "fps": fps}
        fps_map = ctx.camera_cache.get_all_camera_fps()
        return {"status": "ok", "fps": fps_map}

    @router.get("/camera/frame")
    def camera_frame(
        camera_id: Optional[str] = None,
        frame_id: Optional[str] = None,
        fmt: str = "jpg",
        quality: int = 75,
    ) -> Response:
        if frame_id:
            evt = ctx.camera_cache.get_by_frame_id(frame_id, camera_id)
        else:
            evt = (
                ctx.camera_cache.get_latest(camera_id)
                if camera_id
                else ctx.camera_cache.get_latest_any()
            )
        if not evt:
            raise HTTPException(status_code=404, detail="no_frame")
        try:
            payload = _read_rgb_frame(evt)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        frame = payload["frame"]
        fmt = fmt.lower().strip(".")
        if fmt not in ("jpg", "jpeg", "png"):
            fmt = "jpg"
        ext = ".png" if fmt == "png" else ".jpg"
        encode_params: List[int] = []
        if fmt == "png":
            encode_params = [int(cv2.IMWRITE_PNG_COMPRESSION), 3]
            media_type = "image/png"
        else:
            quality = max(10, min(int(quality), 95))
            encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
            media_type = "image/jpeg"
        ok, buf = cv2.imencode(ext, frame, encode_params)
        if not ok:
            raise HTTPException(status_code=500, detail="encode_failed")
        seq = payload.get("sequence_id")
        cam = payload.get("camera_id") or (camera_id or "")
        frame_id = f"{cam}:{seq}" if cam and seq is not None else ""
        return Response(
            content=buf.tobytes(),
            media_type=media_type,
            headers={
                "Cache-Control": "no-store",
                "X-Frame-Id": frame_id,
                "X-Camera-Id": str(cam),
                "X-Sequence-Id": str(seq or ""),
            },
        )

    @router.get("/camera/calibration/status")
    def camera_calibration_status() -> Dict[str, Any]:
        url = f"{_camera_calibration_control_url(ctx)}/calibration/status"
        res = _http_json(url, method="GET")
        if not res.get("ok"):
            raise HTTPException(
                status_code=int(res.get("status", 503)),
                detail=res.get("error", "camera_core_unreachable"),
            )
        return res.get("body", {})

    @router.post("/camera/calibration/start")
    def camera_calibration_start() -> Dict[str, Any]:
        url = f"{_camera_calibration_control_url(ctx)}/calibration/start"
        res = _http_json(url, method="POST")
        if not res.get("ok"):
            raise HTTPException(
                status_code=int(res.get("status", 503)),
                detail=res.get("error", "camera_core_unreachable"),
            )
        return res.get("body", {})

    @router.post("/camera/calibration/stop")
    def camera_calibration_stop() -> Dict[str, Any]:
        url = f"{_camera_calibration_control_url(ctx)}/calibration/stop"
        res = _http_json(url, method="POST")
        if not res.get("ok"):
            raise HTTPException(
                status_code=int(res.get("status", 503)),
                detail=res.get("error", "camera_core_unreachable"),
            )
        return res.get("body", {})

    @router.get("/camera/calibration/frame")
    def camera_calibration_frame(camera_id: str, quality: int = 75) -> Response:
        base_url = _camera_calibration_frame_url(ctx)
        url = f"{base_url}/frame?camera_id={camera_id}&quality={int(quality)}"
        try:
            with urlopen(url, timeout=2.5) as resp:
                payload = resp.read()
                media_type = resp.headers.get("Content-Type", "image/jpeg")
                frame_id = resp.headers.get("X-Frame-Id", "")
                camera_name = resp.headers.get("X-Camera-Id", camera_id)
                sequence_id = resp.headers.get("X-Sequence-Id", "")
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="ignore") if hasattr(exc, "read") else str(exc)
            if int(exc.code or 0) == 404 and "no_frame" in str(detail):
                evt = ctx.camera_cache.get_latest(camera_id)
                if not evt:
                    raise HTTPException(status_code=404, detail="no_frame") from exc
                try:
                    payload_dict = _read_rgb_frame(evt)
                except Exception as fallback_exc:
                    raise HTTPException(status_code=500, detail=str(fallback_exc)) from fallback_exc
                frame = payload_dict["frame"]
                ok, buf = cv2.imencode(
                    ".jpg",
                    frame,
                    [int(cv2.IMWRITE_JPEG_QUALITY), max(20, min(int(quality), 95))],
                )
                if not ok:
                    raise HTTPException(status_code=500, detail="encode_failed") from exc
                seq = payload_dict.get("sequence_id")
                cam = payload_dict.get("camera_id") or camera_id
                frame_id = f"{cam}:{seq}" if cam and seq is not None else ""
                camera_name = str(cam or camera_id)
                sequence_id = str(seq or "")
                return Response(
                    content=buf.tobytes(),
                    media_type="image/jpeg",
                    headers={
                        "Cache-Control": "no-store",
                        "X-Frame-Id": frame_id,
                        "X-Camera-Id": camera_name,
                        "X-Sequence-Id": sequence_id,
                    },
                )
            raise HTTPException(status_code=int(exc.code or 503), detail=detail or "camera_calibration_frame_failed") from exc
        except URLError as exc:
            raise HTTPException(status_code=503, detail=f"camera_calibration_unreachable:{exc}") from exc
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"camera_calibration_frame_failed:{exc}") from exc

        return Response(
            content=payload,
            media_type=media_type,
            headers={
                "Cache-Control": "no-store",
                "X-Frame-Id": frame_id,
                "X-Camera-Id": camera_name,
                "X-Sequence-Id": sequence_id,
            },
        )

    @router.api_route("/camera/calibration/api/{path:path}", methods=["GET", "POST"])
    async def camera_calibration_api_proxy(path: str, request: Request) -> Response:
        base_url = _camera_calibration_api_url(ctx)
        upstream_path = str(path or "").lstrip("/")
        query = request.url.query
        url = f"{base_url}/{upstream_path}"
        if query:
            url = f"{url}?{query}"

        body = await request.body()
        headers: Dict[str, str] = {}
        content_type = request.headers.get("content-type")
        if content_type:
            headers["Content-Type"] = content_type

        req = URLRequest(
            url,
            data=body if request.method.upper() != "GET" else None,
            headers=headers,
            method=request.method.upper(),
        )
        try:
            with urlopen(req, timeout=5.0) as resp:
                payload = resp.read()
                media_type = resp.headers.get("Content-Type", "application/json")
                return Response(content=payload, media_type=media_type)
        except HTTPError as exc:
            detail = exc.read()
            media_type = exc.headers.get("Content-Type", "application/json")
            return Response(content=detail, status_code=int(exc.code or 500), media_type=media_type)
        except URLError as exc:
            raise HTTPException(status_code=503, detail=f"camera_calibration_unreachable:{exc}") from exc

    @router.get("/vision/cameras")
    def vision_cameras() -> Dict[str, Any]:
        url = _vision_http_url(ctx)
        transport = str(
            getattr(ctx.vision, "transport", ctx.config.get("vision_engine", {}).get("transport", "zmq"))
            or "zmq"
        )
        try:
            with urlopen(f"{url}/ready", timeout=1.0) as resp:
                payload = resp.read().decode("utf-8")
            data = json.loads(payload)
        except Exception as exc:
            return {
                "status": "error",
                "transport": transport,
                "known_cameras": [],
                "error": str(exc),
            }
        known = data.get("known_cameras") if isinstance(data, dict) else []
        if not isinstance(known, list):
            known = []
        return {
            "status": "ok",
            "transport": transport,
            "known_cameras": known,
            "engine_running": (
                data.get("engine_running") if isinstance(data, dict) else False
            ),
        }

    @router.get("/vision/latest")
    def vision_latest(
        request_id: Optional[str] = None,
        frame_id: Optional[str] = None,
        include_image: bool = True,
    ) -> Dict[str, Any]:
        if frame_id:
            if request_id:
                evt = ctx.vision_cache.get_by_frame(request_id, frame_id)
            else:
                evt = ctx.vision_cache.get_by_frame_any(frame_id)
        else:
            evt = (
                ctx.vision_cache.get_latest(request_id)
                if request_id
                else ctx.vision_cache.get_latest_any()
            )
        if not evt:
            return {"status": "pending"}
        result = evt.get("result")
        if result and not include_image and isinstance(result, dict):
            result = dict(result)
            result.pop("image_b64", None)
        evt_request_id = evt.get("request_id")
        fps_instant = (
            ctx.vision_cache.get_fps_instant(evt_request_id) if evt_request_id else None
        )
        fps_estimate = (
            ctx.vision_cache.get_fps_estimate(evt_request_id)
            if evt_request_id
            else None
        )
        result_fps_instant = (
            ctx.vision_cache.get_result_fps_instant(evt_request_id)
            if evt_request_id
            else None
        )
        result_fps_estimate = (
            ctx.vision_cache.get_result_fps_estimate(evt_request_id)
            if evt_request_id
            else None
        )
        publish_fps_instant = (
            ctx.vision_cache.get_publish_fps_instant(evt_request_id)
            if evt_request_id
            else None
        )
        publish_fps_estimate = (
            ctx.vision_cache.get_publish_fps_estimate(evt_request_id)
            if evt_request_id
            else None
        )
        return {
            "status": "ok",
            "request_id": evt_request_id,
            "frame_id": evt.get("frame_id"),
            "sequence_id": evt.get("sequence_id"),
            "timestamp_ns": evt.get("timestamp_ns"),
            "produced_timestamp_ns": evt.get("produced_timestamp_ns"),
            "produced_monotonic_ns": evt.get("produced_monotonic_ns"),
            "produced_latency_ms": evt.get("produced_latency_ms"),
            "process_time_ms": evt.get("process_time_ms"),
            "fps_instant": fps_instant,
            "fps_estimate": fps_estimate,
            "result_fps_instant": result_fps_instant,
            "result_fps_estimate": result_fps_estimate,
            "publish_fps_instant": publish_fps_instant,
            "publish_fps_estimate": publish_fps_estimate,
            "result": result,
        }

    @router.get("/vision/frame")
    def vision_frame(
        request_id: Optional[str] = None,
        frame_id: Optional[str] = None,
    ) -> Response:
        if frame_id:
            if request_id:
                evt = ctx.vision_cache.get_by_frame(request_id, frame_id)
            else:
                evt = ctx.vision_cache.get_by_frame_any(frame_id)
        else:
            evt = (
                ctx.vision_cache.get_latest_image(request_id)
                if request_id
                else ctx.vision_cache.get_latest_image_any()
            )
        if not evt:
            raise HTTPException(status_code=404, detail="no_frame")
        result = evt.get("result") or {}
        payload = result.get("image_b64")
        if not payload:
            raise HTTPException(status_code=404, detail="no_image")
        fmt = str(result.get("format") or "jpg").lower()
        if fmt == "jpeg":
            fmt = "jpg"
        media_type = "image/png" if fmt == "png" else "image/jpeg"
        try:
            raw = base64.b64decode(payload)
        except Exception:
            raise HTTPException(status_code=500, detail="image_decode_failed")
        return Response(
            content=raw,
            media_type=media_type,
            headers={
                "Cache-Control": "no-store",
                "X-Frame-Id": str(evt.get("frame_id") or ""),
                "X-Request-Id": str(evt.get("request_id") or ""),
            },
        )

    @router.get("/debug/image")
    def debug_image(path: str) -> Response:
        raw_path = str(path or "").strip()
        if not raw_path:
            raise HTTPException(status_code=400, detail="missing_path")
        try:
            resolved = Path(raw_path).resolve(strict=True)
        except Exception:
            raise HTTPException(status_code=404, detail="not_found")
        data_root = ctx.data_root.resolve()
        try:
            resolved.relative_to(data_root)
        except Exception:
            raise HTTPException(status_code=403, detail="path_outside_data_root")
        suffix = resolved.suffix.lower()
        if suffix not in {".png", ".jpg", ".jpeg"}:
            raise HTTPException(status_code=400, detail="unsupported_image_type")
        media_type = "image/png" if suffix == ".png" else "image/jpeg"
        return FileResponse(str(resolved), media_type=media_type, headers={"Cache-Control": "no-store"})

    @router.get("/debug/file")
    def debug_file(path: str) -> Response:
        raw_path = str(path or "").strip()
        if not raw_path:
            raise HTTPException(status_code=400, detail="missing_path")
        try:
            resolved = Path(raw_path).resolve(strict=True)
        except Exception:
            raise HTTPException(status_code=404, detail="not_found")
        data_root = ctx.data_root.resolve()
        try:
            resolved.relative_to(data_root)
        except Exception:
            raise HTTPException(status_code=403, detail="path_outside_data_root")

        suffix = resolved.suffix.lower()
        media_types = {
            ".glb": "model/gltf-binary",
            ".gltf": "model/gltf+json",
            ".ply": "application/octet-stream",
            ".json": "application/json",
            ".txt": "text/plain; charset=utf-8",
        }
        media_type = media_types.get(suffix)
        if media_type is None:
            raise HTTPException(status_code=400, detail="unsupported_file_type")
        return FileResponse(str(resolved), media_type=media_type, headers={"Cache-Control": "no-store"})

    @router.post("/robot/model")
    def robot_model(req: RobotModelRequest) -> Dict[str, Any]:
        if not req.model:
            raise HTTPException(status_code=400, detail="model is required")
        ctx.robot.set_robot_model(req.model)
        return {"status": "ok", "model": req.model}

    @router.get("/robot/state")
    def robot_state() -> Dict[str, Any]:
        state = ctx.robot.get_state()
        connected = bool(state.get("connected"))
        with state_lock:
            nonlocal robot_connected
            prev = robot_connected
            robot_connected = connected
        if prev is None:
            if connected:
                log.info("robot_connected")
            else:
                log.warning("robot_disconnected")
        elif prev != connected:
            if connected:
                log.info("robot_connected")
            else:
                log.warning("robot_disconnected")
        return _attach_custom_tcp_pose(state)

    @router.get("/robot/state/stream")
    async def robot_state_stream(request: Request) -> StreamingResponse:
        """SSE stream of robot state at ~50 Hz using the PUB/SUB cache.

        The client receives a ``data: <json>\\n\\n`` event whenever the cached
        state changes (joint positions differ by more than float epsilon).
        Falls back to a 100 ms keepalive if the state adapter has no subscriber.
        """
        async def _generate():
            last_q_sig = None
            while True:
                if await request.is_disconnected():
                    break
                state = await asyncio.to_thread(ctx.robot.get_state)
                q = state.get("q") or []
                q_sig = ",".join(f"{v:.5f}" for v in q) if q else ""
                if q_sig != last_q_sig:
                    last_q_sig = q_sig
                    payload = _attach_custom_tcp_pose(state)
                    yield f"data: {json.dumps(payload, separators=(',', ':'))}\n\n"
                else:
                    yield ": keepalive\n\n"
                await asyncio.sleep(0.02)  # 50 Hz ceiling

        return StreamingResponse(
            _generate(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
        )

    @router.post("/recipes/load")
    def recipes_load(req: RecipeLoadRequest) -> Dict[str, Any]:
        if req.path:
            path = Path(req.path)
            if not path.is_absolute():
                path = (ctx.data_paths.legacy_recipes / path).resolve()
            if ctx.data_paths.legacy_recipes.resolve() not in path.parents:
                raise HTTPException(
                    status_code=400, detail="recipe path must be under data/recipes"
                )
            data = _read_recipe_file(path)
            if not data:
                raise HTTPException(status_code=404, detail="recipe_not_found")
            return {
                "status": "loaded",
                "recipe_id": data.get("recipe_id") or path.stem,
                "recipe": data,
            }
        if req.recipe:
            data = dict(req.recipe)
            if req.recipe_id:
                data["recipe_id"] = req.recipe_id
            return {
                "status": "loaded",
                "recipe_id": data.get("recipe_id"),
                "recipe": data,
            }
        raise HTTPException(status_code=400, detail="path or recipe is required")

    @router.get("/recipes/{recipe_id}")
    def recipes_get(recipe_id: str) -> Dict[str, Any]:
        name = _safe_name(recipe_id)
        if not name:
            raise HTTPException(status_code=400, detail="invalid_recipe_id")
        path = _find_recipe_path(ctx.data_paths.legacy_recipes, name)
        if not path:
            raise HTTPException(status_code=404, detail="recipe_not_found")
        data = _read_recipe_file(path)
        if not data:
            raise HTTPException(status_code=404, detail="recipe_not_found")
        return data

    @router.get("/runs/{run_id}/timeline")
    def run_timeline(run_id: str, limit: int = 50) -> Dict[str, Any]:
        safe_id = _safe_name(run_id)
        if not safe_id:
            raise HTTPException(status_code=400, detail="invalid_run_id")
        events = ctx.runs.get_timeline(safe_id, limit=limit)
        return {"run_id": safe_id, "events": events}

    @router.get("/runs/{run_id}/latest-pose")
    def run_latest_pose(run_id: str) -> Dict[str, Any]:
        """Get the latest 6D pose from a run's vision results or timeline."""
        safe_id = _safe_name(run_id)
        if not safe_id:
            raise HTTPException(status_code=400, detail="invalid_run_id")

        run_meta = ctx.runs.get(safe_id) or {}
        handle = ctx.executor.get_handle(safe_id) if getattr(ctx, "executor", None) else None
        request_ids = []
        for candidate in (
            getattr(handle, "vision_request_id", None),
            getattr(handle, "last_vision_request_id", None),
            run_meta.get("last_vision_request_id"),
            f"{safe_id}-vision",
        ):
            value = str(candidate or "").strip()
            if value and value not in request_ids:
                request_ids.append(value)

        # Try vision cache first (if available)
        try:
            for request_id in request_ids:
                latest_vision = ctx.vision_cache.get_latest(request_id)
                if latest_vision and latest_vision.get("event") == "VISION_RESULT":
                    result = latest_vision.get("result") or {}
                    matches = result.get("matches") or []
                    if matches:
                        match = matches[0]  # First match is best
                        return {
                            "status": "ok",
                            "source": "vision_cache",
                            "run_id": safe_id,
                            "request_id": request_id,
                            "frame_id": latest_vision.get("frame_id", ""),
                            "confidence": match.get("confidence", 0),
                            "position_2d": match.get("center_uv", [0, 0]),
                            "yaw_deg": match.get("yaw_deg", 0),
                            "inliers": match.get("inliers", 0),
                            "bbox_xywh": match.get("bbox_xywh"),
                            "obb_points": match.get("obb_points"),
                            "pose_6d": {
                                "confidence": match.get("confidence", 0),
                                "center_uv": match.get("center_uv", [0, 0]),
                                "yaw_deg": match.get("yaw_deg", 0),
                                "inliers": match.get("inliers", 0),
                            },
                        }
        except Exception:
            pass  # Fall through to timeline

        # Fallback: Get from timeline events (for historical data)
        events = ctx.runs.get_timeline(safe_id, limit=100)

        # Look for pose events in reverse order (most recent first)
        for evt in reversed(events):
            evt_type = evt.get("event", "")

            # Try to extract pose from event
            if evt_type in ["FOLLOW_POSE", "PICK_PLACE_POSE"]:
                if "pose" in evt:
                    pose_data = evt.get("pose", {})
                    if pose_data:
                        return {
                            "status": "ok",
                            "source": "timeline_pose",
                            "run_id": safe_id,
                            "event": evt_type,
                            "frame_id": evt.get("frame_id", ""),
                            "confidence": pose_data.get("confidence", 0),
                            "position_m": pose_data.get(
                                "position", evt.get("position", None)
                            ),
                            "quat_xyzw": pose_data.get("quat", evt.get("quat", None)),
                            "yaw_deg": pose_data.get("yaw_deg", 0),
                            "inliers": pose_data.get("inliers", 0),
                            "pose_6d": pose_data,
                        }

            # Alternative: extract pose from debug match data
            if evt_type in ["FOLLOW_DEBUG", "PICK_PLACE_DEBUG"]:
                if evt.get("stage") == "vision_match" and "match" in evt:
                    match_data = evt.get("match", {})
                    if match_data and match_data.get("confidence", 0) > 0:
                        center = match_data.get("center_uv") or match_data.get(
                            "center", [0, 0]
                        )
                        return {
                            "status": "ok",
                            "source": "timeline_debug",
                            "run_id": safe_id,
                            "event": evt_type,
                            "frame_id": evt.get("frame_id", ""),
                            "confidence": match_data.get("confidence", 0),
                            "position_2d": center,
                            "yaw_deg": match_data.get("yaw_deg", 0),
                            "inliers": match_data.get("inliers", 0),
                            "pose_6d": {
                                "confidence": match_data.get("confidence", 0),
                                "center_uv": center,
                                "yaw_deg": match_data.get("yaw_deg", 0),
                                "inliers": match_data.get("inliers", 0),
                            },
                        }

        # No pose found yet: return pending instead of 404 to avoid poll-noise in logs/UI.
        return {
            "status": "pending",
            "source": "none",
            "run_id": safe_id,
            "detail": "no_pose_available",
        }

    @router.get("/stations/{station_id}/calibration/handeye")
    def handeye_get(station_id: str) -> Dict[str, Any]:
        _station_or_404(station_id)
        calib_dir = ctx.data_paths.station_calibration_dir(station_id)
        combined = _read_station_combined_calibration(calib_dir)
        payload = _flatten_hand_eye(combined) if combined else None
        if payload is None:
            legacy_path = calib_dir / "handeye.json"
            if legacy_path.exists():
                payload = json.loads(legacy_path.read_text(encoding="utf-8"))
        if payload is None:
            return {"status": "pending", "detail": "handeye_not_set"}
        if combined:
            payload.update(_tcp_fields_for_handeye(_flatten_custom_tcp(combined)))
            payload["combined_calibration"] = combined
        return payload

    @router.get("/stations/{station_id}/calibration/tcp")
    def calibration_tcp_get(station_id: str) -> Dict[str, Any]:
        _station_or_404(station_id)
        calib_dir = ctx.data_paths.station_calibration_dir(station_id)
        combined = _read_station_combined_calibration(calib_dir)
        if combined:
            tcp_payload = _flatten_custom_tcp(combined, include_status=True)
            if isinstance(combined.get("tool_mount"), dict):
                tcp_payload["tool_mount"] = combined["tool_mount"]
            tcp_payload["combined_calibration"] = combined
            return tcp_payload
        handeye_path = calib_dir / "handeye.json"
        if handeye_path.exists():
            handeye = json.loads(handeye_path.read_text(encoding="utf-8"))
            if _has_tcp_calibration_fields(handeye):
                return _normalize_calibration_tcp_payload(handeye, include_status=True)
        return {"status": "pending", "detail": "tcp_calibration_not_set"}

    @router.post("/stations/{station_id}/calibration/tcp")
    def calibration_tcp_set(station_id: str, req: CalibrationTcpRequest) -> Dict[str, Any]:
        _station_or_404(station_id)
        calib_dir = ctx.data_paths.station_calibration_dir(station_id)
        calib_dir.mkdir(parents=True, exist_ok=True)
        path = calib_dir / "tcp.json"
        now = _calibration_timestamp()
        current = _read_station_combined_calibration(calib_dir)
        payload = _combined_calibration_payload(
            current,
            tcp_raw=req.model_dump(exclude_none=True),
            saved_at=now,
        )
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        handeye_path = calib_dir / "handeye.json"
        if handeye_path.exists():
            handeye_path.unlink()
        return {
            "status": "saved",
            "path": str(path),
            "tcp_calibration": _flatten_custom_tcp(payload, include_status=True),
            "combined_calibration": payload,
        }

    @router.post("/stations/{station_id}/calibration/handeye")
    def handeye_set(station_id: str, req: HandEyeRequest) -> Dict[str, Any]:
        _station_or_404(station_id)
        calib_dir = ctx.data_paths.station_calibration_dir(station_id)
        calib_dir.mkdir(parents=True, exist_ok=True)
        path = calib_dir / "tcp.json"
        now = _calibration_timestamp()
        current = _read_station_combined_calibration(calib_dir)
        hand_eye_payload: Dict[str, Any] = {
            "translation_m": req.translation_m,
            "saved_at": now,
        }
        if req.gripper_x_offset_m is not None:
            hand_eye_payload["gripper_x_offset_m"] = req.gripper_x_offset_m
        if req.gripper_y_offset_m is not None:
            hand_eye_payload["gripper_y_offset_m"] = req.gripper_y_offset_m
        if req.gripper_z_offset_m is not None:
            hand_eye_payload["gripper_z_offset_m"] = req.gripper_z_offset_m
        req_data = req.model_dump(exclude_none=True)
        if req.rotation_rpy_deg is not None:
            hand_eye_payload["rotation_rpy_deg"] = req.rotation_rpy_deg
        elif req.rotation_quat_xyzw is not None:
            hand_eye_payload["rotation_quat_xyzw"] = req.rotation_quat_xyzw
        if req.hand_eye_frame:
            hand_eye_payload["hand_eye_frame"] = req.hand_eye_frame
        if isinstance(req.handeye_result, dict):
            hand_eye_payload["handeye_result"] = req.handeye_result
        payload = _combined_calibration_payload(
            current,
            tcp_raw=req_data if _has_tcp_calibration_fields(req_data) else current,
            hand_eye_raw=hand_eye_payload,
            saved_at=now,
        )
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        legacy_path = calib_dir / "handeye.json"
        if legacy_path.exists():
            legacy_path.unlink()
        return {"status": "saved", "path": str(path)}

    @router.get("/stations/{station_id}/calibration/intrinsics")
    def intrinsics_get(station_id: str) -> Dict[str, Any]:
        _station_or_404(station_id)
        path = ctx.data_paths.station_calibration_dir(station_id) / "intrinsics.json"
        if not path.exists():
            return {"status": "pending", "detail": "intrinsics_not_set"}
        return json.loads(path.read_text(encoding="utf-8"))

    @router.post("/stations/{station_id}/calibration/intrinsics")
    def intrinsics_set(station_id: str, req: IntrinsicsRequest) -> Dict[str, Any]:
        _station_or_404(station_id)
        calib_dir = ctx.data_paths.station_calibration_dir(station_id)
        calib_dir.mkdir(parents=True, exist_ok=True)
        path = calib_dir / "intrinsics.json"
        payload: Dict[str, Any] = {
            "fx": req.fx,
            "fy": req.fy,
            "cx": req.cx,
            "cy": req.cy,
            "saved_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        }
        if isinstance(req.dist_coeffs, list):
            payload["dist_coeffs"] = req.dist_coeffs
        if isinstance(req.resolution, dict):
            payload["resolution"] = req.resolution
        if isinstance(req.intrinsics_result, dict):
            payload["intrinsics_result"] = req.intrinsics_result
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return {"status": "saved", "path": str(path)}

    @router.get("/stations/{station_id}/calibration/target")
    def calibration_target_get(station_id: str) -> Dict[str, Any]:
        _station_or_404(station_id)
        path = ctx.data_paths.station_calibration_dir(station_id) / "target.json"
        if not path.exists():
            return {"status": "pending", "detail": "target_not_set"}
        return json.loads(path.read_text(encoding="utf-8"))

    @router.post("/stations/{station_id}/calibration/target")
    def calibration_target_set(station_id: str, req: CalibrationTargetRequest) -> Dict[str, Any]:
        _station_or_404(station_id)
        calib_dir = ctx.data_paths.station_calibration_dir(station_id)
        calib_dir.mkdir(parents=True, exist_ok=True)
        path = calib_dir / "target.json"
        payload: Dict[str, Any] = {
            "type": str(req.type or "charuco"),
            "rows": int(req.rows),
            "cols": int(req.cols),
            "square_size_m": float(req.square_size_m),
            "saved_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        }
        if req.marker_size_m is not None:
            payload["marker_size_m"] = float(req.marker_size_m)
        if req.aruco_dict is not None:
            payload["aruco_dict"] = str(req.aruco_dict)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return {"status": "saved", "path": str(path)}

    @router.get("/stations/{station_id}/calibration/samples")
    def calibration_samples_get(station_id: str) -> Dict[str, Any]:
        _station_or_404(station_id)
        path = ctx.data_paths.station_calibration_dir(station_id) / "samples.json"
        if not path.exists():
            raise HTTPException(status_code=404, detail="samples_not_set")
        return json.loads(path.read_text(encoding="utf-8"))

    @router.post("/stations/{station_id}/calibration/samples")
    def calibration_samples_set(
        station_id: str, req: CalibrationSamplesRequest
    ) -> Dict[str, Any]:
        _station_or_404(station_id)
        calib_dir = ctx.data_paths.station_calibration_dir(station_id)
        calib_dir.mkdir(parents=True, exist_ok=True)
        path = calib_dir / "samples.json"
        payload = _format_calibration_samples_payload(station_id, req)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return {"status": "saved", "path": str(path), "samples": len(req.samples)}

    @router.get("/calibration/handeye")
    def legacy_handeye_get() -> Dict[str, Any]:
        station_id = _default_station_id()
        return handeye_get(station_id)

    @router.post("/calibration/handeye")
    def legacy_handeye_set(req: HandEyeRequest) -> Dict[str, Any]:
        station_id = _default_station_id()
        return handeye_set(station_id, req)

    @router.get("/calibration/tcp")
    def legacy_calibration_tcp_get() -> Dict[str, Any]:
        station_id = _default_station_id()
        return calibration_tcp_get(station_id)

    @router.post("/calibration/tcp")
    def legacy_calibration_tcp_set(req: CalibrationTcpRequest) -> Dict[str, Any]:
        station_id = _default_station_id()
        return calibration_tcp_set(station_id, req)

    @router.get("/calibration/intrinsics")
    def legacy_intrinsics_get() -> Dict[str, Any]:
        station_id = _default_station_id()
        return intrinsics_get(station_id)

    @router.post("/calibration/intrinsics")
    def legacy_intrinsics_set(req: IntrinsicsRequest) -> Dict[str, Any]:
        station_id = _default_station_id()
        return intrinsics_set(station_id, req)

    @router.get("/calibration/samples")
    def legacy_calibration_samples_get() -> Dict[str, Any]:
        station_id = _default_station_id()
        return calibration_samples_get(station_id)

    @router.post("/calibration/samples")
    def legacy_calibration_samples_set(req: CalibrationSamplesRequest) -> Dict[str, Any]:
        station_id = _default_station_id()
        return calibration_samples_set(station_id, req)

    @router.post("/processes/{process_id}/objects/templates/upload")
    def templates_upload(process_id: str, req: TemplateUploadRequest) -> Dict[str, Any]:
        try:
            data = _decode_image(req.image_b64)
            path = ctx.objects.save_template(
                process_id, req.object_id, req.template_name, req.ext, data
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return {"status": "saved", "path": str(path)}

    @router.get("/processes/{process_id}/objects")
    def objects_list(process_id: str) -> Dict[str, Any]:
        _process_or_404(process_id)
        return {"objects": ctx.objects.list(process_id)}

    @router.post("/processes/{process_id}/objects/rename")
    def objects_rename(process_id: str, req: ObjectRenameRequest) -> Dict[str, Any]:
        try:
            name = ctx.objects.rename_object(process_id, req.object_id, req.new_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except FileExistsError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"status": "renamed", "object_id": name}

    @router.delete("/processes/{process_id}/objects/{object_id}")
    def objects_delete(process_id: str, object_id: str) -> Dict[str, Any]:
        try:
            name = ctx.objects.delete_object(process_id, object_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"status": "deleted", "object_id": name}

    @router.post("/processes/{process_id}/objects/templates/rename")
    def templates_rename(process_id: str, req: TemplateRenameRequest) -> Dict[str, Any]:
        try:
            name = ctx.objects.rename_template(
                process_id, req.object_id, req.template_name, req.new_name
            )
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except FileExistsError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"status": "renamed", "template": name}

    @router.delete(
        "/processes/{process_id}/objects/{object_id}/templates/{template_name}"
    )
    def templates_delete(
        process_id: str, object_id: str, template_name: str
    ) -> Dict[str, Any]:
        try:
            name = ctx.objects.delete_template(process_id, object_id, template_name)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"status": "deleted", "template": name}

    @router.get("/processes/{process_id}/objects/{object_id}/geometry")
    def objects_get_geometry(process_id: str, object_id: str) -> Dict[str, Any]:
        """Get cuboid geometry for an object."""
        _process_or_404(process_id)
        metadata = ctx.objects.get_metadata(process_id, object_id)
        if not metadata or "geometry" not in metadata:
            raise HTTPException(status_code=404, detail="geometry_not_found")
        return {"object_id": object_id, "geometry": metadata.get("geometry", {})}

    @router.put("/processes/{process_id}/objects/{object_id}/geometry")
    def objects_set_geometry(
        process_id: str, object_id: str, req: CuboidGeometryRequest
    ) -> Dict[str, Any]:
        """Set cuboid geometry (L, B, H in mm)."""
        _process_or_404(process_id)
        try:
            # Load or create metadata
            metadata = ctx.objects.get_metadata(process_id, object_id) or {}

            # Update geometry
            metadata["geometry"] = {
                "type": "cuboid",
                "L_mm": req.L_mm,
                "B_mm": req.B_mm,
                "H_mm": req.H_mm,
                "axis_convention": req.axis_convention,
            }

            # Save metadata
            ctx.objects.save_metadata(process_id, object_id, metadata)

            return {
                "status": "saved",
                "object_id": object_id,
                "geometry": metadata["geometry"],
            }
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.get("/processes/{process_id}/objects/{object_id}/megapose")
    def objects_get_megapose(process_id: str, object_id: str) -> Dict[str, Any]:
        _process_or_404(process_id)
        metadata = ctx.objects.get_metadata(process_id, object_id)
        megapose = metadata.get("megapose") if isinstance(metadata, dict) else None
        if not isinstance(megapose, dict):
            raise HTTPException(status_code=404, detail="megapose_not_found")
        return {"object_id": object_id, "megapose": megapose}

    @router.put("/processes/{process_id}/objects/{object_id}/megapose")
    def objects_set_megapose(
        process_id: str, object_id: str, req: MegaPoseObjectConfigRequest
    ) -> Dict[str, Any]:
        _process_or_404(process_id)
        try:
            metadata = ctx.objects.get_metadata(process_id, object_id) or {}
            metadata["megapose"] = {
                "object_folder": str(req.object_folder).strip(),
                "model": str(req.model).strip() or "rgbd",
                "device": str(req.device).strip() or "auto",
                "mesh_units": str(req.mesh_units).strip() or "mm",
                "mesh_scale": float(req.mesh_scale),
                "yolo_conf": float(req.yolo_conf),
                "selection_top_k": max(1, int(req.selection_top_k)),
                "selection_min_confidence": float(req.selection_min_confidence),
                "refiner_iterations": max(0, int(req.refiner_iterations)),
                "coarse_grid_size": max(0, int(req.coarse_grid_size)),
            }
            ctx.objects.save_metadata(process_id, object_id, metadata)
            return {
                "status": "saved",
                "object_id": object_id,
                "megapose": metadata["megapose"],
            }
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.get("/processes/{process_id}/objects/{object_id}/frame")
    def objects_get_frame(process_id: str, object_id: str) -> Dict[str, Any]:
        _process_or_404(process_id)
        metadata = ctx.objects.get_metadata(process_id, object_id)
        frame = metadata.get("bin_picking_frame") if isinstance(metadata, dict) else None
        if not isinstance(frame, dict):
            raise HTTPException(status_code=404, detail="frame_not_found")
        return {"object_id": object_id, "frame": frame}

    def _process_object_dir(process_id: str, object_id: str) -> Path:
        process = _process_or_404(process_id)
        station_id = str(process.get("station_id") or "")
        if not station_id:
            raise HTTPException(status_code=400, detail="station_not_set")
        objects_root = ctx.data_paths.process_objects_dir(station_id, process_id)
        # _safe_relative_path enforces no traversal in object_id
        try:
            rel = _safe_relative_path(object_id)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        obj_dir = (objects_root / rel).resolve()
        if objects_root.resolve() not in obj_dir.parents:
            raise HTTPException(status_code=400, detail="invalid_object_id")
        obj_dir.mkdir(parents=True, exist_ok=True)
        return obj_dir

    @router.get("/processes/{process_id}/objects/{object_id}/cad")
    def objects_get_cad(process_id: str, object_id: str) -> Dict[str, Any]:
        obj_dir = _process_object_dir(process_id, object_id)
        mesh_exts = ("stl", "obj", "step", "stp", "iges", "igs", "dae", "ply")
        candidates: List[Path] = []
        for ext in mesh_exts:
            candidate = obj_dir / f"{object_id}.{ext}"
            if candidate.exists() and candidate.is_file():
                candidates.append(candidate)
        visual_dir = obj_dir / "visual"
        if visual_dir.exists():
            for ext in mesh_exts:
                candidates.extend(sorted(visual_dir.glob(f"model.{ext}")))
                candidates.extend(sorted(visual_dir.glob(f"*.{ext}")))
        if not candidates:
            for ext in mesh_exts:
                candidates.extend(sorted(obj_dir.glob(f"*.{ext}")))
        if not candidates:
            raise HTTPException(status_code=404, detail="cad_not_found")
        chosen = candidates[0]
        rel = chosen.relative_to(obj_dir).as_posix()
        url = f"/processes/{process_id}/objects/{object_id}/cad/{rel}"
        return {"filename": chosen.name, "url": url}

    @router.get("/processes/{process_id}/objects/{object_id}/cad/{filename:path}")
    def objects_serve_cad(process_id: str, object_id: str, filename: str) -> FileResponse:
        obj_dir = _process_object_dir(process_id, object_id)
        try:
            path = _resolve_under(obj_dir, filename)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if not path.exists() or not path.is_file():
            raise HTTPException(status_code=404, detail="cad_file_not_found")
        return FileResponse(path)

    @router.get("/processes/{process_id}/objects/{object_id}/grasps")
    def objects_get_grasps(process_id: str, object_id: str) -> Dict[str, Any]:
        obj_dir = _process_object_dir(process_id, object_id)
        path = obj_dir / "grasp_authoring.json"
        if not path.exists():
            return {"object_id": object_id, "grasps": []}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            data = {}
        grasps = data.get("grasps") if isinstance(data, dict) else None
        if not isinstance(grasps, list):
            grasps = []
        return {"object_id": object_id, "grasps": grasps}

    @router.put("/processes/{process_id}/objects/{object_id}/grasps")
    def objects_put_grasps(
        process_id: str,
        object_id: str,
        payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        obj_dir = _process_object_dir(process_id, object_id)
        grasps = payload.get("grasps") if isinstance(payload, dict) else None
        if not isinstance(grasps, list):
            raise HTTPException(status_code=400, detail="grasps_required")
        path = obj_dir / "grasp_authoring.json"
        path.write_text(
            json.dumps({"grasps": grasps}, indent=2),
            encoding="utf-8",
        )
        return {"status": "saved", "object_id": object_id, "count": len(grasps)}

    @router.put("/processes/{process_id}/objects/{object_id}/frame")
    def objects_set_frame(
        process_id: str, object_id: str, req: ObjectFrameRequest
    ) -> Dict[str, Any]:
        _process_or_404(process_id)
        try:
            metadata = ctx.objects.get_metadata(process_id, object_id) or {}
            metadata["bin_picking_frame"] = {
                "frame": str(req.frame or "object").strip() or "object",
                "position_m": _as_float_triplet(req.position_m, [0.0, 0.0, 0.0]),
                "rotation_rpy_deg": _as_float_triplet(req.rotation_rpy_deg, [0.0, 0.0, 0.0]),
            }
            ctx.objects.save_metadata(process_id, object_id, metadata)
            return {
                "status": "saved",
                "object_id": object_id,
                "frame": metadata["bin_picking_frame"],
            }
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.get("/processes/{process_id}/bin")
    def bin_roi_get(process_id: str) -> Dict[str, Any]:
        """Get the bin ROI (optional 2D OBB) for this asset/process."""
        process = _process_or_404(process_id)
        station_id = process.get("station_id") or ""
        if not station_id:
            raise HTTPException(status_code=400, detail="station_not_set")
        path = ctx.data_paths.process_bin_path(station_id, process_id)
        if not path.exists():
            return {"process_id": process_id, "bin": None}
        try:
            return {
                "process_id": process_id,
                "bin": json.loads(path.read_text(encoding="utf-8")),
            }
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"bin_read_failed: {exc}") from exc

    @router.put("/processes/{process_id}/bin")
    def bin_roi_set(process_id: str, req: BinRoiRequest) -> Dict[str, Any]:
        """Save the bin ROI (2D OBB). Overwrites any existing bin.json."""
        process = _process_or_404(process_id)
        station_id = process.get("station_id") or ""
        if not station_id:
            raise HTTPException(status_code=400, detail="station_not_set")
        try:
            points = [[float(p[0]), float(p[1])] for p in (req.obb_points_uv or [])]
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"invalid_obb_points: {exc}") from exc
        if len(points) != 4:
            raise HTTPException(
                status_code=400,
                detail="obb_points_uv_must_have_4_corners",
            )
        try:
            bbox = [float(v) for v in (req.bbox_xywh or [])]
        except Exception:
            bbox = []
        payload = {
            "format": "vgr_bin_roi/v1",
            "camera_id": str(req.camera_id or ""),
            "frame_source": str(req.frame_source or ""),
            "image_width": int(req.image_width or 0),
            "image_height": int(req.image_height or 0),
            "obb_points_uv": points,
            "yaw_deg": float(req.yaw_deg or 0.0),
            "bbox_xywh": bbox if len(bbox) == 4 else [],
            "notes": str(req.notes or ""),
        }
        bin_dir = ctx.data_paths.process_bin_dir(station_id, process_id)
        bin_dir.mkdir(parents=True, exist_ok=True)
        bin_dir.joinpath("bin.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )
        return {"status": "saved", "process_id": process_id, "bin": payload}

    @router.delete("/processes/{process_id}/bin")
    def bin_roi_clear(process_id: str) -> Dict[str, Any]:
        """Remove any existing bin ROI for this asset/process."""
        process = _process_or_404(process_id)
        station_id = process.get("station_id") or ""
        if not station_id:
            raise HTTPException(status_code=400, detail="station_not_set")
        path = ctx.data_paths.process_bin_path(station_id, process_id)
        removed = False
        if path.exists():
            try:
                path.unlink()
                removed = True
            except Exception as exc:
                raise HTTPException(status_code=500, detail=f"bin_delete_failed: {exc}") from exc
        return {"status": "cleared" if removed else "absent", "process_id": process_id}

    @router.post("/objects/templates/upload")
    def legacy_templates_upload(req: TemplateUploadRequest) -> Dict[str, Any]:
        process_id = _default_process_id()
        return templates_upload(process_id, req)

    @router.get("/objects")
    def legacy_objects_list() -> Dict[str, Any]:
        process_id = _default_process_id()
        return objects_list(process_id)

    @router.post("/objects/rename")
    def legacy_objects_rename(req: ObjectRenameRequest) -> Dict[str, Any]:
        process_id = _default_process_id()
        return objects_rename(process_id, req)

    @router.delete("/objects/{object_id}")
    def legacy_objects_delete(object_id: str) -> Dict[str, Any]:
        process_id = _default_process_id()
        return objects_delete(process_id, object_id)

    @router.post("/objects/templates/rename")
    def legacy_templates_rename(req: TemplateRenameRequest) -> Dict[str, Any]:
        process_id = _default_process_id()
        return templates_rename(process_id, req)

    @router.delete("/objects/{object_id}/templates/{template_name}")
    def legacy_templates_delete(object_id: str, template_name: str) -> Dict[str, Any]:
        process_id = _default_process_id()
        return templates_delete(process_id, object_id, template_name)

    @router.get("/processes/{process_id}/poses")
    def poses_list(process_id: str) -> Dict[str, Any]:
        _process_or_404(process_id)
        return {"poses": ctx.poses.list(process_id)}

    @router.post("/processes/{process_id}/poses")
    def pose_record(process_id: str, req: PoseRecordRequest) -> Dict[str, Any]:
        name = _safe_name(req.name)
        if not name:
            raise HTTPException(status_code=400, detail="pose_name_required")
        state = _attach_custom_tcp_pose(ctx.robot.get_state(), process_id)
        joints = state.get("q") or state.get("joints") or []
        current_tcp_pose = state.get("custom_tcp_pose") or {}
        if not isinstance(current_tcp_pose, dict) or not current_tcp_pose.get("position_m"):
            raise HTTPException(
                status_code=400,
                detail=state.get("custom_tcp_error") or "custom_tcp_pose_missing",
            )
        current_position = [float(v) for v in current_tcp_pose.get("position_m", [0.0, 0.0, 0.0])[:3]]
        current_quat = _normalize_quat_xyzw(
            current_tcp_pose.get("quat_xyzw") or [0.0, 0.0, 0.0, 1.0]
        )
        current_frame = "base"

        requested_position = (
            [float(v) for v in req.position_m]
            if req.position_m is not None
            else [float(v) for v in current_position[:3]]
        )
        if req.quat_xyzw is not None:
            requested_quat = _normalize_quat_xyzw([float(v) for v in req.quat_xyzw])
            requested_rpy = _quat_xyzw_to_rpy_deg(requested_quat)
        elif req.rotation_rpy_deg is not None:
            requested_rpy = [float(v) for v in req.rotation_rpy_deg]
            requested_quat = _rpy_deg_to_quat_xyzw(requested_rpy)
        else:
            requested_quat = current_quat
            requested_rpy = _quat_xyzw_to_rpy_deg(current_quat)
        requested_frame = str(req.frame or current_frame or "base")

        tcp_pose = {
            "position_m": requested_position,
            "quat_xyzw": requested_quat,
            "rotation_rpy_deg": requested_rpy,
            "frame": "base",
            "parent_frame": "base",
            "child_frame": "custom_tcp",
            "tcp_frame": "tool_tcp",
            "pose_source": "custom_tcp",
            "interpretation": "base_to_custom_tcp",
        }

        has_tcp_override = bool(
            req.position_m is not None
            or req.quat_xyzw is not None
            or req.rotation_rpy_deg is not None
            or (req.frame is not None and requested_frame != current_frame)
        )
        if has_tcp_override:
            same_position = np.allclose(
                np.asarray(requested_position, dtype=np.float64),
                np.asarray(current_position[:3], dtype=np.float64),
                atol=1e-9,
                rtol=0.0,
            )
            same_quat = np.allclose(
                np.asarray(requested_quat, dtype=np.float64),
                np.asarray(current_quat, dtype=np.float64),
                atol=1e-9,
                rtol=0.0,
            ) or np.allclose(
                np.asarray(requested_quat, dtype=np.float64),
                -np.asarray(current_quat, dtype=np.float64),
                atol=1e-9,
                rtol=0.0,
            )
            has_tcp_override = not (same_position and same_quat and requested_frame == current_frame)

        payload = {
            "joints": list(joints or []),
            "tcp_pose": tcp_pose,
            "frame_interpretation": {
                "tcp_pose": "base_to_custom_tcp",
                "joints": "robot joint positions captured at the same time",
            },
        }
        mode = req.mode.lower()
        if mode == "joints":
            payload = {
                "joints": list(joints or []),
                "tcp_pose": tcp_pose,
                "frame_interpretation": {
                    "tcp_pose": "base_to_custom_tcp",
                    "joints": "robot joint positions captured at the same time",
                },
            }
        elif mode == "tcp":
            payload = {
                "joints": list(joints or []),
                "tcp_pose": tcp_pose,
                "frame_interpretation": {
                    "tcp_pose": "base_to_custom_tcp",
                    "joints": "robot joint positions captured at the same time",
                },
            }
        elif mode == "auto" and has_tcp_override:
            payload["note"] = "TCP override was supplied; joints are the robot state at save time."
        try:
            path = ctx.poses.save(process_id, name, payload)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"status": "saved", "pose": name, "path": str(path)}

    @router.delete("/processes/{process_id}/poses/{pose_name}")
    def pose_delete(process_id: str, pose_name: str) -> Dict[str, Any]:
        try:
            name = ctx.poses.delete(process_id, pose_name)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"status": "deleted", "pose": name}

    @router.post("/processes/{process_id}/poses/{pose_name}/rename")
    def pose_rename(
        process_id: str, pose_name: str, req: PoseRenameRequest
    ) -> Dict[str, Any]:
        new_name = _safe_name(req.new_name)
        if not new_name:
            raise HTTPException(status_code=400, detail="new_pose_name_required")
        try:
            name = ctx.poses.rename(process_id, pose_name, new_name)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except FileExistsError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"status": "renamed", "pose": name}

    @router.get("/station/poses")
    def legacy_poses_list() -> Dict[str, Any]:
        process_id = _default_process_id()
        return poses_list(process_id)

    @router.post("/station/poses")
    def legacy_pose_record(req: PoseRecordRequest) -> Dict[str, Any]:
        process_id = _default_process_id()
        return pose_record(process_id, req)

    @router.delete("/station/poses/{pose_name}")
    def legacy_pose_delete(pose_name: str) -> Dict[str, Any]:
        process_id = _default_process_id()
        return pose_delete(process_id, pose_name)

    @router.post("/station/poses/{pose_name}/rename")
    def legacy_pose_rename(pose_name: str, req: PoseRenameRequest) -> Dict[str, Any]:
        process_id = _default_process_id()
        return pose_rename(process_id, pose_name, req)

    @router.post("/robot/movej")
    def robot_movej(req: RobotMoveJRequest) -> Dict[str, Any]:
        if not bool(ctx.runtime_state.get("robot_enabled", True)):
            return {"status": "skipped", "reason": "robot_disabled"}
        try:
            ctx.robot.movej(tuple(req.joints), req.profile)
            return {"status": "ok"}
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @router.post("/robot/movel")
    def robot_movel(req: RobotMoveLRequest) -> Dict[str, Any]:
        if not bool(ctx.runtime_state.get("robot_enabled", True)):
            return {"status": "skipped", "reason": "robot_disabled"}
        target = {
            "position_m": req.position_m,
            "quat_xyzw": req.quat_xyzw,
            "frame": req.frame,
        }
        try:
            ctx.robot.movel(target, req.profile)
            return {"status": "ok"}
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @router.post("/robot/movel_custom_tcp")
    def robot_movel_custom_tcp(req: RobotMoveLRequest) -> Dict[str, Any]:
        """Move to a target expressed in custom_tcp frame (fingertip).
        Converts custom_tcp → EE (flange) by applying the inverse TCP offset before movel.
        """
        if not bool(ctx.runtime_state.get("robot_enabled", True)):
            return {"status": "skipped", "reason": "robot_disabled"}
        # Convert station-defined custom TCP target back to the robot command EE pose.
        tcp_offset = _station_tool_tcp_offset()
        if tcp_offset is None:
            raise HTTPException(status_code=400, detail="custom_tcp_transform_missing")
        tcp_mat = quat_to_matrix(req.position_m, req.quat_xyzw)
        target_flange = tcp_mat @ np.linalg.inv(tcp_offset)
        ee_mat = _current_ee_from_target_flange(target_flange, ctx.robot.get_state())
        ee_pose = _matrix_to_pose(ee_mat)
        ee_pos = ee_pose["position_m"]
        ee_quat = ee_pose["quat_xyzw"]
        target = {"position_m": ee_pos, "quat_xyzw": ee_quat, "frame": req.frame}
        try:
            ctx.robot.movel(target, req.profile)
            return {"status": "ok"}
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @router.post("/robot/movel_custom_tcp_ik")
    def robot_movel_custom_tcp_ik(req: RobotMoveIkRequest) -> Dict[str, Any]:
        if not bool(ctx.runtime_state.get("robot_enabled", True)):
            return {"status": "skipped", "reason": "robot_disabled"}
        tcp_offset = _station_tool_tcp_offset()
        if tcp_offset is None:
            raise HTTPException(status_code=400, detail="custom_tcp_transform_missing")
        tcp_mat = quat_to_matrix(req.position_m, req.quat_xyzw)
        target_flange = tcp_mat @ np.linalg.inv(tcp_offset)
        ee_mat = _current_ee_from_target_flange(target_flange, ctx.robot.get_state())
        ee_pose = _matrix_to_pose(ee_mat)
        target = {
            "position_m": ee_pose["position_m"],
            "quat_xyzw": ee_pose["quat_xyzw"],
            "frame": req.frame,
        }
        try:
            result = ctx.robot.move_tcp_ik(
                target,
                req.profile,
                seed_joints=tuple(req.seed_joints) if req.seed_joints is not None else None,
                preferred_joints=(
                    tuple(req.preferred_joints)
                    if req.preferred_joints is not None
                    else None
                ),
                position_tolerance_m=req.position_tolerance_m,
                orientation_tolerance_deg=req.orientation_tolerance_deg,
                approximate_position_tolerance_m=req.approximate_position_tolerance_m,
                approximate_orientation_tolerance_deg=req.approximate_orientation_tolerance_deg,
                max_iterations=req.max_iterations,
            )
            result = dict(result or {})
            result["command_frame"] = "robot_ee"
            result["requested_frame"] = "custom_tcp"
            result["command_pose"] = target
            return {"status": "ok", "result": result}
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @router.post("/robot/movel_ik")
    def robot_movel_ik(req: RobotMoveIkRequest) -> Dict[str, Any]:
        if not bool(ctx.runtime_state.get("robot_enabled", True)):
            return {"status": "skipped", "reason": "robot_disabled"}
        target = {
            "position_m": req.position_m,
            "quat_xyzw": req.quat_xyzw,
            "frame": req.frame,
        }
        try:
            result = ctx.robot.move_tcp_ik(
                target,
                req.profile,
                seed_joints=tuple(req.seed_joints) if req.seed_joints is not None else None,
                preferred_joints=(
                    tuple(req.preferred_joints)
                    if req.preferred_joints is not None
                    else None
                ),
                position_tolerance_m=req.position_tolerance_m,
                orientation_tolerance_deg=req.orientation_tolerance_deg,
                approximate_position_tolerance_m=req.approximate_position_tolerance_m,
                approximate_orientation_tolerance_deg=req.approximate_orientation_tolerance_deg,
                max_iterations=req.max_iterations,
            )
            return {"status": "ok", "result": result}
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @router.post("/robot/gripper/open")
    def robot_gripper_open() -> Dict[str, Any]:
        if not bool(ctx.runtime_state.get("robot_enabled", True)):
            return {"status": "skipped", "reason": "robot_disabled"}
        try:
            ctx.robot.open_gripper()
            return {"status": "ok"}
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @router.post("/robot/gripper/close")
    def robot_gripper_close() -> Dict[str, Any]:
        if not bool(ctx.runtime_state.get("robot_enabled", True)):
            return {"status": "skipped", "reason": "robot_disabled"}
        try:
            ctx.robot.close_gripper()
            return {"status": "ok"}
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @router.post("/robot/freedrive")
    def robot_freedrive(req: RobotFreedriveRequest) -> Dict[str, Any]:
        if not bool(ctx.runtime_state.get("robot_enabled", True)):
            return {"status": "skipped", "reason": "robot_disabled", "enabled": False}
        try:
            ctx.robot.freedrive(req.enable)
            return {"status": "ok", "enabled": req.enable}
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @router.post("/robot/stop")
    def robot_stop() -> Dict[str, Any]:
        try:
            ctx.robot.stop()
            return {"status": "ok"}
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @router.post("/recipes/save")
    def recipes_save(req: RecipeSaveRequest) -> Dict[str, Any]:
        rid = _safe_name(req.recipe_id)
        if not rid:
            raise HTTPException(status_code=400, detail="recipe_id_required")
        fmt = req.fmt.lower()
        if fmt not in ("yaml", "yml", "json"):
            raise HTTPException(status_code=400, detail="unsupported_format")
        filename = req.filename or f"{rid}.{fmt if fmt != 'yml' else 'yaml'}"
        path = Path(filename)
        if not path.is_absolute():
            path = (ctx.data_paths.legacy_recipes / filename).resolve()
        if ctx.data_paths.legacy_recipes.resolve() not in path.parents:
            raise HTTPException(
                status_code=400, detail="recipe path must be under data/recipes"
            )
        payload = dict(req.recipe)
        payload["recipe_id"] = rid
        if path.suffix.lower() in (".yaml", ".yml"):
            path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
        else:
            path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return {"status": "saved", "recipe_id": rid, "path": str(path)}

    # Debug: Save annotated frames for debugging
    @router.post("/debug/save-frame")
    def debug_save_frame(req: Dict[str, Any]) -> Dict[str, Any]:
        """Save annotated vision frame to disk for debugging."""
        try:
            timestamp = req.get("timestamp", "unknown")
            frame_id = req.get("frameId", "unknown")
            data_url = req.get("dataUrl", "")
            match_count = req.get("matchCount", 0)
            has_pose = req.get("hasPose", False)

            if not data_url.startswith("data:image"):
                return {"status": "error", "detail": "invalid_data_url"}

            # Create debug frames directory
            debug_dir = ctx.data_paths.data_root / "debug_frames"
            debug_dir.mkdir(parents=True, exist_ok=True)

            # Extract base64 image data
            header, b64_data = data_url.split(",", 1)
            img_data = base64.b64decode(b64_data)

            # Save frame
            filename = f"{timestamp}_{frame_id}_m{match_count}_p{int(has_pose)}.jpg"
            filepath = debug_dir / filename
            filepath.write_bytes(img_data)

            return {
                "status": "ok",
                "saved": True,
                "path": str(filepath),
                "filename": filename,
                "frame_id": frame_id,
                "matches": match_count,
                "pose_available": has_pose,
            }
        except Exception as e:
            return {"status": "error", "detail": str(e)}

    return router
