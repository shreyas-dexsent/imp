"""Implementation for `camera_core.calibration.server`."""

import base64
import json
import threading
import time
from dataclasses import dataclass, field
from multiprocessing import shared_memory
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import uvicorn
import zmq
from camera_core.shm.header import FLAG_VALID, HEADER_SIZE, unpack_header
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel, Field

HAND_EYE_METHODS = {
    "Tsai": cv2.CALIB_HAND_EYE_TSAI,
    "Park": cv2.CALIB_HAND_EYE_PARK,
    "Horaud": cv2.CALIB_HAND_EYE_HORAUD,
    "Andreff": cv2.CALIB_HAND_EYE_ANDREFF,
    "Daniilidis": cv2.CALIB_HAND_EYE_DANIILIDIS,
}


def _resolve_aruco_dict(name: str) -> Any:
    if not hasattr(cv2, "aruco"):
        raise HTTPException(status_code=400, detail="aruco_not_available")
    key = (name or "DICT_4X4_50").upper()
    if key in ("DICT_4X4", "4X4"):
        key = "DICT_4X4_50"
    if key in ("4X4_50", "4X4-50"):
        key = "DICT_4X4_50"
    aruco_mod = cv2.aruco
    mapping = {
        "DICT_4X4_50": aruco_mod.DICT_4X4_50,
        "DICT_4X4_100": aruco_mod.DICT_4X4_100,
        "DICT_4X4_250": aruco_mod.DICT_4X4_250,
        "DICT_4X4_1000": aruco_mod.DICT_4X4_1000,
        "DICT_5X5_50": aruco_mod.DICT_5X5_50,
        "DICT_5X5_100": aruco_mod.DICT_5X5_100,
        "DICT_5X5_250": aruco_mod.DICT_5X5_250,
        "DICT_5X5_1000": aruco_mod.DICT_5X5_1000,
        "DICT_6X6_50": aruco_mod.DICT_6X6_50,
        "DICT_6X6_100": aruco_mod.DICT_6X6_100,
        "DICT_6X6_250": aruco_mod.DICT_6X6_250,
        "DICT_6X6_1000": aruco_mod.DICT_6X6_1000,
        "DICT_7X7_50": aruco_mod.DICT_7X7_50,
        "DICT_7X7_100": aruco_mod.DICT_7X7_100,
        "DICT_7X7_250": aruco_mod.DICT_7X7_250,
        "DICT_7X7_1000": aruco_mod.DICT_7X7_1000,
    }
    if key not in mapping:
        raise HTTPException(status_code=400, detail="unknown_aruco_dict")
    return aruco_mod.getPredefinedDictionary(mapping[key])


def _charuco_corners(board: Any) -> np.ndarray:
    if hasattr(board, "chessboardCorners"):
        return board.chessboardCorners
    if hasattr(board, "getChessboardCorners"):
        return board.getChessboardCorners()
    raise HTTPException(status_code=500, detail="charuco_board_corners_unavailable")


def _draw_charuco_overlay(
    image: np.ndarray,
    marker_corners: List[np.ndarray],
    charuco_corners: np.ndarray,
    axis_origin_object_point: np.ndarray,
    intrinsics: np.ndarray,
    dist_coeffs: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    axis_len: float,
) -> None:
    detected_points: List[np.ndarray] = []
    if marker_corners:
        for pts in marker_corners:
            poly = np.asarray(pts, dtype=np.float32).reshape(-1, 2)
            detected_points.append(poly)
            poly_i = np.round(poly).astype(np.int32)
            cv2.polylines(
                image, [poly_i.reshape(-1, 1, 2)], True, (80, 220, 80), 2, cv2.LINE_AA
            )

    if charuco_corners is not None and len(charuco_corners) > 0:
        charuco_points = np.asarray(charuco_corners, dtype=np.float32).reshape(-1, 2)
        detected_points.append(charuco_points)
        for pt in charuco_points:
            center = tuple(np.round(pt).astype(int))
            cv2.circle(image, center, 5, (0, 215, 255), -1, cv2.LINE_AA)
            cv2.circle(image, center, 8, (40, 40, 40), 1, cv2.LINE_AA)

    if detected_points:
        all_detected = np.concatenate(detected_points, axis=0)
        if len(all_detected) >= 3:
            hull = cv2.convexHull(all_detected.reshape(-1, 1, 2)).reshape(-1, 2)
            hull_i = np.round(hull).astype(np.int32)
            cv2.polylines(
                image,
                [hull_i.reshape(-1, 1, 2)],
                True,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

    origin_obj = np.asarray(axis_origin_object_point, dtype=np.float32).reshape(3)
    if len(origin_obj) != 3:
        return

    axis_points = np.array(
        [
            origin_obj,
            origin_obj + np.array([axis_len, 0.0, 0.0], dtype=np.float32),
            origin_obj + np.array([0.0, axis_len, 0.0], dtype=np.float32),
            origin_obj + np.array([0.0, 0.0, -axis_len], dtype=np.float32),
        ],
        dtype=np.float32,
    )
    projected_axes, _ = cv2.projectPoints(
        axis_points, rvec, tvec, intrinsics, dist_coeffs
    )
    projected_axes = projected_axes.reshape(-1, 2)
    origin = tuple(np.round(projected_axes[0]).astype(int))
    endpoints = [tuple(np.round(pt).astype(int)) for pt in projected_axes[1:]]
    axis_specs = [
        ("X", (0, 0, 255), endpoints[0]),
        ("Y", (0, 200, 0), endpoints[1]),
        ("Z", (255, 0, 0), endpoints[2]),
    ]
    cv2.circle(image, origin, 6, (255, 255, 255), -1, cv2.LINE_AA)
    cv2.circle(image, origin, 9, (30, 30, 30), 1, cv2.LINE_AA)
    for label, color, endpoint in axis_specs:
        cv2.line(image, origin, endpoint, color, 3, cv2.LINE_AA)
        cv2.circle(image, endpoint, 5, color, -1, cv2.LINE_AA)
        cv2.putText(
            image,
            label,
            (endpoint[0] + 6, endpoint[1] - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            color,
            2,
            cv2.LINE_AA,
        ) 


def _select_charuco_axis_origin(obj_pts: np.ndarray) -> Optional[np.ndarray]:
    pts = np.asarray(obj_pts, dtype=np.float32).reshape(-1, 3)
    if len(pts) == 0:
        return None
    sort_idx = np.lexsort((pts[:, 0], pts[:, 1]))
    return pts[int(sort_idx[0])]


def _charuco_board_center(obj_pts: np.ndarray) -> Optional[np.ndarray]:
    pts = np.asarray(obj_pts, dtype=np.float32).reshape(-1, 3)
    if len(pts) == 0:
        return None
    mins = np.min(pts, axis=0)
    maxs = np.max(pts, axis=0)
    return ((mins + maxs) * 0.5).astype(np.float32)


def quat_xyzw_to_rotmat(quat: List[float]) -> np.ndarray:
    x, y, z, w = quat
    n = x * x + y * y + z * z + w * w
    if n < 1e-12:
        return np.eye(3)
    s = 2.0 / n
    xx = x * x * s
    yy = y * y * s
    zz = z * z * s
    xy = x * y * s
    xz = x * z * s
    yz = y * z * s
    wx = w * x * s
    wy = w * y * s
    wz = w * z * s
    return np.array(
        [
            [1.0 - (yy + zz), xy - wz, xz + wy],
            [xy + wz, 1.0 - (xx + zz), yz - wx],
            [xz - wy, yz + wx, 1.0 - (xx + yy)],
        ],
        dtype=np.float64,
    )


def rpy_deg_to_rotmat(rpy_deg: List[float]) -> np.ndarray:
    roll = float(rpy_deg[0]) * np.pi / 180.0
    pitch = float(rpy_deg[1]) * np.pi / 180.0
    yaw = float(rpy_deg[2]) * np.pi / 180.0
    cr = np.cos(roll)
    sr = np.sin(roll)
    cp = np.cos(pitch)
    sp = np.sin(pitch)
    cy = np.cos(yaw)
    sy = np.sin(yaw)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=np.float64,
    )


def rotmat_to_quat_xyzw(mat: np.ndarray) -> List[float]:
    m = mat
    trace = float(np.trace(m))
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    else:
        if m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
            s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
            w = (m[2, 1] - m[1, 2]) / s
            x = 0.25 * s
            y = (m[0, 1] + m[1, 0]) / s
            z = (m[0, 2] + m[2, 0]) / s
        elif m[1, 1] > m[2, 2]:
            s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
            w = (m[0, 2] - m[2, 0]) / s
            x = (m[0, 1] + m[1, 0]) / s
            y = 0.25 * s
            z = (m[1, 2] + m[2, 1]) / s
        else:
            s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
            w = (m[1, 0] - m[0, 1]) / s
            x = (m[0, 2] + m[2, 0]) / s
            y = (m[1, 2] + m[2, 1]) / s
            z = 0.25 * s
    return [float(x), float(y), float(z), float(w)]


def average_quaternions(quats: List[List[float]]) -> List[float]:
    if not quats:
        return [0.0, 0.0, 0.0, 1.0]
    base = np.array(quats[0], dtype=np.float64)
    accum = np.zeros(4, dtype=np.float64)
    for q in quats:
        qv = np.array(q, dtype=np.float64)
        if np.dot(base, qv) < 0:
            qv = -qv
        accum += qv
    norm = np.linalg.norm(accum)
    if norm < 1e-12:
        return [0.0, 0.0, 0.0, 1.0]
    accum /= norm
    return [float(accum[0]), float(accum[1]), float(accum[2]), float(accum[3])]


def rotation_error_deg(r1: np.ndarray, r2: np.ndarray) -> float:
    rel = r1.T @ r2
    trace = float(np.trace(rel))
    cosang = max(-1.0, min(1.0, (trace - 1.0) / 2.0))
    return float(np.degrees(np.arccos(cosang)))


def summarize_scalar_stats(values: List[float], scale: float = 1.0) -> Optional[Dict[str, float]]:
    if not values:
        return None
    arr = np.asarray(values, dtype=np.float64) * float(scale)
    return {
        "min": float(np.min(arr)),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "max": float(np.max(arr)),
        "std": float(np.std(arr)),
    }


def _recompute_detection_pose(
    session: "CalibrationSession", target_pose: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    try:
        if session.target_type == "chessboard":
            pts2 = target_pose.get("image_points") or []
            pts3 = target_pose.get("object_points") or []
            if not (
                isinstance(pts2, list)
                and isinstance(pts3, list)
                and len(pts2) >= 4
                and len(pts3) >= 4
            ):
                return None
            imgp = np.asarray(pts2, dtype=np.float32).reshape(-1, 2)
            objp = np.asarray(pts3, dtype=np.float32).reshape(-1, 3)
            ok, rvec, tvec = cv2.solvePnP(
                objp, imgp, session.intrinsics, session.dist_coeffs
            )
            if not ok:
                return None
            proj, _ = cv2.projectPoints(
                objp, rvec, tvec, session.intrinsics, session.dist_coeffs
            )
            err = float(
                np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - imgp) ** 2, axis=1)))
            )
            return {
                "rvec": [float(v) for v in rvec.reshape(-1)],
                "tvec": [float(v) for v in tvec.reshape(-1)],
                "reprojection_rmse_px": err,
            }

        if session.target_type == "charuco":
            pts2 = target_pose.get("charuco_image_points") or []
            ids = target_pose.get("charuco_ids") or []
            if not (
                isinstance(pts2, list)
                and isinstance(ids, list)
                and len(pts2) >= 4
                and len(ids) >= 4
            ):
                return None
            aruco_dict = _resolve_aruco_dict(session.aruco_dict)
            if hasattr(cv2.aruco, "CharucoBoard_create"):
                board = cv2.aruco.CharucoBoard_create(
                    int(session.cols),
                    int(session.rows),
                    float(session.square_size_m),
                    float(session.marker_size_m),
                    aruco_dict,
                )
            else:
                board = cv2.aruco.CharucoBoard(
                    (int(session.cols), int(session.rows)),
                    float(session.square_size_m),
                    float(session.marker_size_m),
                    aruco_dict,
                )
            ids_flat = np.asarray(ids, dtype=np.int32).reshape(-1)
            objp_all = _charuco_corners(board)
            objp_all = np.asarray(objp_all, dtype=np.float32).reshape(-1, 3)
            if np.any(ids_flat < 0) or np.any(ids_flat >= len(objp_all)):
                return None
            imgp = np.asarray(pts2, dtype=np.float32).reshape(-1, 2)
            objp = objp_all[ids_flat]
            ok = False
            rvec = None
            tvec = None
            if hasattr(cv2.aruco, "estimatePoseCharucoBoard"):
                try:
                    ok, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
                        imgp.reshape(-1, 1, 2),
                        ids_flat.reshape(-1, 1),
                        board,
                        session.intrinsics,
                        session.dist_coeffs,
                        None,
                        None,
                    )
                except Exception:
                    ok = False
            if not ok:
                ok, rvec, tvec = cv2.solvePnP(
                    objp, imgp, session.intrinsics, session.dist_coeffs
                )
            if not ok:
                return None
            proj, _ = cv2.projectPoints(
                objp, rvec, tvec, session.intrinsics, session.dist_coeffs
            )
            err = float(
                np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - imgp) ** 2, axis=1)))
            )
            return {
                "rvec": [float(v) for v in rvec.reshape(-1)],
                "tvec": [float(v) for v in tvec.reshape(-1)],
                "reprojection_rmse_px": err,
            }
    except Exception:
        return None
    return None


def _read_rgb_frame(evt: Dict[str, Any]) -> Dict[str, Any]:
    name = evt.get("rgb_shm")
    shape = evt.get("rgb_shape")
    dtype = evt.get("rgb_dtype", "uint8")
    if not name or not shape:
        raise ValueError("missing_rgb_shm")
    dtype_np = np.dtype(dtype)
    if not isinstance(shape, (list, tuple)) or len(shape) < 2:
        raise ValueError("invalid_rgb_shape")
    shm = shared_memory.SharedMemory(name=name, create=False)
    try:
        expected = int(np.prod(shape)) * dtype_np.itemsize
        if len(shm.buf) < HEADER_SIZE + expected:
            raise ValueError("rgb_shm_size_mismatch")
        header = bytes(shm.buf[:HEADER_SIZE])
        _, _, _, flags = unpack_header(header)
        if not (flags & FLAG_VALID):
            raise ValueError("rgb_frame_invalid")
        img = np.ndarray(
            shape=tuple(shape), dtype=dtype_np, buffer=shm.buf, offset=HEADER_SIZE
        )
        frame = img.copy()
    finally:
        shm.close()
    return {
        "frame": frame,
        "sequence_id": evt.get("sequence_id"),
        "timestamp_ns": evt.get("timestamp_ns"),
        "camera_id": evt.get("camera_id"),
    }


def _read_depth_frame(evt: Dict[str, Any]) -> Optional[np.ndarray]:
    name = evt.get("depth_shm")
    shape = evt.get("depth_shape")
    dtype = evt.get("depth_dtype", "float32")
    if not name or not shape:
        return None
    dtype_np = np.dtype(dtype)
    if not isinstance(shape, (list, tuple)) or len(shape) < 2:
        raise ValueError("invalid_depth_shape")
    shm = shared_memory.SharedMemory(name=name, create=False)
    try:
        expected = int(np.prod(shape)) * dtype_np.itemsize
        if len(shm.buf) < HEADER_SIZE + expected:
            raise ValueError("depth_shm_size_mismatch")
        header = bytes(shm.buf[:HEADER_SIZE])
        _, _, _, flags = unpack_header(header)
        if not (flags & FLAG_VALID):
            raise ValueError("depth_frame_invalid")
        img = np.ndarray(
            shape=tuple(shape), dtype=dtype_np, buffer=shm.buf, offset=HEADER_SIZE
        )
        return img.copy()
    finally:
        shm.close()


def _mean_depth_xyz_at_uv(
    depth_m: Optional[np.ndarray],
    intrinsics: np.ndarray,
    uv: np.ndarray,
    radius_px: int = 5,
) -> Optional[Dict[str, Any]]:
    if depth_m is None:
        return None
    if depth_m.ndim < 2:
        return None
    h, w = int(depth_m.shape[0]), int(depth_m.shape[1])
    u0 = int(round(float(uv[0])))
    v0 = int(round(float(uv[1])))
    if u0 < 0 or u0 >= w or v0 < 0 or v0 >= h:
        return None
    radius_px = max(1, int(radius_px))
    x0 = max(0, u0 - radius_px)
    x1 = min(w - 1, u0 + radius_px)
    y0 = max(0, v0 - radius_px)
    y1 = min(h - 1, v0 + radius_px)
    grid_x, grid_y = np.meshgrid(
        np.arange(x0, x1 + 1, dtype=np.float32),
        np.arange(y0, y1 + 1, dtype=np.float32),
    )
    circle = (grid_x - float(u0)) ** 2 + (grid_y - float(v0)) ** 2 <= float(
        radius_px * radius_px
    )
    patch_z = depth_m[y0 : y1 + 1, x0 : x1 + 1].astype(np.float32, copy=False)
    valid = circle & np.isfinite(patch_z) & (patch_z > 1e-6)
    if int(np.count_nonzero(valid)) < 3:
        return None
    fx = float(intrinsics[0, 0])
    fy = float(intrinsics[1, 1])
    cx = float(intrinsics[0, 2])
    cy = float(intrinsics[1, 2])
    x = ((grid_x - cx) / fx) * patch_z
    y = ((grid_y - cy) / fy) * patch_z
    xyz = np.stack([x, y, patch_z], axis=-1)[valid]
    mean_xyz = np.mean(xyz, axis=0)
    return {
        "xyz_m": [float(mean_xyz[0]), float(mean_xyz[1]), float(mean_xyz[2])],
        "uv": [int(u0), int(v0)],
        "radius_px": int(radius_px),
        "samples": int(xyz.shape[0]),
    }


def _nearest_depth_xyz_at_uv(
    depth_m: Optional[np.ndarray],
    intrinsics: np.ndarray,
    uv: np.ndarray,
    radius_px: int = 8,
) -> Optional[Dict[str, Any]]:
    if depth_m is None:
        return None
    if depth_m.ndim < 2:
        return None
    h, w = int(depth_m.shape[0]), int(depth_m.shape[1])
    u0 = int(round(float(uv[0])))
    v0 = int(round(float(uv[1])))
    if u0 < 0 or u0 >= w or v0 < 0 or v0 >= h:
        return None
    radius_px = max(1, int(radius_px))
    x0 = max(0, u0 - radius_px)
    x1 = min(w - 1, u0 + radius_px)
    y0 = max(0, v0 - radius_px)
    y1 = min(h - 1, v0 + radius_px)
    grid_x, grid_y = np.meshgrid(
        np.arange(x0, x1 + 1, dtype=np.float32),
        np.arange(y0, y1 + 1, dtype=np.float32),
    )
    patch_z = depth_m[y0 : y1 + 1, x0 : x1 + 1].astype(np.float32, copy=False)
    valid = np.isfinite(patch_z) & (patch_z > 1e-6)
    if int(np.count_nonzero(valid)) < 1:
        return None
    dist2 = (grid_x - float(u0)) ** 2 + (grid_y - float(v0)) ** 2
    dist2 = np.where(valid, dist2, np.inf)
    best_flat = int(np.argmin(dist2.reshape(-1)))
    best_dist2 = float(dist2.reshape(-1)[best_flat])
    if not np.isfinite(best_dist2):
        return None
    best_y, best_x = np.unravel_index(best_flat, dist2.shape)
    sample_u = int(round(float(grid_x[best_y, best_x])))
    sample_v = int(round(float(grid_y[best_y, best_x])))
    z = float(patch_z[best_y, best_x])
    fx = float(intrinsics[0, 0])
    fy = float(intrinsics[1, 1])
    cx = float(intrinsics[0, 2])
    cy = float(intrinsics[1, 2])
    x = ((float(sample_u) - cx) / fx) * z
    y = ((float(sample_v) - cy) / fy) * z
    return {
        "xyz_m": [float(x), float(y), float(z)],
        "uv": [sample_u, sample_v],
        "target_uv": [u0, v0],
        "radius_px": int(radius_px),
        "distance_px": float(np.sqrt(best_dist2)),
        "samples": 1,
    }


class CameraEventSubscriber:
    def __init__(self, endpoint: str, topic: str) -> None:
        self.ctx = zmq.Context.instance()
        self.sock = self.ctx.socket(zmq.SUB)
        self.sock.connect(endpoint)
        self.sock.setsockopt_string(zmq.SUBSCRIBE, topic)
        self.poller = zmq.Poller()
        self.poller.register(self.sock, zmq.POLLIN)

    def recv(self, timeout_ms: int = 100) -> Optional[Dict[str, Any]]:
        try:
            events = dict(self.poller.poll(timeout_ms))
        except zmq.error.ZMQError:
            return None
        if self.sock not in events:
            return None
        raw = self.sock.recv_string()
        if " " not in raw:
            return None
        _, payload = raw.split(" ", 1)
        try:
            return json.loads(payload)
        except Exception:
            return None

    def close(self) -> None:
        self.sock.close()


class FrameCache:
    def __init__(self, endpoint: str, topic: str) -> None:
        self.sub = CameraEventSubscriber(endpoint, topic)
        self._latest: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while self._running:
            evt = self.sub.recv(timeout_ms=200)
            if not evt:
                continue
            if evt.get("event") != "FRAME_READY":
                continue
            camera_id = evt.get("camera_id")
            if not camera_id:
                continue
            with self._lock:
                self._latest[camera_id] = evt

    def latest(self, camera_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            return self._latest.get(camera_id)

    def close(self) -> None:
        self._running = False
        self.sub.close()


class IntrinsicsModel(BaseModel):
    fx: float
    fy: float
    cx: float
    cy: float
    dist_coeffs: List[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0, 0.0])


class TargetModel(BaseModel):
    type: str = "chessboard"
    rows: int = 6
    cols: int = 9
    square_size_m: float = 0.025
    marker_size_m: float = 0.02
    aruco_dict: str = "DICT_4X4_50"


class SessionStartRequest(BaseModel):
    camera_id: str
    target: TargetModel
    intrinsics: IntrinsicsModel
    mode: str = "eye_in_hand"
    method: str = "Tsai"


class SessionStopRequest(BaseModel):
    reason: str = "user"


class DetectRequest(BaseModel):
    camera_id: Optional[str] = None
    return_overlay: bool = True
    min_timestamp_ns: Optional[int] = None
    wait_timeout_ms: int = 1200


class RobotPoseModel(BaseModel):
    position_m: List[float] = Field(..., min_length=3, max_length=3)
    quat_xyzw: List[float] = Field(..., min_length=4, max_length=4)
    frame: str = "base"
    tcp_frame: Optional[str] = None
    pose_source: Optional[str] = None
    timestamp_ns: Optional[int] = None


class SampleAddRequest(BaseModel):
    robot: RobotPoseModel
    camera_id: Optional[str] = None
    use_last_detection: bool = True
    robot_state_debug: Optional[Dict[str, Any]] = None


class TcpCalibrationModel(BaseModel):
    translation_m: List[float] = Field(
        default_factory=lambda: [0.0, 0.0, 0.0], min_length=3, max_length=3
    )
    rotation_rpy_deg: List[float] = Field(
        default_factory=lambda: [0.0, 0.0, 0.0], min_length=3, max_length=3
    )
    frame: str = "flange"


class SamplesImportRequest(BaseModel):
    samples: List[Dict[str, Any]] = Field(default_factory=list)
    camera_id: Optional[str] = None
    method: Optional[str] = None
    mode: Optional[str] = None
    target: Optional[Dict[str, Any]] = None


class ComputeRequest(BaseModel):
    method: Optional[str] = None
    tcp_calibration: Optional[TcpCalibrationModel] = None


@dataclass
class CalibrationSession:
    camera_id: str
    target_type: str
    rows: int
    cols: int
    square_size_m: float
    marker_size_m: float
    aruco_dict: str
    intrinsics: np.ndarray
    dist_coeffs: np.ndarray
    mode: str
    method: str
    created_ns: int = field(default_factory=time.time_ns)


class CalibrationService:
    def __init__(self, pub_endpoint: str, topic: str) -> None:
        self.frame_cache = FrameCache(pub_endpoint, topic)
        self._lock = threading.Lock()
        self._session: Optional[CalibrationSession] = None
        self._samples: List[Dict[str, Any]] = []
        self._last_detection: Optional[Dict[str, Any]] = None
        self._last_result: Optional[Dict[str, Any]] = None

    def shutdown(self) -> None:
        self.frame_cache.close()

    def start_session(self, req: SessionStartRequest) -> Dict[str, Any]:
        if req.mode not in ("eye_in_hand",):
            raise HTTPException(status_code=400, detail="mode_not_supported")
        method = req.method
        if method not in HAND_EYE_METHODS:
            raise HTTPException(status_code=400, detail="unknown_method")
        intr = req.intrinsics
        k = np.array(
            [
                [intr.fx, 0.0, intr.cx],
                [0.0, intr.fy, intr.cy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        dist = np.array(intr.dist_coeffs, dtype=np.float64).reshape(-1, 1)
        tgt = req.target
        session = CalibrationSession(
            camera_id=req.camera_id,
            target_type=tgt.type,
            rows=int(tgt.rows),
            cols=int(tgt.cols),
            square_size_m=float(tgt.square_size_m),
            marker_size_m=float(tgt.marker_size_m),
            aruco_dict=str(tgt.aruco_dict),
            intrinsics=k,
            dist_coeffs=dist,
            mode=req.mode,
            method=method,
        )
        with self._lock:
            self._session = session
            self._samples = []
            self._last_detection = None
            self._last_result = None
        return {
            "status": "started",
            "camera_id": session.camera_id,
            "target": {
                "type": session.target_type,
                "rows": session.rows,
                "cols": session.cols,
                "square_size_m": session.square_size_m,
                "marker_size_m": session.marker_size_m,
                "aruco_dict": session.aruco_dict,
            },
            "mode": session.mode,
            "method": session.method,
        }

    def stop_session(self, reason: str = "user") -> Dict[str, Any]:
        with self._lock:
            self._session = None
            self._samples = []
            self._last_detection = None
            self._last_result = None
        return {"status": "stopped", "reason": reason}

    def status(self) -> Dict[str, Any]:
        with self._lock:
            if not self._session:
                return {"status": "idle"}
            session = self._session
            return {
                "status": "active",
                "camera_id": session.camera_id,
                "mode": session.mode,
                "method": session.method,
                "target": {
                    "type": session.target_type,
                    "rows": session.rows,
                    "cols": session.cols,
                    "square_size_m": session.square_size_m,
                    "marker_size_m": session.marker_size_m,
                    "aruco_dict": session.aruco_dict,
                },
                "samples": len(self._samples),
            }

    def _get_frame(
        self,
        camera_id: str,
        min_timestamp_ns: Optional[int] = None,
        wait_timeout_ms: int = 0,
    ) -> Dict[str, Any]:
        deadline = time.perf_counter() + max(0.0, float(wait_timeout_ms) / 1000.0)
        evt = None
        while True:
            evt = self.frame_cache.latest(camera_id)
            evt_ts = int(evt.get("timestamp_ns") or 0) if evt else 0
            if evt and (not min_timestamp_ns or evt_ts >= int(min_timestamp_ns)):
                break
            if time.perf_counter() >= deadline:
                break
            time.sleep(0.01)
        if not evt:
            raise HTTPException(status_code=404, detail="no_frame")
        if min_timestamp_ns:
            evt_ts = int(evt.get("timestamp_ns") or 0)
            if evt_ts < int(min_timestamp_ns):
                raise HTTPException(
                    status_code=408,
                    detail={
                        "code": "fresh_frame_timeout",
                        "camera_id": camera_id,
                        "min_timestamp_ns": int(min_timestamp_ns),
                        "latest_timestamp_ns": evt_ts,
                    },
                )
        try:
            payload = _read_rgb_frame(evt)
            payload["depth_m"] = _read_depth_frame(evt)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        return payload

    def frame_jpeg(self, camera_id: str, quality: int = 75) -> bytes:
        payload = self._get_frame(camera_id)
        frame = payload["frame"]
        quality = max(20, min(int(quality), 95))
        ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if not ok:
            raise HTTPException(status_code=500, detail="encode_failed")
        return buf.tobytes()

    def camera_intrinsics(self, camera_id: str) -> Dict[str, Any]:
        evt = self.frame_cache.latest(camera_id)
        if not evt:
            raise HTTPException(status_code=404, detail="no_frame")
        camera_data = evt.get("camera_data")
        if not isinstance(camera_data, dict) or not camera_data:
            raise HTTPException(status_code=404, detail="camera_intrinsics_unavailable")
        intr = camera_data.get("intrinsics")
        if not isinstance(intr, dict) or not intr:
            raise HTTPException(status_code=404, detail="camera_intrinsics_unavailable")
        resolution = intr.get("resolution")
        if not isinstance(resolution, dict):
            raw_resolution = camera_data.get("resolution")
            if isinstance(raw_resolution, (list, tuple)) and len(raw_resolution) >= 2:
                resolution = {
                    "width": int(raw_resolution[1]),
                    "height": int(raw_resolution[0]),
                }
            else:
                resolution = {}
        dist_coeffs_raw = intr.get("dist_coeffs")
        if not isinstance(dist_coeffs_raw, list):
            dist_coeffs_raw = (
                camera_data.get("dist_coeffs")
                if isinstance(camera_data.get("dist_coeffs"), list)
                else []
            )
        dist_coeffs = [float(v) for v in dist_coeffs_raw[:5]]
        payload = {
            "fx": float(intr.get("fx", 0.0) or 0.0),
            "fy": float(intr.get("fy", 0.0) or 0.0),
            "cx": float(intr.get("cx", 0.0) or 0.0),
            "cy": float(intr.get("cy", 0.0) or 0.0),
            "dist_coeffs": dist_coeffs,
            "resolution": {
                "width": int(resolution.get("width", 0) or 0),
                "height": int(resolution.get("height", 0) or 0),
            },
            "distortion_model": str(
                intr.get("distortion_model")
                or camera_data.get("distortion_model")
                or ""
            ),
            "depth_scale_m_per_unit": float(
                camera_data.get("depth_scale_m_per_unit", 0.0) or 0.0
            ),
            "source": str(camera_data.get("source") or "camera_frame"),
            "device_model": str(camera_data.get("device_model") or ""),
        }
        return {"status": "ok", "camera_id": camera_id, "intrinsics": payload}

    def detect(
        self,
        camera_id: Optional[str],
        return_overlay: bool,
        min_timestamp_ns: Optional[int] = None,
        wait_timeout_ms: int = 1200,
    ) -> Dict[str, Any]:
        with self._lock:
            session = self._session
        if not session:
            raise HTTPException(status_code=400, detail="session_not_started")
        cam_id = camera_id or session.camera_id
        if cam_id != session.camera_id:
            raise HTTPException(status_code=400, detail="camera_mismatch")
        payload = self._get_frame(
            cam_id,
            min_timestamp_ns=min_timestamp_ns,
            wait_timeout_ms=wait_timeout_ms,
        )
        frame = payload["frame"]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        overlay = None
        overlay_b64 = None
        overlay_fmt = None
        corner_count = 0
        marker_count = 0
        marker_ids: List[int] = []
        marker_centers_uv: List[List[float]] = []
        marker_centroid_uv = None
        charuco_centroid_uv = None
        image_size = [int(frame.shape[1]), int(frame.shape[0])]
        first_corner_uv = None
        first_corner_object_point = None
        first_corner_depth = None
        first_corner_pose_xyz = None
        first_corner_blended_xyz = None
        axis_origin_uv = None
        axis_origin_object_point = None
        axis_origin_pose_xyz = None
        axis_origin_depth = None
        if session.target_type == "chessboard":
            pattern_size = (int(session.cols), int(session.rows))
            flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
            found, corners = cv2.findChessboardCorners(gray, pattern_size, flags)
            if not found or corners is None:
                result = {
                    "status": "not_found",
                    "camera_id": cam_id,
                    "sequence_id": payload.get("sequence_id"),
                }
                with self._lock:
                    self._last_detection = None
                return result
            term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), term)
            objp = np.zeros((session.rows * session.cols, 3), np.float32)
            objp[:, :2] = np.mgrid[0 : session.cols, 0 : session.rows].T.reshape(-1, 2)
            objp *= session.square_size_m
            image_points = corners.reshape(-1, 2)
            object_points = objp.reshape(-1, 3)
            ok, rvec, tvec = cv2.solvePnP(
                objp, corners, session.intrinsics, session.dist_coeffs
            )
            if not ok:
                raise HTTPException(status_code=500, detail="solvepnp_failed")
            proj, _ = cv2.projectPoints(
                objp, rvec, tvec, session.intrinsics, session.dist_coeffs
            )
            err = float(np.sqrt(np.mean(np.sum((proj - corners) ** 2, axis=2))))
            corner_count = int(len(corners))
            if return_overlay:
                overlay = frame.copy()
                cv2.drawChessboardCorners(overlay, pattern_size, corners, found)
                if hasattr(cv2, "drawFrameAxes"):
                    axis_len = float(
                        session.square_size_m
                        * max(2.0, min(session.rows, session.cols))
                    )
                    cv2.drawFrameAxes(
                        overlay,
                        session.intrinsics,
                        session.dist_coeffs,
                        rvec,
                        tvec,
                        axis_len,
                    )
        elif session.target_type == "charuco":
            aruco_dict = _resolve_aruco_dict(session.aruco_dict)
            if hasattr(cv2.aruco, "CharucoBoard_create"):
                board = cv2.aruco.CharucoBoard_create(
                    int(session.cols),
                    int(session.rows),
                    float(session.square_size_m),
                    float(session.marker_size_m),
                    aruco_dict,
                )
            else:
                board = cv2.aruco.CharucoBoard(
                    (int(session.cols), int(session.rows)),
                    float(session.square_size_m),
                    float(session.marker_size_m),
                    aruco_dict,
                )
            if hasattr(cv2.aruco, "DetectorParameters_create"):
                parameters = cv2.aruco.DetectorParameters_create()
            else:
                parameters = cv2.aruco.DetectorParameters()
            corners, ids, rejected = cv2.aruco.detectMarkers(
                gray, aruco_dict, parameters=parameters
            )
            if corners:
                for marker in corners:
                    try:
                        pts = np.asarray(marker, dtype=np.float32).reshape(-1, 2)
                        center = np.mean(pts, axis=0)
                        marker_centers_uv.append([float(center[0]), float(center[1])])
                    except Exception:
                        pass
                if marker_centers_uv:
                    mean_center = np.mean(
                        np.asarray(marker_centers_uv, dtype=np.float32), axis=0
                    )
                    marker_centroid_uv = [
                        float(mean_center[0]),
                        float(mean_center[1]),
                    ]
            if ids is not None:
                marker_count = int(len(ids))
                try:
                    marker_ids = [int(v) for v in ids.flatten().tolist()]
                except Exception:
                    marker_ids = []
            if ids is None or len(ids) == 0:
                result = {
                    "status": "not_found",
                    "camera_id": cam_id,
                    "sequence_id": payload.get("sequence_id"),
                    "aruco_dict": session.aruco_dict,
                    "marker_count": marker_count,
                    "marker_ids": marker_ids[:50],
                    "marker_centers_uv": marker_centers_uv,
                    "marker_centroid_uv": marker_centroid_uv,
                    "image_size": image_size,
                }
                with self._lock:
                    self._last_detection = None
                return result
            try:
                cv2.aruco.refineDetectedMarkers(
                    gray,
                    board,
                    corners,
                    ids,
                    rejected,
                    session.intrinsics,
                    session.dist_coeffs,
                )
            except Exception:
                pass
            retval, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
                corners, ids, gray, board, session.intrinsics, session.dist_coeffs
            )
            if charuco_corners is not None and len(charuco_corners) > 0:
                try:
                    mean_charuco = np.mean(
                        np.asarray(charuco_corners, dtype=np.float32).reshape(-1, 2),
                        axis=0,
                    )
                    charuco_centroid_uv = [
                        float(mean_charuco[0]),
                        float(mean_charuco[1]),
                    ]
                except Exception:
                    charuco_centroid_uv = None
            if charuco_corners is None or charuco_ids is None or len(charuco_ids) < 4:
                if return_overlay:
                    overlay = frame.copy()
                    cv2.aruco.drawDetectedMarkers(overlay, corners, ids)
                if return_overlay and overlay is not None:
                    ok, buf = cv2.imencode(
                        ".jpg", overlay, [int(cv2.IMWRITE_JPEG_QUALITY), 85]
                    )
                    if ok:
                        overlay_b64 = base64.b64encode(buf.tobytes()).decode("ascii")
                        overlay_fmt = "jpg"
                result = {
                    "status": "not_found",
                    "camera_id": cam_id,
                    "sequence_id": payload.get("sequence_id"),
                    "aruco_dict": session.aruco_dict,
                    "marker_count": marker_count,
                    "marker_ids": marker_ids[:50],
                    "marker_centers_uv": marker_centers_uv,
                    "marker_centroid_uv": marker_centroid_uv,
                    "charuco_centroid_uv": charuco_centroid_uv,
                    "charuco_corners": (
                        int(len(charuco_corners)) if charuco_corners is not None else 0
                    ),
                    "overlay_b64": overlay_b64,
                    "overlay_format": overlay_fmt,
                    "image_size": image_size,
                }
                with self._lock:
                    self._last_detection = None
                return result
            ids_flat = charuco_ids.flatten()
            objp_all = _charuco_corners(board)
            objp_all = np.asarray(objp_all, dtype=np.float32).reshape(-1, 3)
            objp = objp_all[ids_flat]
            imgp = charuco_corners.reshape(-1, 2)
            charuco_image_points = imgp
            charuco_ids_flat = ids_flat
            if len(objp) > 0:
                sort_idx = np.lexsort((objp[:, 0], objp[:, 1]))
                first_idx = int(sort_idx[0])
                first_corner_uv = imgp[first_idx]
                first_corner_object_point = objp[first_idx]
                first_corner_depth = _nearest_depth_xyz_at_uv(
                    payload.get("depth_m"),
                    session.intrinsics,
                    first_corner_uv,
                    radius_px=8,
                )
            ok = False
            if hasattr(cv2.aruco, "estimatePoseCharucoBoard"):
                try:
                    ok, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
                        charuco_corners,
                        charuco_ids,
                        board,
                        session.intrinsics,
                        session.dist_coeffs,
                        None,
                        None,
                    )
                except Exception:
                    ok = False
            if not ok:
                ok, rvec, tvec = cv2.solvePnP(
                    objp, imgp, session.intrinsics, session.dist_coeffs
                )
            if not ok:
                raise HTTPException(status_code=500, detail="solvepnp_failed")
            r_target, _ = cv2.Rodrigues(rvec)
            if first_corner_object_point is not None:
                first_corner_cam = (
                    r_target
                    @ np.asarray(first_corner_object_point, dtype=np.float64).reshape(
                        3, 1
                    )
                    + tvec.reshape(3, 1)
                ).reshape(3)
                first_corner_pose_xyz = [float(v) for v in first_corner_cam.tolist()]
                if (
                    first_corner_depth is not None
                    and abs(float(first_corner_cam[2])) > 1e-6
                ):
                    depth_z = float(first_corner_depth["xyz_m"][2])
                    scale = depth_z / float(first_corner_cam[2])
                    first_corner_blended_xyz = [
                        float(first_corner_cam[0] * scale),
                        float(first_corner_cam[1] * scale),
                        depth_z,
                    ]
            axis_origin_object_point = _charuco_board_center(objp_all)
            if axis_origin_object_point is not None:
                origin_proj, _ = cv2.projectPoints(
                    np.asarray([axis_origin_object_point], dtype=np.float32).reshape(
                        -1, 1, 3
                    ),
                    rvec,
                    tvec,
                    session.intrinsics,
                    session.dist_coeffs,
                )
                axis_origin_uv = origin_proj.reshape(-1, 2)[0]
                axis_origin_cam = (
                    r_target
                    @ np.asarray(axis_origin_object_point, dtype=np.float64).reshape(3, 1)
                    + tvec.reshape(3, 1)
                ).reshape(3)
                axis_origin_pose_xyz = [float(v) for v in axis_origin_cam.tolist()]
                axis_origin_depth = _mean_depth_xyz_at_uv(
                    payload.get("depth_m"),
                    session.intrinsics,
                    axis_origin_uv,
                    radius_px=5,
                )
            proj, _ = cv2.projectPoints(
                objp, rvec, tvec, session.intrinsics, session.dist_coeffs
            )
            err = float(
                np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - imgp) ** 2, axis=1)))
            )
            corner_count = int(len(charuco_corners))
            if return_overlay:
                overlay = frame.copy()
                axis_len = float(
                    session.square_size_m * max(1.5, min(session.rows, session.cols) * 0.35)
                )
                _draw_charuco_overlay(
                    overlay,
                    corners if corners is not None else [],
                    charuco_corners,
                    axis_origin_object_point,
                    session.intrinsics,
                    session.dist_coeffs,
                    rvec,
                    tvec,
                    axis_len,
                )
        else:
            raise HTTPException(status_code=400, detail="target_not_supported")
        if return_overlay and overlay is not None:
            ok, buf = cv2.imencode(".jpg", overlay, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            if ok:
                overlay_b64 = base64.b64encode(buf.tobytes()).decode("ascii")
                overlay_fmt = "jpg"
        result = {
            "status": "ok",
            "camera_id": cam_id,
            "sequence_id": payload.get("sequence_id"),
            "timestamp_ns": payload.get("timestamp_ns"),
            "image_size": image_size,
            "rvec": [float(v) for v in rvec.reshape(-1)],
            "tvec": [float(v) for v in tvec.reshape(-1)],
            "reprojection_rmse_px": err,
            "corners": corner_count,
            "marker_count": marker_count,
            "marker_ids": marker_ids[:50],
            "marker_centers_uv": marker_centers_uv,
            "marker_centroid_uv": marker_centroid_uv,
            "charuco_centroid_uv": charuco_centroid_uv,
            "overlay_b64": overlay_b64,
            "overlay_format": overlay_fmt,
            "depth_available": payload.get("depth_m") is not None,
        }
        if session.target_type == "chessboard":
            result["image_points"] = [[float(p[0]), float(p[1])] for p in image_points]
            result["object_points"] = [
                [float(p[0]), float(p[1]), float(p[2])] for p in object_points
            ]
        elif session.target_type == "charuco":
            result["charuco_image_points"] = [
                [float(p[0]), float(p[1])] for p in charuco_image_points
            ]
            result["charuco_ids"] = [int(v) for v in charuco_ids_flat.tolist()]
            if first_corner_uv is not None:
                result["first_corner_uv"] = [
                    float(first_corner_uv[0]),
                    float(first_corner_uv[1]),
                ]
            if first_corner_object_point is not None:
                result["first_corner_object_point_m"] = [
                    float(first_corner_object_point[0]),
                    float(first_corner_object_point[1]),
                    float(first_corner_object_point[2]),
                ]
            if first_corner_pose_xyz is not None:
                result["first_corner_pose_xyz_m"] = first_corner_pose_xyz
            if first_corner_depth is not None:
                result["first_corner_depth_xyz_m"] = first_corner_depth["xyz_m"]
                result["first_corner_xyz_m"] = (
                    first_corner_blended_xyz
                    if first_corner_blended_xyz is not None
                    else first_corner_depth["xyz_m"]
                )
                result["first_corner_depth_samples"] = first_corner_depth["samples"]
                result["first_corner_depth_radius_px"] = first_corner_depth["radius_px"]
                result["first_corner_depth_uv"] = first_corner_depth["uv"]
                result["first_corner_depth_distance_px"] = first_corner_depth[
                    "distance_px"
                ]
            if axis_origin_uv is not None:
                result["axis_origin_uv"] = [
                    float(axis_origin_uv[0]),
                    float(axis_origin_uv[1]),
                ]
            if axis_origin_object_point is not None:
                result["axis_origin_object_point_m"] = [
                    float(axis_origin_object_point[0]),
                    float(axis_origin_object_point[1]),
                    float(axis_origin_object_point[2]),
                ]
            if axis_origin_pose_xyz is not None:
                result["axis_origin_pose_xyz_m"] = axis_origin_pose_xyz
            if axis_origin_depth is not None:
                result["axis_origin_xyz_m"] = axis_origin_depth["xyz_m"]
                result["axis_origin_depth_samples"] = axis_origin_depth["samples"]
                result["axis_origin_depth_radius_px"] = axis_origin_depth["radius_px"]
        with self._lock:
            self._last_detection = result
        return result

    def add_sample(self, req: SampleAddRequest) -> Dict[str, Any]:
        with self._lock:
            session = self._session
            last_detection = self._last_detection
        if not session:
            raise HTTPException(status_code=400, detail="session_not_started")
        cam_id = req.camera_id or session.camera_id
        if cam_id != session.camera_id:
            raise HTTPException(status_code=400, detail="camera_mismatch")
        detection = None
        if (
            req.use_last_detection
            and last_detection
            and last_detection.get("status") == "ok"
        ):
            detection = last_detection
        if not detection:
            detection = self.detect(cam_id, return_overlay=False)
            if detection.get("status") != "ok":
                raise HTTPException(status_code=400, detail="detection_failed")
        sample_id = len(self._samples) + 1
        entry = {
            "sample_id": sample_id,
            "camera_id": cam_id,
            "robot": req.robot.model_dump(exclude_none=True),
            "robot_state_debug": req.robot_state_debug,
            "detection": {
                "rvec": detection["rvec"],
                "tvec": detection["tvec"],
                "reprojection_rmse_px": detection.get("reprojection_rmse_px"),
                "sequence_id": detection.get("sequence_id"),
                "timestamp_ns": detection.get("timestamp_ns"),
                "target_type": session.target_type,
                "image_size": detection.get("image_size"),
                "image_points": detection.get("image_points"),
                "object_points": detection.get("object_points"),
                "charuco_image_points": detection.get("charuco_image_points"),
                "charuco_ids": detection.get("charuco_ids"),
                "first_corner_uv": detection.get("first_corner_uv"),
                "first_corner_xyz_m": detection.get("first_corner_xyz_m"),
                "first_corner_pose_xyz_m": detection.get("first_corner_pose_xyz_m"),
                "first_corner_depth_xyz_m": detection.get("first_corner_depth_xyz_m"),
                "first_corner_depth_uv": detection.get("first_corner_depth_uv"),
                "first_corner_depth_samples": detection.get(
                    "first_corner_depth_samples"
                ),
                "first_corner_depth_radius_px": detection.get(
                    "first_corner_depth_radius_px"
                ),
                "first_corner_depth_distance_px": detection.get(
                    "first_corner_depth_distance_px"
                ),
                "axis_origin_uv": detection.get("axis_origin_uv"),
                "axis_origin_xyz_m": detection.get("axis_origin_xyz_m"),
                "axis_origin_depth_samples": detection.get(
                    "axis_origin_depth_samples"
                ),
            },
        }
        with self._lock:
            self._samples.append(entry)
        return {"status": "ok", "sample": entry, "samples": len(self._samples)}

    def list_samples(self) -> Dict[str, Any]:
        with self._lock:
            return {"samples": list(self._samples)}

    def clear_samples(self) -> Dict[str, Any]:
        with self._lock:
            self._samples = []
        return {"status": "cleared"}

    def import_samples(self, req: SamplesImportRequest) -> Dict[str, Any]:
        with self._lock:
            session = self._session
        if not session:
            raise HTTPException(status_code=400, detail="session_not_started")
        if not isinstance(req.samples, list) or len(req.samples) == 0:
            raise HTTPException(status_code=400, detail="no_samples_to_import")
        if req.camera_id and str(req.camera_id) != str(session.camera_id):
            raise HTTPException(status_code=400, detail="camera_mismatch")
        if req.mode and str(req.mode) != str(session.mode):
            raise HTTPException(status_code=400, detail="mode_mismatch")
        target = req.target or {}
        if target:
            if str(target.get("type") or session.target_type) != str(session.target_type):
                raise HTTPException(status_code=400, detail="target_type_mismatch")
            if int(target.get("rows") or session.rows) != int(session.rows):
                raise HTTPException(status_code=400, detail="target_rows_mismatch")
            if int(target.get("cols") or session.cols) != int(session.cols):
                raise HTTPException(status_code=400, detail="target_cols_mismatch")
            if abs(float(target.get("square_size_m") or session.square_size_m) - float(session.square_size_m)) > 1e-9:
                raise HTTPException(status_code=400, detail="target_square_size_mismatch")
            if abs(float(target.get("marker_size_m") or session.marker_size_m) - float(session.marker_size_m)) > 1e-9:
                raise HTTPException(status_code=400, detail="target_marker_size_mismatch")
            if str(target.get("aruco_dict") or session.aruco_dict) != str(session.aruco_dict):
                raise HTTPException(status_code=400, detail="target_aruco_dict_mismatch")

        imported: List[Dict[str, Any]] = []
        recomputed_count = 0
        for idx, sample in enumerate(req.samples, start=1):
            robot_pose = sample.get("robot_pose") or sample.get("robot") or {}
            target_pose = sample.get("target_pose_in_camera_frame") or sample.get("detection") or {}
            robot_pos = robot_pose.get("translation_m") or robot_pose.get("position_m") or [0.0, 0.0, 0.0]
            robot_quat = robot_pose.get("rotation_quat_xyzw") or robot_pose.get("quat_xyzw") or [0.0, 0.0, 0.0, 1.0]
            det_rvec = target_pose.get("rotation_rvec") or target_pose.get("rvec") or [0.0, 0.0, 0.0]
            det_tvec = target_pose.get("translation_m") or target_pose.get("tvec") or [0.0, 0.0, 0.0]
            if len(robot_pos) < 3 or len(robot_quat) < 4 or len(det_rvec) < 3 or len(det_tvec) < 3:
                raise HTTPException(status_code=400, detail=f"invalid_sample_{idx}")
            recomputed_pose = _recompute_detection_pose(session, target_pose)
            if recomputed_pose is not None:
                det_rvec = recomputed_pose["rvec"]
                det_tvec = recomputed_pose["tvec"]
                recomputed_count += 1
            reproj_rmse = (
                recomputed_pose["reprojection_rmse_px"]
                if recomputed_pose is not None
                else target_pose.get("reprojection_rmse_px")
            )
            imported.append(
                {
                    "sample_id": int(sample.get("sample_id") or idx),
                    "camera_id": str(sample.get("camera_id") or req.camera_id or session.camera_id),
                    "robot": {
                        "position_m": [float(v) for v in robot_pos[:3]],
                        "quat_xyzw": [float(v) for v in robot_quat[:4]],
                        "frame": str(robot_pose.get("frame") or "base"),
                        "tcp_frame": robot_pose.get("tcp_frame"),
                        "pose_source": robot_pose.get("pose_source"),
                        "timestamp_ns": robot_pose.get("timestamp_ns"),
                    },
                    "robot_state_debug": sample.get("robot_state_debug"),
                    "detection": {
                        "rvec": [float(v) for v in det_rvec[:3]],
                        "tvec": [float(v) for v in det_tvec[:3]],
                        "reprojection_rmse_px": reproj_rmse,
                        "sequence_id": sample.get("sequence_id") or target_pose.get("sequence_id"),
                        "timestamp_ns": sample.get("timestamp_ns") or target_pose.get("timestamp_ns"),
                        "target_type": session.target_type,
                        "image_size": target_pose.get("image_size"),
                        "image_points": target_pose.get("image_points"),
                        "object_points": target_pose.get("object_points"),
                        "charuco_image_points": target_pose.get("charuco_image_points"),
                        "charuco_ids": target_pose.get("charuco_ids"),
                        "first_corner_uv": (sample.get("overlay_first_corner_in_camera_frame") or {}).get("image_uv"),
                        "first_corner_xyz_m": (sample.get("overlay_first_corner_in_camera_frame") or {}).get("depth_xyz_m"),
                        "first_corner_pose_xyz_m": (sample.get("overlay_first_corner_in_camera_frame") or {}).get("translation_m"),
                        "first_corner_depth_xyz_m": (sample.get("overlay_first_corner_in_camera_frame") or {}).get("depth_mean_xyz_m"),
                        "first_corner_depth_uv": (sample.get("overlay_first_corner_in_camera_frame") or {}).get("image_uv"),
                        "first_corner_depth_samples": (sample.get("overlay_first_corner_in_camera_frame") or {}).get("depth_sample_count"),
                        "first_corner_depth_radius_px": (sample.get("overlay_first_corner_in_camera_frame") or {}).get("depth_radius_px"),
                        "first_corner_depth_distance_px": (sample.get("overlay_first_corner_in_camera_frame") or {}).get("depth_distance_px"),
                        "axis_origin_uv": (sample.get("overlay_axis_origin_in_camera_frame") or {}).get("image_uv"),
                        "axis_origin_xyz_m": (sample.get("overlay_axis_origin_in_camera_frame") or {}).get("depth_mean_xyz_m"),
                        "axis_origin_pose_xyz_m": (sample.get("overlay_axis_origin_in_camera_frame") or {}).get("translation_m"),
                        "axis_origin_depth_samples": (sample.get("overlay_axis_origin_in_camera_frame") or {}).get("depth_sample_count"),
                    },
                }
            )
        with self._lock:
            self._samples = imported
            self._last_result = None
        return {
            "status": "imported",
            "samples": len(imported),
            "recomputed_with_session_intrinsics": int(recomputed_count),
        }

    def compute(
        self,
        method_override: Optional[str] = None,
        tcp_calibration: Optional[TcpCalibrationModel] = None,
    ) -> Dict[str, Any]:
        with self._lock:
            session = self._session
            samples = list(self._samples)
        if not session:
            raise HTTPException(status_code=400, detail="session_not_started")
        if len(samples) < 4:
            raise HTTPException(status_code=400, detail="need_at_least_4_samples")
        if session.mode != "eye_in_hand":
            raise HTTPException(status_code=400, detail="mode_not_supported")
        raw_sample_ids = [
            int(sample.get("sample_id") or idx)
            for idx, sample in enumerate(samples, start=1)
            if str((sample.get("robot") or {}).get("pose_source") or "").strip().lower()
            != "custom_tcp"
        ]
        if raw_sample_ids:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "samples_must_use_custom_tcp_pose",
                    "sample_ids": raw_sample_ids,
                },
            )
        method = str(method_override or session.method or "Tsai")
        if method not in HAND_EYE_METHODS:
            raise HTTPException(status_code=400, detail="unknown_method")
        r_gripper2base = []
        t_gripper2base = []
        r_target2cam = []
        t_target2cam = []
        reproj = []
        tcp_translation = [0.0, 0.0, 0.0]
        tcp_rpy = [0.0, 0.0, 0.0]
        tcp_frame = "flange"
        tcp_applied = False
        r_tcp_to_flange = np.eye(3, dtype=np.float64)
        t_tcp_to_flange = np.zeros((3, 1), dtype=np.float64)
        if tcp_calibration is not None:
            tcp_translation = [float(v) for v in tcp_calibration.translation_m[:3]]
            tcp_rpy = [float(v) for v in tcp_calibration.rotation_rpy_deg[:3]]
            tcp_frame = str(tcp_calibration.frame or "flange")
            r_flange_to_tcp = rpy_deg_to_rotmat(tcp_rpy)
            t_flange_to_tcp = np.asarray(tcp_translation, dtype=np.float64).reshape(3, 1)
            r_tcp_to_flange = r_flange_to_tcp.T
            t_tcp_to_flange = -r_tcp_to_flange @ t_flange_to_tcp
            tcp_applied = bool(
                np.linalg.norm(t_flange_to_tcp) > 1e-12
                or np.linalg.norm(r_flange_to_tcp - np.eye(3)) > 1e-12
            )
        for sample in samples:
            robot = sample["robot"]
            rvec = np.array(sample["detection"]["rvec"], dtype=np.float64).reshape(3, 1)
            tvec = np.array(sample["detection"]["tvec"], dtype=np.float64).reshape(3, 1)
            r_target, _ = cv2.Rodrigues(rvec)
            r_target2cam.append(r_target)
            t_target2cam.append(tvec)
            reproj_err = sample["detection"].get("reprojection_rmse_px")
            if reproj_err is not None:
                reproj.append(float(reproj_err))
            r_robot = quat_xyzw_to_rotmat(robot["quat_xyzw"])
            t_robot = np.array(robot["position_m"], dtype=np.float64).reshape(3, 1)
            if tcp_applied:
                r_gripper = r_robot @ r_tcp_to_flange
                t_gripper = t_robot + r_robot @ t_tcp_to_flange
            else:
                r_gripper = r_robot
                t_gripper = t_robot
            r_gripper2base.append(r_gripper)
            t_gripper2base.append(t_gripper)
        method_flag = HAND_EYE_METHODS[method]
        r_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
            r_gripper2base,
            t_gripper2base,
            r_target2cam,
            t_target2cam,
            method=method_flag,
        )
        q_cam2gripper = rotmat_to_quat_xyzw(r_cam2gripper)
        base_to_camera = []
        base_to_camera_quat = []
        base_to_camera_trans = []
        target_to_base_rot = []
        target_to_base_trans = []
        timestamp_deltas_ms = []
        for idx in range(len(samples)):
            r_bg = r_gripper2base[idx]
            t_bg = t_gripper2base[idx]
            r_gc = r_cam2gripper
            t_gc = t_cam2gripper
            r_bc = r_bg @ r_gc
            t_bc = r_bg @ t_gc + t_bg
            q_bc = rotmat_to_quat_xyzw(r_bc)
            base_to_camera.append((r_bc, t_bc))
            base_to_camera_quat.append(q_bc)
            base_to_camera_trans.append(t_bc.reshape(3))
            r_tc = r_target2cam[idx]
            t_tc = t_target2cam[idx]
            r_bt = r_bc @ r_tc
            t_bt = r_bc @ t_tc + t_bc
            target_to_base_rot.append(r_bt)
            target_to_base_trans.append(t_bt.reshape(3))
            robot_ts = samples[idx].get("robot", {}).get("timestamp_ns")
            det_ts = samples[idx].get("detection", {}).get("timestamp_ns")
            if robot_ts is not None and det_ts is not None:
                try:
                    timestamp_deltas_ms.append(
                        abs(float(int(det_ts) - int(robot_ts))) / 1e6
                    )
                except Exception:
                    pass
        mean_trans = np.mean(np.stack(base_to_camera_trans), axis=0)
        mean_quat = average_quaternions(base_to_camera_quat)
        mean_rot = quat_xyzw_to_rotmat(mean_quat)
        trans_errors = [
            float(np.linalg.norm(t - mean_trans)) for t in base_to_camera_trans
        ]
        rot_errors = [rotation_error_deg(mean_rot, r) for r, _ in base_to_camera]
        target_trans_mean = np.mean(np.stack(target_to_base_trans), axis=0)
        target_rot_mean = quat_xyzw_to_rotmat(
            average_quaternions([rotmat_to_quat_xyzw(r) for r in target_to_base_rot])
        )
        target_trans_err = [
            float(np.linalg.norm(t - target_trans_mean)) for t in target_to_base_trans
        ]
        target_rot_err = [
            rotation_error_deg(target_rot_mean, r) for r in target_to_base_rot
        ]
        reproj_mean = float(np.mean(reproj)) if reproj else None
        reproj_stats = summarize_scalar_stats(reproj)
        base_trans_stats = summarize_scalar_stats(trans_errors)
        base_rot_stats = summarize_scalar_stats(rot_errors)
        target_trans_stats = summarize_scalar_stats(target_trans_err)
        target_rot_stats = summarize_scalar_stats(target_rot_err)
        timestamp_delta_stats = summarize_scalar_stats(timestamp_deltas_ms)
        tcp_offset_rows = []
        world_offset_rows = []
        for sample in samples:
            debug = sample.get("robot_state_debug") or {}
            tcp_offset = debug.get("tcp_offset_mm_rpy_deg")
            world_offset = debug.get("world_offset_mm_rpy_deg")
            if isinstance(tcp_offset, (list, tuple)) and len(tcp_offset) >= 6:
                tcp_offset_rows.append([float(v) for v in tcp_offset[:6]])
            if isinstance(world_offset, (list, tuple)) and len(world_offset) >= 6:
                world_offset_rows.append([float(v) for v in world_offset[:6]])
        tcp_offset_unique = sorted(
            {
                tuple(round(float(v), 6) for v in row)
                for row in tcp_offset_rows
            }
        )
        world_offset_unique = sorted(
            {
                tuple(round(float(v), 6) for v in row)
                for row in world_offset_rows
            }
        )
        result = {
            "status": "ok",
            "samples": len(samples),
            "method": method,
            "solver_inputs": {
                "robot_pose_expected": "flange/gripper_in_robot_base_frame (^bT_g)",
                "target_pose_expected": "calibration_target_in_camera_frame (^cT_t)",
                "result_transform": "camera_in_gripper (^gT_c)",
                "tcp_calibration": (
                    "reported TCP pose is converted to flange/gripper pose with inverse flange_to_tcp transform"
                    if tcp_applied
                    else "identity; reported robot pose is used directly"
                ),
            },
            "camera_in_gripper": {
                "translation_m": [float(v) for v in t_cam2gripper.reshape(-1)],
                "rotation_quat_xyzw": q_cam2gripper,
            },
            "base_to_camera": {
                "translation_m": [float(v) for v in mean_trans.reshape(-1)],
                "rotation_quat_xyzw": mean_quat,
            },
            "quality": {
                "primary_metric": "target_in_base_residuals",
                "reprojection_rmse_px": {
                    "mean": reproj_mean,
                    "stats": reproj_stats,
                },
                "target_in_base_translation_residual_m": target_trans_stats,
                "target_in_base_rotation_residual_deg": target_rot_stats,
                "robot_detection_timestamp_delta_ms": timestamp_delta_stats,
                "summary": {
                    "translation_mean_mm": (
                        float(target_trans_stats["mean"] * 1000.0)
                        if target_trans_stats
                        else None
                    ),
                    "translation_max_mm": (
                        float(target_trans_stats["max"] * 1000.0)
                        if target_trans_stats
                        else None
                    ),
                    "rotation_mean_deg": (
                        float(target_rot_stats["mean"]) if target_rot_stats else None
                    ),
                    "rotation_max_deg": (
                        float(target_rot_stats["max"]) if target_rot_stats else None
                    ),
                },
                "notes": [
                    "Eye-in-hand quality should be judged mainly from target_in_base residuals.",
                    "Lower translation/rotation residual mean and max values are better.",
                ],
            },
            "debug": {
                "base_to_camera_spread_m": base_trans_stats,
                "base_to_camera_spread_deg": base_rot_stats,
                "robot_frame_audit": {
                    "samples_with_robot_state_debug": int(
                        sum(1 for sample in samples if sample.get("robot_state_debug"))
                    ),
                    "tcp_offset_mm_rpy_deg_unique": [list(row) for row in tcp_offset_unique],
                    "world_offset_mm_rpy_deg_unique": [list(row) for row in world_offset_unique],
                    "calibration_tcp": {
                        "applied_to_solver": tcp_applied,
                        "frame": tcp_frame,
                        "flange_to_tcp_translation_m": tcp_translation,
                        "flange_to_tcp_rotation_rpy_deg": tcp_rpy,
                    },
                    "notes": [
                        "Non-zero TCP/world offsets during capture can shift calibration away from raw flange frame.",
                        "If you intend flange-based hand-eye, TCP/world offsets should usually be zero or explicitly accounted for.",
                    ],
                },
            },
        }
        with self._lock:
            self._last_result = result
        return result

    def compute_intrinsics(self) -> Dict[str, Any]:
        with self._lock:
            session = self._session
            samples = list(self._samples)
        if not session:
            raise HTTPException(status_code=400, detail="session_not_started")
        if len(samples) < 4:
            raise HTTPException(status_code=400, detail="need_at_least_4_samples")

        image_size = None
        if session.target_type == "chessboard":
            objpoints: List[np.ndarray] = []
            imgpoints: List[np.ndarray] = []
            for sample in samples:
                det = sample.get("detection", {})
                pts2 = det.get("image_points") or []
                pts3 = det.get("object_points") or []
                sz = det.get("image_size") or []
                if not (
                    isinstance(pts2, list)
                    and isinstance(pts3, list)
                    and len(pts2) >= 4
                    and len(pts3) >= 4
                ):
                    continue
                if (
                    isinstance(sz, list)
                    and len(sz) >= 2
                    and int(sz[0]) > 0
                    and int(sz[1]) > 0
                ):
                    image_size = (int(sz[0]), int(sz[1]))
                imgp = np.asarray(pts2, dtype=np.float32).reshape(-1, 1, 2)
                objp = np.asarray(pts3, dtype=np.float32).reshape(-1, 1, 3)
                objpoints.append(objp)
                imgpoints.append(imgp)
            valid_samples = len(objpoints)
            if valid_samples < 4:
                raise HTTPException(
                    status_code=400, detail="need_at_least_4_valid_samples"
                )
            if not image_size:
                raise HTTPException(status_code=400, detail="missing_image_size")
            rmse, k, dist, _, _ = cv2.calibrateCamera(
                objpoints,
                imgpoints,
                image_size,
                None,
                None,
            )
        elif session.target_type == "charuco":
            if not hasattr(cv2, "aruco") or not hasattr(
                cv2.aruco, "calibrateCameraCharuco"
            ):
                raise HTTPException(
                    status_code=500, detail="charuco_calibration_not_available"
                )
            aruco_dict = _resolve_aruco_dict(session.aruco_dict)
            if hasattr(cv2.aruco, "CharucoBoard_create"):
                board = cv2.aruco.CharucoBoard_create(
                    int(session.cols),
                    int(session.rows),
                    float(session.square_size_m),
                    float(session.marker_size_m),
                    aruco_dict,
                )
            else:
                board = cv2.aruco.CharucoBoard(
                    (int(session.cols), int(session.rows)),
                    float(session.square_size_m),
                    float(session.marker_size_m),
                    aruco_dict,
                )
            charuco_corners: List[np.ndarray] = []
            charuco_ids: List[np.ndarray] = []
            for sample in samples:
                det = sample.get("detection", {})
                pts2 = det.get("charuco_image_points") or []
                ids = det.get("charuco_ids") or []
                sz = det.get("image_size") or []
                if not (
                    isinstance(pts2, list)
                    and isinstance(ids, list)
                    and len(pts2) >= 4
                    and len(ids) >= 4
                ):
                    continue
                if (
                    isinstance(sz, list)
                    and len(sz) >= 2
                    and int(sz[0]) > 0
                    and int(sz[1]) > 0
                ):
                    image_size = (int(sz[0]), int(sz[1]))
                corners = np.asarray(pts2, dtype=np.float32).reshape(-1, 1, 2)
                cid = np.asarray(ids, dtype=np.int32).reshape(-1, 1)
                charuco_corners.append(corners)
                charuco_ids.append(cid)
            valid_samples = len(charuco_corners)
            if valid_samples < 4:
                raise HTTPException(
                    status_code=400, detail="need_at_least_4_valid_samples"
                )
            if not image_size:
                raise HTTPException(status_code=400, detail="missing_image_size")
            rmse, k, dist, _, _ = cv2.aruco.calibrateCameraCharuco(
                charuco_corners,
                charuco_ids,
                board,
                image_size,
                None,
                None,
            )
        else:
            raise HTTPException(status_code=400, detail="target_not_supported")

        k = np.asarray(k, dtype=np.float64)
        dist = np.asarray(dist, dtype=np.float64).reshape(-1)
        result = {
            "status": "ok",
            "samples_total": len(samples),
            "samples_used": int(valid_samples),
            "target_type": session.target_type,
            "image_size": [int(image_size[0]), int(image_size[1])],
            "intrinsics": {
                "fx": float(k[0, 0]),
                "fy": float(k[1, 1]),
                "cx": float(k[0, 2]),
                "cy": float(k[1, 2]),
                "dist_coeffs": [float(v) for v in dist.tolist()],
            },
            "quality": {
                "reprojection_rmse_px": float(rmse),
            },
        }
        with self._lock:
            if self._session:
                self._session.intrinsics = k
                self._session.dist_coeffs = dist.reshape(-1, 1)
        return result


def create_calibration_app(
    service: CalibrationService, cors_origins: List[str]
) -> FastAPI:
    app = FastAPI(title="Camera Core Calibration")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    def health() -> Dict[str, Any]:
        return {"status": "alive"}

    @app.get("/session")
    def session_status() -> Dict[str, Any]:
        return service.status()

    @app.post("/session/start")
    def session_start(req: SessionStartRequest) -> Dict[str, Any]:
        return service.start_session(req)

    @app.post("/session/stop")
    def session_stop(req: SessionStopRequest) -> Dict[str, Any]:
        return service.stop_session(req.reason)

    @app.get("/frame")
    def frame(camera_id: str, quality: int = 75) -> Response:
        data = service.frame_jpeg(camera_id, quality=quality)
        return Response(
            content=data,
            media_type="image/jpeg",
            headers={"Cache-Control": "no-store"},
        )

    @app.get("/camera/intrinsics")
    def camera_intrinsics(camera_id: str) -> Dict[str, Any]:
        return service.camera_intrinsics(camera_id)

    @app.post("/detect")
    def detect(req: DetectRequest) -> Dict[str, Any]:
        return service.detect(
            req.camera_id,
            req.return_overlay,
            min_timestamp_ns=req.min_timestamp_ns,
            wait_timeout_ms=req.wait_timeout_ms,
        )

    @app.post("/samples/add")
    def samples_add(req: SampleAddRequest) -> Dict[str, Any]:
        return service.add_sample(req)

    @app.get("/samples")
    def samples_list() -> Dict[str, Any]:
        return service.list_samples()

    @app.post("/samples/clear")
    def samples_clear() -> Dict[str, Any]:
        return service.clear_samples()

    @app.post("/samples/import")
    def samples_import(req: SamplesImportRequest) -> Dict[str, Any]:
        return service.import_samples(req)

    @app.post("/compute")
    def compute(req: Optional[ComputeRequest] = None) -> Dict[str, Any]:
        return service.compute(
            req.method if req else None,
            req.tcp_calibration if req else None,
        )

    @app.post("/compute/intrinsics")
    def compute_intrinsics() -> Dict[str, Any]:
        return service.compute_intrinsics()

    return app


class CalibrationServerManager:
    def __init__(
        self,
        host: str,
        port: int,
        pub_endpoint: str,
        topic: str,
        log_level: str,
        cors_origins: Optional[List[str]] = None,
    ) -> None:
        self.host = host
        self.port = port
        self.pub_endpoint = pub_endpoint
        self.topic = topic
        self.log_level = log_level
        self.cors_origins = cors_origins or ["*"]
        self._thread: Optional[threading.Thread] = None
        self._server: Optional[uvicorn.Server] = None
        self._service: Optional[CalibrationService] = None
        self._running = False
        self._starting = False
        self._lock = threading.Lock()

    def _run(self) -> None:
        self._service = CalibrationService(self.pub_endpoint, self.topic)
        app = create_calibration_app(self._service, self.cors_origins)
        config = uvicorn.Config(
            app,
            host=self.host,
            port=self.port,
            log_level=self.log_level.lower(),
            access_log=False,
        )
        self._server = uvicorn.Server(config)
        self._running = True
        self._starting = False
        try:
            self._server.run()
        finally:
            self._running = False
            self._starting = False
            if self._service:
                self._service.shutdown()
            self._service = None

    def start(self) -> Dict[str, Any]:
        with self._lock:
            if self._running:
                return {"status": "running", "url": self.url()}
            if self._starting:
                return {"status": "starting", "url": self.url()}
            self._starting = True
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
        time.sleep(0.1)
        return {"status": "starting", "url": self.url()}

    def stop(self) -> Dict[str, Any]:
        with self._lock:
            if not self._running or not self._server:
                return {"status": "stopped"}
            self._server.should_exit = True
        if self._thread:
            self._thread.join(timeout=2.0)
        return {"status": "stopping"}

    def status(self) -> Dict[str, Any]:
        if self._running:
            state = "running"
        elif self._starting:
            state = "starting"
        else:
            state = "stopped"
        return {"status": state, "url": self.url()}

    def url(self) -> str:
        return f"http://{self.host}:{self.port}"
