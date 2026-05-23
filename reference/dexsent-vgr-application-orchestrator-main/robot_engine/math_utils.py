from __future__ import annotations

import math
from typing import Dict, Iterable, List

import numpy as np

from .interfaces.schemas import Transform3D


def as_matrix(transform: Transform3D) -> np.ndarray:
    mat = np.asarray(transform.matrix, dtype=float)
    if mat.shape != (4, 4) or not np.isfinite(mat).all():
        raise ValueError("transform matrix must be finite 4x4")
    if not np.allclose(mat[3], [0.0, 0.0, 0.0, 1.0]):
        raise ValueError("homogeneous transform bottom row must be [0, 0, 0, 1]")
    return mat


def to_transform(parent: str, child: str, matrix: np.ndarray) -> Transform3D:
    return Transform3D(parent_frame=parent, child_frame=child, matrix=np.asarray(matrix, dtype=float).tolist())


def identity_transform(parent: str, child: str) -> Transform3D:
    return to_transform(parent, child, np.eye(4))


def translation_matrix(xyz: Iterable[float]) -> np.ndarray:
    out = np.eye(4)
    out[:3, 3] = np.asarray(list(xyz), dtype=float)
    return out


def axis_angle_matrix(axis: Iterable[float], angle: float) -> np.ndarray:
    axis = normalize(np.asarray(list(axis), dtype=float))
    x, y, z = axis
    c = math.cos(angle)
    s = math.sin(angle)
    C = 1.0 - c
    rot = np.array(
        [
            [x * x * C + c, x * y * C - z * s, x * z * C + y * s],
            [y * x * C + z * s, y * y * C + c, y * z * C - x * s],
            [z * x * C - y * s, z * y * C + x * s, z * z * C + c],
        ],
        dtype=float,
    )
    out = np.eye(4)
    out[:3, :3] = rot
    return out


def joint_motion_matrix(joint_type: str, axis: Iterable[float], value: float) -> np.ndarray:
    if joint_type == "revolute":
        return axis_angle_matrix(axis, value)
    if joint_type == "prismatic":
        return translation_matrix(normalize(np.asarray(list(axis), dtype=float)) * value)
    return np.eye(4)


def normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 0 or not np.isfinite(norm):
        raise ValueError("axis must be finite and non-zero")
    return vector / norm


def rotation_error(current: np.ndarray, target: np.ndarray) -> np.ndarray:
    rel = target[:3, :3] @ current[:3, :3].T
    cos_angle = max(-1.0, min(1.0, (float(np.trace(rel)) - 1.0) / 2.0))
    angle = math.acos(cos_angle)
    if abs(angle) < 1e-12:
        return np.zeros(3)
    axis = np.array([rel[2, 1] - rel[1, 2], rel[0, 2] - rel[2, 0], rel[1, 0] - rel[0, 1]]) / (2.0 * math.sin(angle))
    return axis * angle


def pose_error(current: np.ndarray, target: np.ndarray) -> np.ndarray:
    return np.r_[target[:3, 3] - current[:3, 3], rotation_error(current, target)]


def quaternion_xyzw_to_matrix(position: List[float], quat: List[float]) -> np.ndarray:
    x, y, z, w = [float(v) for v in quat]
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm <= 0:
        raise ValueError("quaternion must be non-zero")
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    rot = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )
    out = np.eye(4)
    out[:3, :3] = rot
    out[:3, 3] = position
    return out


def ordered_joint_names(joint_positions: Dict[str, float], all_names: Iterable[str]) -> List[str]:
    return [name for name in all_names if name in joint_positions]
