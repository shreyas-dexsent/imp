from __future__ import annotations

import numpy as np

from robot_engine.core_math.rotations import (
    orientation_error,
    quaternion_from_rotation_matrix,
    rotation_matrix_from_quaternion,
    validate_rotation_matrix,
)


def _mat(T):
    T = np.asarray(T, dtype=float)
    validate_transform(T)
    return T


def compose_transform(T_ab, T_bc):
    return _mat(T_ab) @ _mat(T_bc)


def invert_transform(T_ab):
    T = _mat(T_ab)
    out = np.eye(4)
    out[:3, :3] = T[:3, :3].T
    out[:3, 3] = -(T[:3, :3].T @ T[:3, 3])
    return out


def transform_point(T, p):
    p = np.asarray(p, dtype=float).reshape(3)
    return (_mat(T) @ np.r_[p, 1.0])[:3]


def transform_points(T, points):
    points = np.asarray(points, dtype=float)
    return np.asarray([transform_point(T, p) for p in points])


def transform_vector(T, v):
    return _mat(T)[:3, :3] @ np.asarray(v, dtype=float).reshape(3)


def adjoint_SE3(T):
    T = _mat(T)
    R = T[:3, :3]
    p = T[:3, 3]
    px = np.array([[0, -p[2], p[1]], [p[2], 0, -p[0]], [-p[1], p[0], 0]], dtype=float)
    # Twist convention: [vx, vy, vz, wx, wy, wz].
    return np.block([[R, px @ R], [np.zeros((3, 3)), R]])


def transform_twist(T, twist):
    return adjoint_SE3(T) @ np.asarray(twist, dtype=float).reshape(6)


def transform_wrench(T, wrench):
    return np.linalg.inv(adjoint_SE3(T)).T @ np.asarray(wrench, dtype=float).reshape(6)


def relative_transform(T_world_a, T_world_b):
    return invert_transform(T_world_a) @ _mat(T_world_b)


def pose_to_matrix(position, quaternion):
    T = np.eye(4)
    T[:3, :3] = rotation_matrix_from_quaternion(quaternion)
    T[:3, 3] = np.asarray(position, dtype=float).reshape(3)
    validate_transform(T)
    return T


def matrix_to_pose(T):
    T = _mat(T)
    return T[:3, 3].copy(), quaternion_from_rotation_matrix(T[:3, :3])


def validate_transform(T):
    T = np.asarray(T, dtype=float)
    if T.shape != (4, 4):
        raise ValueError("transform must be 4x4")
    if not np.isfinite(T).all():
        raise ValueError("transform must be finite")
    if not np.allclose(T[3], [0.0, 0.0, 0.0, 1.0], atol=1e-9):
        raise ValueError("transform last row must be [0, 0, 0, 1]")
    validate_rotation_matrix(T[:3, :3])
    return True


def is_valid_transform(T):
    try:
        return bool(validate_transform(T))
    except Exception:
        return False


def translation_error(T_current, T_target):
    return _mat(T_target)[:3, 3] - _mat(T_current)[:3, 3]


def rotation_error(T_current, T_target):
    return orientation_error(_mat(T_current)[:3, :3], _mat(T_target)[:3, :3])


def pose_error(T_current, T_target):
    return np.r_[translation_error(T_current, T_target), rotation_error(T_current, T_target)]
