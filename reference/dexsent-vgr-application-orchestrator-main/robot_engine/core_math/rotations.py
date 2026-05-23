from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

from robot_engine.core_math.tolerances import DEFAULT_TOLERANCES

# Quaternion convention throughout robot_engine APIs: [x, y, z, w].


def normalize_quaternion(q):
    q = np.asarray(q, dtype=float).reshape(-1)
    if q.shape != (4,) or not np.isfinite(q).all():
        raise ValueError("quaternion must be finite [x, y, z, w]")
    norm = float(np.linalg.norm(q))
    if norm <= DEFAULT_TOLERANCES.numerical:
        raise ValueError("quaternion norm must be non-zero")
    return q / norm


def rotation_matrix_from_quaternion(q):
    return R.from_quat(normalize_quaternion(q)).as_matrix()


def quaternion_from_rotation_matrix(rot):
    validate_rotation_matrix(rot)
    return R.from_matrix(rot).as_quat()


def validate_rotation_matrix(rot):
    rot = np.asarray(rot, dtype=float)
    if rot.shape != (3, 3):
        raise ValueError("rotation matrix must be 3x3")
    if not np.isfinite(rot).all():
        raise ValueError("rotation matrix must be finite")
    if not np.allclose(rot.T @ rot, np.eye(3), atol=DEFAULT_TOLERANCES.rotation_orthonormal):
        raise ValueError("rotation matrix must be orthonormal")
    det = float(np.linalg.det(rot))
    if not np.isclose(det, 1.0, atol=DEFAULT_TOLERANCES.rotation_determinant):
        raise ValueError("rotation matrix determinant must be +1")
    return True


def is_valid_rotation_matrix(rot):
    try:
        return bool(validate_rotation_matrix(rot))
    except Exception:
        return False


def slerp_quaternion(q0, q1, alpha):
    q0 = normalize_quaternion(q0)
    q1 = normalize_quaternion(q1)
    alpha = float(alpha)
    if alpha <= 0.0:
        return q0.copy()
    if alpha >= 1.0:
        return q1.copy()
    slerp = Slerp([0.0, 1.0], R.from_quat([q0, q1]))
    return slerp([alpha]).as_quat()[0]


def rotation_exp(w):
    w = np.asarray(w, dtype=float).reshape(3)
    return R.from_rotvec(w).as_matrix()


def rotation_log(rot):
    validate_rotation_matrix(rot)
    return R.from_matrix(rot).as_rotvec()


def orientation_error(current, target):
    validate_rotation_matrix(current)
    validate_rotation_matrix(target)
    return rotation_log(np.asarray(target) @ np.asarray(current).T)


def angular_distance(current, target):
    return float(np.linalg.norm(orientation_error(current, target)))


def euler_to_rotation_matrix(euler, convention="xyz"):
    return R.from_euler(convention, euler).as_matrix()


def rotation_matrix_to_euler(rot, convention="xyz"):
    validate_rotation_matrix(rot)
    return R.from_matrix(rot).as_euler(convention)
