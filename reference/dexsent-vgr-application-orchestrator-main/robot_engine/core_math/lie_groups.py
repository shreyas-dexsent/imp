from __future__ import annotations

import numpy as np

from robot_engine.core_math.rotations import rotation_exp, rotation_log
from robot_engine.core_math.transforms import adjoint_SE3, validate_transform

# Twist convention: [vx, vy, vz, wx, wy, wz].


def skew(v):
    v = np.asarray(v, dtype=float).reshape(3)
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]], dtype=float)


def unskew(S):
    S = np.asarray(S, dtype=float)
    return np.array([S[2, 1], S[0, 2], S[1, 0]], dtype=float)


def so3_exp(w):
    return rotation_exp(w)


def so3_log(R):
    return rotation_log(R)


def left_jacobian_SO3(w):
    w = np.asarray(w, dtype=float).reshape(3)
    theta = float(np.linalg.norm(w))
    W = skew(w)
    if theta < 1e-9:
        return np.eye(3) + 0.5 * W + (1.0 / 6.0) * W @ W
    return np.eye(3) + (1 - np.cos(theta)) / theta**2 * W + (theta - np.sin(theta)) / theta**3 * W @ W


def inverse_left_jacobian_SO3(w):
    w = np.asarray(w, dtype=float).reshape(3)
    theta = float(np.linalg.norm(w))
    W = skew(w)
    if theta < 1e-9:
        return np.eye(3) - 0.5 * W + (1.0 / 12.0) * W @ W
    return np.eye(3) - 0.5 * W + (1 / theta**2 - (1 + np.cos(theta)) / (2 * theta * np.sin(theta))) * W @ W


def se3_exp(xi):
    xi = np.asarray(xi, dtype=float).reshape(6)
    v = xi[:3]
    w = xi[3:]
    T = np.eye(4)
    T[:3, :3] = so3_exp(w)
    T[:3, 3] = left_jacobian_SO3(w) @ v
    return T


def se3_log(T):
    T = np.asarray(T, dtype=float)
    validate_transform(T)
    w = so3_log(T[:3, :3])
    v = inverse_left_jacobian_SO3(w) @ T[:3, 3]
    return np.r_[v, w]


def twist_to_matrix(xi):
    xi = np.asarray(xi, dtype=float).reshape(6)
    X = np.zeros((4, 4), dtype=float)
    X[:3, :3] = skew(xi[3:])
    X[:3, 3] = xi[:3]
    return X


def matrix_to_twist(X):
    X = np.asarray(X, dtype=float)
    return np.r_[X[:3, 3], unskew(X[:3, :3])]


def body_twist_error(T_current, T_target):
    return se3_log(np.linalg.inv(T_current) @ T_target)


def spatial_twist_error(T_current, T_target):
    return se3_log(T_target @ np.linalg.inv(T_current))


def adjoint_from_transform(T):
    return adjoint_SE3(T)
