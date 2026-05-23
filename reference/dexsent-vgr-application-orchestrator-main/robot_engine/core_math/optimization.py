from __future__ import annotations

import numpy as np


def pseudoinverse(J):
    return np.linalg.pinv(np.asarray(J, dtype=float))


def damped_pseudoinverse(J, damping=1e-3):
    J = np.asarray(J, dtype=float)
    return J.T @ np.linalg.inv(J @ J.T + float(damping) ** 2 * np.eye(J.shape[0]))


def weighted_damped_pseudoinverse(J, W, damping=1e-3):
    J = np.asarray(J, dtype=float)
    W = np.asarray(W, dtype=float)
    Winv = np.linalg.pinv(W)
    return Winv @ J.T @ np.linalg.pinv(J @ Winv @ J.T + float(damping) ** 2 * np.eye(J.shape[0]))


def nullspace_projector(J):
    J = np.asarray(J, dtype=float)
    return np.eye(J.shape[1]) - pseudoinverse(J) @ J


def weighted_nullspace_projector(J, W):
    Jp = weighted_damped_pseudoinverse(J, W, 1e-9)
    return np.eye(np.asarray(J).shape[1]) - Jp @ J


def weighted_least_squares(A, b, W=None):
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    if W is None:
        return np.linalg.lstsq(A, b, rcond=None)[0]
    W = np.asarray(W, dtype=float)
    return np.linalg.lstsq(W @ A, W @ b, rcond=None)[0]


def clamp_to_joint_limits(q, lower, upper):
    return np.minimum(np.asarray(upper, dtype=float), np.maximum(np.asarray(lower, dtype=float), np.asarray(q, dtype=float)))


def joint_limit_margin(q, lower, upper):
    q = np.asarray(q, dtype=float)
    return np.minimum(q - np.asarray(lower, dtype=float), np.asarray(upper, dtype=float) - q)


def residual_norm(r, weights=None):
    r = np.asarray(r, dtype=float)
    if weights is not None:
        r = np.asarray(weights, dtype=float) * r
    return float(np.linalg.norm(r))
