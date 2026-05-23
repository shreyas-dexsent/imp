from __future__ import annotations

import numpy as np

from robot_engine.core_math.rotations import angular_distance as _angular_distance
from robot_engine.core_math.interpolation import compute_path_length_cartesian, compute_path_length_joint


def euclidean_distance(a, b):
    return float(np.linalg.norm(np.asarray(a, dtype=float) - np.asarray(b, dtype=float)))


def angular_distance(R_a, R_b):
    return _angular_distance(R_a, R_b)


def pose_distance(T_a, T_b, position_weight=1.0, orientation_weight=1.0):
    return float(position_weight * euclidean_distance(np.asarray(T_a)[:3, 3], np.asarray(T_b)[:3, 3]) + orientation_weight * angular_distance(np.asarray(T_a)[:3, :3], np.asarray(T_b)[:3, :3]))


def joint_distance(q_a, q_b, weights=None):
    diff = np.asarray(q_a, dtype=float) - np.asarray(q_b, dtype=float)
    if weights is not None:
        diff = diff * np.asarray(weights, dtype=float)
    return float(np.linalg.norm(diff))


def path_length_joint(q_waypoints):
    return compute_path_length_joint(q_waypoints)


def path_length_cartesian(T_waypoints):
    return compute_path_length_cartesian(T_waypoints)


def max_joint_delta(q0, q1):
    return float(np.max(np.abs(np.asarray(q1, dtype=float) - np.asarray(q0, dtype=float))))


def rms_error(errors):
    errors = np.asarray(errors, dtype=float)
    return float(np.sqrt(np.mean(errors**2)))
