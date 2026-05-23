from __future__ import annotations

import numpy as np

from robot_engine.core_math.lie_groups import se3_exp, se3_log
from robot_engine.core_math.rotations import quaternion_from_rotation_matrix, rotation_matrix_from_quaternion, slerp_quaternion
from robot_engine.core_math.transforms import matrix_to_pose, pose_to_matrix, validate_transform


def interpolate_joint(q0, q1, alpha):
    q0 = np.asarray(q0, dtype=float)
    q1 = np.asarray(q1, dtype=float)
    return q0 + (q1 - q0) * float(alpha)


def interpolate_position(p0, p1, alpha):
    return interpolate_joint(p0, p1, alpha)


def interpolate_quaternion(q0, q1, alpha):
    return slerp_quaternion(q0, q1, alpha)


def interpolate_pose_SE3(T0, T1, alpha):
    validate_transform(T0)
    validate_transform(T1)
    if alpha <= 0:
        return np.asarray(T0, dtype=float).copy()
    if alpha >= 1:
        return np.asarray(T1, dtype=float).copy()
    return np.asarray(T0) @ se3_exp(se3_log(np.linalg.inv(T0) @ T1) * float(alpha))


def interpolate_pose_position_slerp(T0, T1, alpha):
    if alpha <= 0:
        return np.asarray(T0, dtype=float).copy()
    if alpha >= 1:
        return np.asarray(T1, dtype=float).copy()
    p0, q0 = matrix_to_pose(T0)
    p1, q1 = matrix_to_pose(T1)
    return pose_to_matrix(interpolate_position(p0, p1, alpha), interpolate_quaternion(q0, q1, alpha))


def sample_joint_path(q0, q1, max_joint_step):
    if max_joint_step <= 0:
        raise ValueError("max_joint_step must be positive")
    q0 = np.asarray(q0, dtype=float)
    q1 = np.asarray(q1, dtype=float)
    steps = max(2, int(np.ceil(np.max(np.abs(q1 - q0)) / max_joint_step)) + 1)
    return [interpolate_joint(q0, q1, a) for a in np.linspace(0, 1, steps)]


def sample_cartesian_path(T0, T1, translation_step, rotation_step):
    if translation_step <= 0 or rotation_step <= 0:
        raise ValueError("step sizes must be positive")
    dist = float(np.linalg.norm(np.asarray(T1)[:3, 3] - np.asarray(T0)[:3, 3]))
    rot_dist = float(np.linalg.norm(se3_log(np.linalg.inv(T0) @ T1)[3:]))
    steps = max(2, int(np.ceil(max(dist / translation_step, rot_dist / rotation_step))) + 1)
    return [interpolate_pose_position_slerp(T0, T1, a) for a in np.linspace(0, 1, steps)]


def compute_path_length_joint(q_waypoints):
    return sum(float(np.linalg.norm(np.asarray(b) - np.asarray(a))) for a, b in zip(q_waypoints[:-1], q_waypoints[1:]))


def compute_path_length_cartesian(T_waypoints):
    return sum(float(np.linalg.norm(np.asarray(b)[:3, 3] - np.asarray(a)[:3, 3])) for a, b in zip(T_waypoints[:-1], T_waypoints[1:]))
