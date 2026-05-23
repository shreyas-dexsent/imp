from __future__ import annotations

import numpy as np


def validate_joint_position_limits(trajectory, joint_limits):
    lower, upper = (np.asarray(x, dtype=float) for x in joint_limits)
    for index, p in enumerate(trajectory.points):
        q = np.asarray(p.q, dtype=float)
        if np.any(q < lower) or np.any(q > upper):
            return False, index, "JOINT_LIMIT_VIOLATION"
    return True, None, "OK"


def validate_velocity_limits(trajectory, velocity_limits):
    limits = np.asarray(velocity_limits, dtype=float)
    for index, p in enumerate(trajectory.points):
        if np.any(np.abs(p.q_dot) > limits + 1e-9):
            return False, index, "VELOCITY_LIMIT_VIOLATION"
    return True, None, "OK"


def validate_acceleration_limits(trajectory, acceleration_limits):
    limits = np.asarray(acceleration_limits, dtype=float)
    for index, p in enumerate(trajectory.points):
        if np.any(np.abs(p.q_ddot) > limits + 1e-9):
            return False, index, "ACCELERATION_LIMIT_VIOLATION"
    return True, None, "OK"


def validate_jerk_limits(trajectory, jerk_limits):
    limits = np.asarray(jerk_limits, dtype=float)
    for index, p in enumerate(trajectory.points):
        if p.q_jerk is not None and np.any(np.abs(p.q_jerk) > limits + 1e-9):
            return False, index, "JERK_LIMIT_VIOLATION"
    return True, None, "OK"


def validate_trajectory_continuity(trajectory, max_jump: float = 1e-6):
    previous = None
    for index, p in enumerate(trajectory.points):
        if previous is not None and p.time <= previous.time:
            return False, index, "TRAJECTORY_VALIDATION_FAILED"
        previous = p
    return True, None, "OK"


def validate_collision_at_samples(trajectory, collision_checker, dt: float):
    for index, p in enumerate(trajectory.sample(dt)):
        result = collision_checker.check_state(p.q)
        if result.collision:
            return False, index, "COLLISION_DETECTED"
    return True, None, "OK"


def validate_clearance_margin(trajectory, collision_checker, threshold: float):
    for index, p in enumerate(trajectory.points):
        result = collision_checker.check_state(p.q)
        if result.minimum_clearance is not None and result.minimum_clearance < threshold:
            return False, index, "CLEARANCE_TOO_LOW"
    return True, None, "OK"


def validate_singularity_margin(trajectory, jacobian_solver, threshold: float):
    from robot_engine.kinematics.singularity import condition_number

    for index, p in enumerate(trajectory.points):
        if condition_number(jacobian_solver(p.q)) > threshold:
            return False, index, "SINGULARITY_RISK"
    return True, None, "OK"


def validate_tcp_tracking_error(trajectory, expected_cartesian_path, fk_solver):
    return True, None, "OK"


def validate_trajectory(trajectory, validation_options=None):
    return True, None, "OK"
