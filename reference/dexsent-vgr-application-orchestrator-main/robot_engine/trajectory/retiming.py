from __future__ import annotations

import numpy as np

from robot_engine.trajectory.time_scaling import time_scale_path_trapezoidal
from robot_engine.trajectory.trajectory_base import JointTrajectory, JointTrajectoryPoint


def retime_joint_path(q_waypoints, velocity_limits, acceleration_limits, method: str = "trapezoidal"):
    q = np.asarray(q_waypoints, dtype=float)
    durations = time_scale_path_trapezoidal(q, velocity_limits, acceleration_limits)
    points = [JointTrajectoryPoint(0.0, q[0].tolist(), np.zeros(q.shape[1]).tolist(), np.zeros(q.shape[1]).tolist())]
    t = 0.0
    for dt, qa, qb in zip(durations, q[:-1], q[1:]):
        t += dt
        vel = (qb - qa) / max(dt, 1e-9)
        points.append(JointTrajectoryPoint(t, qb.tolist(), vel.tolist(), np.zeros_like(vel).tolist()))
    return JointTrajectory(points, generation_method=method)


def enforce_velocity_limits(trajectory):
    return trajectory


def enforce_acceleration_limits(trajectory):
    return trajectory


def enforce_jerk_limits(trajectory):
    return trajectory


def synchronize_by_slowest_joint(q_waypoints, velocity_limits, acceleration_limits):
    return retime_joint_path(q_waypoints, velocity_limits, acceleration_limits)


def validate_retiming_result(trajectory):
    return bool(trajectory.points and all(b.time >= a.time for a, b in zip(trajectory.points[:-1], trajectory.points[1:])))

