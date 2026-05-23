from __future__ import annotations

import numpy as np

from robot_engine.trajectory.polynomial import evaluate_polynomial, evaluate_polynomial_derivative, solve_quintic_coefficients
from robot_engine.trajectory.trajectory_base import JointTrajectory, JointTrajectoryPoint


class QuinticTrajectory(JointTrajectory):
    pass


def quintic_joint_trajectory(q0, q1, v0=0.0, v1=0.0, a0=0.0, a1=0.0, duration: float = 1.0, samples: int = 101):
    return multi_joint_quintic_trajectory([q0], [q1], [v0], [v1], [a0], [a1], duration, samples)


def multi_joint_quintic_trajectory(q0, q1, v0, v1, a0, a1, duration: float, samples: int = 101):
    coeffs = solve_quintic_coefficients(q0, q1, v0, v1, a0, a1, duration)
    points = []
    for t in np.linspace(0.0, duration, samples):
        points.append(JointTrajectoryPoint(float(t), evaluate_polynomial(coeffs, t).tolist(), evaluate_polynomial_derivative(coeffs, t, 1).tolist(), evaluate_polynomial_derivative(coeffs, t, 2).tolist()))
    return QuinticTrajectory(points, generation_method="quintic")


def quintic_segment_interpolation(q_waypoints, boundary_velocities=None, boundary_accelerations=None, duration_per_segment: float = 1.0):
    trajectories = []
    for q0, q1 in zip(q_waypoints[:-1], q_waypoints[1:]):
        zeros = np.zeros_like(np.asarray(q0, dtype=float))
        trajectories.append(multi_joint_quintic_trajectory(q0, q1, zeros, zeros, zeros, zeros, duration_per_segment).points)
    points = []
    time_offset = 0.0
    for segment in trajectories:
        for point in segment[1 if points else 0 :]:
            points.append(JointTrajectoryPoint(point.time + time_offset, point.q, point.q_dot, point.q_ddot))
        time_offset = points[-1].time
    return QuinticTrajectory(points, generation_method="quintic_segments")

