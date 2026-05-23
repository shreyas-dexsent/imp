from __future__ import annotations

import numpy as np

from robot_engine.trajectory.polynomial import evaluate_polynomial, evaluate_polynomial_derivative, solve_cubic_coefficients
from robot_engine.trajectory.trajectory_base import JointTrajectory, JointTrajectoryPoint


class CubicTrajectory(JointTrajectory):
    pass


def cubic_joint_trajectory(q0, q1, v0=0.0, v1=0.0, duration: float = 1.0, samples: int = 101):
    return multi_joint_cubic_trajectory([q0], [q1], [v0], [v1], duration, samples)


def multi_joint_cubic_trajectory(q0, q1, v0, v1, duration: float, samples: int = 101):
    coeffs = solve_cubic_coefficients(q0, q1, v0, v1, duration)
    points = []
    for t in np.linspace(0.0, duration, samples):
        points.append(JointTrajectoryPoint(float(t), evaluate_polynomial(coeffs, t).tolist(), evaluate_polynomial_derivative(coeffs, t, 1).tolist(), evaluate_polynomial_derivative(coeffs, t, 2).tolist()))
    return CubicTrajectory(points, generation_method="cubic")

