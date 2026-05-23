from __future__ import annotations

from typing import Iterable

import numpy as np

from robot_engine.core_math.optimization import nullspace_projector


def select_minimum_joint_motion_solution(solutions: Iterable, q_current, weights=None):
    q0 = np.asarray(q_current, dtype=float)
    W = np.ones_like(q0) if weights is None else np.asarray(weights, dtype=float)
    best = None
    best_cost = float("inf")
    for solution in solutions:
        q = np.asarray(solution, dtype=float)
        cost = float(np.linalg.norm((q - q0) * W))
        if cost < best_cost:
            best = q
            best_cost = cost
    return best


def joint_limit_avoidance_gradient(q, lower, upper):
    q = np.asarray(q, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    mid = 0.5 * (lower + upper)
    span = np.maximum(upper - lower, 1e-9)
    return -2.0 * (q - mid) / (span * span)


def preferred_posture_cost(q, q_nominal, weights=None):
    q = np.asarray(q, dtype=float)
    q_nominal = np.asarray(q_nominal, dtype=float)
    W = np.ones_like(q) if weights is None else np.asarray(weights, dtype=float)
    return float(0.5 * np.sum(W * (q - q_nominal) ** 2))


def manipulability_gradient_numeric(q, jacobian_fn, eps: float = 1e-5):
    from robot_engine.kinematics.singularity import manipulability_index

    q = np.asarray(q, dtype=float)
    grad = np.zeros_like(q)
    for i in range(q.size):
        qp = q.copy()
        qm = q.copy()
        qp[i] += eps
        qm[i] -= eps
        grad[i] = (manipulability_index(jacobian_fn(qp)) - manipulability_index(jacobian_fn(qm))) / (2.0 * eps)
    return grad


def nullspace_secondary_objective(primary_update, J, secondary_gradient):
    return np.asarray(primary_update, dtype=float) + nullspace_projector(J) @ np.asarray(secondary_gradient, dtype=float)

