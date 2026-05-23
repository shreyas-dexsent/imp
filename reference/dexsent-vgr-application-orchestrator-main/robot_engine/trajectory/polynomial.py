from __future__ import annotations

import numpy as np


def solve_cubic_coefficients(q0, q1, v0, v1, T: float):
    q0, q1, v0, v1 = (np.asarray(x, dtype=float) for x in (q0, q1, v0, v1))
    if T <= 0:
        raise ValueError("duration must be positive")
    a0 = q0
    a1 = v0
    a2 = (3 * (q1 - q0) / T**2) - (2 * v0 + v1) / T
    a3 = (-2 * (q1 - q0) / T**3) + (v0 + v1) / T**2
    return np.stack([a0, a1, a2, a3], axis=0)


def solve_quintic_coefficients(q0, q1, v0, v1, a0, a1, T: float):
    q0, q1, v0, v1, a0, a1 = (np.asarray(x, dtype=float) for x in (q0, q1, v0, v1, a0, a1))
    if T <= 0:
        raise ValueError("duration must be positive")
    c0 = q0
    c1 = v0
    c2 = a0 / 2.0
    A = np.array([[T**3, T**4, T**5], [3*T**2, 4*T**3, 5*T**4], [6*T, 12*T**2, 20*T**3]], dtype=float)
    b = np.vstack([q1 - (c0 + c1*T + c2*T**2), v1 - (c1 + 2*c2*T), a1 - 2*c2])
    rest = np.linalg.solve(A, b).T
    return np.vstack([c0, c1, c2, rest[:, 0], rest[:, 1], rest[:, 2]])


def evaluate_polynomial(coeffs, t: float):
    coeffs = np.asarray(coeffs, dtype=float)
    powers = np.asarray([t**i for i in range(coeffs.shape[0])])
    return powers @ coeffs


def evaluate_polynomial_derivative(coeffs, t: float, derivative_order: int):
    coeffs = np.asarray(coeffs, dtype=float).copy()
    for _ in range(derivative_order):
        coeffs = np.asarray([i * coeffs[i] for i in range(1, coeffs.shape[0])])
        if coeffs.size == 0:
            return np.zeros(coeffs.shape[1] if coeffs.ndim == 2 else 1)
    return evaluate_polynomial(coeffs, t)

