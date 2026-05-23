from __future__ import annotations

import numpy as np


def finite_difference_jacobian(function, x, eps=1e-6):
    x = np.asarray(x, dtype=float)
    f0 = np.asarray(function(x), dtype=float).reshape(-1)
    J = np.zeros((f0.size, x.size))
    for i in range(x.size):
        dx = np.zeros_like(x)
        dx[i] = eps
        J[:, i] = (np.asarray(function(x + dx)).reshape(-1) - f0) / eps
    return J


def central_difference_jacobian(function, x, eps=1e-6):
    x = np.asarray(x, dtype=float)
    f0 = np.asarray(function(x), dtype=float).reshape(-1)
    J = np.zeros((f0.size, x.size))
    for i in range(x.size):
        dx = np.zeros_like(x)
        dx[i] = eps
        J[:, i] = (np.asarray(function(x + dx)).reshape(-1) - np.asarray(function(x - dx)).reshape(-1)) / (2 * eps)
    return J


def check_jacobian_analytic_vs_numeric(analytic_J, numeric_J, tolerance=1e-6):
    diff = np.asarray(analytic_J, dtype=float) - np.asarray(numeric_J, dtype=float)
    return {"ok": bool(np.linalg.norm(diff) <= tolerance), "error_norm": float(np.linalg.norm(diff)), "max_abs_error": float(np.max(np.abs(diff)))}


def finite_difference_gradient(function, x, eps=1e-6):
    return finite_difference_jacobian(lambda z: [function(z)], x, eps).reshape(-1)


def finite_difference_hessian(function, x, eps=1e-5):
    return central_difference_jacobian(lambda z: finite_difference_gradient(function, z, eps), x, eps)
