from __future__ import annotations

import numpy as np


def _singular_values(jacobian) -> np.ndarray:
    J = np.asarray(jacobian, dtype=float)
    if J.size == 0:
        return np.array([], dtype=float)
    return np.linalg.svd(J, compute_uv=False)


def manipulability_index(jacobian) -> float:
    values = _singular_values(jacobian)
    return float(np.prod(values)) if values.size else 0.0


def condition_number(jacobian) -> float:
    values = _singular_values(jacobian)
    if values.size == 0:
        return float("inf")
    smallest = float(np.min(values))
    return float("inf") if smallest <= 1e-12 else float(np.max(values) / smallest)


def minimum_singular_value(jacobian) -> float:
    values = _singular_values(jacobian)
    return float(np.min(values)) if values.size else 0.0


def is_near_singularity(jacobian, threshold: float) -> bool:
    return condition_number(jacobian) >= threshold


def singularity_report(jacobian) -> dict:
    return {
        "manipulability": manipulability_index(jacobian),
        "condition_number": condition_number(jacobian),
        "minimum_singular_value": minimum_singular_value(jacobian),
    }

