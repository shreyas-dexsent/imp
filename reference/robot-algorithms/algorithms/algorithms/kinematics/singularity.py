# Copyright (c) 2026 DexSent Robotics Pvt Ltd.
# All Rights Reserved.
#
# SPDX-License-Identifier: LicenseRef-Proprietary
# See LICENSE file for details.
#
# Author: Shreyas <shreyas.kumar@dexsentrobotics.com>
# Proprietary and confidential.

"""Singularity metrics derived from a frame Jacobian.

All functions in this module take a Jacobian matrix `J` of shape
`(6, dof)` and return a scalar. `singularity_report` returns all common
metrics from one shared SVD, so callers that need several values should
prefer it over calling individual helpers repeatedly.

Use these to:

* annotate IK solutions with a singularity-proximity score
  (planning/IK can prefer well-conditioned solutions);
* annotate a planned path with per-waypoint singularity metrics
  (post-plan validators flag paths that cross singularities).
"""
from __future__ import annotations

from typing import Dict

import numpy as np


def _singular_values(jacobian: np.ndarray) -> np.ndarray:
    """Return the singular values of `jacobian` (descending order)."""
    J = np.asarray(jacobian, dtype=float)
    if J.size == 0:
        return np.array([], dtype=float)
    return np.linalg.svd(J, compute_uv=False)


def manipulability(jacobian: np.ndarray) -> float:
    """Yoshikawa's manipulability index.

    Defined as `sqrt(det(J J^T))` = product of singular values.
    Geometrically the volume of the manipulability ellipsoid. Approaches
    zero at a singularity and grows with kinematic dexterity. Values are
    typically compared relatively (across configurations) rather than to
    an absolute threshold.
    """
    sigma = _singular_values(jacobian)
    if sigma.size == 0:
        return 0.0
    return float(np.prod(sigma))


def condition_number(jacobian: np.ndarray) -> float:
    """Ratio of largest to smallest singular value of `J`.

    Diverges to `inf` at a singularity. Useful as a scalar with a
    clear threshold semantics (e.g. `cond > 50` flags a marginal pose).
    """
    sigma = _singular_values(jacobian)
    if sigma.size == 0:
        return float("inf")
    smallest = float(np.min(sigma))
    if smallest <= 1e-12:
        return float("inf")
    return float(np.max(sigma) / smallest)


def inverse_condition_number(jacobian: np.ndarray) -> float:
    """Numerically stable inverse of :func:`condition_number`.

    Returns `sigma_min / sigma_max` in `[0, 1]`; zero at a singularity,
    one for an isotropic configuration. Useful as a smooth gradient signal
    in optimization-based IK.
    """
    sigma = _singular_values(jacobian)
    if sigma.size == 0:
        return 0.0
    largest = float(np.max(sigma))
    if largest <= 1e-12:
        return 0.0
    return float(np.min(sigma) / largest)


def min_singular_value(jacobian: np.ndarray) -> float:
    """Smallest singular value of `J`.

    Direct proxy for distance-to-singularity. Zero exactly at a singularity.
    """
    sigma = _singular_values(jacobian)
    return float(np.min(sigma)) if sigma.size else 0.0


def singularity_report(jacobian: np.ndarray) -> Dict[str, float]:
    """Compute all metrics plus numerical rank from a single SVD.

    Returns a dict with keys `manipulability`, `condition_number`,
    `inverse_condition_number`, `min_singular_value`,
    `max_singular_value`, and `rank`. Use this when several metrics
    are needed at once; the SVD is shared.
    """
    sigma = _singular_values(jacobian)
    if sigma.size == 0:
        return {
            "manipulability": 0.0,
            "condition_number": float("inf"),
            "inverse_condition_number": 0.0,
            "min_singular_value": 0.0,
            "max_singular_value": 0.0,
            "rank": 0,
        }

    smallest = float(np.min(sigma))
    largest = float(np.max(sigma))
    rank = int(np.sum(sigma > 1e-6 * largest)) if largest > 0 else 0

    return {
        "manipulability": float(np.prod(sigma)),
        "condition_number": (
            float("inf") if smallest <= 1e-12 else float(largest / smallest)
        ),
        "inverse_condition_number": (
            0.0 if largest <= 1e-12 else float(smallest / largest)
        ),
        "min_singular_value": smallest,
        "max_singular_value": largest,
        "rank": rank,
    }
