# Copyright (c) 2026 DexSent Robotics Pvt Ltd.
# All Rights Reserved.
#
# SPDX-License-Identifier: LicenseRef-Proprietary
# See LICENSE file for details.
#
# Author: Shreyas <shreyas.kumar@dexsentrobotics.com>
# Proprietary and confidential.

"""Polynomial spline fit through a path's waypoints.

Geometric pass. Input: Path. Output: Path (with more, smoother
waypoints). No timing is assigned — that's the trajectory layer's job.

The fit uses chord-length parameterisation to assign a notional
"distance" coordinate ``s`` along the path, computes interior
velocities at each waypoint via Catmull-Rom finite differences (so
the curve does not have to stop at every via point), and solves a
piecewise polynomial per segment.

Cubic (4 coefficients) gives C^1 continuity. Quintic (6 coefficients)
gives C^2 continuity with zero acceleration at the start and goal.
"""
from __future__ import annotations

from typing import Literal

import numpy as np

from algorithms.planning.path import Path


def spline_fit(
    path: Path,
    *,
    order: Literal["cubic", "quintic"] = "quintic",
    samples: int = 200,
) -> Path:
    """Fit a smooth polynomial spline through ``path``'s waypoints.

    Returns a new Path with ``samples`` waypoints, sampled uniformly in
    chord-length s. Geometry only; no velocities or timestamps are
    attached to the returned waypoints — those are the trajectory
    layer's concern.

    Parameters
    ----------
    path
        Input path; must have at least 2 waypoints.
    order
        ``"quintic"`` (C^2 by construction, zero accel at endpoints) or
        ``"cubic"`` (C^1, zero velocity at endpoints).
    samples
        Total number of waypoints in the returned path.
    """
    if path.num_waypoints < 2:
        raise ValueError("spline_fit requires a path with at least 2 waypoints")
    if order not in ("cubic", "quintic"):
        raise ValueError(f"unsupported order: {order!r}; choose 'cubic' or 'quintic'")
    if samples < 2:
        raise ValueError(f"samples must be >= 2; got {samples}")

    qs = path.waypoints
    n = qs.shape[1]

    # Special case: two waypoints. The spline collapses to a single
    # segment; sample it directly.
    if path.num_waypoints == 2:
        seg_durations = [1.0]
        cum = np.array([0.0, 1.0])
    else:
        seg_durations = _chord_length_segments(qs)
        cum = np.concatenate([[0.0], np.cumsum(seg_durations)])

    # Interior velocities via Catmull-Rom finite differences. Endpoints
    # at rest (v=0).
    vels = _catmull_rom_velocities(qs, seg_durations)

    # For quintic, accelerations are zero at start and goal.
    zeros = np.zeros(n, dtype=float)

    # Sample s uniformly along the parameterised path.
    total_s = float(cum[-1])
    s_samples = np.linspace(0.0, total_s, samples)

    out_waypoints = np.zeros((samples, n), dtype=float)
    for k, s in enumerate(s_samples):
        seg_idx = _find_segment(cum, s)
        s_local = s - cum[seg_idx]
        T = seg_durations[seg_idx]
        q_local = _evaluate_segment(
            qs[seg_idx], qs[seg_idx + 1],
            vels[seg_idx], vels[seg_idx + 1],
            zeros, zeros,
            T, s_local, order,
        )
        out_waypoints[k] = q_local

    # Pin the endpoints exactly to avoid numerical drift.
    out_waypoints[0] = qs[0]
    out_waypoints[-1] = qs[-1]

    return Path(
        waypoints=out_waypoints,
        joint_names=path.joint_names,
        cartesian_waypoints=None,
        metadata={**path.metadata, "smoothed_by": f"spline_{order}"},
    )


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _chord_length_segments(qs: np.ndarray) -> list[float]:
    """Per-segment 'distance' coordinate. Avoids zero-length segments."""
    durations = []
    for i in range(qs.shape[0] - 1):
        d = float(np.linalg.norm(qs[i + 1] - qs[i]))
        durations.append(max(d, 1e-9))
    return durations


def _catmull_rom_velocities(
    qs: np.ndarray,
    seg_durations: list[float],
) -> np.ndarray:
    """Velocity at each waypoint via centered finite differences.

    v[i] = (q[i+1] - q[i-1]) / (s[i+1] - s[i-1])

    Endpoints set to zero so the path starts and ends at rest.
    """
    n_pts = qs.shape[0]
    vels = np.zeros_like(qs)
    cum = [0.0]
    for d in seg_durations:
        cum.append(cum[-1] + d)
    for i in range(1, n_pts - 1):
        denom = max(cum[i + 1] - cum[i - 1], 1e-9)
        vels[i] = (qs[i + 1] - qs[i - 1]) / denom
    return vels


def _find_segment(cum: np.ndarray, s: float) -> int:
    """Return the index i such that cum[i] <= s < cum[i+1]. Clamps."""
    if s <= cum[0]:
        return 0
    if s >= cum[-1]:
        return len(cum) - 2
    return int(np.searchsorted(cum, s, side="right") - 1)


def _evaluate_segment(q0, q1, v0, v1, a0, a1, T, s_local, order) -> np.ndarray:
    """Evaluate a single cubic or quintic segment at local parameter s_local."""
    if order == "cubic":
        coeffs = _solve_cubic(q0, q1, v0, v1, T)
    else:
        coeffs = _solve_quintic(q0, q1, v0, v1, a0, a1, T)
    return _eval_poly(coeffs, s_local)


def _solve_cubic(q0, q1, v0, v1, T) -> np.ndarray:
    """Closed-form cubic boundary-value solution. Returns (4, dof)."""
    T2, T3 = T * T, T * T * T
    a0 = q0
    a1 = v0
    a2 = 3.0 * (q1 - q0) / T2 - (2.0 * v0 + v1) / T
    a3 = -2.0 * (q1 - q0) / T3 + (v0 + v1) / T2
    return np.stack([a0, a1, a2, a3])


def _solve_quintic(q0, q1, v0, v1, a0_acc, a1_acc, T) -> np.ndarray:
    """Closed-form quintic boundary-value solution. Returns (6, dof).

    c0..c2 directly: q0, v0, a0/2. Remaining three from a 3x3 solve.
    """
    c0 = q0
    c1 = v0
    c2 = 0.5 * a0_acc
    T2, T3, T4, T5 = T * T, T * T * T, T**4, T**5
    A = np.array([
        [T3, T4, T5],
        [3 * T2, 4 * T3, 5 * T4],
        [6 * T, 12 * T2, 20 * T3],
    ])
    rhs = np.stack([
        q1 - (c0 + c1 * T + c2 * T2),
        v1 - (c1 + 2 * c2 * T),
        a1_acc - 2 * c2,
    ])
    c345 = np.linalg.solve(A, rhs)
    c3, c4, c5 = c345[0], c345[1], c345[2]
    return np.stack([c0, c1, c2, c3, c4, c5])


def _eval_poly(coeffs: np.ndarray, t: float) -> np.ndarray:
    """Horner-form polynomial evaluation. coeffs shape (order+1, dof)."""
    result = np.zeros_like(coeffs[0])
    t_pow = 1.0
    for c in coeffs:
        result = result + c * t_pow
        t_pow *= t
    return result
