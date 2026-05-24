# Copyright (c) 2026 DexSent Robotics Pvt Ltd.
# All Rights Reserved.
#
# SPDX-License-Identifier: LicenseRef-Proprietary
# See LICENSE file for details.
#
# Author: Shreyas <shreyas.kumar@dexsentrobotics.com>
# Proprietary and confidential.

"""Pure-Python polynomial trajectory backend.

Fits ONE continuous piecewise polynomial through the path's
waypoints, with C^2 continuity at every interior knot by
construction. Uses ``scipy.interpolate.CubicSpline`` with clamped
boundary conditions (v=0 at start and end). For a two-waypoint path
this collapses to a single cubic; for a many-waypoint path the
interior derivatives are computed from the geometry, not pinned to
zero.

Why this design: an earlier per-segment quintic with forced
acceleration=0 at every interior knot produced visible wiggles on
real hardware. Each interior knot became a mini "ramp acceleration
up, then back to zero" event, and a dense path (33 collision-validation
samples for a 90 deg move, ~20 samples for a 10 cm Cartesian line)
turned into 20-30 wiggles in position-vs-time. The single-spline
approach removes those artifacts entirely: interior accelerations
are derived, not forced.

Velocity and acceleration envelopes are enforced by chord-length
based timing. If the chosen timing makes the spline exceed v/a
limits, the backend reports ``LIMITS_INFEASIBLE`` so the caller
knows to relax limits or scale down v_scale / a_scale.
"""
from __future__ import annotations

import numpy as np
from scipy.interpolate import CubicSpline

from algorithms.planning.path import Path
from algorithms.trajectory.backends.base import (
    RawTrajectoryResult,
    TimeParameterizationBackend,
)
from algorithms.trajectory.options import TimeParameterizationOptions
from algorithms.trajectory.result import TrajectoryStatus


class PolynomialBackend:
    """Pure-Python polynomial backend. Always available."""

    name: str = "polynomial"

    def parameterize(
        self,
        path: Path,
        v_limits: np.ndarray,
        a_limits: np.ndarray,
        j_limits: np.ndarray | None,  # unused; cubic spline doesn't model jerk
        options: TimeParameterizationOptions,
    ) -> RawTrajectoryResult:
        try:
            return self._do_parameterize(path, v_limits, a_limits, options)
        except Exception as exc:  # pragma: no cover - safety net
            return RawTrajectoryResult(
                status=TrajectoryStatus.NUMERICAL_FAILURE,
                times=None, positions=None, velocities=None, accelerations=None,
                message=f"polynomial backend raised: {exc}",
            )

    def _do_parameterize(
        self,
        path: Path,
        v_limits: np.ndarray,
        a_limits: np.ndarray,
        options: TimeParameterizationOptions,
    ) -> RawTrajectoryResult:
        qs = np.asarray(path.waypoints, dtype=float)
        if qs.shape[0] < 2:
            return RawTrajectoryResult(
                status=TrajectoryStatus.NO_WAYPOINTS,
                times=None, positions=None, velocities=None, accelerations=None,
                message="path needs at least 2 waypoints",
            )

        # Choose a total duration that satisfies v_max and a_max for
        # the straight-line distance between every pair of waypoints.
        # The spline's chord-length parameterisation maps this duration
        # onto interior knot times.
        cum_dist = _cumulative_chord_lengths(qs)
        total_chord = float(cum_dist[-1])
        if total_chord < 1e-12:
            # Degenerate: all waypoints coincide. Emit a two-sample
            # stationary trajectory so downstream code doesn't choke.
            return _stationary_result(qs[0], options.dt)

        n = qs.shape[1]
        start_v = (
            np.full(n, options.start_velocity, dtype=float)
            if options.start_velocity is not None
            else np.zeros(n, dtype=float)
        )
        end_v = (
            np.full(n, options.end_velocity, dtype=float)
            if options.end_velocity is not None
            else np.zeros(n, dtype=float)
        )

        # Duration is chosen so the spline's peak velocity and
        # acceleration stay inside the requested envelopes. The
        # closed-form heuristic in `_duration_for_envelope` gets us
        # close, then we adaptively stretch until the actual sampled
        # derivatives fit. Iteration is bounded; for any reasonable
        # path we converge in 1-3 rounds.
        duration = _duration_for_envelope(qs, v_limits, a_limits, total_chord)
        spline, positions, velocities, accelerations, times, duration = (
            _fit_and_stretch_to_envelope(
                qs=qs,
                cum_dist=cum_dist,
                total_chord=total_chord,
                start_v=start_v,
                end_v=end_v,
                v_limits=v_limits,
                a_limits=a_limits,
                dt=options.dt,
                initial_duration=duration,
            )
        )
        if spline is None:
            return RawTrajectoryResult(
                status=TrajectoryStatus.LIMITS_INFEASIBLE,
                times=None, positions=None, velocities=None, accelerations=None,
                message=(
                    "polynomial backend cannot fit a clamped cubic spline inside "
                    "v/a limits even after stretching duration; simplify the path "
                    "or relax v_scale / a_scale."
                ),
            )

        # Pin endpoints to avoid numerical drift.
        positions[0] = qs[0]
        positions[-1] = qs[-1]
        velocities[0] = start_v
        velocities[-1] = end_v

        return RawTrajectoryResult(
            status=TrajectoryStatus.SUCCESS,
            times=times,
            positions=positions,
            velocities=velocities,
            accelerations=accelerations,
            message=(
                f"polynomial backend, one CubicSpline through {qs.shape[0]} "
                f"waypoints, duration {duration:.3f} s"
            ),
            extra={
                "duration": float(duration),
                "num_knots": int(qs.shape[0]),
                "chord_length": float(total_chord),
            },
        )


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _cumulative_chord_lengths(qs: np.ndarray) -> np.ndarray:
    """Cumulative chord length along the waypoint sequence, starting at 0.

    Returns an array of shape ``(N,)``. ``cum[0] == 0``.
    """
    segs = np.linalg.norm(np.diff(qs, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(segs)])


def _duration_for_envelope(
    qs: np.ndarray,
    v_limits: np.ndarray,
    a_limits: np.ndarray,
    total_chord: float,
) -> float:
    """Choose a total duration so a clamped cubic spline through ``qs``
    with v=0 endpoints stays inside v_limits and a_limits.

    Heuristic: assume the spline's peak velocity is roughly
    ``1.5 * (total_chord / duration)`` (the cubic-with-zero-endpoints
    velocity-shape factor) and its peak acceleration is roughly
    ``6 * total_chord / duration^2``. Solve for duration; take the
    larger of the velocity-bound and the acceleration-bound result,
    floored at 1 ms.

    For multi-waypoint paths the heuristic is approximate; the post-
    parameterisation envelope check catches violations and surfaces
    ``LIMITS_INFEASIBLE``.
    """
    # Per-joint cumulative travel (used to pick the worst-case joint
    # for the velocity / acceleration bound).
    per_joint_total = np.sum(np.abs(np.diff(qs, axis=0)), axis=0)
    v_bound = 1.5 * float(np.max(per_joint_total / np.maximum(v_limits, 1e-9)))
    a_bound = float(
        np.max(np.sqrt(6.0 * per_joint_total / np.maximum(a_limits, 1e-9)))
    )
    # Multi-waypoint paths need a bit of slack on top because the
    # spline's peak derivative is higher than the constant-velocity
    # heuristic. 15% empirical.
    safety = 1.15 if qs.shape[0] > 2 else 1.05
    return max(safety * v_bound, safety * a_bound, 1e-3)


def _fit_and_stretch_to_envelope(
    *,
    qs: np.ndarray,
    cum_dist: np.ndarray,
    total_chord: float,
    start_v: np.ndarray,
    end_v: np.ndarray,
    v_limits: np.ndarray,
    a_limits: np.ndarray,
    dt: float,
    initial_duration: float,
    max_iters: int = 8,
    safety: float = 1.05,
):
    """Fit a clamped CubicSpline at successive durations until the
    sampled velocity and acceleration both fit inside the requested
    envelopes.

    The closed-form heuristic in :func:`_duration_for_envelope` is
    exact for a two-waypoint path with rest BCs, but it underestimates
    for paths with many interior knots — the spline's peak derivative
    is larger than the average. We stretch by the worst observed
    ratio (plus a small safety factor) until the envelope holds, or
    return ``None`` if it never does.

    Returns
    -------
    (spline, positions, velocities, accelerations, times, duration)
        On success; ``(None, None, None, None, None, None)`` on
        failure.
    """
    duration = float(initial_duration)
    bc_type = ((1, start_v), (1, end_v))
    for _ in range(max_iters):
        knot_times = (cum_dist / total_chord) * duration
        spline = CubicSpline(knot_times, qs, bc_type=bc_type, axis=0)

        n_samples = max(2, int(np.floor(duration / dt)) + 1)
        times = np.linspace(0.0, duration, n_samples)
        positions = spline(times)
        velocities = spline(times, 1)
        accelerations = spline(times, 2)

        v_ratio = float(np.max(np.abs(velocities) / np.maximum(v_limits, 1e-9)))
        a_ratio = float(np.max(np.abs(accelerations) / np.maximum(a_limits, 1e-9)))
        worst = max(v_ratio, a_ratio)
        if worst <= 1.0 + 1e-6:
            return spline, positions, velocities, accelerations, times, duration

        # Velocity scales as 1/duration, acceleration as 1/duration^2.
        # Stretch by enough to push the worst ratio under 1 in one go,
        # picking the more-demanding of the two scalings.
        stretch = max(v_ratio, np.sqrt(a_ratio)) * safety
        duration *= max(stretch, 1.01)

    return None, None, None, None, None, None


def _stationary_result(q: np.ndarray, dt: float) -> RawTrajectoryResult:
    """Emit a minimal two-sample stationary trajectory for zero-length paths."""
    dt = max(dt, 1e-3)
    times = np.array([0.0, dt], dtype=float)
    positions = np.stack([q, q])
    velocities = np.zeros_like(positions)
    accelerations = np.zeros_like(positions)
    return RawTrajectoryResult(
        status=TrajectoryStatus.SUCCESS,
        times=times,
        positions=positions,
        velocities=velocities,
        accelerations=accelerations,
        message="polynomial backend: zero-length path; stationary trajectory",
        extra={"duration": float(dt), "num_knots": 1, "chord_length": 0.0},
    )
