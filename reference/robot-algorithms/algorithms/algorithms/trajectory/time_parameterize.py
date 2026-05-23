# Copyright (c) 2026 DexSent Robotics Pvt Ltd.
# All Rights Reserved.
#
# SPDX-License-Identifier: LicenseRef-Proprietary
# See LICENSE file for details.
#
# Author: Shreyas <shreyas.kumar@dexsentrobotics.com>
# Proprietary and confidential.

"""Time parameterization entrypoint (Phase 6d).

Takes a `Path` (geometry) and produces a `Trajectory` (geometry + time).
Default behaviour is pass-through: interior waypoints have non-zero
velocity and the robot does NOT pause at any of them.
"""
from __future__ import annotations

import time
from typing import Dict, Type

import numpy as np

from algorithms.planning.path import Path
from algorithms.resolved.kinematic_model import KinematicModel
from algorithms.trajectory.backends.base import (
    RawTrajectoryResult,
    TimeParameterizationBackend,
)
from algorithms.trajectory.backends.polynomial import PolynomialBackend
from algorithms.trajectory.backends.ruckig_backend import RuckigBackend
from algorithms.trajectory.options import TimeParameterizationOptions
from algorithms.trajectory.result import (
    TrajectoryDiagnostics,
    TrajectoryResult,
    TrajectoryStatus,
)
from algorithms.trajectory.trajectory import Trajectory


_BACKENDS: Dict[str, Type[TimeParameterizationBackend]] = {
    "ruckig": RuckigBackend,
    "polynomial": PolynomialBackend,
}


def time_parameterize(
    path: Path,
    model: KinematicModel,
    *,
    options: TimeParameterizationOptions | None = None,
) -> TrajectoryResult:
    """Assign timing to a path and return a `Trajectory`.

    Pass-through by default — the robot moves smoothly through every
    interior waypoint without stopping. Set `options.rest_to_rest=True`
    for the legacy behaviour where the robot comes to rest at each via
    point.

    Backend selection (`options.backend`):

    * ``"auto"`` (default) — Ruckig if importable, else polynomial.
    * ``"ruckig"`` — Force Ruckig. Errors if the package is missing.
    * ``"polynomial"`` — Force pure-Python quintic. Always available.

    Parameters
    ----------
    path
        Output of the planning / optimization layer. Must have at least
        2 waypoints and joint_names matching `model.active_joint_names`.
    model
        Resolved kinematic model. Used to pull velocity / acceleration /
        jerk limits.
    options
        :class:`TimeParameterizationOptions`. Defaults give 1 kHz pass-
        through motion with the model's full v/a/j envelope.
    """
    started = time.perf_counter()
    opts = options or TimeParameterizationOptions()

    if path.num_waypoints < 2:
        return _result(
            TrajectoryStatus.NO_WAYPOINTS,
            None,
            started,
            "path needs at least 2 waypoints",
            backends_tried=(),
        )

    expected = tuple(model.active_joint_names)
    # Composite paths (multi-robot) carry a different joint_names tuple.
    # For Phase 6d we time-parameterize single-robot paths only; the
    # composite case can be added when Phase 7 primitives need it.
    if path.joint_names != expected:
        # Allow paths whose dof matches even if names differ (e.g., a
        # caller that constructed Path with synthetic names). Reject if
        # the dof itself is wrong.
        if path.dof != len(expected):
            return _result(
                TrajectoryStatus.INVALID_INPUT,
                None,
                started,
                f"path.dof={path.dof} but model has {len(expected)} active joints",
                backends_tried=(),
            )

    v_limits = model.active_velocity_limits() * opts.v_scale
    a_limits = model.active_acceleration_limits() * opts.a_scale
    try:
        j_limits = model.active_jerk_limits() * opts.j_scale
    except Exception:
        j_limits = None

    # Pre-pass: collapse waypoints that lie on a straight line between
    # their neighbours. Dense collision-validation samples (from the
    # direct planner or plan_cartesian) become two real endpoints, so
    # the backend doesn't have to interpolate through spurious via
    # points with forced zero-acceleration boundary conditions. This
    # is what causes the visible wiggle on direct A->B moves.
    path = _collapse_collinear_waypoints(path)

    selected = _select_backend(opts.backend, path)
    if isinstance(selected, str):
        return _result(
            TrajectoryStatus.INVALID_INPUT,
            None,
            started,
            selected,
            backends_tried=(),
        )

    tried: list[str] = []
    raw: RawTrajectoryResult | None = None
    for backend_cls in selected:
        backend = backend_cls()
        tried.append(backend.name)
        raw = backend.parameterize(path, v_limits, a_limits, j_limits, opts)
        if raw.status is TrajectoryStatus.SUCCESS:
            break

    assert raw is not None  # selected is non-empty by construction

    if raw.status is not TrajectoryStatus.SUCCESS:
        return _result(
            raw.status,
            None,
            started,
            raw.message,
            backends_tried=tuple(tried),
            extra=raw.extra,
        )

    trajectory = Trajectory(
        times=raw.times,
        positions=raw.positions,
        velocities=raw.velocities,
        accelerations=raw.accelerations,
        joint_names=path.joint_names,
        backend_used=tried[-1],
        metadata={
            "dt": opts.dt,
            "v_scale": opts.v_scale,
            "a_scale": opts.a_scale,
            "j_scale": opts.j_scale,
            "pass_through": not opts.rest_to_rest,
            **raw.extra,
        },
    )
    return TrajectoryResult(
        status=TrajectoryStatus.SUCCESS,
        trajectory=trajectory,
        elapsed_ms=(time.perf_counter() - started) * 1000.0,
        diagnostics=TrajectoryDiagnostics(
            message=raw.message,
            backend_attempted=tuple(tried),
            extra=raw.extra,
        ),
    )


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------


def _select_backend(
    name: str,
    path: Path,
) -> list[Type[TimeParameterizationBackend]] | str:
    """Return the ordered list of backends to try, or an error message.

    Auto-dispatch heuristic:

    * **2-waypoint path** — Ruckig is preferred. It is time-optimal
      under the v/a/j envelope, producing the fastest valid S-curve.
      Polynomial is fallback if Ruckig is unavailable.

    * **3+ waypoint path** — Polynomial is preferred. It fits a single
      C^2 cubic spline through every waypoint, so dense paths (e.g.,
      IK samples from ``plan_cartesian``, or OMPL via-point chains)
      become one smooth motion instead of N chained Ruckig segments.
      A per-segment Ruckig chain pins acceleration at every interior
      knot and is much less smooth on these inputs. Ruckig is fallback
      if the polynomial backend reports the geometry as infeasible
      under the requested envelope.
    """
    if name == "auto":
        try:
            import ruckig  # noqa: F401
            ruckig_available = True
        except Exception:
            ruckig_available = False

        if path.num_waypoints <= 2:
            return (
                [RuckigBackend, PolynomialBackend]
                if ruckig_available
                else [PolynomialBackend]
            )
        return (
            [PolynomialBackend, RuckigBackend]
            if ruckig_available
            else [PolynomialBackend]
        )
    if name in _BACKENDS:
        return [_BACKENDS[name]]
    return f"unknown backend: {name!r}. Choose 'auto', 'ruckig', or 'polynomial'."


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _collapse_collinear_waypoints(
    path: Path,
    *,
    angle_tol: float = 1e-3,
    distance_tol: float = 1e-6,
) -> Path:
    """Drop interior waypoints that lie on a straight line between
    their neighbours in joint space.

    Why: dense waypoints from `StraightLineBackend` (one sample per
    `max_joint_step` for collision validation) and `plan_cartesian`
    (one sample per IK call) are not real via points the robot has
    to pass through with specified boundary conditions. Treating them
    as such forces the backend to pin acceleration to zero at every
    interior knot, which produces visible wiggles on the controller
    side. Collapsing collinear stretches in joint space restores the
    expected single smooth motion for direct-line moves while leaving
    genuine OMPL-style via points untouched (their direction changes
    fail the collinearity test).

    Parameters
    ----------
    path
        Input path.
    angle_tol
        Maximum allowed deviation of the segment-to-segment angle from
        180 degrees (in radians of the joint-space turn). Smaller =
        stricter collinearity.
    distance_tol
        Numerical floor for segment lengths below which a waypoint is
        treated as a duplicate.

    Returns
    -------
    Path
        A new path with collinear interior waypoints removed. Always
        at least 2 waypoints. Endpoints and direction-change waypoints
        are preserved exactly.
    """
    if path.num_waypoints <= 2:
        return path

    qs = path.waypoints
    kept_indices: list[int] = [0]
    for i in range(1, qs.shape[0] - 1):
        prev = qs[kept_indices[-1]]
        curr = qs[i]
        nxt = qs[i + 1]

        v_in = curr - prev
        v_out = nxt - curr
        len_in = float(np.linalg.norm(v_in))
        len_out = float(np.linalg.norm(v_out))

        # Treat zero-length segments as duplicates; skip the waypoint.
        if len_in < distance_tol:
            continue
        if len_out < distance_tol:
            kept_indices.append(i)
            continue

        # Collinearity test: the unit direction shouldn't change.
        cos_angle = float(np.dot(v_in, v_out) / (len_in * len_out))
        cos_angle = max(-1.0, min(1.0, cos_angle))
        angle = float(np.arccos(cos_angle))
        if angle > angle_tol:
            # Real direction change; keep this waypoint.
            kept_indices.append(i)

    kept_indices.append(qs.shape[0] - 1)
    if len(kept_indices) == path.num_waypoints:
        return path

    new_waypoints = qs[kept_indices]
    # `cartesian_waypoints` is dropped: the joint-space collapse
    # doesn't preserve the per-sample IK metadata, and downstream
    # consumers only use joint geometry from this point on.
    return Path(
        waypoints=new_waypoints,
        joint_names=path.joint_names,
        cartesian_waypoints=None,
        metadata={
            **path.metadata,
            "collinear_collapsed_from": path.num_waypoints,
            "collinear_collapsed_to": new_waypoints.shape[0],
        },
    )


def _result(
    status: TrajectoryStatus,
    trajectory: Trajectory | None,
    started: float,
    message: str,
    *,
    backends_tried: tuple,
    extra: dict | None = None,
) -> TrajectoryResult:
    return TrajectoryResult(
        status=status,
        trajectory=trajectory,
        elapsed_ms=(time.perf_counter() - started) * 1000.0,
        diagnostics=TrajectoryDiagnostics(
            message=message,
            backend_attempted=backends_tried,
            extra=dict(extra or {}),
        ),
    )
