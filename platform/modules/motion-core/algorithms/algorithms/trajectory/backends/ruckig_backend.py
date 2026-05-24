# Copyright (c) 2026 DexSent Robotics Pvt Ltd.
# All Rights Reserved.
#
# SPDX-License-Identifier: LicenseRef-Proprietary
# See LICENSE file for details.
#
# Author: Shreyas <shreyas.kumar@dexsentrobotics.com>
# Proprietary and confidential.

"""Ruckig trajectory backend.

Chains per-segment Ruckig calls with **pass-through** velocity AND
acceleration at every interior waypoint. The robot does not pause at
any interior knot.

Interior boundary conditions (target_velocity and target_acceleration
at the end of each non-final segment) come from a single
``scipy.interpolate.CubicSpline`` fit through all waypoints with
clamped-velocity boundary conditions. Why a spline: an earlier version
of this backend pinned ``target_acceleration = 0`` at every interior
waypoint. With many interior knots (dense collision-validation samples
on a direct A->B move, or the per-IK output of ``plan_cartesian``)
that produced one acceleration ramp per segment and the robot
visibly wiggled. Fitting a single C^2 spline first and using its
derivatives at the knots gives Ruckig a self-consistent set of
interior boundary conditions, so each segment is a smooth jerk-limited
continuation of the previous one.

Local computation only. Does NOT use Ruckig's ``intermediate_positions``
API, which routes through the Ruckig Community cloud server and is
production-unsafe. Each segment's ``update()`` loop is solved purely
on-device.

If ``ruckig`` is not installed this module returns ``BACKEND_FAILURE``
with a clear message. The dispatcher in :mod:`time_parameterize`
falls back to the polynomial backend when that happens.
"""
from __future__ import annotations

from typing import List

import numpy as np
from scipy.interpolate import CubicSpline

from algorithms.planning.path import Path
from algorithms.trajectory.backends.base import (
    RawTrajectoryResult,
    TimeParameterizationBackend,
)
from algorithms.trajectory.options import TimeParameterizationOptions
from algorithms.trajectory.result import TrajectoryStatus


class RuckigBackend:
    """Ruckig backend with chained pass-through segments.

    Each segment is one Ruckig online-update loop with start / target
    velocities AND accelerations taken from a single C^2 spline fit
    through the entire waypoint sequence. Interior knots are pass-
    through by construction; no acceleration ramps at via points.
    """

    name: str = "ruckig"

    def parameterize(
        self,
        path: Path,
        v_limits: np.ndarray,
        a_limits: np.ndarray,
        j_limits: np.ndarray | None,
        options: TimeParameterizationOptions,
    ) -> RawTrajectoryResult:
        try:
            from ruckig import InputParameter, OutputParameter, Result, Ruckig
        except Exception as exc:
            return RawTrajectoryResult(
                status=TrajectoryStatus.BACKEND_FAILURE,
                times=None, positions=None, velocities=None, accelerations=None,
                message=(
                    f"ruckig not importable: {exc}. Install via "
                    "`pip install ruckig` or use backend='polynomial'."
                ),
            )

        try:
            return self._parameterize(
                InputParameter, OutputParameter, Result, Ruckig,
                path, v_limits, a_limits, j_limits, options,
            )
        except Exception as exc:  # pragma: no cover - safety net
            return RawTrajectoryResult(
                status=TrajectoryStatus.NUMERICAL_FAILURE,
                times=None, positions=None, velocities=None, accelerations=None,
                message=f"ruckig backend raised: {exc}",
            )

    def _parameterize(
        self,
        InputParameter, OutputParameter, Result, Ruckig,
        path: Path,
        v_limits: np.ndarray,
        a_limits: np.ndarray,
        j_limits: np.ndarray | None,
        options: TimeParameterizationOptions,
    ) -> RawTrajectoryResult:
        qs = np.asarray(path.waypoints, dtype=float)
        n = qs.shape[1]
        if path.num_segments < 1:
            return RawTrajectoryResult(
                status=TrajectoryStatus.NO_WAYPOINTS,
                times=None, positions=None, velocities=None, accelerations=None,
                message="path needs at least 2 waypoints",
            )

        if j_limits is None:
            # Pinocchio doesn't provide jerk in URDF; the resolved layer
            # raises if jerk is missing for chain joints. If somebody
            # passes None here anyway, default to 10x acceleration as a
            # reasonable fallback.
            j_limits = 10.0 * a_limits

        # Endpoint boundary conditions.
        start_v = (
            float(options.start_velocity)
            if options.start_velocity is not None
            else 0.0
        )
        end_v = (
            float(options.end_velocity)
            if options.end_velocity is not None
            else 0.0
        )
        a_start = (
            float(options.start_acceleration)
            if options.start_acceleration is not None
            else 0.0
        )
        a_end = (
            float(options.end_acceleration)
            if options.end_acceleration is not None
            else 0.0
        )

        # Interior boundary conditions from a single clamped CubicSpline
        # through all waypoints. Endpoints honour rest-to-rest if asked.
        target_vels, target_accs = _spline_interior_bcs(
            qs,
            v_limits=v_limits,
            a_limits=a_limits,
            start_v=np.full(n, start_v),
            end_v=np.full(n, end_v) if options.rest_to_rest else np.full(n, end_v),
            rest_to_rest=options.rest_to_rest,
            interior_scale=options.interior_velocity_scale,
        )

        otg = Ruckig(n, options.dt)
        inp = InputParameter(n)
        out = OutputParameter(n)
        inp.max_velocity = v_limits.tolist()
        inp.max_acceleration = a_limits.tolist()
        inp.max_jerk = j_limits.tolist()

        all_t: List[float] = []
        all_q: List[np.ndarray] = []
        all_qd: List[np.ndarray] = []
        all_qdd: List[np.ndarray] = []

        # Running state at the segment boundary. Starts at the very
        # first waypoint with the requested start velocity / acceleration.
        cur_q = qs[0].astype(float).tolist()
        cur_v = target_vels[0].astype(float).tolist()
        cur_a = target_accs[0].astype(float).tolist()
        t_global = 0.0

        # Capture the initial sample so the trajectory starts at t=0.
        all_t.append(t_global)
        all_q.append(np.asarray(cur_q, dtype=float))
        all_qd.append(np.asarray(cur_v, dtype=float))
        all_qdd.append(np.asarray(cur_a, dtype=float))

        for seg_idx in range(path.num_segments):
            inp.current_position = cur_q
            inp.current_velocity = cur_v
            inp.current_acceleration = cur_a

            inp.target_position = qs[seg_idx + 1].astype(float).tolist()
            inp.target_velocity = target_vels[seg_idx + 1].astype(float).tolist()
            inp.target_acceleration = target_accs[seg_idx + 1].astype(float).tolist()

            while True:
                res = otg.update(inp, out)
                t_global += options.dt
                all_t.append(t_global)
                all_q.append(np.asarray(out.new_position, dtype=float))
                all_qd.append(np.asarray(out.new_velocity, dtype=float))
                all_qdd.append(np.asarray(out.new_acceleration, dtype=float))

                if res == Result.Finished:
                    cur_q = list(out.new_position)
                    cur_v = list(out.new_velocity)
                    cur_a = list(out.new_acceleration)
                    break
                if res == Result.Error:
                    return RawTrajectoryResult(
                        status=TrajectoryStatus.BACKEND_FAILURE,
                        times=None, positions=None, velocities=None, accelerations=None,
                        message=f"ruckig Result.Error on segment {seg_idx}",
                    )

                inp.current_position = list(out.new_position)
                inp.current_velocity = list(out.new_velocity)
                inp.current_acceleration = list(out.new_acceleration)

        # Override the very last target so downstream consumers see
        # exact endpoint state (Ruckig is usually within tolerance, but
        # we don't want sub-tolerance drift to look like a tracking
        # error).
        all_q[-1] = qs[-1].astype(float)
        all_qd[-1] = target_vels[-1].astype(float)
        all_qdd[-1] = target_accs[-1].astype(float)

        return RawTrajectoryResult(
            status=TrajectoryStatus.SUCCESS,
            times=np.asarray(all_t, dtype=float),
            positions=np.stack(all_q),
            velocities=np.stack(all_qd),
            accelerations=np.stack(all_qdd),
            message=(
                f"ruckig backend, {path.num_segments} segments, "
                f"{'pass-through' if not options.rest_to_rest else 'rest-to-rest'}, "
                f"duration {t_global:.3f} s"
            ),
            extra={
                "num_segments": int(path.num_segments),
                "duration": float(t_global),
            },
        )


# ---------------------------------------------------------------------------
# Interior boundary conditions
# ---------------------------------------------------------------------------


def _spline_interior_bcs(
    qs: np.ndarray,
    *,
    v_limits: np.ndarray,
    a_limits: np.ndarray,
    start_v: np.ndarray,
    end_v: np.ndarray,
    rest_to_rest: bool,
    interior_scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-waypoint velocity and acceleration targets for Ruckig.

    Strategy: fit a single C^2 ``CubicSpline`` through all waypoints
    with clamped velocity boundary conditions (``start_v`` and
    ``end_v``), then sample its first and second derivatives at every
    knot. Clip the result to per-joint envelopes so Ruckig is never
    asked for a boundary condition it can't realize.

    Endpoints get exactly ``start_v`` / ``end_v`` for velocity and
    zero acceleration (Ruckig's standard rest start/end). Interior
    knots get spline-derived velocity and acceleration so each
    segment's target lines up with the next segment's start —
    eliminating the per-knot acceleration ramps that the old
    ``target_acceleration = 0`` strategy produced.

    Parameters
    ----------
    qs
        Waypoint array, shape ``(N, dof)``.
    v_limits, a_limits
        Per-joint velocity / acceleration envelopes. Targets are
        clipped to ``interior_scale * v_limits`` / ``a_limits``.
    start_v, end_v
        Endpoint velocities, shape ``(dof,)``.
    rest_to_rest
        If True, all interior velocities and accelerations are zero
        (legacy behaviour). Endpoint velocities still come from
        ``start_v`` / ``end_v``.
    interior_scale
        Multiplier applied to ``v_limits`` when clipping interior
        velocity targets.

    Returns
    -------
    (target_vels, target_accs)
        Two ``(N, dof)`` arrays. Row ``i`` is the BC for waypoint ``i``.
    """
    n_pts, dof = qs.shape
    target_vels = np.zeros_like(qs)
    target_accs = np.zeros_like(qs)

    target_vels[0] = start_v
    target_vels[-1] = end_v
    # Endpoint accelerations stay at zero — this matches Ruckig's
    # default "start from rest, end at rest" assumption. Any caller
    # that needs a non-zero endpoint acceleration can extend this.

    if rest_to_rest or n_pts <= 2:
        return target_vels, target_accs

    # Chord-length parameterization for the spline knot times. The
    # absolute duration doesn't matter for Ruckig (it picks its own
    # timing) but the *ratios* between knot intervals must reflect
    # geometry, so the spline shape is meaningful.
    seg_lens = np.linalg.norm(np.diff(qs, axis=0), axis=1)
    seg_lens = np.maximum(seg_lens, 1e-9)
    knot_s = np.concatenate([[0.0], np.cumsum(seg_lens)])
    total = float(knot_s[-1])
    if total < 1e-12:
        return target_vels, target_accs

    # Pick a duration matched to the velocity envelope so derivatives
    # come out in sensible q/s units. Heuristic: peak speed roughly
    # 1.5 * total_chord / duration for a clamped-rest spline; pick
    # duration so the slowest joint stays under interior_scale * v_max.
    per_joint_total = np.sum(np.abs(np.diff(qs, axis=0)), axis=0)
    cap = np.maximum(v_limits * float(interior_scale), 1e-9)
    duration = 1.5 * float(np.max(per_joint_total / cap))
    duration = max(duration, 1e-3)

    knot_t = (knot_s / total) * duration

    bc_type = ((1, start_v), (1, end_v))
    try:
        spline = CubicSpline(knot_t, qs, bc_type=bc_type, axis=0)
    except Exception:
        # Degenerate knot times (rare; happens for duplicated waypoints
        # we failed to dedupe upstream). Fall back to rest-at-interior.
        return target_vels, target_accs

    v_cap = v_limits * float(interior_scale)
    a_cap = a_limits  # full envelope for accel; Ruckig will rate-limit
    for i in range(1, n_pts - 1):
        v_i = np.asarray(spline(knot_t[i], 1), dtype=float)
        a_i = np.asarray(spline(knot_t[i], 2), dtype=float)
        target_vels[i] = np.clip(v_i, -v_cap, v_cap)
        target_accs[i] = np.clip(a_i, -a_cap, a_cap)

    return target_vels, target_accs
