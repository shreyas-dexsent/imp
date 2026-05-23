# Copyright (c) 2026 DexSent Robotics Pvt Ltd.
# All Rights Reserved.
#
# SPDX-License-Identifier: LicenseRef-Proprietary
# See LICENSE file for details.
#
# Author: Shreyas <shreyas.kumar@dexsentrobotics.com>
# Proprietary and confidential.

"""Trajectory validator (Phase 6e).

Walks the trajectory at `options.validation_dt` and runs:

1. Joint position inside limits with margin
2. |qd| <= v_max * v_scale (with numerical slack)
3. |qdd| <= a_max * a_scale
4. Numerical jerk (finite difference of qdd) <= j_max * j_scale
5. Dense-time collision (when scene supplied with collision_model)
6. TCP linear / angular speed (optional, requires tcp_frame_id)
7. Stored dt fine enough for the controller tick (optional)

Short-circuits on first failure with `(t, reason)` recorded.
"""
from __future__ import annotations

import numpy as np

from algorithms.kinematics.fk import fk_local
from algorithms.resolved.kinematic_model import KinematicModel
from algorithms.resolved.scene import Scene
from algorithms.trajectory.options import TrajectoryValidationOptions
from algorithms.trajectory.result import (
    TrajectoryCheckResult,
    TrajectoryValidationReport,
)
from algorithms.trajectory.trajectory import Trajectory


def validate_trajectory(
    trajectory: Trajectory,
    model: KinematicModel,
    scene: Scene,
    *,
    options: TrajectoryValidationOptions | None = None,
) -> TrajectoryValidationReport:
    """Run dense-time validation on a trajectory."""
    opts = options or TrajectoryValidationOptions()
    checks: list[TrajectoryCheckResult] = []

    if trajectory.num_samples < 2:
        return _fail(checks, 0.0, "trajectory has < 2 samples", "structure")
    checks.append(TrajectoryCheckResult("structure", True))

    # Build the dense-time validation grid.
    dt = opts.validation_dt
    if dt <= 0:
        return _fail(checks, 0.0, "validation_dt must be > 0", "structure")
    n_check = max(2, int(np.floor(trajectory.duration / dt)) + 1)
    sample_times = np.linspace(0.0, trajectory.duration, n_check)
    qs = np.zeros((n_check, trajectory.dof))
    qds = np.zeros((n_check, trajectory.dof))
    qdds = np.zeros((n_check, trajectory.dof))
    for i, t in enumerate(sample_times):
        qs[i], qds[i], qdds[i] = trajectory.at(float(t))

    # Limits
    lower, upper = model.active_position_limits()
    lower_m, upper_m = lower + opts.joint_margin, upper - opts.joint_margin
    v_lim = model.active_velocity_limits() * opts.v_scale + opts.numerical_slack
    a_lim = model.active_acceleration_limits() * opts.a_scale + opts.numerical_slack
    try:
        j_lim = model.active_jerk_limits() * opts.j_scale + opts.numerical_slack
    except Exception:
        j_lim = None

    # 1. Joint positions
    for i, t in enumerate(sample_times):
        q = qs[i]
        if np.any(q < lower_m) or np.any(q > upper_m):
            return _fail(checks, float(t), "joint position limit violated", "joint_limits")
    checks.append(TrajectoryCheckResult("joint_limits", True))

    # 2. Velocities
    for i, t in enumerate(sample_times):
        if np.any(np.abs(qds[i]) > v_lim):
            return _fail(checks, float(t), "velocity limit violated", "velocity_limits")
    checks.append(TrajectoryCheckResult("velocity_limits", True))

    # 3. Accelerations
    for i, t in enumerate(sample_times):
        if np.any(np.abs(qdds[i]) > a_lim):
            return _fail(checks, float(t), "acceleration limit violated", "acceleration_limits")
    checks.append(TrajectoryCheckResult("acceleration_limits", True))

    # 4. Jerk (numerical finite difference of qdd between successive
    # samples). Only meaningful when j_lim is known.
    if j_lim is not None and n_check >= 2:
        # Compute numerical jerk; allow generous slack for the
        # finite-difference noise that small dt introduces.
        slack = j_lim + 5.0 * opts.numerical_slack
        for i in range(1, n_check):
            jerk = (qdds[i] - qdds[i - 1]) / max(sample_times[i] - sample_times[i - 1], 1e-9)
            if np.any(np.abs(jerk) > slack):
                return _fail(
                    checks, float(sample_times[i]),
                    "jerk envelope exceeded", "jerk_limits",
                )
        checks.append(TrajectoryCheckResult("jerk_limits", True))

    # 5. Dense-time collision
    if opts.check_collision and scene.collision_model is not None:
        from algorithms.collision import is_in_collision

        for i, t in enumerate(sample_times):
            if is_in_collision(model, scene, qs[i]).in_collision:
                return _fail(checks, float(t), "collision at time t", "collision")
        checks.append(TrajectoryCheckResult("collision", True))

    # 6. TCP speed
    if (opts.tcp_v_max is not None or opts.tcp_omega_max is not None) and opts.tcp_frame_id:
        # Linear / angular speed via FK + finite difference.
        positions_tcp = np.zeros((n_check, 3))
        rotations_tcp = []
        for i in range(n_check):
            T = fk_local(model, qs[i], opts.tcp_frame_id)
            positions_tcp[i] = T[:3, 3]
            rotations_tcp.append(T[:3, :3])
        # Linear speeds
        for i in range(1, n_check):
            dt_i = max(sample_times[i] - sample_times[i - 1], 1e-9)
            if opts.tcp_v_max is not None:
                v = float(np.linalg.norm(positions_tcp[i] - positions_tcp[i - 1])) / dt_i
                if v > opts.tcp_v_max + opts.numerical_slack:
                    return _fail(checks, float(sample_times[i]),
                                 f"TCP linear speed {v:.3f} > {opts.tcp_v_max:.3f}",
                                 "tcp_linear_speed")
            if opts.tcp_omega_max is not None:
                R_rel = rotations_tcp[i] @ rotations_tcp[i - 1].T
                # rotation angle from a rotation matrix
                cos_th = max(-1.0, min(1.0, (np.trace(R_rel) - 1.0) / 2.0))
                angle = float(np.arccos(cos_th))
                w = angle / dt_i
                if w > opts.tcp_omega_max + opts.numerical_slack:
                    return _fail(checks, float(sample_times[i]),
                                 f"TCP angular speed {w:.3f} > {opts.tcp_omega_max:.3f}",
                                 "tcp_angular_speed")
        checks.append(TrajectoryCheckResult("tcp_speed", True))

    # 7. Controller-rate compatibility
    if opts.controller_dt is not None:
        traj_dt = float(trajectory.metadata.get("dt", trajectory.times[1] - trajectory.times[0]))
        if traj_dt > opts.controller_dt + opts.numerical_slack:
            return _fail(
                checks, 0.0,
                f"trajectory dt {traj_dt:.4f} > controller tick {opts.controller_dt:.4f}",
                "controller_rate",
            )
        checks.append(TrajectoryCheckResult("controller_rate", True))

    return TrajectoryValidationReport(
        passed=True, checks=tuple(checks), first_failure=None,
    )


def _fail(
    checks: list[TrajectoryCheckResult],
    t: float,
    reason: str,
    name: str,
) -> TrajectoryValidationReport:
    checks.append(TrajectoryCheckResult(name, False, {"reason": reason, "t": t}))
    return TrajectoryValidationReport(
        passed=False,
        checks=tuple(checks),
        first_failure=(t, reason),
    )
