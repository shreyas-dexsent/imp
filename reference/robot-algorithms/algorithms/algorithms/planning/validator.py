# Copyright (c) 2026 DexSent Robotics Pvt Ltd.
# All Rights Reserved.
#
# SPDX-License-Identifier: LicenseRef-Proprietary
# See LICENSE file for details.
#
# Author: Shreyas <shreyas.kumar@dexsentrobotics.com>
# Proprietary and confidential.

"""Path validator — post-plan advisory checks.

The planner already ran a discrete-state validity check on every
sample; this validator runs the higher-level advisory checks:
clearance, singularity, continuous collision between waypoints, and
branch-jump detection. Short-circuits on first failure with a
`PathValidationReport` carrying the failing waypoint index and reason.
"""
from __future__ import annotations

import math

import numpy as np

from algorithms.kinematics.fk import fk_local
from algorithms.kinematics.jacobian import jacobian
from algorithms.kinematics.singularity import condition_number, min_singular_value
from algorithms.planning.options import PathValidationOptions
from algorithms.planning.path import Path
from algorithms.planning.result import CheckResult, PathValidationReport
from algorithms.resolved.kinematic_model import KinematicModel
from algorithms.resolved.scene import Scene


def validate_path(
    model: KinematicModel,
    scene: Scene,
    path: Path,
    *,
    options: PathValidationOptions | None = None,
) -> PathValidationReport:
    """Run post-plan advisory checks on a Path.

    Returns a `PathValidationReport` whose `passed` is True iff every
    check passed at every waypoint. `first_failure` is `(waypoint_index,
    reason)` when a check fails.
    """
    opts = options or PathValidationOptions()
    checks: list[CheckResult] = []

    # 1. Structural.
    if path.num_waypoints < 2:
        return _fail(checks, 0, "path requires at least 2 waypoints", "structure")
    checks.append(CheckResult("structure", True))

    # 2. Joint limits at every waypoint.
    lower, upper = model.active_position_limits()
    lower_m, upper_m = lower + opts.joint_margin, upper - opts.joint_margin
    for i, wp in enumerate(path.waypoints):
        if np.any(wp < lower_m) or np.any(wp > upper_m):
            return _fail(checks, i, "joint limits violated", "joint_limits")
    checks.append(CheckResult("joint_limits", True))

    # 3. Discrete collision at every waypoint.
    if scene.collision_model is not None:
        from algorithms.collision import is_in_collision

        for i, wp in enumerate(path.waypoints):
            if is_in_collision(model, scene, wp).in_collision:
                return _fail(checks, i, "collision at waypoint", "collision")
        checks.append(CheckResult("collision", True))

    # 4. Continuous collision sweep between consecutive waypoints.
    if scene.collision_model is not None:
        from algorithms.collision import is_in_collision

        for i in range(path.num_segments):
            qa, qb = path.waypoints[i], path.waypoints[i + 1]
            step = max(opts.collision_step, 1e-9)
            n = max(1, math.ceil(float(np.max(np.abs(qb - qa))) / step))
            for k in range(1, n):
                alpha = k / n
                q = (1 - alpha) * qa + alpha * qb
                if is_in_collision(model, scene, q).in_collision:
                    return _fail(
                        checks, i, f"continuous collision at segment {i}, alpha={alpha:.2f}",
                        "continuous_collision",
                    )
        checks.append(CheckResult("continuous_collision", True))

    # 5. Singularity metric per waypoint.
    if opts.reject_singular:
        # Use the path's metadata frame_id when present (Cartesian paths);
        # for joint paths, walk every waypoint and check the most relevant
        # body Jacobian. The default body is the model's primary chain
        # tip if exposed; otherwise we skip silently.
        frame_id = path.metadata.get("frame_id")
        if frame_id is not None:
            for i, wp in enumerate(path.waypoints):
                J = jacobian(model, wp, frame_id)
                sigma = min_singular_value(J)
                cond = condition_number(J)
                if sigma < opts.min_sigma_limit or cond > opts.condition_number_limit:
                    return _fail(
                        checks, i,
                        f"singularity at waypoint {i}: sigma={sigma:.2e}, cond={cond:.2e}",
                        "singularity",
                    )
            checks.append(CheckResult("singularity", True))

    # 6. Branch-jump detection: large joint jump with tiny TCP delta.
    if path.cartesian_waypoints is not None:
        for i in range(path.num_segments):
            dq = float(np.max(np.abs(path.waypoints[i + 1] - path.waypoints[i])))
            dp = float(np.linalg.norm(
                path.cartesian_waypoints[i + 1][:3, 3] - path.cartesian_waypoints[i][:3, 3]
            ))
            if (
                dq > opts.branch_jump_joint_threshold
                and dp < opts.branch_jump_tcp_threshold
            ):
                return _fail(
                    checks, i + 1,
                    f"branch jump at waypoint {i+1}: |dq|_inf={dq:.3f}, |dp|={dp:.3e}",
                    "branch_jump",
                )
        checks.append(CheckResult("branch_jump", True))

    # 7. Optional velocity-envelope estimate when a nominal segment time
    # is supplied.
    if opts.nominal_segment_time is not None and opts.nominal_segment_time > 0:
        v_lim = model.active_velocity_limits()
        for i in range(path.num_segments):
            dq = path.waypoints[i + 1] - path.waypoints[i]
            v_est = np.abs(dq) / opts.nominal_segment_time
            if np.any(v_est > v_lim):
                return _fail(
                    checks, i + 1,
                    f"velocity envelope exceeded at segment {i}",
                    "velocity_envelope",
                )
        checks.append(CheckResult("velocity_envelope", True))

    return PathValidationReport(passed=True, checks=tuple(checks), first_failure=None)


def _fail(
    checks: list[CheckResult],
    waypoint_index: int,
    reason: str,
    check_name: str,
) -> PathValidationReport:
    checks.append(CheckResult(check_name, False, {"reason": reason, "waypoint": waypoint_index}))
    return PathValidationReport(
        passed=False,
        checks=tuple(checks),
        first_failure=(waypoint_index, reason),
    )
