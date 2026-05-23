# Copyright (c) 2026 DexSent Robotics Pvt Ltd.
# All Rights Reserved.
#
# SPDX-License-Identifier: LicenseRef-Proprietary
# See LICENSE file for details.
#
# Author: Shreyas <shreyas.kumar@dexsentrobotics.com>
# Proprietary and confidential.

"""Shared "path -> validated trajectory" composition for primitives.

Each primitive resolves to a `Path` (via plan_joint, plan_cartesian, or
direct construction) and then funnels through this helper to get a
validated `Trajectory`. Keeps the smoothing / parameterization / dual
validation flow consistent across primitives.
"""
from __future__ import annotations

import time

from algorithms.optimization import shortcut_smooth, spline_fit
from algorithms.planning import validate_path
from algorithms.planning.path import Path
from algorithms.planning.result import PathPlanResult, PathValidationReport
from algorithms.primitives.options import MoveOptions
from algorithms.primitives.result import MoveDiagnostics, MoveResult, MoveStatus
from algorithms.resolved.kinematic_model import KinematicModel
from algorithms.resolved.scene import Scene
from algorithms.trajectory import (
    TrajectoryStatus,
    time_parameterize,
    validate_trajectory,
)
from algorithms.trajectory.result import (
    TrajectoryResult,
    TrajectoryValidationReport,
)


def finalize_path_to_trajectory(
    *,
    primitive_name: str,
    model: KinematicModel,
    scene: Scene,
    path: Path,
    options: MoveOptions,
    started: float,
    plan_result: PathPlanResult | None = None,
    smooth: bool | None = None,
    spline: bool | None = None,
    ik_result=None,
) -> MoveResult:
    """Run shortcut + spline + validation + time_parameterize on a Path.

    `smooth` and `spline` override `options.smooth_path` / `spline_fit`
    when set — primitives that produce Cartesian paths set
    `smooth=False, spline=False` to preserve the line.
    """
    do_smooth = options.smooth_path if smooth is None else bool(smooth)
    do_spline = options.spline_fit if spline is None else bool(spline)

    current_path = path

    if do_smooth and current_path.num_waypoints >= 3:
        try:
            current_path, _stats = shortcut_smooth(
                current_path, model, scene,
                iterations=options.smoothing_iterations,
                random_seed=options.plan.random_seed,
            )
        except Exception as exc:  # pragma: no cover
            return _fail(
                primitive_name, MoveStatus.OPTIMIZATION_FAILED, started,
                stage="shortcut_smooth", message=str(exc),
                path=current_path, plan_result=plan_result, ik_result=ik_result,
            )

    if do_spline and current_path.num_waypoints >= 2:
        try:
            current_path = spline_fit(
                current_path,
                order="quintic",
                samples=options.spline_samples,
            )
        except Exception as exc:  # pragma: no cover
            return _fail(
                primitive_name, MoveStatus.OPTIMIZATION_FAILED, started,
                stage="spline_fit", message=str(exc),
                path=current_path, plan_result=plan_result, ik_result=ik_result,
            )

    path_validation: PathValidationReport | None = None
    if options.validate_path:
        path_validation = validate_path(
            model, scene, current_path,
            options=options.path_validation,
        )
        if not path_validation.passed:
            return _fail(
                primitive_name, MoveStatus.PATH_VALIDATION_FAILED, started,
                stage="validate_path",
                message=(
                    f"validate_path failed: {path_validation.first_failure}"
                    if path_validation.first_failure
                    else "validate_path failed"
                ),
                path=current_path, plan_result=plan_result, ik_result=ik_result,
                path_validation=path_validation,
            )

    traj_result: TrajectoryResult = time_parameterize(
        current_path, model, options=options.time_parameterize,
    )
    if traj_result.status is not TrajectoryStatus.SUCCESS or traj_result.trajectory is None:
        return _fail(
            primitive_name, MoveStatus.TRAJECTORY_FAILED, started,
            stage="time_parameterize",
            message=traj_result.diagnostics.message or traj_result.status.value,
            path=current_path, plan_result=plan_result, ik_result=ik_result,
            path_validation=path_validation, trajectory_result=traj_result,
        )

    traj_validation: TrajectoryValidationReport | None = None
    if options.validate_trajectory:
        traj_validation = validate_trajectory(
            traj_result.trajectory, model, scene,
            options=options.trajectory_validation,
        )
        if not traj_validation.passed:
            return _fail(
                primitive_name, MoveStatus.TRAJECTORY_VALIDATION_FAILED, started,
                stage="validate_trajectory",
                message=(
                    f"validate_trajectory failed: {traj_validation.first_failure}"
                    if traj_validation.first_failure
                    else "validate_trajectory failed"
                ),
                path=current_path, plan_result=plan_result, ik_result=ik_result,
                path_validation=path_validation, trajectory_result=traj_result,
                trajectory_validation=traj_validation,
            )

    return MoveResult(
        status=MoveStatus.SUCCESS,
        trajectory=traj_result.trajectory,
        path=current_path,
        ik_result=ik_result,
        plan_result=plan_result,
        trajectory_result=traj_result,
        path_validation=path_validation,
        trajectory_validation=traj_validation,
        primitive_used=primitive_name,
        elapsed_ms=(time.perf_counter() - started) * 1000.0,
        diagnostics=MoveDiagnostics(stage="success", message="trajectory ready"),
    )


def _fail(
    primitive_name: str,
    status: MoveStatus,
    started: float,
    *,
    stage: str,
    message: str,
    path=None, plan_result=None, ik_result=None,
    path_validation=None, trajectory_result=None, trajectory_validation=None,
) -> MoveResult:
    return MoveResult(
        status=status,
        trajectory=None,
        path=path,
        ik_result=ik_result,
        plan_result=plan_result,
        trajectory_result=trajectory_result,
        path_validation=path_validation,
        trajectory_validation=trajectory_validation,
        primitive_used=primitive_name,
        elapsed_ms=(time.perf_counter() - started) * 1000.0,
        diagnostics=MoveDiagnostics(stage=stage, message=message),
    )
