# Copyright (c) 2026 DexSent Robotics Pvt Ltd.
# All Rights Reserved.
#
# SPDX-License-Identifier: LicenseRef-Proprietary
# See LICENSE file for details.
#
# Author: Shreyas <shreyas.kumar@dexsentrobotics.com>
# Proprietary and confidential.

"""Joint-space goto primitive."""
from __future__ import annotations

import time

import numpy as np

from algorithms.planning import PathStatus, plan_joint
from algorithms.primitives._compose import _fail, finalize_path_to_trajectory
from algorithms.primitives.options import MoveOptions
from algorithms.primitives.result import MoveResult, MoveStatus
from algorithms.resolved.kinematic_model import KinematicModel
from algorithms.resolved.scene import Scene


def move_joint(
    model: KinematicModel,
    scene: Scene,
    q_goal: np.ndarray,
    q_seed: np.ndarray,
    *,
    options: MoveOptions | None = None,
) -> MoveResult:
    """Generate a smooth, validated trajectory from `q_seed` to `q_goal`.

    Composes `plan_joint` -> optional `shortcut_smooth` -> optional `spline_fit` ->
    `time_parameterize` (pass-through) -> `validate_trajectory` into
    one call. The robot moves continuously from `q_seed` to `q_goal`;
    it does NOT pause at any interior point of the planned path.

    Parameters
    ----------
    model
        Resolved kinematic model.
    scene
        Scene the planner runs against.
    q_goal
        Goal joint configuration.
    q_seed
        Starting joint configuration.
    options
        :class:`MoveOptions`. Default produces pass-through motion via
        OMPL + smoothing + spline + auto trajectory backend. For a
        simple straight joint-space command, pass
        `MoveOptions(planner_backend="direct", smooth_path=False,
        spline_fit=False)`.
    """
    started = time.perf_counter()
    opts = options or MoveOptions()

    plan_result = plan_joint(
        model,
        scene,
        q_seed,
        q_goal,
        backend=opts.planner_backend,
        options=opts.plan,
    )
    if plan_result.status is not PathStatus.SUCCESS or plan_result.path is None:
        return _fail(
            "move_joint", MoveStatus.PLAN_FAILED, started,
            stage="plan_joint",
            message=(plan_result.diagnostics.message or plan_result.status.value),
            plan_result=plan_result,
        )

    return finalize_path_to_trajectory(
        primitive_name="move_joint",
        model=model,
        scene=scene,
        path=plan_result.path,
        options=opts,
        started=started,
        plan_result=plan_result,
    )
