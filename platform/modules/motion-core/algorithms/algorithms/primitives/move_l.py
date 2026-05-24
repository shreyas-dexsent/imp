# Copyright (c) 2026 DexSent Robotics Pvt Ltd.
# All Rights Reserved.
#
# SPDX-License-Identifier: LicenseRef-Proprietary
# See LICENSE file for details.
#
# Author: Shreyas <shreyas.kumar@dexsentrobotics.com>
# Proprietary and confidential.

"""Linear Cartesian goto primitive (MoveL)."""
from __future__ import annotations

import time

import numpy as np

from algorithms.planning import PathStatus, plan_cartesian
from algorithms.primitives._compose import _fail, finalize_path_to_trajectory
from algorithms.primitives.options import MoveOptions
from algorithms.primitives.result import MoveResult, MoveStatus
from algorithms.resolved.scene import Scene
from algorithms.resolved.kinematic_model import KinematicModel


def move_l(
    scene: Scene,
    robot_id: str,
    frame_id: str,
    T_goal: np.ndarray,
    q_seed: np.ndarray,
    *,
    T_start: np.ndarray | None = None,
    options: MoveOptions | None = None,
) -> MoveResult:
    """Generate a linear Cartesian trajectory ending at `T_goal`.

    Composes `plan_cartesian` -> `time_parameterize` (pass-through) ->
    `validate_trajectory`. Path smoothing / spline fit is DISABLED for
    Cartesian moves because both would deviate from the requested
    straight TCP line.

    Parameters
    ----------
    scene
        Scene to plan against.
    robot_id
        Which world robot to move.
    frame_id
        TCP frame whose pose follows the Cartesian line.
    T_goal
        Desired final pose in world coordinates.
    q_seed
        Joint state at the start of the move. The TCP at `q_seed` is
        the start of the Cartesian line (or pass `T_start` explicitly).
    T_start
        Optional override for the start TCP pose. When `None`, the
        planner uses FK on `q_seed`.
    options
        :class:`MoveOptions`. Smoothing / spline fit are forced off
        regardless of the options; everything else applies.
    """
    started = time.perf_counter()
    opts = options or MoveOptions()

    plan_result = plan_cartesian(
        scene, robot_id, frame_id, T_start, T_goal, q_seed,
        options=opts.plan,
    )
    if plan_result.status is not PathStatus.SUCCESS or plan_result.path is None:
        return _fail(
            "move_l", MoveStatus.PLAN_FAILED, started,
            stage="plan_cartesian",
            message=(plan_result.diagnostics.message or plan_result.status.value),
            plan_result=plan_result,
        )

    # KinematicModel for the trajectory step.
    model = KinematicModel.from_robot_system(scene.world.robot(robot_id).robot_system)

    return finalize_path_to_trajectory(
        primitive_name="move_l",
        model=model,
        scene=scene,
        path=plan_result.path,
        options=opts,
        started=started,
        plan_result=plan_result,
        smooth=False,
        spline=False,
    )
