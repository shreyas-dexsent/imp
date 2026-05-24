# Copyright (c) 2026 DexSent Robotics Pvt Ltd.
# All Rights Reserved.
#
# SPDX-License-Identifier: LicenseRef-Proprietary
# See LICENSE file for details.
#
# Author: Shreyas <shreyas.kumar@dexsentrobotics.com>
# Proprietary and confidential.

"""Smooth motion through a sequence of joint-space via-points.

Plans `q_seed -> q_via_1 -> q_via_2 -> ... -> q_final` then runs
shortcut smoothing + spline fit + time parameterization. Interior
via-points are pass-through by default (the robot does NOT stop at any
of them). Use this when a high-level controller has produced a
sequence of joint targets that should be visited in order without rest.
"""
from __future__ import annotations

import time
from typing import Sequence

import numpy as np

from algorithms.planning import PathStatus, plan_joint
from algorithms.planning.path import Path
from algorithms.primitives._compose import _fail, finalize_path_to_trajectory
from algorithms.primitives.options import MoveOptions
from algorithms.primitives.result import MoveResult, MoveStatus
from algorithms.resolved.kinematic_model import KinematicModel
from algorithms.resolved.scene import Scene


def via_motion(
    model: KinematicModel,
    scene: Scene,
    q_waypoints: Sequence[np.ndarray],
    *,
    options: MoveOptions | None = None,
) -> MoveResult:
    """Plan a smooth motion through every q in `q_waypoints`, in order.

    The first element is the start; the last is the goal; interior
    elements are via-points the trajectory must visit but pass through
    without stopping (pass-through behaviour from time_parameterize).

    Internally: `plan_joint` runs once per consecutive pair, producing
    segment paths. The segment paths are concatenated into one Path
    (with duplicate waypoints at junctions removed) and the
    composition helper takes it from there.

    Parameters
    ----------
    model, scene
        Resolved model and scene.
    q_waypoints
        Sequence of joint configurations, length >= 2.
    options
        :class:`MoveOptions`. Defaults: smooth + spline + pass-through.
    """
    started = time.perf_counter()
    opts = options or MoveOptions()

    wps = [np.asarray(q, dtype=float) for q in q_waypoints]
    if len(wps) < 2:
        return _fail(
            "via_motion", MoveStatus.INVALID_INPUT, started,
            stage="input_check",
            message=f"via_motion requires >= 2 via-points; got {len(wps)}",
        )

    dof = wps[0].shape[0]
    if any(q.shape != (dof,) for q in wps):
        return _fail(
            "via_motion", MoveStatus.INVALID_INPUT, started,
            stage="input_check",
            message="all via-points must have the same shape",
        )

    # Plan each pair separately and concatenate. plan_joint may produce
    # a multi-waypoint OMPL path per segment in cluttered space; the
    # final concatenated path is then smoothed end-to-end.
    all_waypoints: list[np.ndarray] = [wps[0].copy()]
    last_plan_result = None
    for i in range(len(wps) - 1):
        plan_result = plan_joint(
            model, scene, wps[i], wps[i + 1], options=opts.plan,
        )
        last_plan_result = plan_result
        if plan_result.status is not PathStatus.SUCCESS or plan_result.path is None:
            return _fail(
                "via_motion", MoveStatus.PLAN_FAILED, started,
                stage=f"plan_joint segment {i}",
                message=(
                    plan_result.diagnostics.message or plan_result.status.value
                ),
                plan_result=plan_result,
            )
        # Skip the start of this segment because it duplicates the end
        # of the previous one.
        all_waypoints.extend(plan_result.path.waypoints[1:].tolist())

    combined_waypoints = np.asarray(all_waypoints, dtype=float)
    joint_names = (
        last_plan_result.path.joint_names
        if last_plan_result and last_plan_result.path
        else tuple(model.active_joint_names)
    )
    combined_path = Path(
        waypoints=combined_waypoints,
        joint_names=joint_names,
        cartesian_waypoints=None,
        metadata={"primitive": "via_motion", "num_via_points": len(wps)},
    )

    return finalize_path_to_trajectory(
        primitive_name="via_motion",
        model=model,
        scene=scene,
        path=combined_path,
        options=opts,
        started=started,
        plan_result=last_plan_result,
    )
