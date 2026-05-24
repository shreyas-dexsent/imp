# Copyright (c) 2026 DexSent Robotics Pvt Ltd.
# All Rights Reserved.
#
# SPDX-License-Identifier: LicenseRef-Proprietary
# See LICENSE file for details.
#
# Author: Shreyas <shreyas.kumar@dexsentrobotics.com>
# Proprietary and confidential.

"""Cartesian straight-line path planning entrypoint.

Samples Cartesian poses along the start->goal TCP path, solves IK at
every sample with the previous q as seed (for joint continuity), and
returns a joint-space Path whose `cartesian_waypoints` field carries
the realised TCP poses.
"""
from __future__ import annotations

import math
import time

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from algorithms.kinematics.fk import fk_local
from algorithms.kinematics.ik import IKStatus, ik_local
from algorithms.planning.options import PlanOptions
from algorithms.planning.path import Path, PathStatus
from algorithms.planning.result import PathDiagnostics, PathPlanResult
from algorithms.planning.state_validity import make_state_validity_fn
from algorithms.resolved.kinematic_model import KinematicModel
from algorithms.resolved.scene import Scene


def plan_cartesian(
    scene: Scene,
    robot_id: str,
    frame_id: str,
    T_start: np.ndarray | None,
    T_goal: np.ndarray,
    q_seed: np.ndarray,
    *,
    options: PlanOptions | None = None,
) -> PathPlanResult:
    """Plan a straight-line Cartesian path for one robot's TCP.

    The TCP at `frame_id` follows a linear path in world coordinates
    from `T_start` (or FK on `q_seed` if `None`) to `T_goal`. IK is
    solved at each sample with the previous q as the seed; large joint
    discontinuities between samples are reported as `IK_DISCONTINUITY`.

    Returns a joint-space `Path` whose `cartesian_waypoints` field
    carries the sampled TCP poses (in the robot's base frame).
    """
    started = time.perf_counter()
    opts = options or PlanOptions()

    world_robot = scene.world.robot(robot_id)
    model = KinematicModel.from_robot_system(world_robot.robot_system)

    # Materialise T_start from current state if missing.
    q_seed = np.asarray(q_seed, dtype=float)
    if T_start is None:
        T_start = fk_local(model, q_seed, frame_id)
    T_start = np.asarray(T_start, dtype=float)
    T_goal = np.asarray(T_goal, dtype=float)

    p_start, p_goal = T_start[:3, 3], T_goal[:3, 3]
    R_start = Rotation.from_matrix(T_start[:3, :3])
    R_goal = Rotation.from_matrix(T_goal[:3, :3])

    # Choose sample count from the larger of translation / rotation steps.
    p_dist = float(np.linalg.norm(p_goal - p_start))
    rot_angle = float((R_goal * R_start.inv()).magnitude())
    n_trans = math.ceil(p_dist / max(opts.cartesian_translation_step, 1e-9))
    n_rot = math.ceil(rot_angle / max(opts.cartesian_rotation_step, 1e-9))
    n_samples = max(2, max(n_trans, n_rot) + 1)

    # Sample Cartesian poses (translate linearly, slerp orientation).
    alphas = np.linspace(0.0, 1.0, n_samples)
    slerp = Slerp([0.0, 1.0], Rotation.concatenate([R_start, R_goal]))
    sampled_poses = []
    line_dir = p_goal - p_start
    line_dot = float(line_dir @ line_dir)
    for a in alphas:
        T = np.eye(4)
        T[:3, 3] = p_start + a * line_dir
        T[:3, :3] = slerp([a])[0].as_matrix()
        sampled_poses.append(T)

        # Straight-line tolerance check (perpendicular deviation).
        if line_dot > 1e-12:
            p = T[:3, 3]
            projected_alpha = float((p - p_start) @ line_dir) / line_dot
            projected = p_start + projected_alpha * line_dir
            dev = float(np.linalg.norm(p - projected))
            if dev > opts.cartesian_line_tolerance:
                return _fail(
                    PathStatus.CARTESIAN_DEVIATION,
                    "cartesian",
                    0,
                    started,
                    f"sample {len(sampled_poses) - 1} deviates {dev:.3e} m from start->goal line",
                )

    # Cheap up-front validity check on seed.
    validity_fn = make_state_validity_fn(model, scene, margin=opts.joint_margin)
    if not validity_fn(q_seed):
        return _fail(
            PathStatus.START_IN_COLLISION,
            "cartesian",
            0,
            started,
            "q_seed is invalid (joint limits or collision at start)",
        )

    # IK per sample with continuity check.
    q_waypoints = [q_seed.copy()]
    q_prev = q_seed.copy()
    for i, T in enumerate(sampled_poses):
        if i == 0:
            continue
        ik_result = ik_local(model, frame_id, T, q_prev, scene=scene)
        if ik_result.status is not IKStatus.SUCCESS or ik_result.q is None:
            return _fail(
                PathStatus.IK_FAILED,
                "cartesian",
                i,
                started,
                f"IK failed at sample {i} with status {ik_result.status.name}",
                extra={"failed_sample_index": i, "ik_status": ik_result.status.name},
            )
        if np.max(np.abs(ik_result.q - q_prev)) > opts.cartesian_ik_continuity:
            return _fail(
                PathStatus.IK_DISCONTINUITY,
                "cartesian",
                i,
                started,
                f"IK branch jump at sample {i}: "
                f"|delta q|_inf = {float(np.max(np.abs(ik_result.q - q_prev))):.3f}",
                extra={"failed_sample_index": i},
            )
        q_waypoints.append(ik_result.q)
        q_prev = ik_result.q

    waypoints = np.array(q_waypoints, dtype=float)
    cartesian_arr = np.array(sampled_poses, dtype=float)

    path = Path(
        waypoints=waypoints,
        joint_names=tuple(model.active_joint_names),
        cartesian_waypoints=cartesian_arr,
        metadata={
            "planner_used": "cartesian",
            "frame_id": frame_id,
            "robot_id": robot_id,
            "n_samples": n_samples,
        },
    )

    return PathPlanResult(
        status=PathStatus.SUCCESS,
        path=path,
        planner_used="cartesian",
        iterations=n_samples,
        elapsed_ms=(time.perf_counter() - started) * 1000.0,
        diagnostics=PathDiagnostics(
            message=f"cartesian path resolved at {n_samples} samples",
        ),
    )


def _fail(
    status: PathStatus,
    planner: str,
    iterations: int,
    started: float,
    message: str,
    *,
    extra: dict | None = None,
) -> PathPlanResult:
    return PathPlanResult(
        status=status,
        path=None,
        planner_used=planner,
        iterations=iterations,
        elapsed_ms=(time.perf_counter() - started) * 1000.0,
        diagnostics=PathDiagnostics(message=message, extra=dict(extra or {})),
    )
