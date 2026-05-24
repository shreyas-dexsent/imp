# Copyright (c) 2026 DexSent Robotics Pvt Ltd.
# All Rights Reserved.
#
# SPDX-License-Identifier: LicenseRef-Proprietary
# See LICENSE file for details.
#
# Author: Shreyas <shreyas.kumar@dexsentrobotics.com>
# Proprietary and confidential.

"""State validity factories.

The state validity contract is the single signature `q -> bool` (or
`dict[str, q] -> bool` for multi-robot) that the planner calls on every
sample. It combines two checks:

* `q` lies inside the active joint limits with the configured margin.
* The robot at `q` does not self-collide and does not collide with the
  world (when a `Scene.collision_model` is available).

The closure captures `model` + `scene` + chain + margin once; per-call
work is one validity test (joint limits) plus one collision query.
"""
from __future__ import annotations

from typing import Callable, Mapping

import numpy as np

from algorithms.resolved.kinematic_model import KinematicModel
from algorithms.resolved.scene import Scene


def make_state_validity_fn(
    model: KinematicModel,
    scene: Scene,
    *,
    chain_id: str | None = None,
    margin: float = 0.0,
) -> Callable[[np.ndarray], bool]:
    """Return a `q -> bool` callable that combines joint limits + collision.

    Single-robot path. The closure assumes `scene.world` has exactly one
    robot and applies the validity check to that robot.

    Parameters
    ----------
    model
        Resolved kinematic model of the (single) robot to be checked.
    scene
        Scene the planner is operating on. The closure reads
        `scene.collision_model` to run collision queries; if the scene
        has no collision model, the closure only checks joint limits.
    chain_id
        Restrict the collision check to a specific chain. Useful when
        only one chain of the robot is in motion. `None` means check
        every robot-robot and robot-world pair.
    margin
        Distance inside the joint limits the state must lie for joint
        limits to pass. Defaults to 0 (limits inclusive).
    """
    q_min, q_max = model.active_position_limits()
    q_min_with_margin = q_min + margin
    q_max_with_margin = q_max - margin
    has_collision = scene.collision_model is not None

    def is_valid(q: np.ndarray) -> bool:
        q_arr = np.asarray(q, dtype=float)
        if q_arr.shape != q_min.shape:
            return False
        if not np.all(np.isfinite(q_arr)):
            return False
        if np.any(q_arr < q_min_with_margin) or np.any(q_arr > q_max_with_margin):
            return False
        if not has_collision:
            return True
        # Local import to avoid a circular dependency through collision/.
        from algorithms.collision import is_in_collision

        report = is_in_collision(model, scene, q_arr, chain_id=chain_id)
        return not report.in_collision

    return is_valid


def make_composite_state_validity_fn(
    scene: Scene,
    *,
    margin: float = 0.0,
) -> Callable[[Mapping[str, np.ndarray]], bool]:
    """Multi-robot version of `make_state_validity_fn`.

    The returned closure accepts a dict keyed by `robot_id` and returns
    `True` only if every robot is inside joint limits AND no self- or
    cross-robot collision exists at the composite configuration. The
    collision check uses the multi-robot-aware `is_in_collision`
    pipeline, which walks every world robot's geometry and includes
    cross-robot pairs by construction.
    """
    # Cache per-robot limits up front.
    robot_models = {}
    limits_with_margin = {}
    for world_robot in scene.world.robots:
        robot_models[world_robot.id] = KinematicModel.from_robot_system(
            world_robot.robot_system
        )
        q_min, q_max = robot_models[world_robot.id].active_position_limits()
        limits_with_margin[world_robot.id] = (q_min + margin, q_max - margin)

    has_collision = scene.collision_model is not None
    first_robot = scene.world.robots[0]
    first_model = robot_models[first_robot.id]

    def is_valid(q_by_robot: Mapping[str, np.ndarray]) -> bool:
        # Joint-limits check per robot.
        for robot_id, (q_min_m, q_max_m) in limits_with_margin.items():
            if robot_id not in q_by_robot:
                return False
            q_arr = np.asarray(q_by_robot[robot_id], dtype=float)
            if q_arr.shape != q_min_m.shape:
                return False
            if not np.all(np.isfinite(q_arr)):
                return False
            if np.any(q_arr < q_min_m) or np.any(q_arr > q_max_m):
                return False

        if not has_collision:
            return True

        from algorithms.collision import is_in_collision

        # KinematicModel arg is ignored in the dict path beyond carrying
        # the active-joint shape for the first robot; the runtime walks
        # every robot via the dict.
        report = is_in_collision(first_model, scene, dict(q_by_robot))
        return not report.in_collision

    return is_valid
