# Copyright (c) 2026 DexSent Robotics Pvt Ltd.
# All Rights Reserved.
#
# SPDX-License-Identifier: LicenseRef-Proprietary
# See LICENSE file for details.
#
# Author: Shreyas <shreyas.kumar@dexsentrobotics.com>
# Proprietary and confidential.

"""Shared runtime placement helpers for collision queries.

Multi-robot aware: `geometry_entries` walks every world robot, runs FK
on each with its own `KinematicModel` + `pin.Data`, and composes the
per-robot base pose into every geometry placement. Cross-robot
collision pairs appear automatically because both robots' links land
in the same entries dict and `pairs.active_pairs` enumerates all
unordered combinations.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Dict, Mapping, Union

import numpy as np

from algorithms.resolved.collision_model import CollisionModel
from algorithms.resolved.kinematic_model import KinematicModel
from algorithms.resolved.scene import Scene


# Composite-q type: either a single ndarray (single-robot back-compat,
# resolved to the sole robot in the world) or a dict keyed by robot id.
QInput = Union[np.ndarray, Mapping[str, np.ndarray]]


@dataclass(frozen=True)
class CollisionGeometryEntry:
    """One geometry object plus its query-time world placement."""

    name: str
    geometry: object
    placement: object


def geometry_entries(
    model: KinematicModel,
    scene: Scene,
    q: QInput,
) -> Dict[str, CollisionGeometryEntry]:
    """Build query-time geometry entries for every robot plus world objects.

    The single-robot caller passes `q` as a plain ndarray; we resolve it
    against the world's only robot. Multi-robot callers pass a dict
    keyed by `robot_id`. Robots whose state is not supplied in the dict
    fall back to `Scene.robot_states[robot_id]` if present; otherwise a
    KeyError is raised.

    Each robot's geometry is placed via FK on its own `KinematicModel`
    (looked up from the cache per `robot_system`) and pre-multiplied by
    its `T_world_base`. Cross-robot collision pairs emerge for free
    because all entries share one dict.
    """
    import pinocchio as pin

    collision_model = _require_collision_model(scene)
    q_by_robot = _normalise_q_input(scene, q, model)

    entries: Dict[str, CollisionGeometryEntry] = {}

    # Per-robot FK + geometry placement. Each robot keeps its own
    # `KinematicModel` (cache returns the same instance for two world
    # robots sharing a robot_system YAML) and per-call pin.Data /
    # pin.GeometryData scratch buffers. We use the per-robot
    # `GeometryModel` from `collision_model.robot_geoms_by_id`, NOT the
    # combined `robot_geom`, because parent-joint indices on the combined
    # catalogue are only valid against the originating robot's pin_model.
    for world_robot in scene.world.robots:
        robot_id = world_robot.id
        robot_q = q_by_robot[robot_id]
        robot_model = KinematicModel.from_robot_system(world_robot.robot_system)
        robot_geom_per_id = collision_model.robot_geoms_by_id.get(robot_id)
        if robot_geom_per_id is None:
            # Backwards compatibility: a CollisionModel built before this
            # field was added. Fall back to the combined geom — correct
            # for single-robot worlds, undefined for multi-robot.
            robot_geom_per_id = collision_model.robot_geom
        T_world_base = (
            world_robot.base_pose.as_matrix()
            if world_robot.base_pose is not None
            else np.eye(4)
        )

        q_full = robot_model.expand(robot_q)
        data = robot_model.pin_model.createData()
        robot_gd = pin.GeometryData(robot_geom_per_id)

        pin.forwardKinematics(robot_model.pin_model, data, q_full)
        pin.updateFramePlacements(robot_model.pin_model, data)
        pin.updateGeometryPlacements(
            robot_model.pin_model,
            data,
            robot_geom_per_id,
            robot_gd,
        )

        for idx, go in enumerate(robot_geom_per_id.geometryObjects):
            T_world_geom = T_world_base @ _se3_to_matrix(robot_gd.oMg[idx])
            entries[go.name] = CollisionGeometryEntry(
                name=go.name,
                geometry=go.geometry,
                placement=_se3_from_matrix(T_world_geom),
            )

    # World objects (universe-parented). Pose comes from
    # Scene.object_poses or, for attached objects, from FK on the
    # attaching robot's parent frame.
    for go in collision_model.world_geom.geometryObjects:
        T_world_obj = _world_object_pose(scene, go.name, q_by_robot)
        T_world_geom = T_world_obj @ _se3_to_matrix(go.placement)
        entries[go.name] = CollisionGeometryEntry(
            name=go.name,
            geometry=go.geometry,
            placement=_se3_from_matrix(T_world_geom),
        )

    return entries


def skipped_pair_count(collision_model: CollisionModel, active_pair_count: int) -> int:
    """Return how many materialised candidate pairs were skipped."""
    total = len(list(combinations(collision_model.object_names(), 2)))
    return max(0, total - active_pair_count)


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _require_collision_model(scene: Scene) -> CollisionModel:
    if scene.collision_model is None:
        raise ValueError("scene.collision_model is required for collision queries")
    return scene.collision_model


def _normalise_q_input(
    scene: Scene,
    q: QInput,
    model_hint: KinematicModel,
) -> Dict[str, np.ndarray]:
    """Coerce single-robot ndarray or multi-robot dict into a per-robot dict.

    A bare ndarray is resolved to the sole world robot whose
    `robot_system` matches `model_hint` (otherwise to `scene.world.robots[0]`
    when only one robot exists). Multi-robot worlds must use the dict form.
    """
    robots = scene.world.robots
    if isinstance(q, Mapping):
        # Multi-robot path. Fill from scene.robot_states for any robot
        # the caller did not supply.
        result: Dict[str, np.ndarray] = {}
        for world_robot in robots:
            if world_robot.id in q:
                result[world_robot.id] = np.asarray(q[world_robot.id], dtype=float)
            elif world_robot.id in scene.robot_states:
                result[world_robot.id] = scene.robot_states[world_robot.id]
            else:
                raise KeyError(
                    f"q missing for robot {world_robot.id!r}; pass it in the q dict "
                    f"or set scene.robot_states[{world_robot.id!r}] first."
                )
        return result

    # Plain ndarray. Only valid when one robot is in the world, or when
    # the caller is interacting with the model_hint-backed robot only.
    q_arr = np.asarray(q, dtype=float)
    if len(robots) == 1:
        return {robots[0].id: q_arr}

    # Multi-robot world with an ndarray: pick the robot whose
    # KinematicModel matches `model_hint`, fall back to scene.robot_states
    # for the others.
    target_id: str | None = None
    for world_robot in robots:
        candidate = KinematicModel.from_robot_system(world_robot.robot_system)
        if candidate is model_hint:
            target_id = world_robot.id
            break
    if target_id is None:
        raise ValueError(
            "ambiguous q for multi-robot world: pass q as a dict keyed by robot_id, "
            "or ensure the bare q matches one of the world's robot systems."
        )

    result = {target_id: q_arr}
    for world_robot in robots:
        if world_robot.id == target_id:
            continue
        if world_robot.id not in scene.robot_states:
            raise KeyError(
                f"q for robot {world_robot.id!r} not supplied. In a multi-robot world, "
                "pass q as a dict or populate scene.robot_states for every robot."
            )
        result[world_robot.id] = scene.robot_states[world_robot.id]
    return result


def _world_object_pose(
    scene: Scene,
    object_id: str,
    q_by_robot: Dict[str, np.ndarray],
) -> np.ndarray:
    if object_id in scene.attached:
        attached = scene.attached[object_id]
        robot_id = _resolve_attached_robot(scene, attached)
        robot = scene.world.robot(robot_id)
        robot_model = KinematicModel.from_robot_system(robot.robot_system)
        T_world_base = (
            robot.base_pose.as_matrix() if robot.base_pose is not None else np.eye(4)
        )
        T_base_parent = _frame_pose(robot_model, q_by_robot[robot_id], attached.parent_frame)
        return T_world_base @ T_base_parent @ attached.T_parent_obj
    return scene.get_object_pose(object_id)


def _resolve_attached_robot(scene: Scene, attached) -> str:
    if attached.robot_id is not None:
        return attached.robot_id
    if len(scene.world.robots) == 1:
        return scene.world.robots[0].id
    raise ValueError(
        f"attached object {attached.object_id!r} has no robot_id and the scene "
        "has multiple robots; set robot_id on the AttachedObject."
    )


def _frame_pose(model: KinematicModel, q: np.ndarray, frame_id: str) -> np.ndarray:
    from algorithms.kinematics.fk import fk_local

    return fk_local(model, q, frame_id)


def _se3_from_matrix(matrix: np.ndarray):
    import pinocchio as pin

    se3 = pin.SE3.Identity()
    se3.rotation = np.asarray(matrix[:3, :3], dtype=float)
    se3.translation = np.asarray(matrix[:3, 3], dtype=float)
    return se3


def _se3_to_matrix(se3) -> np.ndarray:
    matrix = np.eye(4, dtype=float)
    matrix[:3, :3] = np.asarray(se3.rotation, dtype=float)
    matrix[:3, 3] = np.asarray(se3.translation, dtype=float).reshape(3)
    return matrix
