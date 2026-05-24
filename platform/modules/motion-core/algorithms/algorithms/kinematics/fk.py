# Copyright (c) 2026 DexSent Robotics Pvt Ltd.
# All Rights Reserved.
#
# SPDX-License-Identifier: LicenseRef-Proprietary
# See LICENSE file for details.
#
# Author: Shreyas <shreyas.kumar@dexsentrobotics.com>
# Proprietary and confidential.

"""Forward kinematics.

Two API layers are exposed:

* :func:`fk_local` - pose of a frame in the robot's base frame, given a
  `KinematicModel` and an active-DOF q vector. Low level; no world knowledge.
* :func:`fk` - pose of a frame in the world frame, given a `Scene` and
  a robot id. Composes `world_T_base @ fk_local` for the chosen robot.

Frame ids are resolved-model frame names. They may come from the robot URDF,
the gripper URDF, or YAML TCPs injected by `KinematicModel`.

Batch variants :func:`fk_local_many` and :func:`fk_many` evaluate multiple
frames with a single underlying Pinocchio FK pass, amortizing the per-call
setup cost.

A fresh `pin.Data` is allocated per call so concurrent or interleaved
multi-robot queries do not corrupt each other (see `docs/architecture.md`).
"""
from __future__ import annotations

from typing import Dict, Iterable

import numpy as np

from algorithms.descriptions import WorldDescription
from algorithms.resolved.kinematic_model import KinematicModel
from algorithms.resolved.scene import Scene


# ---------------------------------------------------------------------------
# Local (robot base frame) FK
# ---------------------------------------------------------------------------


def fk_local(model: KinematicModel, q: np.ndarray, frame_id: str) -> np.ndarray:
    """Forward kinematics returning a 4x4 in the robot's base frame.

    Parameters
    ----------
    model : KinematicModel
        Resolved kinematic model for the robot system.
    q : np.ndarray
        Active-DOF joint values, shape `(len(model.active_joint_names),)`.
        Values must be ordered exactly as `model.active_joint_names`.
        Mimic followers are expanded internally.
    frame_id : str
        Resolved model frame name. The frame may come from the robot URDF,
        gripper URDF, or a YAML TCP injected during model resolution.

    Returns
    -------
    np.ndarray
        4x4 homogeneous transform from the robot's base frame to `frame_id`.

    Raises
    ------
    KeyError
        If `frame_id` is not present in the resolved kinematic model.
    ValueError
        If `q` does not have shape `(len(model.active_joint_names),)`.
    """
    import pinocchio as pin

    q_full = model.expand(q)
    data = model.pin_model.createData()
    pin.forwardKinematics(model.pin_model, data, q_full)
    pin.updateFramePlacements(model.pin_model, data)

    return _frame_pose_in_base(model, data, frame_id)


def fk_local_many(
    model: KinematicModel,
    q: np.ndarray,
    frame_ids: Iterable[str],
) -> Dict[str, np.ndarray]:
    """Batch FK: one Pinocchio pass for many frames.

    `q` must be ordered as `model.active_joint_names`. Every frame id must
    exist in the resolved kinematic model.

    Roughly 5x faster than calling :func:`fk_local` in a loop when several
    frames are needed from the same configuration.
    """
    import pinocchio as pin

    q_full = model.expand(q)
    data = model.pin_model.createData()
    pin.forwardKinematics(model.pin_model, data, q_full)
    pin.updateFramePlacements(model.pin_model, data)

    return {fid: _frame_pose_in_base(model, data, fid) for fid in frame_ids}


# ---------------------------------------------------------------------------
# World-frame FK
# ---------------------------------------------------------------------------


def fk(
    scene: Scene,
    robot_id: str,
    q: np.ndarray,
    frame_id: str,
) -> np.ndarray:
    """Forward kinematics returning a 4x4 in the world frame.

    The frame is named by its local resolved-model identifier, e.g.
    `"fr3_link8"`, `"fr3_hand_tcp"`, or `"robot_tcp"`. For a multi-robot
    world the returned pose is always in world coordinates; callers never
    write namespaced frame names like `"left/fr3_link8"`.

    Internally this function fetches the cached `KinematicModel` for
    `scene.world.robot(robot_id).robot_system`, computes base-frame FK, and
    composes it with the robot instance's world base pose:

        T_world_frame = T_world_base @ T_base_frame

    Parameters
    ----------
    scene : Scene
        Carries the world description (base_pose for each robot lives there).
    robot_id : str
        Which world robot to evaluate FK for.
    q : np.ndarray
        Active-DOF joint values for that robot, ordered as
        `model.active_joint_names` for the robot system referenced by
        `robot_id`.
    frame_id : str
        Local resolved-model frame name. The frame may come from URDF or
        YAML TCP injection.

    Returns
    -------
    np.ndarray
        4x4 homogeneous transform from world to `frame_id`.
    """
    world_robot = scene.world.robot(robot_id)
    model = KinematicModel.from_robot_system(world_robot.robot_system)

    T_base_frame = fk_local(model, q, frame_id)
    T_world_base = _world_base_pose(scene.world, robot_id)
    return T_world_base @ T_base_frame


def fk_many(
    scene: Scene,
    robot_id: str,
    q: np.ndarray,
    frame_ids: Iterable[str],
) -> Dict[str, np.ndarray]:
    """Batch world-frame FK: one Pinocchio pass for many frames.

    Frame names are local resolved-model frame names, not namespaced world
    frame names. For example, use `"fr3_hand_tcp"`, not
    `"left/fr3_hand_tcp"`.
    """
    world_robot = scene.world.robot(robot_id)
    model = KinematicModel.from_robot_system(world_robot.robot_system)

    locals_ = fk_local_many(model, q, frame_ids)
    T_world_base = _world_base_pose(scene.world, robot_id)
    return {fid: T_world_base @ T_base_frame for fid, T_base_frame in locals_.items()}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _frame_pose_in_base(model: KinematicModel, data, frame_id: str) -> np.ndarray:
    """Return the pose of `frame_id` relative to the robot's declared base.

    Pinocchio stores frames relative to its own universe joint. We invert
    through the base frame so the returned pose is expressed in the
    user-facing base frame declared in the YAML.
    """
    base_frame = model.system.robot.base_frame
    if not model.pin_model.existFrame(base_frame):
        raise KeyError(
            f"robot base frame not found in resolved kinematic model: {base_frame!r}"
        )
    if not model.pin_model.existFrame(frame_id):
        raise KeyError(
            f"frame not found in resolved kinematic model: {frame_id!r}"
        )

    base_idx = model.pin_model.getFrameId(base_frame)
    frame_idx = model.pin_model.getFrameId(frame_id)

    universe_to_base = _se3_to_matrix(data.oMf[base_idx])
    universe_to_frame = _se3_to_matrix(data.oMf[frame_idx])
    return np.linalg.inv(universe_to_base) @ universe_to_frame


def _se3_to_matrix(se3) -> np.ndarray:
    """Convert a `pin.SE3` to a 4x4 NumPy matrix."""
    matrix = np.eye(4, dtype=float)
    matrix[:3, :3] = np.asarray(se3.rotation, dtype=float)
    matrix[:3, 3] = np.asarray(se3.translation, dtype=float).reshape(3)
    return matrix


def _world_base_pose(world: WorldDescription, robot_id: str) -> np.ndarray:
    """World-frame placement of a robot's base. Identity if base_pose unset."""
    robot = world.robot(robot_id)
    if robot.base_pose is None:
        return np.eye(4, dtype=float)
    return robot.base_pose.as_matrix()
