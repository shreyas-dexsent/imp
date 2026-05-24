# Copyright (c) 2026 DexSent Robotics Pvt Ltd.
# All Rights Reserved.
#
# SPDX-License-Identifier: LicenseRef-Proprietary
# See LICENSE file for details.
#
# Author: Shreyas <shreyas.kumar@dexsentrobotics.com>
# Proprietary and confidential.

"""Approach primitive — linear descent to a target pose.

`approach` produces a straight-line Cartesian trajectory from a
pre-approach pose (`T_target` offset by `distance` along `axis` in the
target's local frame) to `T_target` itself. The caller is responsible
for getting the arm to the pre-approach pose first (typically via
`move_joint`).

Use this for the final linear descent in a pick / place / insertion
sequence. Returning a pre-approach pose lets the application stitch
the joint-space approach onto a linear final approach with one
controller-side handoff.
"""
from __future__ import annotations

from typing import Literal

import numpy as np

from algorithms.primitives.move_l import move_l
from algorithms.primitives.options import MoveOptions
from algorithms.primitives.result import MoveResult
from algorithms.resolved.scene import Scene


AxisLiteral = Literal["x", "y", "z", "-x", "-y", "-z"]


def approach(
    scene: Scene,
    robot_id: str,
    frame_id: str,
    T_target: np.ndarray,
    q_seed: np.ndarray,
    *,
    distance: float = 0.05,
    axis: AxisLiteral = "-z",
    reference: Literal["target", "world"] = "target",
    options: MoveOptions | None = None,
) -> MoveResult:
    """Linear Cartesian descent from a pre-approach pose to `T_target`.

    The pre-approach pose is computed as ``T_target offset by
    `distance` along `axis``` in the chosen `reference` frame.
    `q_seed` is expected to already be at the pre-approach pose
    (typically the output of a `move_joint` that landed there). The
    returned trajectory starts at the pre-approach pose and ends at
    `T_target`.

    Defaults: 5 cm descent along the target's negative z axis (the
    typical "approach from above" pattern for top-down grasping).

    Parameters
    ----------
    scene
        Scene to plan against.
    robot_id, frame_id
        Robot and TCP frame for the linear motion.
    T_target
        Final TCP pose in world coordinates.
    q_seed
        Joint state at the pre-approach pose. Caller's responsibility
        to ensure the seed is at `pre_approach_pose(T_target, distance, axis, reference)`
        before calling.
    distance
        Approach distance in metres. The pre-approach pose sits this
        far from `T_target` along `axis`.
    axis
        Direction the pre-approach pose is offset from `T_target`.
        Default `-z` (above the target along its z-axis when reference
        is "target").
    reference
        Frame `axis` is interpreted in: `"target"` (local to T_target)
        or `"world"`.
    options
        :class:`MoveOptions`.
    """
    T_pre = pre_approach_pose(T_target, distance, axis, reference)
    return move_l(
        scene, robot_id, frame_id,
        T_goal=T_target, q_seed=q_seed,
        T_start=T_pre,
        options=options,
    )


def pre_approach_pose(
    T_target: np.ndarray,
    distance: float,
    axis: AxisLiteral,
    reference: Literal["target", "world"] = "target",
) -> np.ndarray:
    """Compute the pre-approach pose `distance` away from `T_target`
    along `axis` in the chosen frame.

    This is the "where should the arm be just before approach" pose.
    Useful in application code that runs `move_joint` to this pose,
    then calls `approach`.
    """
    T_target = np.asarray(T_target, dtype=float)
    axis_vec = _axis_to_unit_vec(axis)
    offset_world = (
        T_target[:3, :3] @ axis_vec if reference == "target" else axis_vec
    )
    T_pre = T_target.copy()
    # pre_pose is target moved BACKWARD along the approach direction by `distance`,
    # so that travelling FORWARD `distance` lands on target.
    T_pre[:3, 3] = T_target[:3, 3] - distance * offset_world
    return T_pre


def _axis_to_unit_vec(axis: str) -> np.ndarray:
    table = {
        "x":  np.array([1.0, 0.0, 0.0]),
        "y":  np.array([0.0, 1.0, 0.0]),
        "z":  np.array([0.0, 0.0, 1.0]),
        "-x": np.array([-1.0, 0.0, 0.0]),
        "-y": np.array([0.0, -1.0, 0.0]),
        "-z": np.array([0.0, 0.0, -1.0]),
    }
    if axis not in table:
        raise ValueError(f"axis must be one of {sorted(table.keys())}; got {axis!r}")
    return table[axis]
