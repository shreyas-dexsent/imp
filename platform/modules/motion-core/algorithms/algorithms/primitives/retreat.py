# Copyright (c) 2026 DexSent Robotics Pvt Ltd.
# All Rights Reserved.
#
# SPDX-License-Identifier: LicenseRef-Proprietary
# See LICENSE file for details.
#
# Author: Shreyas <shreyas.kumar@dexsentrobotics.com>
# Proprietary and confidential.

"""Retreat primitive — linear motion away from the current TCP pose.

`retreat` produces a straight-line Cartesian trajectory from the
current TCP pose (via FK on `q_seed`) to a pose `distance` away along
`axis`. The dual of `approach`: where `approach` lands on a target
from a pre-approach pose, `retreat` lifts off a target back to a
post-retreat pose.

Use this for the lift-off after a grasp / place / insertion.
"""
from __future__ import annotations

from typing import Literal

import numpy as np

from algorithms.kinematics import fk
from algorithms.primitives.approach import _axis_to_unit_vec
from algorithms.primitives.move_l import move_l
from algorithms.primitives.options import MoveOptions
from algorithms.primitives.result import MoveResult
from algorithms.resolved.scene import Scene


AxisLiteral = Literal["x", "y", "z", "-x", "-y", "-z"]


def retreat(
    scene: Scene,
    robot_id: str,
    frame_id: str,
    q_seed: np.ndarray,
    *,
    distance: float = 0.05,
    axis: AxisLiteral = "z",
    reference: Literal["tcp", "world"] = "tcp",
    options: MoveOptions | None = None,
) -> MoveResult:
    """Linear Cartesian motion `distance` away from the current TCP pose.

    Default direction: `+z` in the TCP's local frame (the typical
    "lift up after grasp" pattern). `reference="world"` interprets
    `axis` in the world frame instead.

    Parameters
    ----------
    scene
        Scene to plan against.
    robot_id, frame_id
        Robot and TCP frame.
    q_seed
        Joint state at the start of the retreat. The motion begins at
        `fk(scene, robot_id, q_seed, frame_id)`.
    distance
        Retreat distance in metres.
    axis
        Direction to retreat. Default `+z`.
    reference
        Frame `axis` is interpreted in: `"tcp"` (local to the current
        TCP pose) or `"world"`.
    options
        :class:`MoveOptions`.
    """
    T_start = fk(scene, robot_id, q_seed, frame_id)
    axis_vec = _axis_to_unit_vec(axis)
    offset_world = (
        T_start[:3, :3] @ axis_vec if reference == "tcp" else axis_vec
    )
    T_goal = T_start.copy()
    T_goal[:3, 3] = T_start[:3, 3] + distance * offset_world

    return move_l(
        scene, robot_id, frame_id,
        T_goal=T_goal, q_seed=q_seed,
        T_start=T_start,
        options=options,
    )
