# Copyright (c) 2026 DexSent Robotics Pvt Ltd.
# All Rights Reserved.
#
# SPDX-License-Identifier: LicenseRef-Proprietary
# See LICENSE file for details.
#
# Author: Shreyas <shreyas.kumar@dexsentrobotics.com>
# Proprietary and confidential.

"""Path data type and status taxonomy.

A `Path` is the canonical output of every planner and the input/output
of every optimizer. It carries geometry only: an ordered sequence of
waypoints, no timing, no velocity, no acceleration. Time appears at
the trajectory layer.

See docs/plan.md §6.0 Glossary for the locked vocabulary.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict

import numpy as np


class PathStatus(Enum):
    """Machine-readable status for every plan attempt."""

    SUCCESS = "success"
    INVALID_INPUT = "invalid_input"
    START_IN_COLLISION = "start_in_collision"
    GOAL_IN_COLLISION = "goal_in_collision"
    START_OUT_OF_LIMITS = "start_out_of_limits"
    GOAL_OUT_OF_LIMITS = "goal_out_of_limits"
    NO_PATH_FOUND = "no_path_found"
    TIMEOUT = "timeout"
    MAX_ITERATIONS = "max_iterations"
    NUMERICAL_FAILURE = "numerical_failure"
    IK_FAILED = "ik_failed"
    IK_DISCONTINUITY = "ik_discontinuity"
    CARTESIAN_DEVIATION = "cartesian_deviation"
    POST_PLAN_INVALID = "post_plan_invalid"


@dataclass(frozen=True)
class Path:
    """An ordered sequence of waypoints describing the geometry of a motion.

    A path carries geometry only. It has no time, no velocity, no
    acceleration. Talking about "the velocity at waypoint i" is a
    category error until time parameterization runs.

    Attributes
    ----------
    waypoints : np.ndarray, shape (N, dof)
        Joint configurations in the order they should be visited.
    joint_names : tuple[str, ...]
        Active-joint order; must match `model.active_joint_names` of
        the robot the path is for.
    cartesian_waypoints : np.ndarray, shape (N, 4, 4), optional
        World-frame poses of the planning frame at each waypoint.
        Populated only by `plan_cartesian`. None for joint-space paths.
    metadata : dict
        Free-form planner-side annotations: planner_used, frame_id for
        Cartesian paths, robot_id in multi-robot scenarios.

    Invariants
    ----------
    1. `N >= 2`.
    2. Every waypoint inside joint limits with the configured margin.
    3. Every segment collision-free at the planner's sampling resolution.
    4. `waypoints[0]` is the start, `waypoints[-1]` is the goal.
    """

    waypoints: np.ndarray
    joint_names: tuple[str, ...]
    cartesian_waypoints: np.ndarray | None = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Pydantic-style cheap structural validation; full validation is
        # the validator's job.
        if not isinstance(self.waypoints, np.ndarray) or self.waypoints.ndim != 2:
            raise ValueError(
                f"Path.waypoints must be a 2D ndarray of shape (N, dof); "
                f"got {type(self.waypoints).__name__} {getattr(self.waypoints, 'shape', None)}"
            )
        if self.waypoints.shape[0] < 2:
            raise ValueError(
                f"Path requires at least 2 waypoints; got {self.waypoints.shape[0]}"
            )
        if self.waypoints.shape[1] != len(self.joint_names):
            raise ValueError(
                f"Path.waypoints has dof={self.waypoints.shape[1]} but "
                f"joint_names has {len(self.joint_names)} entries"
            )
        if self.cartesian_waypoints is not None:
            cw = self.cartesian_waypoints
            if cw.shape != (self.waypoints.shape[0], 4, 4):
                raise ValueError(
                    f"Path.cartesian_waypoints must have shape (N, 4, 4); "
                    f"got {cw.shape} (N={self.waypoints.shape[0]})"
                )

    @property
    def num_waypoints(self) -> int:
        return int(self.waypoints.shape[0])

    @property
    def dof(self) -> int:
        return int(self.waypoints.shape[1])

    @property
    def num_segments(self) -> int:
        return self.num_waypoints - 1

    def length(self) -> float:
        """Sum of Euclidean joint-space segment norms. A rough proxy for
        'how much joint motion is in this path.' Used by smoothers and
        for diagnostics."""
        diffs = np.diff(self.waypoints, axis=0)
        return float(np.linalg.norm(diffs, axis=1).sum())
