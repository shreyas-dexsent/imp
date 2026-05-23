# Copyright (c) 2026 DexSent Robotics Pvt Ltd.
# All Rights Reserved.
#
# SPDX-License-Identifier: LicenseRef-Proprietary
# See LICENSE file for details.
#
# Author: Shreyas <shreyas.kumar@dexsentrobotics.com>
# Proprietary and confidential.

"""Drake-style modular IK problem builder."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from algorithms.kinematics.ik.constraints import JointPositionBounds, PoseTarget
from algorithms.kinematics.ik.costs import (
    JointCenteringCost,
    ManipulabilityCost,
    SeedRegularization,
    SingularValuePenalty,
)


Task = PoseTarget
Constraint = JointPositionBounds
Cost = SeedRegularization | JointCenteringCost | ManipulabilityCost | SingularValuePenalty


@dataclass(frozen=True)
class IKProblemSpec:
    """Frozen IK problem snapshot consumed by backends and validators."""

    tasks: tuple[Task, ...]
    constraints: tuple[Constraint, ...]
    costs: tuple[Cost, ...]
    metadata: dict[str, Any]

    @property
    def pose_target(self) -> PoseTarget:
        """Return the single pose target supported by v1."""
        pose_tasks = [task for task in self.tasks if isinstance(task, PoseTarget)]
        if len(pose_tasks) != 1:
            raise ValueError("v1 IK requires exactly one PoseTarget task")
        return pose_tasks[0]


class IKProblem:
    """Mutable builder for an inverse-kinematics problem."""

    def __init__(self) -> None:
        self._tasks: list[Task] = []
        self._constraints: list[Constraint] = []
        self._costs: list[Cost] = []
        self._metadata: dict[str, Any] = {}

    def add_task(self, task: Task) -> None:
        """Add a v1 task. Only PoseTarget is currently supported."""
        if not isinstance(task, PoseTarget):
            raise TypeError(f"unsupported IK task type: {type(task).__name__}")
        self._tasks.append(task)

    def add_constraint(self, constraint: Constraint) -> None:
        """Add a v1 hard constraint."""
        if not isinstance(constraint, JointPositionBounds):
            raise TypeError(
                f"unsupported IK constraint type: {type(constraint).__name__}"
            )
        self._constraints.append(constraint)

    def add_cost(self, cost: Cost) -> None:
        """Add a v1 soft cost."""
        if not isinstance(
            cost,
            (
                SeedRegularization,
                JointCenteringCost,
                ManipulabilityCost,
                SingularValuePenalty,
            ),
        ):
            raise TypeError(f"unsupported IK cost type: {type(cost).__name__}")
        self._costs.append(cost)

    def set_metadata(self, key: str, value: Any) -> None:
        """Attach non-solver metadata to the frozen spec."""
        self._metadata[key] = value

    def freeze(self) -> IKProblemSpec:
        """Return an immutable problem snapshot."""
        return IKProblemSpec(
            tasks=tuple(self._tasks),
            constraints=tuple(self._constraints),
            costs=tuple(self._costs),
            metadata=dict(self._metadata),
        )
