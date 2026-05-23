# Copyright (c) 2026 DexSent Robotics Pvt Ltd.
# All Rights Reserved.
#
# SPDX-License-Identifier: LicenseRef-Proprietary
# See LICENSE file for details.
#
# Author: Shreyas <shreyas.kumar@dexsentrobotics.com>
# Proprietary and confidential.

"""Result + status types for motion primitives.

Every primitive returns a `MoveResult` with an enumerated status, the
final `Trajectory` on success, and the intermediate artifacts
(`Path`, `IKResult`, planner / trajectory results) for inspection.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict

from algorithms.kinematics.ik import IKResult
from algorithms.planning.path import Path
from algorithms.planning.result import PathPlanResult, PathValidationReport
from algorithms.trajectory.result import (
    TrajectoryResult,
    TrajectoryValidationReport,
)
from algorithms.trajectory.trajectory import Trajectory


class MoveStatus(Enum):
    """Result of a motion primitive."""

    SUCCESS = "success"
    INVALID_INPUT = "invalid_input"
    IK_FAILED = "ik_failed"
    PLAN_FAILED = "plan_failed"
    OPTIMIZATION_FAILED = "optimization_failed"
    PATH_VALIDATION_FAILED = "path_validation_failed"
    TRAJECTORY_FAILED = "trajectory_failed"
    TRAJECTORY_VALIDATION_FAILED = "trajectory_validation_failed"
    NUMERICAL_FAILURE = "numerical_failure"


@dataclass(frozen=True)
class MoveDiagnostics:
    """Debug payload for a motion primitive call."""

    message: str = ""
    stage: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MoveResult:
    """Final result of a motion primitive.

    `trajectory` is populated only when `status == MoveStatus.SUCCESS`.
    The intermediate artifacts (`path`, `ik_result`, `plan_result`,
    `trajectory_result`, `path_validation`, `trajectory_validation`)
    are populated when their corresponding stage ran, regardless of
    overall success — useful for diagnosing where the primitive failed.
    """

    status: MoveStatus
    trajectory: Trajectory | None
    path: Path | None = None
    ik_result: IKResult | None = None
    plan_result: PathPlanResult | None = None
    trajectory_result: TrajectoryResult | None = None
    path_validation: PathValidationReport | None = None
    trajectory_validation: TrajectoryValidationReport | None = None
    primitive_used: str = ""
    elapsed_ms: float = 0.0
    diagnostics: MoveDiagnostics = field(default_factory=MoveDiagnostics)

    @property
    def success(self) -> bool:
        return self.status is MoveStatus.SUCCESS
