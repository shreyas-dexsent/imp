# Copyright (c) 2026 DexSent Robotics Pvt Ltd.
# All Rights Reserved.
#
# SPDX-License-Identifier: LicenseRef-Proprietary
# See LICENSE file for details.
#
# Author: Shreyas <shreyas.kumar@dexsentrobotics.com>
# Proprietary and confidential.

"""Result + diagnostic types for trajectory generation and validation."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Tuple

from algorithms.trajectory.trajectory import Trajectory


class TrajectoryStatus(Enum):
    """Machine-readable status for every time_parameterize call."""

    SUCCESS = "success"
    INVALID_INPUT = "invalid_input"
    LIMITS_INFEASIBLE = "limits_infeasible"
    BACKEND_FAILURE = "backend_failure"
    NUMERICAL_FAILURE = "numerical_failure"
    NO_WAYPOINTS = "no_waypoints"


@dataclass(frozen=True)
class TrajectoryDiagnostics:
    """Debug payload returned with every TrajectoryResult."""

    message: str = ""
    backend_attempted: Tuple[str, ...] = ()
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TrajectoryResult:
    """Final result of a time_parameterize call.

    `trajectory` is populated only when `status == TrajectoryStatus.SUCCESS`.
    """

    status: TrajectoryStatus
    trajectory: Trajectory | None
    elapsed_ms: float
    diagnostics: TrajectoryDiagnostics

    @property
    def success(self) -> bool:
        return self.status is TrajectoryStatus.SUCCESS


@dataclass(frozen=True)
class TrajectoryCheckResult:
    """One row in a trajectory validator report."""

    name: str
    ok: bool
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TrajectoryValidationReport:
    """Result of `validate_trajectory(...)`.

    `passed` is the AND of every check. `first_failure` is `(t, reason)`
    in seconds when a check failed.
    """

    passed: bool
    checks: Tuple[TrajectoryCheckResult, ...]
    first_failure: Tuple[float, str] | None = None
