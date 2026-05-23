# Copyright (c) 2026 DexSent Robotics Pvt Ltd.
# All Rights Reserved.
#
# SPDX-License-Identifier: LicenseRef-Proprietary
# See LICENSE file for details.
#
# Author: Shreyas <shreyas.kumar@dexsentrobotics.com>
# Proprietary and confidential.

"""Structured IK result types."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np


class IKStatus(Enum):
    """Machine-readable status for every IK attempt."""

    SUCCESS = "success"
    INVALID_INPUT = "invalid_input"
    UNREACHABLE = "unreachable"
    MAX_ITERATIONS = "max_iterations"
    TIMEOUT = "timeout"
    JOINT_LIMIT_VIOLATION = "joint_limit_violation"
    POSE_ERROR_TOO_HIGH = "pose_error_too_high"
    SINGULARITY_RISK = "singularity_risk"
    FINAL_COLLISION = "final_collision"
    CONSTRAINT_VIOLATION = "constraint_violation"
    NO_VALID_CANDIDATE = "no_valid_candidate"
    NUMERICAL_FAILURE = "numerical_failure"


@dataclass(frozen=True)
class IKCandidate:
    """One candidate q returned by a backend and accepted by validation."""

    q: np.ndarray
    pose_error: tuple[float, float]
    score: float
    backend: str
    seed_index: int = 0


@dataclass(frozen=True)
class IKDiagnostics:
    """Debug information about the solve and validation process."""

    message: str = ""
    seed_reports: tuple[dict[str, Any], ...] = ()
    validation_reports: tuple[Any, ...] = ()
    backend_statuses: tuple[IKStatus, ...] = ()
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class IKResult:
    """Final IK solve result.

    `q` is populated only when `status == IKStatus.SUCCESS`.
    """

    status: IKStatus
    q: np.ndarray | None
    pose_error: tuple[float, float]
    iterations: int
    elapsed_ms: float
    backend_used: str
    candidates: tuple[IKCandidate, ...]
    diagnostics: IKDiagnostics = field(default_factory=IKDiagnostics)

    @property
    def success(self) -> bool:
        """Convenience boolean for application code; status remains canonical."""
        return self.status is IKStatus.SUCCESS
