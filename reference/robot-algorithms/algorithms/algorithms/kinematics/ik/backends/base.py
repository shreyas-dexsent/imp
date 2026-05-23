# Copyright (c) 2026 DexSent Robotics Pvt Ltd.
# All Rights Reserved.
#
# SPDX-License-Identifier: LicenseRef-Proprietary
# See LICENSE file for details.
#
# Author: Shreyas <shreyas.kumar@dexsentrobotics.com>
# Proprietary and confidential.

"""Backend interfaces for IK solvers."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from algorithms.kinematics.ik.options import IKOptions
from algorithms.kinematics.ik.problem import IKProblemSpec
from algorithms.kinematics.ik.result import IKStatus
from algorithms.resolved.kinematic_model import KinematicModel


@dataclass(frozen=True)
class BackendCandidate:
    """Candidate produced by a backend before validation."""

    q: np.ndarray
    pose_error: tuple[float, float]
    iterations: int
    seed_index: int = 0
    cost: float = 0.0


@dataclass(frozen=True)
class BackendResult:
    """Raw backend result before validation."""

    status: IKStatus
    candidates: tuple[BackendCandidate, ...]
    iterations: int
    elapsed_ms: float
    message: str = ""
    seed_reports: tuple[dict[str, Any], ...] = field(default_factory=tuple)


class NonlinearIKBackend(Protocol):
    """Protocol implemented by pose-IK backends."""

    name: str

    def solve(
        self,
        model: KinematicModel,
        spec: IKProblemSpec,
        q_seed: np.ndarray,
        options: IKOptions,
    ) -> BackendResult:
        """Solve an IK problem and return unvalidated candidates."""
