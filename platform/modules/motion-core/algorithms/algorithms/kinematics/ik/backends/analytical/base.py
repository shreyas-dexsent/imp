# Copyright (c) 2026 DexSent Robotics Pvt Ltd.
# All Rights Reserved.
#
# SPDX-License-Identifier: LicenseRef-Proprietary
# See LICENSE file for details.
#
# Author: Shreyas <shreyas.kumar@dexsentrobotics.com>
# Proprietary and confidential.

"""Analytical IK backend interface."""
from __future__ import annotations

from typing import Protocol

import numpy as np

from algorithms.kinematics.ik.problem import IKProblemSpec
from algorithms.resolved.kinematic_model import KinematicModel


class AnalyticalIK(Protocol):
    """Protocol for closed-form solvers that return solution branches."""

    name: str

    def solve_branches(
        self,
        model: KinematicModel,
        spec: IKProblemSpec,
        q_seed: np.ndarray,
    ) -> tuple[np.ndarray, ...]:
        """Return zero or more active-q branches."""
