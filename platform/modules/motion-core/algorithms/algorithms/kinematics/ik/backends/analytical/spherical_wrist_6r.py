# Copyright (c) 2026 DexSent Robotics Pvt Ltd.
# All Rights Reserved.
#
# SPDX-License-Identifier: LicenseRef-Proprietary
# See LICENSE file for details.
#
# Author: Shreyas <shreyas.kumar@dexsentrobotics.com>
# Proprietary and confidential.

"""Spherical-wrist 6R analytical IK backend placeholder."""
from __future__ import annotations

import numpy as np

from algorithms.kinematics.ik.problem import IKProblemSpec
from algorithms.resolved.kinematic_model import KinematicModel


class SphericalWrist6RIK:
    """Closed-form 6R spherical-wrist interface."""

    name = "spherical_wrist_6r"

    def solve_branches(
        self,
        model: KinematicModel,
        spec: IKProblemSpec,
        q_seed: np.ndarray,
    ) -> tuple[np.ndarray, ...]:
        raise ValueError(
            "SphericalWrist6RIK requires a verified 6R spherical-wrist "
            f"structure; {model.system.robot.id!r} is not registered"
        )
