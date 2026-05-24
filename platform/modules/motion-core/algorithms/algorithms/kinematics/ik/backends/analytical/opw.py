# Copyright (c) 2026 DexSent Robotics Pvt Ltd.
# All Rights Reserved.
#
# SPDX-License-Identifier: LicenseRef-Proprietary
# See LICENSE file for details.
#
# Author: Shreyas <shreyas.kumar@dexsentrobotics.com>
# Proprietary and confidential.

"""OPW analytical IK backend placeholder."""
from __future__ import annotations

import numpy as np

from algorithms.kinematics.ik.problem import IKProblemSpec
from algorithms.resolved.kinematic_model import KinematicModel


class OPWIK:
    """OPW analytical solver interface.

    Robot-specific OPW parameters are not present in the current YAML
    descriptions, so this backend fails explicitly instead of guessing.
    """

    name = "opw"

    def solve_branches(
        self,
        model: KinematicModel,
        spec: IKProblemSpec,
        q_seed: np.ndarray,
    ) -> tuple[np.ndarray, ...]:
        raise ValueError(
            "OPWIK requires registered OPW parameters for this robot; "
            f"none are available for {model.system.robot.id!r}"
        )
