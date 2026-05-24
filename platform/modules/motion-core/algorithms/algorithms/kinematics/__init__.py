# Copyright (c) 2026 DexSent Robotics Pvt Ltd.
# All Rights Reserved.
#
# SPDX-License-Identifier: LicenseRef-Proprietary
# See LICENSE file for details.
#
# Author: Shreyas <shreyas.kumar@dexsentrobotics.com>
# Proprietary and confidential.

"""Kinematic operations on resolved models: FK, Jacobian, singularity metrics.

Each function in this package is stateless and side-effect free. It takes
a `KinematicModel` (and optionally a `Scene`) plus inputs and returns
a NumPy result. No function holds a long-lived `pin.Data` or caches
results; the same model can therefore back multiple world robots without
cross-contamination.
"""

from algorithms.kinematics.fk import (
    fk,
    fk_local,
    fk_local_many,
    fk_many,
)
from algorithms.kinematics.jacobian import (
    jacobian,
)
from algorithms.kinematics.ik import (
    IKOptions,
    IKProblem,
    IKResult,
    IKStatus,
    PoseTarget,
    ik,
    ik_local,
    ik_velocity,
    solve_problem,
)
from algorithms.kinematics.singularity import (
    condition_number,
    inverse_condition_number,
    manipulability,
    min_singular_value,
    singularity_report,
)

__all__ = [
    # Forward kinematics
    "fk",
    "fk_local",
    "fk_local_many",
    "fk_many",
    # Jacobian
    "jacobian",
    # Inverse kinematics — ergonomic entrypoints
    "ik",
    "ik_local",
    "ik_velocity",
    # Inverse kinematics — types
    "IKOptions",
    "IKResult",
    "IKStatus",
    # Inverse kinematics — Drake-style modular construction
    "IKProblem",
    "PoseTarget",
    "solve_problem",
    # Singularity metrics
    "condition_number",
    "inverse_condition_number",
    "manipulability",
    "min_singular_value",
    "singularity_report",
]
