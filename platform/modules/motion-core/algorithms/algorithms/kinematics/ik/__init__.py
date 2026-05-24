# Copyright (c) 2026 DexSent Robotics Pvt Ltd.
# All Rights Reserved.
#
# SPDX-License-Identifier: LicenseRef-Proprietary
# See LICENSE file for details.
#
# Author: Shreyas <shreyas.kumar@dexsentrobotics.com>
# Proprietary and confidential.

"""Inverse kinematics package.

Three public functions cover every use case:

* :func:`ik_local` — pose IK in the robot's base frame.
* :func:`ik` — pose IK in the world frame.
* :func:`ik_velocity` — Cartesian velocity IK for servo loops.

For Drake-style modular problems (custom constraints, custom costs),
build an :class:`IKProblem` and call :func:`solve_problem` directly.
"""

from algorithms.kinematics.ik.backends.analytical import (
    AnalyticalIK,
    lookup_analytical,
    register_analytical,
)
from algorithms.kinematics.ik.constraints import JointPositionBounds, PoseTarget
from algorithms.kinematics.ik.costs import (
    JointCenteringCost,
    ManipulabilityCost,
    SeedRegularization,
)
from algorithms.kinematics.ik.options import IKOptions
from algorithms.kinematics.ik.problem import IKProblem, IKProblemSpec
from algorithms.kinematics.ik.result import (
    IKCandidate,
    IKDiagnostics,
    IKResult,
    IKStatus,
)
from algorithms.kinematics.ik.solver import (
    ik,
    ik_local,
    ik_velocity,
    solve_problem,
)
from algorithms.kinematics.ik.validator import ValidationReport, validate

__all__ = [
    # Tier 1 — ergonomic entrypoints
    "ik",
    "ik_local",
    "ik_velocity",
    # Result + status taxonomy
    "IKResult",
    "IKStatus",
    "IKCandidate",
    "IKDiagnostics",
    "IKOptions",
    # Validator (public so external solvers can be checked against the
    # same acceptance rules)
    "ValidationReport",
    "validate",
    # Tier 3 — Drake-style modular construction
    "IKProblem",
    "IKProblemSpec",
    "PoseTarget",
    "JointPositionBounds",
    "JointCenteringCost",
    "ManipulabilityCost",
    "SeedRegularization",
    "solve_problem",
    # Analytical IK registry
    "AnalyticalIK",
    "register_analytical",
    "lookup_analytical",
]
