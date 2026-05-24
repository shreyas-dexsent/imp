# Copyright (c) 2026 DexSent Robotics Pvt Ltd.
# All Rights Reserved.
#
# SPDX-License-Identifier: LicenseRef-Proprietary
# See LICENSE file for details.
#
# Author: Shreyas <shreyas.kumar@dexsentrobotics.com>
# Proprietary and confidential.

"""Path planning and path validation (Phase 6a + 6b).

Public entrypoints, parallel to FK / IK / collision:

* :func:`plan_joint` — joint-space path planning (OMPL or straight-line).
* :func:`plan_cartesian` — straight-line Cartesian TCP path with IK
  resolution and continuity checks.
* :func:`validate_path` — post-plan advisory validator (clearance,
  singularity, branch-jump, continuous collision).

The Drake-style modular path stays available for power users who want
to plug a different planner backend; see
:class:`algorithms.planning.backends.base.PathPlannerBackend`.
"""

from algorithms.planning.cartesian import plan_cartesian
from algorithms.planning.joint_space import plan_joint
from algorithms.planning.options import PathValidationOptions, PlanOptions
from algorithms.planning.path import Path, PathStatus
from algorithms.planning.result import (
    CheckResult,
    PathDiagnostics,
    PathPlanResult,
    PathValidationReport,
)
from algorithms.planning.state_validity import (
    make_composite_state_validity_fn,
    make_state_validity_fn,
)
from algorithms.planning.validator import validate_path

__all__ = [
    # Tier 1 — ergonomic entrypoints
    "plan_joint",
    "plan_cartesian",
    "validate_path",
    # Data types
    "Path",
    "PathStatus",
    "PathPlanResult",
    "PathDiagnostics",
    "PathValidationReport",
    "CheckResult",
    "PlanOptions",
    "PathValidationOptions",
    # State validity factories
    "make_state_validity_fn",
    "make_composite_state_validity_fn",
]
