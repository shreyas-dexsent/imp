# Copyright (c) 2026 DexSent Robotics Pvt Ltd.
# All Rights Reserved.
#
# SPDX-License-Identifier: LicenseRef-Proprietary
# See LICENSE file for details.
#
# Author: Shreyas <shreyas.kumar@dexsentrobotics.com>
# Proprietary and confidential.

"""Trajectory generation and validation (Phase 6d + 6e).

Two public entrypoints:

* :func:`time_parameterize` — assign timing to a Path; produces a
  Trajectory. Pass-through by default (no rest at interior waypoints).
* :func:`validate_trajectory` — dense-time validation of v/a/j envelopes,
  collision, TCP speed, controller-rate compatibility.

The :class:`Trajectory` data type exposes ``at(t)`` and
``sample(dt)`` for both one-off queries and streaming controller use.
"""

from algorithms.trajectory.options import (
    TimeParameterizationOptions,
    TrajectoryValidationOptions,
)
from algorithms.trajectory.result import (
    TrajectoryCheckResult,
    TrajectoryDiagnostics,
    TrajectoryResult,
    TrajectoryStatus,
    TrajectoryValidationReport,
)
from algorithms.trajectory.time_parameterize import time_parameterize
from algorithms.trajectory.trajectory import Trajectory
from algorithms.trajectory.validator import validate_trajectory

__all__ = [
    # Tier 1 — ergonomic entrypoints
    "time_parameterize",
    "validate_trajectory",
    # Data types
    "Trajectory",
    "TrajectoryStatus",
    "TrajectoryResult",
    "TrajectoryDiagnostics",
    "TrajectoryValidationReport",
    "TrajectoryCheckResult",
    "TimeParameterizationOptions",
    "TrajectoryValidationOptions",
]
