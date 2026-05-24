# Copyright (c) 2026 DexSent Robotics Pvt Ltd.
# All Rights Reserved.
#
# SPDX-License-Identifier: LicenseRef-Proprietary
# See LICENSE file for details.
#
# Author: Shreyas <shreyas.kumar@dexsentrobotics.com>
# Proprietary and confidential.

"""Options for motion primitives.

A `MoveOptions` is the knob bag every primitive accepts. It groups
planning, optimization, time-parameterization, and validation knobs
so callers tune one object instead of three.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from algorithms.planning.options import PathValidationOptions, PlanOptions
from algorithms.trajectory.options import (
    TimeParameterizationOptions,
    TrajectoryValidationOptions,
)


@dataclass(frozen=True)
class MoveOptions:
    """Knobs for every motion primitive.

    Defaults: OMPL planner, shortcut smoothing 200 iterations, quintic
    spline at 60 samples, auto trajectory backend (Ruckig if
    installed), pass-through interior velocities, dt = 1 ms, both
    validators run.
    """

    plan: PlanOptions = field(default_factory=PlanOptions)
    """Planner knobs (max_joint_step, OMPL planner name, budgets)."""

    time_parameterize: TimeParameterizationOptions = field(
        default_factory=TimeParameterizationOptions
    )
    """Trajectory knobs (backend, dt, v/a/j scale, pass-through vs rest-to-rest)."""

    path_validation: PathValidationOptions = field(default_factory=PathValidationOptions)
    """Path validator knobs."""

    trajectory_validation: TrajectoryValidationOptions = field(
        default_factory=TrajectoryValidationOptions
    )
    """Trajectory validator knobs."""

    # Pipeline toggles
    planner_backend: Literal["ompl", "direct"] = "ompl"
    """Planner backend for joint-space primitives. Use `"direct"` for
    simple known joint moves where the straight joint-space segment is
    expected to be valid."""

    smooth_path: bool = True
    """Apply shortcut smoothing to joint-space plans before time
    parameterization. Disabled for plan_cartesian outputs (would deviate
    from the requested Cartesian line)."""

    smoothing_iterations: int = 200
    """Shortcut iterations when `smooth_path=True`."""

    spline_fit: bool = True
    """Re-densify the path with a quintic spline after shortcut.
    Recommended for joint-space moves so the trajectory layer sees a
    C^2 path."""

    spline_samples: int = 60
    """Sample count for `spline_fit` when enabled."""

    validate_path: bool = True
    """Run `validate_path` after smoothing / spline-fitting."""

    validate_trajectory: bool = True
    """Run `validate_trajectory` after parameterization."""

    cartesian_backend: Literal["cartesian"] = "cartesian"
    """Reserved for future Cartesian backends (arc, spline)."""
