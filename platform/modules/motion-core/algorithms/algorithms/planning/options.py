# Copyright (c) 2026 DexSent Robotics Pvt Ltd.
# All Rights Reserved.
#
# SPDX-License-Identifier: LicenseRef-Proprietary
# See LICENSE file for details.
#
# Author: Shreyas <shreyas.kumar@dexsentrobotics.com>
# Proprietary and confidential.

"""Configuration knobs for path planning and validation.

Two frozen dataclasses, both intentionally flat (no nested options) so
they are trivially serialisable and easy to tweak per call.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PlanOptions:
    """Options for `plan_joint` and `plan_cartesian`.

    Defaults are starting points. The benchmark script
    (`scripts/bench_planning.py`) is the source of truth for the
    numbers that ship as defaults.
    """

    # ---- Joint-space planner ----
    max_joint_step: float = 0.05
    """Edge / waypoint sampling granularity in radians (or metres for
    prismatic). Smaller = tighter validity checks, larger = faster."""

    max_iterations: int = 5000
    """OMPL node cap. Higher = more search depth."""

    max_time_ms: float = 2000.0
    """Wall-clock budget for the whole solve, in milliseconds."""

    goal_bias: float = 0.10
    """OMPL goal-sampling probability. Higher = converges faster on easy
    problems, worse on cluttered ones."""

    interpolation_waypoints: int = 100
    """After OMPL solves, interpolate the path to this many waypoints
    for a smoother handoff to the optimizer."""

    # ---- Cartesian planner ----
    cartesian_translation_step: float = 0.005
    """Sample resolution along the TCP path in metres."""

    cartesian_rotation_step: float = 0.02
    """Sample resolution along the TCP rotation in radians (slerp angle
    per sample)."""

    cartesian_ik_continuity: float = 0.5
    """Maximum allowed `|q_new - q_prev|_inf` between consecutive
    Cartesian samples. Larger jumps indicate a branch flip and the
    planner returns `IK_DISCONTINUITY`."""

    cartesian_line_tolerance: float = 5e-4
    """Maximum allowed perpendicular deviation of any sampled frame
    from the start->goal line, in metres."""

    # ---- Common ----
    joint_margin: float = 1e-3
    """Validity-check margin inside joint limits. Same semantics as the
    IK validator's joint_margin."""

    random_seed: int = 0
    """RNG seed for OMPL and other sampling. Pass `None` (via
    `replace(random_seed=None)`) for nondeterministic runs."""

    planner_name: str = "RRTConnect"
    """OMPL planner choice. Other useful values: 'RRT', 'PRM',
    'RRTstar', 'BIT*'. RRTConnect is the locked default."""


@dataclass(frozen=True)
class PathValidationOptions:
    """Options for `validate_path(...)`."""

    joint_margin: float = 1e-3
    """Validity margin inside joint limits."""

    collision_step: float = 0.02
    """Continuous-collision sweep resolution between consecutive
    waypoints. Smaller = denser check, slower."""

    min_clearance: float = 0.0
    """Minimum allowed clearance to any obstacle. Default 0 (touching
    permitted). Set higher for safety-conservative scenes."""

    condition_number_limit: float = 1000.0
    """Max acceptable cond(J) at any waypoint. Same number as the IK
    validator uses for SUCCESS."""

    min_sigma_limit: float = 1e-4
    """Min acceptable sigma_min(J) at any waypoint."""

    branch_jump_joint_threshold: float = 0.5
    """Joint-space jump > this with TCP delta below the TCP threshold
    is flagged as a branch jump (IK flipped to a different elbow / wrist
    posture)."""

    branch_jump_tcp_threshold: float = 0.05
    """TCP-position delta below which a large joint jump is suspicious."""

    reject_singular: bool = True
    """Whether to enforce the singularity thresholds."""

    nominal_segment_time: float | None = None
    """If set, validator estimates the per-joint velocity needed and
    flags segments where it exceeds the model's velocity limits."""
