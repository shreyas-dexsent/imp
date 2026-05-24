# Copyright (c) 2026 DexSent Robotics Pvt Ltd.
# All Rights Reserved.
#
# SPDX-License-Identifier: LicenseRef-Proprietary
# See LICENSE file for details.
#
# Author: Shreyas <shreyas.kumar@dexsentrobotics.com>
# Proprietary and confidential.

"""algorithms - robotics algorithms layer for industrial manipulation stacks.

The package is organised in three layers (see `docs/architecture.md`):

* `descriptions` - YAML pydantic models. Parse and validate; no computation.
* `resolved`     - Heavy objects built once from descriptions
                     (KinematicModel, CollisionModel, Scene).
* `kinematics`   - Stateless kinematic operations on resolved objects
                     (FK, Jacobian, singularity metrics, IK).
* `collision`    - Stateless low-level geometry queries on resolved models
                     and runtime Scene state.
* `planning`     - Joint-space and Cartesian path planning.
* `optimization` - Geometric path cleanup.
* `trajectory`   - Time-parameterization and dense trajectory validation.
* `primitives`   - High-level motion commands composed from lower layers.

Robot drivers, controller sessions, UI synchronization, and runtime monitoring
remain application concerns. See `docs/architecture.md` for the long form.
"""

from algorithms.descriptions import RobotSystemDescription, WorldDescription
from algorithms.optimization import (
    remove_redundant_waypoints,
    shortcut_smooth,
    spline_fit,
)
from algorithms.planning import (
    Path,
    PathPlanResult,
    PathStatus,
    PathValidationOptions,
    PathValidationReport,
    PlanOptions,
    plan_cartesian,
    plan_joint,
    validate_path,
)
from algorithms.primitives import (
    MoveOptions,
    MoveResult,
    MoveStatus,
    approach,
    move_joint,
    move_l,
    pre_approach_pose,
    retreat,
    via_motion,
)
from algorithms.resolved import CollisionModel, KinematicModel, Scene
from algorithms.trajectory import (
    TimeParameterizationOptions,
    Trajectory,
    TrajectoryResult,
    TrajectoryStatus,
    TrajectoryValidationOptions,
    TrajectoryValidationReport,
    time_parameterize,
    validate_trajectory,
)

__all__ = [
    "CollisionModel",
    "KinematicModel",
    "MoveOptions",
    "MoveResult",
    "MoveStatus",
    "Path",
    "PathPlanResult",
    "PathStatus",
    "PathValidationOptions",
    "PathValidationReport",
    "PlanOptions",
    "RobotSystemDescription",
    "Scene",
    "TimeParameterizationOptions",
    "Trajectory",
    "TrajectoryResult",
    "TrajectoryStatus",
    "TrajectoryValidationOptions",
    "TrajectoryValidationReport",
    "WorldDescription",
    "approach",
    "move_joint",
    "move_l",
    "plan_cartesian",
    "plan_joint",
    "pre_approach_pose",
    "remove_redundant_waypoints",
    "retreat",
    "shortcut_smooth",
    "spline_fit",
    "time_parameterize",
    "validate_path",
    "validate_trajectory",
    "via_motion",
]
