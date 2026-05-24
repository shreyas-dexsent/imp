# Copyright (c) 2026 DexSent Robotics Pvt Ltd.
# All Rights Reserved.
#
# SPDX-License-Identifier: LicenseRef-Proprietary
# See LICENSE file for details.
#
# Author: Shreyas <shreyas.kumar@dexsentrobotics.com>
# Proprietary and confidential.

"""Options for inverse-kinematics solve and validation."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class IKOptions:
    """Configuration for pose IK.

    Tolerances and validation settings live here, not in YAML. The same
    physical descriptions can therefore be reused across different solver
    policies without editing static asset files.
    """

    pos_tol: float = 1e-4
    rot_tol: float = 1e-3
    joint_margin: float = 1e-3
    # `max_iters` is forwarded to scipy.optimize.least_squares as
    # `max_nfev` (function evaluations, not gradient steps). With the
    # analytical Jacobian supplied, one evaluation per iteration is the
    # norm, so 200 covers far-from-seed targets comfortably.
    max_iters: int = 200
    # Total wall-clock budget across all multi-start seeds in one solve.
    # 200 ms is enough for ~10 seeds with the analytical Jacobian on FR3;
    # bump higher when running larger seed lists or harder targets.
    max_time_ms: float = 200.0
    multi_start: bool = True
    num_random_seeds: int = 8
    random_seed: int = 0
    reject_singular: bool = True
    condition_number_limit: float = 1000.0
    min_sigma_limit: float = 1e-4
    validate_collision_if_available: bool = True
    return_all_candidates: bool = True
    # Soft costs act as tie-breakers between equally-valid pose solutions,
    # not as forces that compete with the pose objective. Keep these very
    # small. At larger values the solver converges on `ftol` to a
    # compromise between pose error and softness, and the pose tolerance
    # is never met — see git history for the regression we hit at 1e-3.
    seed_regularization_weight: float = 1e-6
    joint_centering_weight: float = 1e-7
    manipulability_weight: float = 0.0
    dls_damping: float = 1e-2
