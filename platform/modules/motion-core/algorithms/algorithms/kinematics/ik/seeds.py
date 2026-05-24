# Copyright (c) 2026 DexSent Robotics Pvt Ltd.
# All Rights Reserved.
#
# SPDX-License-Identifier: LicenseRef-Proprietary
# See LICENSE file for details.
#
# Author: Shreyas <shreyas.kumar@dexsentrobotics.com>
# Proprietary and confidential.

"""Deterministic multi-start seed generation."""
from __future__ import annotations

import numpy as np

from algorithms.resolved.kinematic_model import KinematicModel
from algorithms.kinematics.ik.options import IKOptions


def generate_seeds(
    model: KinematicModel,
    q_current: np.ndarray,
    options: IKOptions,
    *,
    q_home: np.ndarray | None = None,
    q_last_success: np.ndarray | None = None,
    q_nominal: np.ndarray | None = None,
) -> list[np.ndarray]:
    """Generate deterministic bounded seeds in active-joint order."""
    q_current = np.asarray(q_current, dtype=float)
    q_min, q_max = model.active_position_limits()
    q_center = 0.5 * (q_min + q_max)

    seeds: list[np.ndarray] = []
    for candidate in (q_current, q_last_success, q_home, q_center, q_nominal):
        if candidate is None:
            continue
        arr = np.asarray(candidate, dtype=float)
        if arr.shape == q_current.shape:
            seeds.append(np.clip(arr, q_min, q_max))

    if options.multi_start:
        rng = np.random.default_rng(options.random_seed)
        for _ in range(options.num_random_seeds):
            seeds.append(rng.uniform(q_min, q_max))

    unique: list[np.ndarray] = []
    for seed in seeds:
        if not any(np.allclose(seed, existing, atol=1e-12) for existing in unique):
            unique.append(seed)
    return unique
