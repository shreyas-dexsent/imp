# Copyright (c) 2026 DexSent Robotics Pvt Ltd.
# All Rights Reserved.
#
# SPDX-License-Identifier: LicenseRef-Proprietary
# See LICENSE file for details.
#
# Author: Shreyas <shreyas.kumar@dexsentrobotics.com>
# Proprietary and confidential.

"""Velocity IK for Cartesian servoing."""
from __future__ import annotations

import numpy as np
from scipy.optimize import lsq_linear

from algorithms.kinematics.jacobian import jacobian
from algorithms.resolved.kinematic_model import KinematicModel


class QPVelocityIK:
    """Small bounded least-squares velocity IK backend.

    This is not pose IK. It emits `qdot` for a servo loop; the caller owns
    live safety validation.
    """

    name = "qp_velocity"

    def solve_velocity(
        self,
        model: KinematicModel,
        frame_id: str,
        target_twist: np.ndarray,
        q_current: np.ndarray,
        *,
        dt: float = 0.01,
    ) -> np.ndarray:
        """Return bounded qdot that best matches a desired 6D frame twist."""
        target_twist = np.asarray(target_twist, dtype=float)
        if target_twist.shape != (6,):
            raise ValueError(f"target_twist must have shape (6,); got {target_twist.shape}")

        q_current = np.asarray(q_current, dtype=float)
        q_min, q_max = model.active_position_limits()
        v_lim = model.active_velocity_limits()
        J = jacobian(model, q_current, frame_id)

        lower = np.maximum(-v_lim, (q_min - q_current) / dt)
        upper = np.minimum(v_lim, (q_max - q_current) / dt)
        result = lsq_linear(J, target_twist, bounds=(lower, upper), lsmr_tol="auto")
        return np.asarray(result.x, dtype=float)
