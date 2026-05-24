# Copyright (c) 2026 DexSent Robotics Pvt Ltd.
# All Rights Reserved.
#
# SPDX-License-Identifier: LicenseRef-Proprietary
# See LICENSE file for details.
#
# Author: Shreyas <shreyas.kumar@dexsentrobotics.com>
# Proprietary and confidential.

"""Damped least-squares IK backend."""
from __future__ import annotations

import time

import numpy as np

from algorithms.kinematics.ik._math import pose_error_norms, pose_error_vector
from algorithms.kinematics.ik.backends.base import BackendCandidate, BackendResult
from algorithms.kinematics.ik.options import IKOptions
from algorithms.kinematics.ik.problem import IKProblemSpec
from algorithms.kinematics.ik.result import IKStatus
from algorithms.kinematics.jacobian import jacobian
from algorithms.resolved.kinematic_model import KinematicModel


class DLSIK:
    """Opt-in damped pseudoinverse backend."""

    name = "dls"

    def solve(
        self,
        model: KinematicModel,
        spec: IKProblemSpec,
        q_seed: np.ndarray,
        options: IKOptions,
    ) -> BackendResult:
        start = time.perf_counter()
        target = spec.pose_target
        q = np.asarray(q_seed, dtype=float).copy()
        q_min, q_max = model.active_position_limits()

        for iteration in range(options.max_iters):
            err = pose_error_vector(model, q, target.frame_id, target.T_target)
            pos_err, rot_err = pose_error_norms(err)
            if pos_err <= target.pos_tol and rot_err <= target.rot_tol:
                return BackendResult(
                    status=IKStatus.SUCCESS,
                    candidates=(
                        BackendCandidate(
                            q=q,
                            pose_error=(pos_err, rot_err),
                            iterations=iteration,
                            cost=float(np.linalg.norm(err)),
                        ),
                    ),
                    iterations=iteration,
                    elapsed_ms=(time.perf_counter() - start) * 1000.0,
                    message="DLS converged",
                )

            J = jacobian(model, q, target.frame_id)
            damping = options.dls_damping
            lhs = J @ J.T + (damping * damping) * np.eye(6)
            dq = J.T @ np.linalg.solve(lhs, err)
            q = np.clip(q + dq, q_min, q_max)

            if (time.perf_counter() - start) * 1000.0 > options.max_time_ms:
                return BackendResult(
                    status=IKStatus.TIMEOUT,
                    candidates=(),
                    iterations=iteration + 1,
                    elapsed_ms=(time.perf_counter() - start) * 1000.0,
                    message="DLS timed out",
                )

        err = pose_error_vector(model, q, target.frame_id, target.T_target)
        return BackendResult(
            status=IKStatus.MAX_ITERATIONS,
            candidates=(
                BackendCandidate(
                    q=q,
                    pose_error=pose_error_norms(err),
                    iterations=options.max_iters,
                    cost=float(np.linalg.norm(err)),
                ),
            ),
            iterations=options.max_iters,
            elapsed_ms=(time.perf_counter() - start) * 1000.0,
            message="DLS reached max iterations",
        )
