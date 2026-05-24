# Copyright (c) 2026 DexSent Robotics Pvt Ltd.
# All Rights Reserved.
#
# SPDX-License-Identifier: LicenseRef-Proprietary
# See LICENSE file for details.
#
# Author: Shreyas <shreyas.kumar@dexsentrobotics.com>
# Proprietary and confidential.

"""Generic constrained pose IK backend."""
from __future__ import annotations

import time

import numpy as np
from scipy.optimize import least_squares

from algorithms.kinematics.ik._math import pose_error_norms, pose_error_vector
from algorithms.kinematics.ik.backends.base import BackendCandidate, BackendResult
from algorithms.kinematics.ik.constraints import JointPositionBounds
from algorithms.kinematics.ik.costs import (
    JointCenteringCost,
    ManipulabilityCost,
    SeedRegularization,
)
from algorithms.kinematics.ik.options import IKOptions
from algorithms.kinematics.ik.problem import IKProblemSpec
from algorithms.kinematics.ik.result import IKStatus
from algorithms.kinematics.ik.seeds import generate_seeds
from algorithms.kinematics.jacobian import jacobian
from algorithms.kinematics.singularity import manipulability
from algorithms.resolved.kinematic_model import KinematicModel


class GenericConstrainedIK:
    """Default multi-start bounded nonlinear least-squares IK backend."""

    name = "generic"

    def solve(
        self,
        model: KinematicModel,
        spec: IKProblemSpec,
        q_seed: np.ndarray,
        options: IKOptions,
    ) -> BackendResult:
        start = time.perf_counter()
        target = spec.pose_target
        q_min, q_max = _bounds_for_spec(model, spec, options)
        q_home = _home_q(model)
        seeds = generate_seeds(
            model,
            q_seed,
            options,
            q_home=q_home,
            q_nominal=q_home,
        )

        candidates: list[BackendCandidate] = []
        seed_reports: list[dict[str, object]] = []
        total_iters = 0

        for seed_index, seed in enumerate(seeds):
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            if elapsed_ms > options.max_time_ms:
                return BackendResult(
                    status=IKStatus.TIMEOUT,
                    candidates=tuple(candidates),
                    iterations=total_iters,
                    elapsed_ms=elapsed_ms,
                    message="IK solve timed out before all seeds completed.",
                    seed_reports=tuple(seed_reports),
                )

            seed_err = pose_error_vector(model, seed, target.frame_id, target.T_target)
            seed_pose_error = pose_error_norms(seed_err)
            if seed_pose_error[0] <= target.pos_tol and seed_pose_error[1] <= target.rot_tol:
                candidate = BackendCandidate(
                    q=np.asarray(seed, dtype=float),
                    pose_error=seed_pose_error,
                    iterations=0,
                    seed_index=seed_index,
                    cost=float(np.linalg.norm(seed_err)),
                )
                candidates.append(candidate)
                seed_reports.append(
                    {
                        "seed_index": seed_index,
                        "success": True,
                        "status": 0,
                        "cost": float(np.linalg.norm(seed_err)),
                        "pose_error": seed_pose_error,
                        "message": "seed already satisfies pose tolerance",
                    }
                )
                if seed_index == 0:
                    elapsed_ms = (time.perf_counter() - start) * 1000.0
                    return BackendResult(
                        status=IKStatus.SUCCESS,
                        candidates=(candidate,),
                        iterations=0,
                        elapsed_ms=elapsed_ms,
                        message="current seed already satisfies pose tolerance",
                        seed_reports=tuple(seed_reports),
                    )
                continue

            residual_fn = lambda q: _residual(model, spec, q, q_seed, options)
            jacobian_fn = lambda q: _residual_jacobian(model, spec, q, q_seed, options)
            result = least_squares(
                residual_fn,
                np.clip(seed, q_min, q_max),
                jac=jacobian_fn,
                bounds=(q_min, q_max),
                method="trf",
                max_nfev=options.max_iters,
                xtol=1e-10,
                ftol=1e-10,
                gtol=1e-10,
            )
            total_iters += int(result.nfev)
            err = pose_error_vector(model, result.x, target.frame_id, target.T_target)
            pose_error = pose_error_norms(err)
            cost = float(np.linalg.norm(result.fun))
            seed_reports.append(
                {
                    "seed_index": seed_index,
                    "success": bool(result.success),
                    "status": int(result.status),
                    "cost": cost,
                    "pose_error": pose_error,
                    "message": str(result.message),
                }
            )

            if result.success:
                candidate = BackendCandidate(
                    q=np.asarray(result.x, dtype=float),
                    pose_error=pose_error,
                    iterations=int(result.nfev),
                    seed_index=seed_index,
                    cost=cost,
                )
                candidates.append(candidate)
                # Once a candidate meets pose tolerance, stop searching.
                # The validator may still reject (joint margin, singularity,
                # collision), but other seeds rarely beat a hit on pose,
                # and they eat the time budget that costs the call its
                # next-iteration fall-back.
                if (
                    pose_error[0] <= target.pos_tol
                    and pose_error[1] <= target.rot_tol
                ):
                    elapsed_ms = (time.perf_counter() - start) * 1000.0
                    return BackendResult(
                        status=IKStatus.SUCCESS,
                        candidates=tuple(candidates),
                        iterations=total_iters,
                        elapsed_ms=elapsed_ms,
                        message="pose tolerance met after NLS",
                        seed_reports=tuple(seed_reports),
                    )

        elapsed_ms = (time.perf_counter() - start) * 1000.0
        status = IKStatus.SUCCESS if candidates else IKStatus.UNREACHABLE
        return BackendResult(
            status=status,
            candidates=tuple(candidates),
            iterations=total_iters,
            elapsed_ms=elapsed_ms,
            message="generic IK completed",
            seed_reports=tuple(seed_reports),
        )


def _residual(
    model: KinematicModel,
    spec: IKProblemSpec,
    q: np.ndarray,
    q_seed: np.ndarray,
    options: IKOptions,
) -> np.ndarray:
    """Residual vector for the bounded NLS solver.

    Layout: ``[w_p * p_err (3,), w_r * r_err (3,), seed_reg (n,),
    joint_centering (n,), manipulability (1,)]``. The pose residual
    keeps its natural scale (metres / radians) so scipy's xtol/ftol/gtol
    behave predictably; pose tolerances are enforced by the validator
    after the solve, not by inflating the residual here.
    """
    target = spec.pose_target
    err = pose_error_vector(model, q, target.frame_id, target.T_target)
    residuals = [
        target.position_weight * err[:3],
        target.rotation_weight * err[3:],
    ]

    for cost in spec.costs:
        if isinstance(cost, SeedRegularization):
            residuals.append(np.sqrt(cost.weight) * (q - cost.q_seed))
        elif isinstance(cost, JointCenteringCost):
            q_min, q_max = model.active_position_limits()
            center = 0.5 * (q_min + q_max)
            span = np.maximum(q_max - q_min, 1e-9)
            residuals.append(np.sqrt(cost.weight) * ((q - center) / span))
        elif isinstance(cost, ManipulabilityCost) and cost.weight > 0:
            J = jacobian(model, q, target.frame_id)
            m = max(manipulability(J), 1e-12)
            residuals.append(np.array([np.sqrt(cost.weight) * -np.log(m)]))

    if not any(isinstance(c, SeedRegularization) for c in spec.costs):
        residuals.append(np.sqrt(options.seed_regularization_weight) * (q - q_seed))

    return np.concatenate(residuals)


def _residual_jacobian(
    model: KinematicModel,
    spec: IKProblemSpec,
    q: np.ndarray,
    q_seed: np.ndarray,
    options: IKOptions,
) -> np.ndarray:
    """Analytical Jacobian of `_residual` w.r.t. `q`.

    The pose-residual block is `-J(q)` (the negative robot Jacobian at
    the target frame) because the residual is `p_target - p(q)`. Cost
    blocks are linear in `q` so their Jacobians are constant.

    Supplying this to `scipy.optimize.least_squares` is the single
    biggest performance win for the default backend: it replaces an
    `O(n+1)` finite-difference Jacobian per iteration with one
    Pinocchio call.
    """
    target = spec.pose_target
    n = q.size

    J_robot = jacobian(model, q, target.frame_id)
    blocks: list[np.ndarray] = [
        -target.position_weight * J_robot[:3, :],
        -target.rotation_weight * J_robot[3:, :],
    ]

    for cost in spec.costs:
        if isinstance(cost, SeedRegularization):
            blocks.append(np.sqrt(cost.weight) * np.eye(n))
        elif isinstance(cost, JointCenteringCost):
            q_min, q_max = model.active_position_limits()
            span = np.maximum(q_max - q_min, 1e-9)
            blocks.append(np.sqrt(cost.weight) * np.diag(1.0 / span))
        elif isinstance(cost, ManipulabilityCost) and cost.weight > 0:
            # Analytical d/dq log(sqrt(det(J J^T))) is expensive; fall
            # back to a one-row finite difference for the rare case
            # someone enables this cost.
            row = np.zeros((1, n))
            base = manipulability(J_robot)
            eps = 1e-6
            for i in range(n):
                q_eps = q.copy()
                q_eps[i] += eps
                J_eps = jacobian(model, q_eps, target.frame_id)
                m = max(manipulability(J_eps), 1e-12)
                row[0, i] = -(np.log(m) - np.log(max(base, 1e-12))) / eps
            blocks.append(np.sqrt(cost.weight) * row)

    if not any(isinstance(c, SeedRegularization) for c in spec.costs):
        blocks.append(np.sqrt(options.seed_regularization_weight) * np.eye(n))

    return np.vstack(blocks)


def _bounds_for_spec(
    model: KinematicModel,
    spec: IKProblemSpec,
    options: IKOptions,
) -> tuple[np.ndarray, np.ndarray]:
    q_min, q_max = model.active_position_limits()
    margin = options.joint_margin
    for constraint in spec.constraints:
        if isinstance(constraint, JointPositionBounds):
            q_min = np.maximum(q_min, constraint.q_min + constraint.margin)
            q_max = np.minimum(q_max, constraint.q_max - constraint.margin)
    return q_min + margin, q_max - margin


def _home_q(model: KinematicModel) -> np.ndarray | None:
    try:
        home = model.system.named_joint_state("home")
    except KeyError:
        return None
    values = []
    for name in model.active_joint_names:
        if name not in home:
            return None
        values.append(home[name])
    return np.asarray(values, dtype=float)
