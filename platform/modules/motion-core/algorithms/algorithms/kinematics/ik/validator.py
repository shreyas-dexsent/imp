# Copyright (c) 2026 DexSent Robotics Pvt Ltd.
# All Rights Reserved.
#
# SPDX-License-Identifier: LicenseRef-Proprietary
# See LICENSE file for details.
#
# Author: Shreyas <shreyas.kumar@dexsentrobotics.com>
# Proprietary and confidential.

"""IK candidate validation pipeline."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from algorithms.kinematics.ik._math import (
    ensure_rotation_matrix,
    pose_error_norms,
    pose_error_vector,
)
from algorithms.kinematics.ik.options import IKOptions
from algorithms.kinematics.ik.problem import IKProblemSpec
from algorithms.kinematics.ik.result import IKStatus
from algorithms.kinematics.jacobian import jacobian
from algorithms.kinematics.singularity import condition_number, min_singular_value
from algorithms.resolved.kinematic_model import KinematicModel
from algorithms.resolved.scene import Scene


@dataclass(frozen=True)
class ValidationReport:
    """Result of validating one candidate q."""

    status: IKStatus
    pose_error: tuple[float, float]
    checks: tuple[dict[str, Any], ...]
    message: str = ""

    @property
    def success(self) -> bool:
        return self.status is IKStatus.SUCCESS


def validate(
    model: KinematicModel,
    spec: IKProblemSpec,
    q_candidate: np.ndarray,
    options: IKOptions,
    *,
    scene: Scene | None = None,
) -> ValidationReport:
    """Validate one IK candidate against the v1 post-solve contract."""
    checks: list[dict[str, Any]] = []
    target = spec.pose_target

    if not model.pin_model.existFrame(target.frame_id):
        return _fail(
            IKStatus.INVALID_INPUT,
            checks,
            "frame not found in resolved kinematic model",
        )

    if not ensure_rotation_matrix(target.T_target):
        return _fail(IKStatus.INVALID_INPUT, checks, "target pose is invalid")

    q = np.asarray(q_candidate, dtype=float)
    if q.shape != (len(model.active_joint_names),):
        return _fail(IKStatus.INVALID_INPUT, checks, f"q has wrong shape {q.shape}")

    if not np.all(np.isfinite(q)):
        return _fail(IKStatus.NUMERICAL_FAILURE, checks, "q contains non-finite values")
    checks.append({"name": "finite_q", "ok": True})

    q_min, q_max = model.active_position_limits()
    if np.any(q < q_min + options.joint_margin) or np.any(
        q > q_max - options.joint_margin
    ):
        return _fail(
            IKStatus.JOINT_LIMIT_VIOLATION,
            checks,
            "q violates position limits or joint margin",
        )
    checks.append({"name": "joint_limits", "ok": True})

    err = pose_error_vector(model, q, target.frame_id, target.T_target)
    pose_error = pose_error_norms(err)
    if pose_error[0] > target.pos_tol:
        return _fail(
            IKStatus.POSE_ERROR_TOO_HIGH,
            checks,
            "position error exceeds tolerance",
            pose_error=pose_error,
        )
    checks.append({"name": "position_error", "ok": True, "value": pose_error[0]})

    if pose_error[1] > target.rot_tol:
        return _fail(
            IKStatus.POSE_ERROR_TOO_HIGH,
            checks,
            "orientation error exceeds tolerance",
            pose_error=pose_error,
        )
    checks.append({"name": "orientation_error", "ok": True, "value": pose_error[1]})

    if options.reject_singular:
        J = jacobian(model, q, target.frame_id)
        min_sigma = min_singular_value(J)
        cond = condition_number(J)
        if min_sigma < options.min_sigma_limit or cond > options.condition_number_limit:
            return _fail(
                IKStatus.SINGULARITY_RISK,
                checks,
                "candidate is near singular",
                pose_error=pose_error,
                min_sigma=min_sigma,
                condition_number=cond,
            )
        checks.append(
            {
                "name": "singularity",
                "ok": True,
                "min_sigma": min_sigma,
                "condition_number": cond,
            }
        )

    if (
        options.validate_collision_if_available
        and scene is not None
        and scene.collision_model is not None
    ):
        from algorithms.collision import is_in_collision

        report = is_in_collision(model, scene, q)
        if report.in_collision:
            return _fail(
                IKStatus.FINAL_COLLISION,
                checks,
                "candidate is in final collision",
                pose_error=pose_error,
                checked_pairs=report.checked_pairs,
            )
        checks.append({"name": "collision", "ok": True})

    return ValidationReport(
        status=IKStatus.SUCCESS,
        pose_error=pose_error,
        checks=tuple(checks),
        message="candidate validated",
    )


def _fail(
    status: IKStatus,
    checks: list[dict[str, Any]],
    message: str,
    *,
    pose_error: tuple[float, float] = (float("inf"), float("inf")),
    **extra: Any,
) -> ValidationReport:
    checks.append({"name": status.value, "ok": False, "message": message, **extra})
    return ValidationReport(
        status=status,
        pose_error=pose_error,
        checks=tuple(checks),
        message=message,
    )
