# Copyright (c) 2026 DexSent Robotics Pvt Ltd.
# All Rights Reserved.
#
# SPDX-License-Identifier: LicenseRef-Proprietary
# See LICENSE file for details.
#
# Author: Shreyas <shreyas.kumar@dexsentrobotics.com>
# Proprietary and confidential.

"""Shared IK math helpers."""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from algorithms.kinematics.fk import fk_local
from algorithms.resolved.kinematic_model import KinematicModel


def pose_error_vector(
    model: KinematicModel,
    q: np.ndarray,
    frame_id: str,
    T_target: np.ndarray,
) -> np.ndarray:
    """Return `[position_error, rotation_vector_error]` in target/world axes."""
    T_current = fk_local(model, q, frame_id)
    p_err = np.asarray(T_target[:3, 3] - T_current[:3, 3], dtype=float)
    R_err = np.asarray(T_target[:3, :3] @ T_current[:3, :3].T, dtype=float)
    r_err = Rotation.from_matrix(R_err).as_rotvec()
    return np.concatenate([p_err, r_err])


def pose_error_norms(error: np.ndarray) -> tuple[float, float]:
    """Return `(position_norm, rotation_norm)` from a 6D pose error."""
    return float(np.linalg.norm(error[:3])), float(np.linalg.norm(error[3:]))


def ensure_rotation_matrix(T: np.ndarray) -> bool:
    """Return True if T has a finite, approximately orthonormal rotation."""
    T = np.asarray(T, dtype=float)
    if T.shape != (4, 4) or not np.all(np.isfinite(T)):
        return False
    R = T[:3, :3]
    if not np.allclose(R.T @ R, np.eye(3), atol=1e-6):
        return False
    return np.isclose(np.linalg.det(R), 1.0, atol=1e-6)
