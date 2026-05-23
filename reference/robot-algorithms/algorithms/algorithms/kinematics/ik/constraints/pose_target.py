# Copyright (c) 2026 DexSent Robotics Pvt Ltd.
# All Rights Reserved.
#
# SPDX-License-Identifier: LicenseRef-Proprietary
# See LICENSE file for details.
#
# Author: Shreyas <shreyas.kumar@dexsentrobotics.com>
# Proprietary and confidential.

"""Pose target task for inverse kinematics."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PoseTarget:
    """Target pose for a resolved model frame."""

    frame_id: str
    T_target: np.ndarray
    pos_tol: float = 1e-4
    rot_tol: float = 1e-3
    position_weight: float = 1.0
    rotation_weight: float = 1.0
    chain_id: str | None = None

    def __post_init__(self) -> None:
        T = np.asarray(self.T_target, dtype=float)
        if T.shape != (4, 4):
            raise ValueError(f"T_target must be shape (4,4); got {T.shape}")
        object.__setattr__(self, "T_target", T)
