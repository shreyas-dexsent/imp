# Copyright (c) 2026 DexSent Robotics Pvt Ltd.
# All Rights Reserved.
#
# SPDX-License-Identifier: LicenseRef-Proprietary
# See LICENSE file for details.
#
# Author: Shreyas <shreyas.kumar@dexsentrobotics.com>
# Proprietary and confidential.

"""Joint-position bounds constraint."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class JointPositionBounds:
    """Hard active-q bounds."""

    q_min: np.ndarray
    q_max: np.ndarray
    margin: float = 0.0

    def __post_init__(self) -> None:
        q_min = np.asarray(self.q_min, dtype=float)
        q_max = np.asarray(self.q_max, dtype=float)
        if q_min.shape != q_max.shape:
            raise ValueError("q_min and q_max must have the same shape")
        object.__setattr__(self, "q_min", q_min)
        object.__setattr__(self, "q_max", q_max)
