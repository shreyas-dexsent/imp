# Copyright (c) 2026 DexSent Robotics Pvt Ltd.
# All Rights Reserved.
#
# SPDX-License-Identifier: LicenseRef-Proprietary
# See LICENSE file for details.
#
# Author: Shreyas <shreyas.kumar@dexsentrobotics.com>
# Proprietary and confidential.

"""Seed regularization cost."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SeedRegularization:
    """Soft cost keeping the solution near a preferred seed."""

    q_seed: np.ndarray
    weight: float = 1e-3

    def __post_init__(self) -> None:
        object.__setattr__(self, "q_seed", np.asarray(self.q_seed, dtype=float))
