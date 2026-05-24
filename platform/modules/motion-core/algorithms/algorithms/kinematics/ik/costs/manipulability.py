# Copyright (c) 2026 DexSent Robotics Pvt Ltd.
# All Rights Reserved.
#
# SPDX-License-Identifier: LicenseRef-Proprietary
# See LICENSE file for details.
#
# Author: Shreyas <shreyas.kumar@dexsentrobotics.com>
# Proprietary and confidential.

"""Manipulability-related IK soft costs."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ManipulabilityCost:
    """Optional soft cost preferring well-conditioned Jacobians."""

    weight: float = 1e-3


@dataclass(frozen=True)
class SingularValuePenalty:
    """Experimental singular-value penalty.

    TODO(next-engineer): only enable after rigorous edge-case testing on
    near-singular configurations.
    """

    weight: float = 1e-3
    epsilon: float = 1e-6
    enabled: bool = False
