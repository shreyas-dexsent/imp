# Copyright (c) 2026 DexSent Robotics Pvt Ltd.
# All Rights Reserved.
#
# SPDX-License-Identifier: LicenseRef-Proprietary
# See LICENSE file for details.
#
# Author: Shreyas <shreyas.kumar@dexsentrobotics.com>
# Proprietary and confidential.

"""Joint-centering soft cost."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class JointCenteringCost:
    """Soft cost preferring the middle of joint limits."""

    weight: float = 1e-4
