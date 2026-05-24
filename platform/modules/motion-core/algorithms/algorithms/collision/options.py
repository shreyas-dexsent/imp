# Copyright (c) 2026 DexSent Robotics Pvt Ltd.
# All Rights Reserved.
#
# SPDX-License-Identifier: LicenseRef-Proprietary
# See LICENSE file for details.
#
# Author: Shreyas <shreyas.kumar@dexsentrobotics.com>
# Proprietary and confidential.

"""Options for collision query operations."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class CollisionOptions:
    """Options shared by discrete contact and distance queries."""

    stop_at_first_contact: bool = True
    collect_contacts: bool = False
    broadphase: Literal["naive"] = "naive"


@dataclass(frozen=True)
class EdgeCollisionOptions:
    """Options for sampled edge collision checking."""

    method: Literal["sampled"] = "sampled"
    max_joint_step: float = 0.02
    include_endpoints: bool = True
