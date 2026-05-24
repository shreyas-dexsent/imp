# Copyright (c) 2026 DexSent Robotics Pvt Ltd.
# All Rights Reserved.
#
# SPDX-License-Identifier: LicenseRef-Proprietary
# See LICENSE file for details.
#
# Author: Shreyas <shreyas.kumar@dexsentrobotics.com>
# Proprietary and confidential.

"""Registry for robot-specific analytical IK backends."""
from __future__ import annotations

from typing import Type

from algorithms.kinematics.ik.backends.analytical.base import AnalyticalIK


_REGISTRY: dict[str, Type[AnalyticalIK]] = {}


def register(robot_id: str, backend_cls: Type[AnalyticalIK]) -> None:
    """Register a robot-specific analytical backend."""
    _REGISTRY[robot_id] = backend_cls


def lookup(robot_id: str) -> Type[AnalyticalIK] | None:
    """Return the registered backend class for a robot id, if any."""
    return _REGISTRY.get(robot_id)


def clear() -> None:
    """Clear the registry. Intended for tests."""
    _REGISTRY.clear()
