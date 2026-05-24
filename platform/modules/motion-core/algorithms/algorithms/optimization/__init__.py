# Copyright (c) 2026 DexSent Robotics Pvt Ltd.
# All Rights Reserved.
#
# SPDX-License-Identifier: LicenseRef-Proprietary
# See LICENSE file for details.
#
# Author: Shreyas <shreyas.kumar@dexsentrobotics.com>
# Proprietary and confidential.

"""Path optimization (Phase 6c).

Geometric passes only. Every public function takes a `Path` and
returns a `Path`. No time, no velocity, no acceleration — those live
at the trajectory layer (Phase 6d).

Two routines ship:

* :func:`shortcut_smooth` — random shortcut smoothing with a
  collision-aware validity check.
* :func:`spline_fit` — quintic / cubic spline fit with Catmull-Rom
  finite-difference interior velocities.

Plus :func:`remove_redundant_waypoints` for trimming clusters from
OMPL output before downstream passes.
"""

from algorithms.optimization.shortcut import (
    ShortcutStats,
    remove_redundant_waypoints,
    shortcut_smooth,
)
from algorithms.optimization.spline import spline_fit

__all__ = [
    "ShortcutStats",
    "remove_redundant_waypoints",
    "shortcut_smooth",
    "spline_fit",
]
