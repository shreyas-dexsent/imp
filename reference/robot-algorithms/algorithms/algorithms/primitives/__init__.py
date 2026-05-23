# Copyright (c) 2026 DexSent Robotics Pvt Ltd.
# All Rights Reserved.
#
# SPDX-License-Identifier: LicenseRef-Proprietary
# See LICENSE file for details.
#
# Author: Shreyas <shreyas.kumar@dexsentrobotics.com>
# Proprietary and confidential.

"""Motion primitives (Phase 7).

High-level "do this motion" verbs. Each primitive composes Phase 5 IK,
Phase 6 planning, optimization, and trajectory generation into one
call that returns a validated `Trajectory`.

Five primitives ship in v1:

* :func:`move_joint` — joint-space goto with shortcut + spline + Ruckig.
* :func:`move_l` — linear Cartesian goto (MoveL). No smoothing (would
  deviate from the line).
* :func:`approach` — linear descent from a pre-approach pose to a
  target pose.
* :func:`retreat` — linear lift-off from the current TCP pose.
* :func:`via_motion` — smooth pass-through motion across a sequence of
  joint-space via-points.

All return a :class:`MoveResult` whose ``trajectory`` is the
controller-ready output on success. Diagnostics carry the intermediate
artifacts (path, IK result, plan result, validators) for inspection.
"""

from algorithms.primitives.approach import approach, pre_approach_pose
from algorithms.primitives.move_joint import move_joint
from algorithms.primitives.move_l import move_l
from algorithms.primitives.options import MoveOptions
from algorithms.primitives.result import MoveDiagnostics, MoveResult, MoveStatus
from algorithms.primitives.retreat import retreat
from algorithms.primitives.via_motion import via_motion

__all__ = [
    # Primitives
    "move_joint",
    "move_l",
    "approach",
    "retreat",
    "via_motion",
    # Helpers
    "pre_approach_pose",
    # Data types
    "MoveOptions",
    "MoveResult",
    "MoveStatus",
    "MoveDiagnostics",
]
