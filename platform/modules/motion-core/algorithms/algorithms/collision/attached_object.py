# Copyright (c) 2026 DexSent Robotics Pvt Ltd.
# All Rights Reserved.
#
# SPDX-License-Identifier: LicenseRef-Proprietary
# See LICENSE file for details.
#
# Author: Shreyas <shreyas.kumar@dexsentrobotics.com>
# Proprietary and confidential.

"""Attached-object pose helpers."""
from __future__ import annotations

import numpy as np

from algorithms.kinematics.fk import fk_local
from algorithms.resolved.kinematic_model import KinematicModel
from algorithms.resolved.scene import AttachedObject


def attached_object_world_pose(
    model: KinematicModel,
    q: np.ndarray,
    attached: AttachedObject,
) -> np.ndarray:
    """Return the attached object's 4x4 pose in the model/base frame.

    The name is kept as `world_pose` because callers use it after
    composing the robot base into the query world. For the low-level
    helper, `model` has no `Scene`, so the returned pose is relative to
    the resolved model's base frame.
    """
    T_base_parent = fk_local(model, q, attached.parent_frame)
    return T_base_parent @ np.asarray(attached.T_parent_obj, dtype=float)
