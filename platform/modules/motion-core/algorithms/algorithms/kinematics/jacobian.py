# Copyright (c) 2026 DexSent Robotics Pvt Ltd.
# All Rights Reserved.
#
# SPDX-License-Identifier: LicenseRef-Proprietary
# See LICENSE file for details.
#
# Author: Shreyas <shreyas.kumar@dexsentrobotics.com>
# Proprietary and confidential.

"""Frame Jacobian computation on a resolved kinematic model.

Returns a `(6, active_dof)` NumPy matrix mapping joint velocities to
a 6-vector twist of the requested resolved-model frame.

The frame can come from the robot URDF, the gripper URDF, or a YAML TCP
that was injected while building `KinematicModel`. The q vector must be
ordered exactly as `model.active_joint_names`. Mimic-follower columns are
folded into the driver column via `model.active_to_full`, keeping the
column count aligned with the user-facing q vector.

Reference frame conventions
---------------------------
* `"local"` - body-frame twist.
* `"world"` - world-axis translation and rotation.
* `"local_world_aligned"` - origin at body frame, axes aligned with
  world. Matches MoveIt convention; default.

A fresh `pin.Data` is allocated per call so concurrent or interleaved
multi-robot queries do not corrupt each other.
"""
from __future__ import annotations

from typing import Literal

import numpy as np

from algorithms.resolved.kinematic_model import KinematicModel


ReferenceFrame = Literal["local", "world", "local_world_aligned"]


def jacobian(
    model: KinematicModel,
    q: np.ndarray,
    frame_id: str,
    *,
    reference: ReferenceFrame = "local_world_aligned",
) -> np.ndarray:
    """Compute the 6-by-active-DOF Jacobian of a frame.

    Twist convention (rows of J):

    * rows 0-2 = linear velocity components
    * rows 3-5 = angular velocity components

    Columns are in `model.active_joint_names` order. To slice down to one
    chain's DOF, index by `model.chain_indices(chain_id)`.

    Parameters
    ----------
    model : KinematicModel
        Resolved kinematic model.
    q : np.ndarray
        Active-DOF joint values in `model.active_joint_names` order, shape
        `(len(model.active_joint_names),)`. This is the same q ordering used
        by FK.
    frame_id : str
        Resolved model frame name, such as `"fr3_link8"` or
        `"fr3_hand_tcp"`. The frame may come from the robot URDF, gripper
        URDF, or a YAML TCP injected during model resolution.
    reference : Literal["local", "world", "local_world_aligned"]
        Reference frame for the twist. Default `"local_world_aligned"`.

    Returns
    -------
    np.ndarray
        Shape `(6, len(model.active_joint_names))`.

    Raises
    ------
    KeyError
        If `frame_id` is not present in the resolved kinematic model.
    ValueError
        If `q` has the wrong shape or `reference` is not one of the
        documented values.
    """
    import pinocchio as pin

    if not model.pin_model.existFrame(frame_id):
        raise KeyError(f"frame not found in resolved kinematic model: {frame_id!r}")

    q_full = model.expand(q)
    data = model.pin_model.createData()

    pin.computeJointJacobians(model.pin_model, data, q_full)
    pin.updateFramePlacements(model.pin_model, data)

    frame_idx = model.pin_model.getFrameId(frame_id)
    ref = _to_pinocchio_reference(reference)
    J_full = pin.getFrameJacobian(model.pin_model, data, frame_idx, ref)

    # J_full has shape (6, full_dof). Map to active-DOF columns via the same
    # active_to_full matrix used for q expansion: dq_full = active_to_full @ dq_active
    # implies twist = J_full @ dq_full = (J_full @ active_to_full) @ dq_active.
    return np.asarray(J_full, dtype=float) @ model.active_to_full


def _to_pinocchio_reference(reference: ReferenceFrame):
    """Map a string reference to the Pinocchio enum value."""
    import pinocchio as pin

    mapping = {
        "local": pin.ReferenceFrame.LOCAL,
        "world": pin.ReferenceFrame.WORLD,
        "local_world_aligned": pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
    }
    if reference not in mapping:
        raise ValueError(
            f"unknown reference frame {reference!r}; expected one of {sorted(mapping)}"
        )
    return mapping[reference]
