# Copyright (c) 2026 DexSent Robotics Pvt Ltd.
# All Rights Reserved.
#
# SPDX-License-Identifier: LicenseRef-Proprietary
# See LICENSE file for details.
#
# Author: Shreyas <shreyas.kumar@dexsentrobotics.com>
# Proprietary and confidential.

"""IK backend dispatch."""
from __future__ import annotations

from algorithms.kinematics.ik.backends.analytical import (
    OPWIK,
    SphericalWrist6RIK,
    lookup,
)
from algorithms.kinematics.ik.backends.dls import DLSIK
from algorithms.kinematics.ik.backends.generic import GenericConstrainedIK
from algorithms.kinematics.ik.backends.qp_velocity import QPVelocityIK
from algorithms.resolved.kinematic_model import KinematicModel


def choose_backend(model: KinematicModel, backend_hint: str | None):
    """Choose an IK backend using the locked dispatch order."""
    if backend_hint == "opw":
        return OPWIK()
    if backend_hint == "spherical_wrist_6r":
        return SphericalWrist6RIK()
    if backend_hint == "dls":
        return DLSIK()
    if backend_hint == "qp_velocity":
        raise ValueError("qp_velocity is not a pose-IK backend; use ik_velocity")
    if backend_hint is not None:
        raise ValueError(f"unknown IK backend: {backend_hint!r}")

    registered = lookup(model.system.robot.id)
    if registered is not None:
        return registered()

    return GenericConstrainedIK()


def velocity_backend() -> QPVelocityIK:
    """Return the Cartesian velocity IK backend."""
    return QPVelocityIK()
