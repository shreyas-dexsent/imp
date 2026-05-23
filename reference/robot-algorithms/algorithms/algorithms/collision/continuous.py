# Copyright (c) 2026 DexSent Robotics Pvt Ltd.
# All Rights Reserved.
#
# SPDX-License-Identifier: LicenseRef-Proprietary
# See LICENSE file for details.
#
# Author: Shreyas <shreyas.kumar@dexsentrobotics.com>
# Proprietary and confidential.

"""Sampled edge collision checking."""
from __future__ import annotations

import math

import numpy as np

from algorithms.collision.checker import is_in_collision
from algorithms.collision.options import CollisionOptions, EdgeCollisionOptions
from algorithms.collision.types import EdgeCollisionReport
from algorithms.resolved.kinematic_model import KinematicModel
from algorithms.resolved.scene import Scene


def check_edge_collision(
    model: KinematicModel,
    scene: Scene,
    q_a: np.ndarray,
    q_b: np.ndarray,
    *,
    chain_id: str | None = None,
    options: EdgeCollisionOptions | None = None,
) -> EdgeCollisionReport:
    """Check a joint-space edge by sampled linear interpolation."""
    opts = options or EdgeCollisionOptions()
    if opts.method != "sampled":
        raise ValueError(f"unsupported edge collision method: {opts.method!r}")
    if opts.max_joint_step <= 0:
        raise ValueError("max_joint_step must be positive")

    q_a = np.asarray(q_a, dtype=float)
    q_b = np.asarray(q_b, dtype=float)
    if q_a.shape != q_b.shape:
        raise ValueError(f"q_a and q_b must have the same shape; got {q_a.shape} and {q_b.shape}")

    max_delta = float(np.max(np.abs(q_b - q_a))) if q_a.size else 0.0
    steps = max(1, int(math.ceil(max_delta / opts.max_joint_step)))

    if opts.include_endpoints:
        indices = range(0, steps + 1)
    else:
        indices = range(1, steps)

    checked_states = 0
    contact_options = CollisionOptions(
        stop_at_first_contact=True,
        collect_contacts=True,
    )

    for idx in indices:
        alpha = idx / steps
        q = (1.0 - alpha) * q_a + alpha * q_b
        report = is_in_collision(
            model,
            scene,
            q,
            chain_id=chain_id,
            options=contact_options,
        )
        checked_states += 1
        if report.in_collision:
            return EdgeCollisionReport(
                in_collision=True,
                first_collision_alpha=alpha,
                contact_report=report,
                checked_states=checked_states,
            )

    return EdgeCollisionReport(
        in_collision=False,
        first_collision_alpha=None,
        contact_report=None,
        checked_states=checked_states,
    )
