# Copyright (c) 2026 DexSent Robotics Pvt Ltd.
# All Rights Reserved.
#
# SPDX-License-Identifier: LicenseRef-Proprietary
# See LICENSE file for details.
#
# Author: Shreyas <shreyas.kumar@dexsentrobotics.com>
# Proprietary and confidential.

"""Straight-line connector backend.

The geometric route is the straight line from `q_start` to `q_goal`.
Collision validation samples the line internally at `max_joint_step`,
but those samples do NOT leak into the output `Path`. The output is
exactly `[q_start, q_goal]`: two waypoints, one segment, no spurious
via points the trajectory layer would otherwise have to interpolate
through.

Sampling resolution and via-point count are two different things.
Earlier versions of this backend conflated them, which caused visible
wiggles in the time-parameterized trajectory whenever the user picked
the direct planner. Fixed.
"""
from __future__ import annotations

import math
from typing import Callable

import numpy as np

from algorithms.planning.backends.base import PathPlannerBackend, RawPlanResult
from algorithms.planning.options import PlanOptions
from algorithms.planning.path import PathStatus


class StraightLineBackend:
    """Linear interpolation + collision sweep at `max_joint_step`.

    Output waypoints: exactly `[q_start, q_goal]`. The internal
    sampling at `options.max_joint_step` is used only for the validity
    check; those samples are not stored on the returned Path.
    """

    name: str = "straight_line"

    def plan(
        self,
        q_start: np.ndarray,
        q_goal: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
        state_validity_fn: Callable[[np.ndarray], bool],
        options: PlanOptions,
    ) -> RawPlanResult:
        delta = q_goal - q_start
        step = max(options.max_joint_step, 1e-9)
        n_steps = max(1, int(math.ceil(float(np.max(np.abs(delta))) / step)))

        # Internal validity sweep. Not exposed on the output.
        for i in range(n_steps + 1):
            alpha = i / n_steps
            wp = q_start + alpha * delta
            if not state_validity_fn(wp):
                return RawPlanResult(
                    status=PathStatus.NO_PATH_FOUND,
                    waypoints=None,
                    iterations=i + 1,
                    message=f"straight-line invalid at sample {i} / {n_steps}",
                    extra={"failed_sample_index": i, "num_validity_samples": n_steps + 1},
                )

        # Output: just the two endpoints. The motion is geometrically
        # a single straight segment; the trajectory layer handles the
        # smooth single-segment parameterization without spurious via
        # points.
        waypoints = np.stack([q_start, q_goal])

        return RawPlanResult(
            status=PathStatus.SUCCESS,
            waypoints=waypoints,
            iterations=n_steps + 1,
            message="straight-line connection valid",
            extra={"num_validity_samples": n_steps + 1},
        )
