# Copyright (c) 2026 DexSent Robotics Pvt Ltd.
# All Rights Reserved.
#
# SPDX-License-Identifier: LicenseRef-Proprietary
# See LICENSE file for details.
#
# Author: Shreyas <shreyas.kumar@dexsentrobotics.com>
# Proprietary and confidential.

"""OMPL adapter — the production path planner backend.

Wraps `ompl.geometric` planners (RRTConnect default; RRT, RRTstar,
PRM, BITstar, KPIECE1, LBKPIECE1 also exposed via `options.planner_name`).

The user does not write OMPL code. The validity callback is adapted to
the OMPL `StateValidityChecker` interface internally; everything else
flows through `PlanOptions` and `Path`.
"""
from __future__ import annotations

import os
import time
from typing import Callable

import numpy as np

from algorithms.planning.backends.base import PathPlannerBackend, RawPlanResult
from algorithms.planning.options import PlanOptions
from algorithms.planning.path import PathStatus


_PLANNER_MAP = {
    "RRT": "RRT",
    "RRTConnect": "RRTConnect",
    "RRTstar": "RRTstar",
    "PRM": "PRM",
    "BITstar": "BITstar",
    "KPIECE1": "KPIECE1",
    "LBKPIECE1": "LBKPIECE1",
}


class OMPLBackend:
    """OMPL geometric planner adapter."""

    name: str = "ompl"

    def plan(
        self,
        q_start: np.ndarray,
        q_goal: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
        state_validity_fn: Callable[[np.ndarray], bool],
        options: PlanOptions,
    ) -> RawPlanResult:
        try:
            from ompl import base as ob, geometric as og, util as ou
        except Exception as exc:
            return RawPlanResult(
                status=PathStatus.NUMERICAL_FAILURE,
                waypoints=None,
                message=(
                    f"ompl Python bindings not importable: {exc}. "
                    "Install via `pip install ompl` or use backend='direct'."
                ),
            )

        planner_cls_name = _PLANNER_MAP.get(options.planner_name)
        if planner_cls_name is None or not hasattr(og, planner_cls_name):
            return RawPlanResult(
                status=PathStatus.INVALID_INPUT,
                waypoints=None,
                message=(
                    f"planner_name={options.planner_name!r} not available in OMPL; "
                    f"choose one of {sorted(_PLANNER_MAP.keys())}."
                ),
            )
        planner_cls = getattr(og, planner_cls_name)

        n = len(q_start)
        seed = options.random_seed
        if seed is None:
            seed = int(os.urandom(4).hex(), 16)
        ou.RNG.setSeed(int(seed))

        space = ob.RealVectorStateSpace(n)
        bounds = ob.RealVectorBounds(n)
        for i in range(n):
            bounds.setLow(i, float(lower[i]))
            bounds.setHigh(i, float(upper[i]))
        space.setBounds(bounds)

        si = ob.SpaceInformation(space)

        validity_fn = state_validity_fn

        class _ValidityChecker(ob.StateValidityChecker):
            def __init__(self, space_information):
                super().__init__(space_information)

            def isValid(self, state):  # noqa: N802 - OMPL binding method name
                q = np.array([state[i] for i in range(n)], dtype=float)
                return bool(validity_fn(q))

        si.setStateValidityChecker(_ValidityChecker(si))
        si.setStateValidityCheckingResolution(
            options.max_joint_step / max(float(np.max(upper - lower)), 1e-9)
        )
        si.setup()

        start_state = space.allocState()
        goal_state = space.allocState()
        for i in range(n):
            start_state[i] = float(q_start[i])
            goal_state[i] = float(q_goal[i])

        pdef = ob.ProblemDefinition(si)
        pdef.setStartAndGoalStates(start_state, goal_state)

        planner = planner_cls(si)
        if hasattr(planner, "setRange"):
            planner.setRange(options.max_joint_step)
        if planner_cls_name in ("RRT", "RRTConnect") and hasattr(planner, "setGoalBias"):
            planner.setGoalBias(options.goal_bias)
        planner.setProblemDefinition(pdef)
        planner.setup()

        timeout_s = options.max_time_ms / 1000.0
        t0 = time.monotonic()
        status = planner.solve(timeout_s)
        elapsed = time.monotonic() - t0

        if not status:
            return RawPlanResult(
                status=PathStatus.NO_PATH_FOUND,
                waypoints=None,
                iterations=0,
                message=f"OMPL {planner_cls_name} did not find a solution in {elapsed:.2f}s",
            )

        # OMPL reports "approximate solution" when it bailed at the
        # iteration / time cap with a partial result. Surface that
        # distinctly so the caller can retry with a larger budget.
        status_str = ""
        as_string = getattr(status, "asString", None)
        if callable(as_string):
            try:
                status_str = str(as_string())
            except Exception:
                status_str = ""
        is_approximate = status_str.lower().startswith("approximate")

        path = pdef.getSolutionPath()
        path.interpolate(options.interpolation_waypoints)
        states = path.getStates()
        waypoints = np.array(
            [[s[i] for i in range(n)] for s in states], dtype=float
        )

        final_status = (
            PathStatus.TIMEOUT if is_approximate else PathStatus.SUCCESS
        )

        return RawPlanResult(
            status=final_status,
            waypoints=waypoints if final_status is PathStatus.SUCCESS else None,
            iterations=len(waypoints),
            message=(
                f"OMPL {planner_cls_name} solved in {elapsed:.2f}s"
                + (" (approximate)" if is_approximate else "")
            ),
            extra={"planner_class": planner_cls_name},
        )
