from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import numpy as np

@dataclass
class PlannerResult:
    success: bool
    q_waypoints: List[np.ndarray] = field(default_factory=list)
    rejection_reason: str = ""
    planning_time: float = 0.0
    planner_used: str = ""


class OMPLPlanner:
    """
    Joint-space motion planner.

    Uses the real OMPL Python bindings. The planning-core stack intentionally
    uses OMPL directly so the production backend is explicit and auditable.

    Parameters
    ----------
    lower_limits : (n,) array
    upper_limits : (n,) array
    is_state_valid : callable(q) -> bool
    timeout : float
        Planning time budget in seconds.
    max_joint_step : float
        Maximum per-joint step during tree extension.
    goal_bias : float
        Fraction of iterations that sample the goal directly.
    interpolation_waypoints : int
        Number of points used to interpolate the final path.
    """

    def __init__(
        self,
        lower_limits: np.ndarray,
        upper_limits: np.ndarray,
        is_state_valid: Callable[[np.ndarray], bool],
        timeout: float = 5.0,
        max_joint_step: float = 0.1,
        goal_bias: float = 0.15,
        interpolation_waypoints: int = 100,
        rng_seed: Optional[int] = None,
    ) -> None:
        self._lower = np.asarray(lower_limits, dtype=float)
        self._upper = np.asarray(upper_limits, dtype=float)
        self._is_valid = is_state_valid
        self.timeout = timeout
        self.max_joint_step = max_joint_step
        self.goal_bias = goal_bias
        self.interpolation_waypoints = interpolation_waypoints
        self.rng_seed = rng_seed

    def plan(
        self,
        q_start: np.ndarray,
        q_goal: np.ndarray,
    ) -> PlannerResult:
        """Plan from q_start to q_goal.  Returns waypoints in joint space."""
        return self._plan_ompl(q_start, q_goal)

    # ------------------------------------------------------------------
    # OMPL (real bindings)
    # ------------------------------------------------------------------

    def _plan_ompl(self, q_start: np.ndarray, q_goal: np.ndarray) -> PlannerResult:
        from ompl import base as ob, geometric as og, util as ou  # type: ignore

        n = len(q_start)
        seed = self.rng_seed if self.rng_seed is not None else int(os.urandom(4).hex(), 16)
        ou.RNG.setSeed(seed)
        space = ob.RealVectorStateSpace(n)
        bounds = ob.RealVectorBounds(n)
        for i in range(n):
            bounds.setLow(i, float(self._lower[i]))
            bounds.setHigh(i, float(self._upper[i]))
        space.setBounds(bounds)

        si = ob.SpaceInformation(space)

        user_is_valid = self._is_valid

        class _ValidityChecker(ob.StateValidityChecker):
            def __init__(self, space_information):
                super().__init__(space_information)

            def isValid(self, state):  # noqa: N802 - OMPL binding method name
                q = np.array([state[i] for i in range(n)])
                return bool(user_is_valid(q))

        si.setStateValidityChecker(_ValidityChecker(si))
        si.setup()

        start_state = space.allocState()
        goal_state = space.allocState()
        for i in range(n):
            start_state[i] = float(q_start[i])
            goal_state[i] = float(q_goal[i])

        pdef = ob.ProblemDefinition(si)
        pdef.setStartAndGoalStates(start_state, goal_state)

        planner = og.RRTConnect(si)
        planner.setRange(self.max_joint_step)
        planner.setProblemDefinition(pdef)
        planner.setup()

        t0 = time.monotonic()
        status = planner.solve(self.timeout)
        elapsed = time.monotonic() - t0

        if not status:
            return PlannerResult(
                success=False,
                rejection_reason="RRT_FAILED",
                planning_time=elapsed,
                planner_used="OMPL_RRTConnect",
            )

        path = pdef.getSolutionPath()
        path.interpolate(self.interpolation_waypoints)
        states = path.getStates()
        waypoints = [np.array([s[i] for i in range(n)]) for s in states]

        return PlannerResult(
            success=True,
            q_waypoints=waypoints,
            planning_time=elapsed,
            planner_used="OMPL_RRTConnect",
        )

    # ------------------------------------------------------------------
