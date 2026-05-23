from __future__ import annotations

import math

import numpy as np

from robot_engine.path_planning.planner_base import PathRequest, PathResult, PlannerBase


def _path_length(path):
    return float(sum(np.linalg.norm(np.asarray(b) - np.asarray(a)) for a, b in zip(path[:-1], path[1:])))


class JointDirectPlanner(PlannerBase):
    planner_name = "JOINT_DIRECT"

    def plan(self, request: PathRequest) -> PathResult:
        error = self.validate_request(request)
        if error:
            return PathResult(False, "JOINT", planner_used=self.planner_name, failed_stage="request", rejection_reason=error)
        start = np.asarray(request.start, dtype=float)
        goal = np.asarray(request.goal, dtype=float)
        if request.joint_limits is not None:
            lower, upper = (np.asarray(x, dtype=float) for x in request.joint_limits)
            if np.any(start < lower) or np.any(start > upper) or np.any(goal < lower) or np.any(goal > upper):
                return PathResult(False, "JOINT", planner_used=self.planner_name, failed_stage="joint_limits", rejection_reason="JOINT_LIMIT_VIOLATION")
        count = max(2, int(math.ceil(np.max(np.abs(goal - start)) / max(request.max_joint_step, 1e-9))) + 1)
        path = [start + (goal - start) * a for a in np.linspace(0.0, 1.0, count)]
        if request.state_validity_fn:
            for i, q in enumerate(path):
                if not request.state_validity_fn(q):
                    return PathResult(False, "JOINT", path, planner_used=self.planner_name, failed_stage="collision", failed_waypoint_index=i, rejection_reason="COLLISION_DETECTED")
        return PathResult(True, "JOINT", path, planner_used=self.planner_name, length=_path_length(path))

