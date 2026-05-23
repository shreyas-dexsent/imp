from __future__ import annotations

import time

import numpy as np

from robot_engine.path_planning.planner_base import PathRequest, PathResult, PlannerBase
from robot_engine.path_planning.joint_direct_planner import _path_length


class RRTPlanner(PlannerBase):
    planner_name = "RRT"

    def _valid(self, request, q):
        return request.state_validity_fn(q) if request.state_validity_fn else True

    def _edge_valid(self, request, a, b):
        count = max(2, int(np.ceil(np.max(np.abs(b - a)) / max(request.max_joint_step, 1e-9))) + 1)
        return all(self._valid(request, a + (b - a) * t) for t in np.linspace(0.0, 1.0, count))

    def plan(self, request: PathRequest) -> PathResult:
        start_time = time.time()
        start = np.asarray(request.start, dtype=float)
        goal = np.asarray(request.goal, dtype=float)
        lower, upper = request.joint_limits if request.joint_limits is not None else (np.full_like(start, -np.pi), np.full_like(start, np.pi))
        lower = np.asarray(lower, dtype=float)
        upper = np.asarray(upper, dtype=float)
        nodes = [start]
        parents = [-1]
        rng = np.random.default_rng(7)
        for _ in range(request.max_iterations):
            sample = goal if rng.random() < request.goal_bias else rng.uniform(lower, upper)
            nearest_i = int(np.argmin([np.linalg.norm(n - sample) for n in nodes]))
            direction = sample - nodes[nearest_i]
            norm = np.linalg.norm(direction)
            if norm < 1e-12:
                continue
            new = nodes[nearest_i] + direction / norm * min(request.max_joint_step, norm)
            if not self._edge_valid(request, nodes[nearest_i], new):
                continue
            nodes.append(new)
            parents.append(nearest_i)
            if np.linalg.norm(new - goal) <= request.max_joint_step and self._edge_valid(request, new, goal):
                path = [goal, new]
                parent = len(nodes) - 1
                while parent >= 0:
                    path.append(nodes[parent])
                    parent = parents[parent]
                path = list(reversed(path))
                return PathResult(True, "JOINT", path, planner_used=self.planner_name, length=_path_length(path), planning_time=time.time() - start_time, debug_info={"nodes": len(nodes)})
            if time.time() - start_time > request.timeout:
                break
        return PathResult(False, "JOINT", planner_used=self.planner_name, planning_time=time.time() - start_time, rejection_reason="RRT_FAILED", debug_info={"nodes": len(nodes)})

