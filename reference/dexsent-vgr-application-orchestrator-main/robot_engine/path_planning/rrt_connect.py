from __future__ import annotations

import time

import numpy as np

from robot_engine.path_planning.joint_direct_planner import _path_length
from robot_engine.path_planning.planner_base import PathRequest, PathResult
from robot_engine.path_planning.rrt import RRTPlanner


class RRTConnectPlanner(RRTPlanner):
    planner_name = "RRT_CONNECT"

    def plan(self, request: PathRequest) -> PathResult:
        started = time.monotonic()
        start = np.asarray(request.start, dtype=float)
        goal = np.asarray(request.goal, dtype=float)
        lower, upper = request.joint_limits if request.joint_limits is not None else (np.full_like(start, -np.pi), np.full_like(start, np.pi))
        lower = np.asarray(lower, dtype=float)
        upper = np.asarray(upper, dtype=float)
        seed = request.debug_info.get("seed")
        rng = np.random.default_rng(int(seed) if seed is not None else None)
        ta = _Tree(start)
        tb = _Tree(goal)
        for iteration in range(request.max_iterations):
            sample = goal if rng.random() < request.goal_bias else rng.uniform(lower, upper)
            new_a = _extend(ta, sample, request, self)
            if new_a is not None:
                new_b = _connect(tb, new_a, request, self)
                if new_b is not None and np.linalg.norm(new_a - new_b) <= request.max_joint_step:
                    path_a = ta.path_to(len(ta.nodes) - 1)
                    path_b = tb.path_to(len(tb.nodes) - 1)
                    if np.allclose(path_a[0], start):
                        path = path_a + list(reversed(path_b))
                    else:
                        path = path_b + list(reversed(path_a))
                    return PathResult(True, "JOINT", path, planner_used=self.planner_name, length=_path_length(path), planning_time=time.monotonic() - started, debug_info={"nodes": len(ta.nodes) + len(tb.nodes), "iterations": iteration + 1})
            ta, tb = tb, ta
            if time.monotonic() - started > request.timeout:
                break
        return PathResult(False, "JOINT", planner_used=self.planner_name, planning_time=time.monotonic() - started, rejection_reason="RRT_FAILED", debug_info={"nodes": len(ta.nodes) + len(tb.nodes)})


class _Tree:
    def __init__(self, root):
        self.nodes = [np.asarray(root, dtype=float)]
        self.parents = [-1]

    def nearest(self, q):
        return int(np.argmin([np.linalg.norm(node - q) for node in self.nodes]))

    def add(self, q, parent):
        self.nodes.append(np.asarray(q, dtype=float))
        self.parents.append(parent)
        return len(self.nodes) - 1

    def path_to(self, index):
        path = []
        while index >= 0:
            path.append(self.nodes[index])
            index = self.parents[index]
        return list(reversed(path))


def _steer(a, b, step):
    direction = np.asarray(b, dtype=float) - np.asarray(a, dtype=float)
    norm = np.linalg.norm(direction)
    if norm <= 1e-12:
        return None
    return np.asarray(a, dtype=float) + direction / norm * min(step, norm)


def _extend(tree, target, request, planner):
    nearest_i = tree.nearest(target)
    new = _steer(tree.nodes[nearest_i], target, request.max_joint_step)
    if new is None or not planner._edge_valid(request, tree.nodes[nearest_i], new):
        return None
    tree.add(new, nearest_i)
    return new


def _connect(tree, target, request, planner):
    last = None
    while True:
        new = _extend(tree, target, request, planner)
        if new is None:
            return last
        last = new
        if np.linalg.norm(new - target) <= request.max_joint_step:
            return new

