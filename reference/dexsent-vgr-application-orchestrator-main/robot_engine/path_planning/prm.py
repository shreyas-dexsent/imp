from __future__ import annotations

import heapq
import time

import numpy as np

from robot_engine.path_planning.joint_direct_planner import _path_length
from robot_engine.path_planning.planner_base import PathRequest, PathResult, PlannerBase


class PRMPlanner(PlannerBase):
    planner_name = "PRM"

    def plan(self, request: PathRequest) -> PathResult:
        started = time.monotonic()
        start = np.asarray(request.start, dtype=float)
        goal = np.asarray(request.goal, dtype=float)
        lower, upper = request.joint_limits if request.joint_limits is not None else (np.full_like(start, -np.pi), np.full_like(start, np.pi))
        lower = np.asarray(lower, dtype=float)
        upper = np.asarray(upper, dtype=float)
        max_samples = int(request.debug_info.get("max_samples", request.max_iterations))
        k_nearest = int(request.debug_info.get("k_nearest", 10))
        max_edge_length = float(request.debug_info.get("max_edge_length", request.max_joint_step * 8.0))
        seed = int(request.debug_info.get("seed", 13))
        rng = np.random.default_rng(seed)

        if start.shape != goal.shape:
            return PathResult(False, "JOINT", planner_used=self.planner_name, failed_stage="request", rejection_reason="INVALID_REQUEST")
        if np.any(start < lower) or np.any(start > upper) or np.any(goal < lower) or np.any(goal > upper):
            return PathResult(False, "JOINT", planner_used=self.planner_name, failed_stage="joint_limits", rejection_reason="JOINT_LIMIT_VIOLATION")
        if not self._valid(request, start) or not self._valid(request, goal):
            return PathResult(False, "JOINT", planner_used=self.planner_name, failed_stage="state_validity", rejection_reason="COLLISION_DETECTED")

        nodes = [start, goal]
        attempts = 0
        while len(nodes) < max_samples + 2 and time.monotonic() - started <= request.timeout:
            attempts += 1
            q = rng.uniform(lower, upper)
            if self._valid(request, q):
                nodes.append(q)

        graph = {i: [] for i in range(len(nodes))}
        edge_count = 0
        for i, node in enumerate(nodes):
            distances = [(float(np.linalg.norm(node - other)), j) for j, other in enumerate(nodes) if j != i]
            for dist, j in sorted(distances)[:k_nearest]:
                if dist > max_edge_length:
                    continue
                if self._edge_valid(request, node, nodes[j]):
                    graph[i].append((j, dist))
                    edge_count += 1

        indices = _dijkstra(graph, 0, 1)
        if indices is None:
            return PathResult(False, "JOINT", planner_used=self.planner_name, planning_time=time.monotonic() - started, rejection_reason="PRM_FAILED", debug_info={"sampled_nodes": len(nodes), "edges": edge_count, "attempts": attempts})
        path = [nodes[i] for i in indices]
        if not self._path_valid(request, path):
            return PathResult(False, "JOINT", planner_used=self.planner_name, failed_stage="final_validation", rejection_reason="PRM_FAILED", debug_info={"sampled_nodes": len(nodes), "edges": edge_count})
        return PathResult(True, "JOINT", path, planner_used=self.planner_name, length=_path_length(path), planning_time=time.monotonic() - started, debug_info={"sampled_nodes": len(nodes), "edges": edge_count, "attempts": attempts})

    def _valid(self, request, q):
        return bool(request.state_validity_fn(q)) if request.state_validity_fn else True

    def _edge_valid(self, request, a, b):
        count = max(2, int(np.ceil(np.max(np.abs(np.asarray(b) - np.asarray(a))) / max(request.max_joint_step, 1e-9))) + 1)
        return all(self._valid(request, np.asarray(a) + (np.asarray(b) - np.asarray(a)) * t) for t in np.linspace(0.0, 1.0, count))

    def _path_valid(self, request, path):
        return all(self._edge_valid(request, a, b) for a, b in zip(path[:-1], path[1:]))


def _dijkstra(graph, start, goal):
    heap = [(0.0, start)]
    dist = {start: 0.0}
    parent = {start: None}
    while heap:
        cost, node = heapq.heappop(heap)
        if node == goal:
            path = []
            while node is not None:
                path.append(node)
                node = parent[node]
            return list(reversed(path))
        if cost > dist[node]:
            continue
        for nxt, weight in graph[node]:
            new = cost + weight
            if new < dist.get(nxt, float("inf")):
                dist[nxt] = new
                parent[nxt] = node
                heapq.heappush(heap, (new, nxt))
    return None

