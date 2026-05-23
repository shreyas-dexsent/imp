from __future__ import annotations

import numpy as np

from robot_engine.core_math.interpolation import sample_cartesian_path
from robot_engine.path_planning.planner_base import PathRequest, PathResult, PlannerBase
from robot_engine.path_planning.joint_direct_planner import _path_length


class CartesianLinearPlanner(PlannerBase):
    planner_name = "CARTESIAN_LINEAR"

    def plan(self, request: PathRequest) -> PathResult:
        ik_fn = request.debug_info.get("ik_fn") or _ik_fn_from_registry(request)
        if ik_fn is None:
            return PathResult(False, "CARTESIAN", planner_used=self.planner_name, failed_stage="ik", rejection_reason="IK_BACKEND_UNAVAILABLE", debug_info={"error_message": "CartesianLinearPlanner requires debug_info['ik_fn'] or debug_info['ik_request_factory']; it will not fake MoveL with joint interpolation."})
        frames = sample_cartesian_path(request.start, request.goal, request.debug_info.get("translation_step", 0.02), request.debug_info.get("rotation_step", 0.1))
        straight_error = _straight_line_error(frames)
        if straight_error > request.debug_info.get("cartesian_line_tolerance", 1e-9):
            return PathResult(False, "CARTESIAN", planner_used=self.planner_name, failed_stage="cartesian_interpolation", rejection_reason="INVALID_TRANSFORM_CHAIN", debug_info={"straight_line_error": straight_error})
        q_path = []
        seed = request.debug_info.get("seed")
        for i, frame in enumerate(frames):
            result = ik_fn(frame, seed)
            if result is None:
                return PathResult(False, "CARTESIAN", q_path, frames, self.planner_name, failed_stage="ik", failed_waypoint_index=i, rejection_reason="IK_FAILED")
            q = np.asarray(result, dtype=float)
            if q_path and np.max(np.abs(q - q_path[-1])) > request.debug_info.get("continuity_joint_step", 0.75):
                return PathResult(False, "CARTESIAN", q_path, frames, self.planner_name, failed_stage="ik_continuity", failed_waypoint_index=i, rejection_reason="IK_DISCONTINUITY")
            if request.state_validity_fn and not request.state_validity_fn(q):
                return PathResult(False, "CARTESIAN", q_path, frames, self.planner_name, failed_stage="collision", failed_waypoint_index=i, rejection_reason="COLLISION_DETECTED")
            q_path.append(q)
            seed = q
        return PathResult(True, "CARTESIAN", q_path, frames, self.planner_name, length=_path_length(q_path))


def _ik_fn_from_registry(request: PathRequest):
    factory = request.debug_info.get("ik_request_factory")
    if factory is None:
        return None
    backend = request.debug_info.get("ik_backend", "auto")

    def solve(frame, seed):
        from robot_engine.kinematics.ik_solver import solve_ik_with_backend

        ik_request = factory(frame, seed)
        result = solve_ik_with_backend(ik_request, backend)
        if not result.ok:
            return None
        names = request.debug_info.get("joint_names") or list(result.joint_positions.keys())
        return [result.joint_positions[name] for name in names]

    return solve


def _straight_line_error(frames):
    if len(frames) <= 2:
        return 0.0
    start = np.asarray(frames[0])[:3, 3]
    goal = np.asarray(frames[-1])[:3, 3]
    direction = goal - start
    denom = float(direction @ direction)
    if denom <= 1e-15:
        return max(float(np.linalg.norm(np.asarray(frame)[:3, 3] - start)) for frame in frames)
    worst = 0.0
    for frame in frames:
        p = np.asarray(frame)[:3, 3]
        alpha = float(((p - start) @ direction) / denom)
        projected = start + direction * alpha
        worst = max(worst, float(np.linalg.norm(p - projected)))
    return worst
