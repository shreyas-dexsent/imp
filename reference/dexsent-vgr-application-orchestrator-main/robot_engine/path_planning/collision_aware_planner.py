from __future__ import annotations

from robot_engine.collision.path_collision_checker import PathCollisionChecker
from robot_engine.path_planning.joint_direct_planner import _path_length
from robot_engine.path_planning.joint_direct_planner import JointDirectPlanner
from robot_engine.path_planning.planner_base import PathRequest, PathResult, PlannerBase
from robot_engine.path_planning.rrt_connect import RRTConnectPlanner
from robot_engine.path_planning.shortcut_smoothing import shortcut_smooth_path


class CollisionAwarePlanner(PlannerBase):
    planner_name = "COLLISION_AWARE"

    def plan(self, request: PathRequest) -> PathResult:
        # Skip direct planner if collision-aware planning is required
        direct = None
        if not request.require_collision_aware_planning:
            direct = JointDirectPlanner().plan(request)
            if direct.success:
                direct.planner_used = "JOINT_DIRECT"
                return direct
        rrt = RRTConnectPlanner().plan(request)
        if not rrt.success:
            return rrt
        checker = PathCollisionChecker(state_checker=lambda q: not request.state_validity_fn(q) if request.state_validity_fn else False, max_joint_delta=request.max_joint_step)
        smoothed, stats = shortcut_smooth_path(rrt.q_waypoints, checker, iterations=100)
        validation = checker.check_path(smoothed)
        if not validation.success:
            return PathResult(False, "JOINT", smoothed, planner_used=self.planner_name, failed_stage="smoothing_validation", rejection_reason=validation.rejection_reason)
        debug_info = {"smoothing": stats}
        if direct is not None:
            debug_info["direct_rejection_reason"] = direct.rejection_reason
        return PathResult(True, "JOINT", smoothed, planner_used="RRT_CONNECT", length=_path_length(smoothed), minimum_clearance=validation.minimum_clearance, planning_time=rrt.planning_time, debug_info=debug_info)
