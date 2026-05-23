from __future__ import annotations

from robot_engine.path_planning.planner_base import PathRequest
from robot_engine.path_planning.rrt_connect import RRTConnectPlanner
from robot_engine.path_planning.shortcut_smoothing import shortcut_smooth_path


def find_first_colliding_segment(q_waypoints, collision_checker):
    if collision_checker is None:
        return None
    result = collision_checker.check_path(q_waypoints)
    return result.first_collision_segment


def local_repair_with_rrt(q_before, q_after, scene):
    state_validity = getattr(scene, "state_validity_fn", None)
    joint_limits = getattr(scene, "joint_limits", None)
    max_joint_step = getattr(scene, "max_joint_step", 0.1)
    max_iterations = getattr(scene, "max_iterations", 2000)
    timeout = getattr(scene, "timeout", 5.0)
    return RRTConnectPlanner().plan(PathRequest(start=q_before, goal=q_after, joint_limits=joint_limits, state_validity_fn=state_validity, max_joint_step=max_joint_step, max_iterations=max_iterations, timeout=timeout))


def splice_repaired_segment(original_path, repaired_segment, start_index: int = 0, end_index: int | None = None):
    end = len(original_path) - 1 if end_index is None else end_index
    segment = list(repaired_segment)
    if segment and _same(segment[0], original_path[start_index]):
        segment = segment[1:]
    if segment and _same(segment[-1], original_path[end]):
        segment = segment[:-1]
    return list(original_path[: start_index + 1]) + segment + list(original_path[end:])


def repair_path(q_waypoints, collision_checker=None, scene=None, smoothing_iterations: int = 100):
    if collision_checker is None or scene is None:
        return q_waypoints, {"repaired": False, "reason": "INVALID_REQUEST", "error_code": "INVALID_REQUEST", "error_message": "collision_checker and scene are required for path repair"}
    segment = find_first_colliding_segment(q_waypoints, collision_checker)
    if segment is None:
        result = collision_checker.check_path(q_waypoints)
        if result.success and not result.collision:
            return q_waypoints, {"repaired": False, "reason": "OK"}
        segment = result.first_collision_segment
    if segment is None or segment + 1 >= len(q_waypoints):
        return q_waypoints, {"repaired": False, "reason": "PATH_REPAIR_FAILED", "error_code": "PATH_REPAIR_FAILED", "error_message": "Could not identify colliding segment"}
    repair = local_repair_with_rrt(q_waypoints[segment], q_waypoints[segment + 1], scene)
    if not repair.success:
        return q_waypoints, {"repaired": False, "reason": "PATH_REPAIR_FAILED", "error_code": "PATH_REPAIR_FAILED", "error_message": repair.rejection_reason, "debug_info": repair.debug_info}
    repaired = splice_repaired_segment(q_waypoints, repair.q_waypoints, segment, segment + 1)
    smoothed, smoothing = shortcut_smooth_path(repaired, collision_checker, iterations=smoothing_iterations)
    final = collision_checker.check_path(smoothed)
    if not final.success or final.collision:
        return repaired, {"repaired": False, "reason": "PATH_REPAIR_FAILED", "error_code": "PATH_REPAIR_FAILED", "error_message": final.rejection_reason, "debug_info": {"smoothing": smoothing}}
    return smoothed, {"repaired": True, "reason": "OK", "bad_segment": segment, "repair_waypoints": len(repair.q_waypoints), "smoothing": smoothing}


def _same(a, b):
    import numpy as np

    return bool(np.allclose(a, b))

