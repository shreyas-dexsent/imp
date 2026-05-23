from __future__ import annotations

from robot_engine.collision.path_collision_checker import PathCollisionChecker


def clearance_margin_for_state(q, checker: PathCollisionChecker):
    return checker.check_state(q).minimum_clearance


def clearance_margin_for_path(q_waypoints, checker: PathCollisionChecker):
    return checker.check_path(q_waypoints).minimum_clearance


def check_clearance_above_threshold(q_waypoints, checker: PathCollisionChecker, threshold: float):
    result = checker.check_path(q_waypoints)
    if result.minimum_clearance is not None and result.minimum_clearance < threshold:
        result.success = False
        result.rejection_reason = "CLEARANCE_TOO_LOW"
    return result

