from __future__ import annotations

from robot_engine.collision.path_collision_checker import PathCollisionChecker


def conservative_continuous_collision(q0, q1, state_checker=None, world=None, resolution: float = 0.05):
    """Conservative interpolated continuous collision validation.

    This is not exact swept mesh collision. It samples the joint segment with a
    resolution based on maximum joint delta and checks active collision pairs at
    every interpolation sample.
    """
    if state_checker is None and world is None:
        from robot_engine.collision.path_collision_checker import PathCollisionResult

        return PathCollisionResult(success=False, rejection_reason="INVALID_REQUEST", debug_info={"message": "state_checker or collision world is required"})
    return PathCollisionChecker(state_checker=state_checker, world=world, max_joint_delta=resolution).check_segment(q0, q1)


def conservative_continuous_path_collision(q_waypoints, state_checker=None, world=None, resolution: float = 0.05):
    if state_checker is None and world is None:
        from robot_engine.collision.path_collision_checker import PathCollisionResult

        return PathCollisionResult(success=False, rejection_reason="INVALID_REQUEST", debug_info={"message": "state_checker or collision world is required"})
    return PathCollisionChecker(state_checker=state_checker, world=world, max_joint_delta=resolution).check_path(q_waypoints)
