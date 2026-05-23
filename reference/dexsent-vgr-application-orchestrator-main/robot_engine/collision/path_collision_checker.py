from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, List

import numpy as np

from robot_engine.collision.distance_queries import minimum_distances_active_pairs


@dataclass
class PathCollisionResult:
    success: bool
    collision: bool = False
    first_collision_waypoint: int | None = None
    first_collision_segment: int | None = None
    colliding_pair: list[str] | None = None
    minimum_clearance: float | None = None
    interpolation_samples: int = 0
    resolution: float | None = None
    rejection_reason: str = "OK"
    debug_info: dict | None = None


class PathCollisionChecker:
    def __init__(self, state_checker: Callable | None = None, world=None, max_joint_delta: float = 0.1):
        self.state_checker = state_checker
        self.world = world
        self.max_joint_delta = max_joint_delta

    def check_state(self, q) -> PathCollisionResult:
        if self.state_checker is not None:
            result = self.state_checker(q)
            if isinstance(result, PathCollisionResult):
                return result
            if isinstance(result, dict):
                collision = bool(result.get("collision", not result.get("valid", True)))
                return PathCollisionResult(success=not collision, collision=collision, colliding_pair=result.get("colliding_pair"), minimum_clearance=result.get("minimum_clearance"), rejection_reason="COLLISION_DETECTED" if collision else "OK", debug_info=result)
            return PathCollisionResult(success=not bool(result), collision=bool(result), rejection_reason="COLLISION_DETECTED" if result else "OK")
        if self.world is None:
            return PathCollisionResult(success=False, rejection_reason="INVALID_REQUEST", debug_info={"message": "collision world or state checker is required"})
        from robot_engine.collision.collision_checker import check_active_pairs

        result = check_active_pairs(self.world)
        clearance = None
        distances = minimum_distances_active_pairs(self.world)
        if distances:
            clearance = min((d.distance for d in distances if d.distance is not None), default=None)
        return PathCollisionResult(success=not result.collision and result.ok, collision=result.collision, colliding_pair=result.colliding_pairs[0] if result.colliding_pairs else None, minimum_clearance=clearance, rejection_reason="COLLISION_DETECTED" if result.collision else "OK")

    def adaptive_subdivision(self, q0, q1, max_joint_delta: float | None = None):
        q0 = np.asarray(q0, dtype=float)
        q1 = np.asarray(q1, dtype=float)
        step = max_joint_delta or self.max_joint_delta
        count = max(2, int(np.ceil(np.max(np.abs(q1 - q0)) / max(step, 1e-9))) + 1)
        return [q0 + (q1 - q0) * a for a in np.linspace(0.0, 1.0, count)]

    def check_segment(self, q0, q1) -> PathCollisionResult:
        samples = self.adaptive_subdivision(q0, q1)
        minimum = float("inf")
        for index, q in enumerate(samples):
            result = self.check_state(q)
            if result.minimum_clearance is not None:
                minimum = min(minimum, result.minimum_clearance)
            if result.collision or not result.success:
                result.first_collision_waypoint = index
                result.interpolation_samples = len(samples)
                result.resolution = self.max_joint_delta
                if minimum != float("inf"):
                    result.minimum_clearance = minimum
                return result
        return PathCollisionResult(success=True, minimum_clearance=None if minimum == float("inf") else minimum, interpolation_samples=len(samples), resolution=self.max_joint_delta)

    def check_waypoints(self, q_waypoints: Iterable) -> PathCollisionResult:
        minimum = float("inf")
        for index, q in enumerate(q_waypoints):
            result = self.check_state(q)
            if result.minimum_clearance is not None:
                minimum = min(minimum, result.minimum_clearance)
            if result.collision or not result.success:
                result.first_collision_waypoint = index
                return result
        return PathCollisionResult(success=True, minimum_clearance=None if minimum == float("inf") else minimum)

    def check_path(self, q_waypoints: List) -> PathCollisionResult:
        waypoint_result = self.check_waypoints(q_waypoints)
        if not waypoint_result.success:
            return waypoint_result
        for index, (q0, q1) in enumerate(zip(q_waypoints[:-1], q_waypoints[1:])):
            result = self.check_segment(q0, q1)
            if result.collision or not result.success:
                result.first_collision_segment = index
                return result
        return waypoint_result

    def report_first_collision(self, q_waypoints: List) -> PathCollisionResult:
        return self.check_path(q_waypoints)
