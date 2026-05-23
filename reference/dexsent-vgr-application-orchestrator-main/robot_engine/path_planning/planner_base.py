from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, List

import numpy as np


@dataclass
class PathRequest:
    start: Any
    goal: Any
    joint_limits: tuple[Any, Any] | None = None
    state_validity_fn: Callable[[Any], bool] | None = None
    max_joint_step: float = 0.1
    max_iterations: int = 1000
    timeout: float = 5.0
    goal_bias: float = 0.1
    motion_type: str = "JOINT"
    debug_info: dict = field(default_factory=dict)
    require_collision_aware_planning: bool = False


@dataclass
class PathResult:
    success: bool
    path_type: str
    q_waypoints: List[Any] = field(default_factory=list)
    cartesian_waypoints: List[Any] = field(default_factory=list)
    planner_used: str = ""
    length: float = 0.0
    minimum_clearance: float | None = None
    planning_time: float = 0.0
    failed_stage: str | None = None
    failed_waypoint_index: int | None = None
    failed_segment_index: int | None = None
    colliding_pair: list[str] | None = None
    rejection_reason: str = "OK"
    debug_info: dict = field(default_factory=dict)


class PlannerBase:
    planner_name = "BASE"

    def validate_request(self, request: PathRequest) -> str | None:
        if np.asarray(request.start, dtype=float).shape != np.asarray(request.goal, dtype=float).shape:
            return "INVALID_TRANSFORM_CHAIN"
        return None

    def plan(self, request: PathRequest) -> PathResult:
        raise NotImplementedError

