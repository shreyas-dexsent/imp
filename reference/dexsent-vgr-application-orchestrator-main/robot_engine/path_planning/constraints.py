from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PathPlanningConstraints:
    joint_limits: tuple | None = None
    singularity_threshold: float = 1e8
    minimum_clearance_threshold: float = 0.0
    max_joint_step: float = 0.1
    max_planning_time: float = 5.0
    max_iterations: int = 1000
    keep_gripper_upright: bool = False
    maintain_approach_direction: bool = False
    avoid_bin_walls: bool = False

