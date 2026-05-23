from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PlanningScene:
    robot_model: object | None = None
    collision_world: object | None = None
    collision_matrix: object | None = None
    frame_graph: object | None = None
    current_joint_state: object | None = None
    attached_objects: list = field(default_factory=list)
    joint_limits: tuple | None = None
    velocity_limits: object | None = None
    acceleration_limits: object | None = None

