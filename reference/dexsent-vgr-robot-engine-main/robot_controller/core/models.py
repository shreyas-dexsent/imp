"""Implementation for `robot_controller.core.models`."""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class PoseSE3:
    position_m: List[float]
    quat_xyzw: List[float]
    frame: str = "base"


@dataclass
class TcpIkRequest:
    target: PoseSE3
    seed_joints: Optional[List[float]] = None
    preferred_joints: Optional[List[float]] = None
    position_tolerance_m: float = 0.002
    orientation_tolerance_deg: float = 2.0
    approximate_position_tolerance_m: float = 0.015
    approximate_orientation_tolerance_deg: float = 3.0
    max_iterations: int = 120


@dataclass
class CartesianVelocity:
    linear_mps: List[float]
    angular_dps: List[float]
    frame: str = "base"


@dataclass
class MotionProfile:
    speed: str = "normal"
    timeout_s: float = 10.0


@dataclass
class RobotState:
    timestamp_ns: int
    mode: str
    tcp_pose: PoseSE3
    q: List[float] = field(default_factory=list)
    dq: List[float] = field(default_factory=list)
    active_motion_id: Optional[str] = None
