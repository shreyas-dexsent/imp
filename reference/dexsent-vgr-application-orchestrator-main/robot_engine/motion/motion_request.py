from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from robot_engine.interfaces.schemas import KinematicChainConfig, Transform3D


class MotionType(str, Enum):
    JOINT = "JOINT"
    LINEAR = "LINEAR"


class Axis(str, Enum):
    X = "X"
    Y = "Y"
    Z = "Z"


class AxisDirection(str, Enum):
    POSITIVE = "POSITIVE"
    NEGATIVE = "NEGATIVE"


class AxisFrame(str, Enum):
    BASE = "BASE"
    TCP = "TCP"
    OBJECT = "OBJECT"
    GRASP = "GRASP"
    BIN = "BIN"
    CUSTOM = "CUSTOM"


class IKOptions(BaseModel):
    seed: Dict[str, float] = Field(default_factory=dict)
    seed_states: List[Dict[str, float]] = Field(default_factory=list)
    max_iterations: int = 100
    position_tolerance: float = 1e-4
    orientation_tolerance: float = 1e-3
    damping: float = 1e-3
    singularity_threshold: float = 1e8
    continuity_joint_step: float = 0.75


class CollisionValidationOptions(BaseModel):
    enabled: bool = True
    world: Any = None
    minimum_clearance: float = 0.0
    tcp_clearance_radius: float = 0.0


class TrajectoryOptions(BaseModel):
    joint_waypoint_count: int = 20
    linear_waypoint_count: int = 20
    max_joint_step: float = 0.1
    max_joint_velocity: float = 1.0
    max_joint_acceleration: float = 2.0
    time_step: float = 0.1
    blend_radius: float = 0.0


class MotionPlanningOptions(BaseModel):
    minimum_clearance: float = 0.0
    max_joint_effort: Optional[float] = None
    singularity_threshold: float = 1e8
    validate_trajectory: bool = True


class FrameOffsetRequest(BaseModel):
    frame: Transform3D
    axis: Axis
    direction: AxisDirection = AxisDirection.POSITIVE
    distance: float
    reference_frame: AxisFrame = AxisFrame.BASE
    reference_transform: Optional[Transform3D] = None
    output_child_frame: str = "offset"


class ApproachOptions(BaseModel):
    enabled: bool = False
    axis: Axis = Axis.Z
    direction: AxisDirection = AxisDirection.NEGATIVE
    distance: float = 0.1
    reference_frame: AxisFrame = AxisFrame.TCP
    wait_seconds: float = 0.0
    motion_type: MotionType = MotionType.JOINT


class RetreatOptions(BaseModel):
    enabled: bool = False
    axis: Axis = Axis.Z
    direction: AxisDirection = AxisDirection.POSITIVE
    distance: float = 0.1
    reference_frame: AxisFrame = AxisFrame.TCP
    wait_seconds: float = 0.0
    motion_type: MotionType = MotionType.JOINT


class LiftOptions(BaseModel):
    axis: Axis = Axis.Z
    direction: AxisDirection = AxisDirection.POSITIVE
    distance: float = 0.1
    reference_frame: AxisFrame = AxisFrame.BASE
    motion_type: MotionType = MotionType.LINEAR


class MotionRequest(BaseModel):
    motion_type: MotionType
    chain: KinematicChainConfig
    current_joint_state: Dict[str, float]
    target_frame: Transform3D
    start_frame: Optional[Transform3D] = None
    named_frames: Dict[str, Transform3D] = Field(default_factory=dict)
    ik_options: IKOptions = Field(default_factory=IKOptions)
    collision_options: CollisionValidationOptions = Field(default_factory=CollisionValidationOptions)
    trajectory_options: TrajectoryOptions = Field(default_factory=TrajectoryOptions)
    planning_options: MotionPlanningOptions = Field(default_factory=MotionPlanningOptions)
    approach: Optional[ApproachOptions] = None
    retreat: Optional[RetreatOptions] = None
    lift: Optional[LiftOptions] = None
    label: str = ""


class TrajectoryValidationRequest(BaseModel):
    chain: KinematicChainConfig
    trajectory: Any
    collision_options: CollisionValidationOptions = Field(default_factory=CollisionValidationOptions)
    trajectory_options: TrajectoryOptions = Field(default_factory=TrajectoryOptions)
    ik_options: IKOptions = Field(default_factory=IKOptions)


class ExportTrajectoryRequest(BaseModel):
    trajectory: Any
    format: str = "dict"
