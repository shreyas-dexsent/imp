from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from robot_engine.interfaces.schemas import Transform3D
from robot_engine.motion.motion_request import MotionType


class MotionRejectionReason(str, Enum):
    FRAME_NOT_FOUND = "FRAME_NOT_FOUND"
    INVALID_TRANSFORM_CHAIN = "INVALID_TRANSFORM_CHAIN"
    IK_FAILED = "IK_FAILED"
    IK_DISCONTINUITY = "IK_DISCONTINUITY"
    JOINT_LIMIT_VIOLATION = "JOINT_LIMIT_VIOLATION"
    VELOCITY_LIMIT_VIOLATION = "VELOCITY_LIMIT_VIOLATION"
    ACCELERATION_LIMIT_VIOLATION = "ACCELERATION_LIMIT_VIOLATION"
    SINGULARITY_RISK = "SINGULARITY_RISK"
    COLLISION_DETECTED = "COLLISION_DETECTED"
    CLEARANCE_TOO_LOW = "CLEARANCE_TOO_LOW"
    LINEAR_PATH_FAILED = "LINEAR_PATH_FAILED"
    TRAJECTORY_VALIDATION_FAILED = "TRAJECTORY_VALIDATION_FAILED"
    UNSUPPORTED_MOTION_TYPE = "UNSUPPORTED_MOTION_TYPE"
    INVALID_AXIS = "INVALID_AXIS"
    INVALID_DISTANCE = "INVALID_DISTANCE"


class JointTrajectory(BaseModel):
    joint_names: List[str]
    positions: List[List[float]]
    times: List[float] = Field(default_factory=list)
    velocities: List[List[float]] = Field(default_factory=list)
    accelerations: List[List[float]] = Field(default_factory=list)


class CartesianPath(BaseModel):
    frames: List[Transform3D] = Field(default_factory=list)


class TrajectoryValidationResult(BaseModel):
    success: bool
    minimum_clearance: Optional[float] = None
    max_joint_motion: float = 0.0
    failed_stage: Optional[str] = None
    failed_waypoint_index: Optional[int] = None
    rejection_reason: Optional[MotionRejectionReason] = None
    debug_info: Dict[str, Any] = Field(default_factory=dict)


class MotionSegment(BaseModel):
    name: str
    motion_type: MotionType
    result: Optional["MotionResult"] = None
    wait_seconds: float = 0.0


class MotionResult(BaseModel):
    success: bool
    motion_type: MotionType
    start_frame: Optional[Transform3D] = None
    target_frame: Optional[Transform3D] = None
    generated_frames: List[Transform3D] = Field(default_factory=list)
    cartesian_waypoints: List[Transform3D] = Field(default_factory=list)
    joint_waypoints: List[Dict[str, float]] = Field(default_factory=list)
    trajectory: Optional[JointTrajectory] = None
    estimated_duration: float = 0.0
    minimum_clearance: Optional[float] = None
    max_joint_motion: float = 0.0
    failed_stage: Optional[str] = None
    failed_waypoint_index: Optional[int] = None
    rejection_reason: Optional[MotionRejectionReason] = None
    debug_info: Dict[str, Any] = Field(default_factory=dict)


class MotionSequence(BaseModel):
    name: str = "motion_sequence"
    segments: List[Any] = Field(default_factory=list)


class MotionSequenceResult(BaseModel):
    success: bool
    segments: List[MotionSegment] = Field(default_factory=list)
    trajectory: Optional[JointTrajectory] = None
    estimated_duration: float = 0.0
    minimum_clearance: Optional[float] = None
    failed_stage: Optional[str] = None
    failed_segment_index: Optional[int] = None
    failed_waypoint_index: Optional[int] = None
    rejection_reason: Optional[MotionRejectionReason] = None
    debug_info: Dict[str, Any] = Field(default_factory=dict)
