from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Sequence

from pydantic import BaseModel, Field, field_validator


ReasonCode = Literal[
    "OK",
    "INVALID_REQUEST",
    "INVALID_TRANSFORM",
    "FRAME_NOT_FOUND",
    "INVALID_FRAME_CHAIN",
    "INVALID_QUATERNION",
    "INVALID_ROTATION_MATRIX",
    "INVALID_MESH",
    "EMPTY_MESH",
    "NON_FINITE_VERTICES",
    "POINT_CLOUD_UNSUPPORTED",
    "UNSUPPORTED_FORMAT",
    "UNSUPPORTED_ASSET_FORMAT",
    "COLLISION_BACKEND_UNAVAILABLE",
    "OBJECT_NOT_FOUND",
    "GRASP_NOT_FOUND",
    "WORLD_NOT_BUILT",
    "PAIR_IGNORED",
    "COLLISION",
    "NO_COLLISION",
    "MODEL_LOAD_FAILED",
    "IK_CONVERGED",
    "IK_UNREACHABLE",
    "IK_FAILED",
    "IK_BACKEND_UNAVAILABLE",
    "IK_DISCONTINUITY",
    "JOINT_LIMIT_VIOLATION",
    "VELOCITY_LIMIT_VIOLATION",
    "ACCELERATION_LIMIT_VIOLATION",
    "JERK_LIMIT_VIOLATION",
    "SINGULAR_JACOBIAN",
    "SINGULARITY_RISK",
    "COLLISION_DETECTED",
    "CLEARANCE_TOO_LOW",
    "PATH_PLANNING_FAILED",
    "RRT_FAILED",
    "PRM_FAILED",
    "PATH_REPAIR_FAILED",
    "TRAJECTORY_GENERATION_FAILED",
    "TRAJECTORY_VALIDATION_FAILED",
    "UNSUPPORTED_ROBOT_MODEL",
    "UNSUPPORTED_IK_BACKEND",
    "INVALID_AXIS",
    "INVALID_DISTANCE",
    "BACKEND_UNAVAILABLE",
    "NOT_IMPLEMENTED",
    "TOLERANCE_NOT_MET",
    "MAX_ITERATIONS",
]


class AlgorithmError(BaseModel):
    code: ReasonCode
    message: str
    details: Dict[str, Any] = Field(default_factory=dict)


class Transform3D(BaseModel):
    parent_frame: str
    child_frame: str
    matrix: List[List[float]]

    @field_validator("matrix")
    @classmethod
    def _shape(cls, value):
        if len(value) != 4 or any(len(row) != 4 for row in value):
            raise ValueError("matrix must be 4x4")
        return value


class Pose6D(BaseModel):
    frame_id: str
    position: List[float] = Field(..., min_length=3, max_length=3)
    quaternion_xyzw: List[float] = Field(..., min_length=4, max_length=4)


class RobotModelConfig(BaseModel):
    robot_id: str
    urdf_path: Optional[str] = None
    package_dirs: List[str] = Field(default_factory=list)
    base_frame: str = "base"


class TCPConfig(BaseModel):
    tcp_id: str = "tcp"
    transform: Transform3D


class GripperConfig(BaseModel):
    gripper_id: str
    mesh_path: Optional[str] = None
    root_frame: str = "gripper"
    tcp: Optional[TCPConfig] = None


class ObjectAssetConfig(BaseModel):
    object_id: str
    mesh_path: str
    frame_id: str
    scale: float = 1.0
    point_cloud_mode: Literal["reject", "convex_hull"] = "reject"


class BinAssetConfig(BaseModel):
    bin_id: str
    mesh_path: Optional[str] = None
    frame_id: str = "bin"
    size_xyz: Optional[List[float]] = Field(None, min_length=3, max_length=3)


class CollisionObjectConfig(BaseModel):
    object_id: str
    asset_path: Optional[str] = None
    frame_id: str
    pose: Transform3D
    group: Literal["robot", "gripper", "object", "bin", "fixture", "world"] = "world"
    scale: float = 1.0
    size_xyz: Optional[List[float]] = Field(None, min_length=3, max_length=3)


class CollisionPairRule(BaseModel):
    object_a: str
    object_b: str
    action: Literal["check", "allow", "ignore"] = "check"
    reason: Optional[str] = None


class CollisionMatrix(BaseModel):
    default_action: Literal["check", "allow", "ignore"] = "check"
    rules: List[CollisionPairRule] = Field(default_factory=list)


class DistanceQueryResult(BaseModel):
    object_a: str
    object_b: str
    distance: Optional[float] = None
    nearest_point_a: Optional[List[float]] = None
    nearest_point_b: Optional[List[float]] = None
    in_collision: bool = False
    ok: bool = True
    error: Optional[AlgorithmError] = None


class CollisionCheckResult(BaseModel):
    collision: bool
    colliding_pairs: List[List[str]] = Field(default_factory=list)
    contacts: List[Dict[str, Any]] = Field(default_factory=list)
    ok: bool = True
    error: Optional[AlgorithmError] = None


class KinematicJointConfig(BaseModel):
    name: str
    parent_frame: str
    child_frame: str
    joint_type: Literal["revolute", "prismatic", "fixed"] = "revolute"
    axis: List[float] = Field(default_factory=lambda: [0.0, 0.0, 1.0], min_length=3, max_length=3)
    origin: Transform3D
    lower: float = -3.141592653589793
    upper: float = 3.141592653589793


class KinematicChainConfig(BaseModel):
    chain_id: str
    base_frame: str
    tip_frame: str
    joints: List[KinematicJointConfig]
    tcp: Optional[TCPConfig] = None


class FKRequest(BaseModel):
    chain: KinematicChainConfig
    joint_positions: Dict[str, float]
    target_frame: Optional[str] = None


class FKResult(BaseModel):
    ok: bool
    transforms: Dict[str, Transform3D] = Field(default_factory=dict)
    error: Optional[AlgorithmError] = None


class JacobianRequest(BaseModel):
    chain: KinematicChainConfig
    joint_positions: Dict[str, float]
    frame_id: Optional[str] = None


class JacobianResult(BaseModel):
    ok: bool
    frame_id: str
    jacobian: List[List[float]] = Field(default_factory=list)
    condition_number: Optional[float] = None
    error: Optional[AlgorithmError] = None


class IKRequest(BaseModel):
    chain: KinematicChainConfig
    target: Transform3D
    seed: Dict[str, float] = Field(default_factory=dict)
    max_iterations: int = 100
    position_tolerance: float = 1e-4
    orientation_tolerance: float = 1e-3
    damping: float = 1e-3
    singularity_threshold: float = 1e8
    mode: Literal["full", "position", "orientation"] = "full"
    position_weight: float = 1.0
    orientation_weight: float = 1.0
    minimum_joint_motion_weight: float = 0.0
    preferred_posture: Dict[str, float] = Field(default_factory=dict)
    preferred_posture_weight: float = 0.0
    joint_limit_avoidance_weight: float = 0.0
    collision_callback: Any = None
    jacobian_callback: Any = None
    robot_model: Any = None
    return_all_solutions: bool = False
    timeout: Optional[float] = None


class IKResult(BaseModel):
    ok: bool
    joint_positions: Dict[str, float] = Field(default_factory=dict)
    all_candidate_solutions: List[Dict[str, float]] = Field(default_factory=list)
    best_solution: Dict[str, float] = Field(default_factory=dict)
    iterations: int = 0
    position_error: Optional[float] = None
    orientation_error: Optional[float] = None
    residual_total: Optional[float] = None
    backend_used: Optional[str] = None
    singularity_metric: Optional[float] = None
    collision_status: Optional[bool] = None
    reason: ReasonCode = "OK"
    error: Optional[AlgorithmError] = None
    debug_info: Dict[str, Any] = Field(default_factory=dict)


class GraspCandidate(BaseModel):
    grasp_id: str
    object_id: str
    tcp_in_object: Transform3D
    pregrasp_in_object: Optional[Transform3D] = None
    score: float = 0.0
    metadata: Dict[str, Any] = Field(default_factory=dict)


class GraspFeasibilityResult(BaseModel):
    grasp_id: str
    feasible: bool
    ik: Optional[IKResult] = None
    collision: Optional[CollisionCheckResult] = None
    distance_results: List[DistanceQueryResult] = Field(default_factory=list)
    rejection_reasons: List[AlgorithmError] = Field(default_factory=list)


class MinimumDistanceRequest(BaseModel):
    object_a: Optional[str] = None
    object_b: Optional[str] = None
    object_id: Optional[str] = None


class CollisionCheckRequest(BaseModel):
    object_a: Optional[str] = None
    object_b: Optional[str] = None


class GraspFeasibilityRequest(BaseModel):
    candidate: GraspCandidate
    ik_request: Optional[IKRequest] = None
    collision_request: Optional[CollisionCheckRequest] = None


class LoadedAssetStatus(BaseModel):
    asset_id: str
    asset_type: Literal["robot", "gripper", "object", "bin", "fixture", "world"]
    ok: bool
    frame_id: Optional[str] = None
    path: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[AlgorithmError] = None


class CollisionGeometryStatus(BaseModel):
    object_id: str
    ok: bool
    frame_id: Optional[str] = None
    coal_ready: bool = False
    vertex_count: int = 0
    face_count: int = 0
    aabb_min: Optional[List[float]] = None
    aabb_max: Optional[List[float]] = None
    error: Optional[AlgorithmError] = None


class FrameValidationResult(BaseModel):
    ok: bool
    checked_frames: List[str] = Field(default_factory=list)
    errors: List[AlgorithmError] = Field(default_factory=list)


class UISceneRequest(BaseModel):
    robot: Optional[RobotModelConfig] = None
    gripper: Optional[GripperConfig] = None
    gripper_pose: Optional[Transform3D] = None
    tcp: Optional[TCPConfig] = None
    object_asset: Optional[ObjectAssetConfig] = None
    object_pose: Optional[Transform3D] = None
    bin: Optional[BinAssetConfig] = None
    bin_pose: Optional[Transform3D] = None
    collision_objects: List[CollisionObjectConfig] = Field(default_factory=list)
    collision_matrix: CollisionMatrix = Field(default_factory=CollisionMatrix)
    grasp_candidates: List[GraspCandidate] = Field(default_factory=list)
    joint_state: Dict[str, float] = Field(default_factory=dict)
    target_grasp_id: Optional[str] = None
    chain: Optional[KinematicChainConfig] = None


class UISceneStatus(BaseModel):
    ok: bool
    loaded_assets: List[LoadedAssetStatus] = Field(default_factory=list)
    collision_geometry: List[CollisionGeometryStatus] = Field(default_factory=list)
    frame_validation: FrameValidationResult
    active_collision_pairs: List[List[str]] = Field(default_factory=list)
    world_object_ids: List[str] = Field(default_factory=list)
    errors: List[AlgorithmError] = Field(default_factory=list)


class GraspByIdRequest(BaseModel):
    target_grasp_id: str
    ik_request: Optional[IKRequest] = None
    collision_request: Optional[CollisionCheckRequest] = None
    distance_request: Optional[MinimumDistanceRequest] = None


class UISceneEvaluationRequest(BaseModel):
    scene: UISceneRequest
    distance_request: Optional[MinimumDistanceRequest] = None
    collision_request: Optional[CollisionCheckRequest] = None
    fk_request: Optional[FKRequest] = None
    jacobian_request: Optional[JacobianRequest] = None
    ik_request: Optional[IKRequest] = None
    target_grasp_id: Optional[str] = None


class UISceneEvaluationResult(BaseModel):
    ok: bool
    scene_status: UISceneStatus
    distance_results: List[DistanceQueryResult] = Field(default_factory=list)
    collision: Optional[CollisionCheckResult] = None
    fk: Optional[FKResult] = None
    jacobian: Optional[JacobianResult] = None
    ik: Optional[IKResult] = None
    grasp_feasibility: Optional[GraspFeasibilityResult] = None
    errors: List[AlgorithmError] = Field(default_factory=list)


class FrameRef(BaseModel):
    frame_id: str


class FrameTransform(Transform3D):
    pass


class ErrorInfo(BaseModel):
    error_code: str = "OK"
    error_message: str = ""
    details: Dict[str, Any] = Field(default_factory=dict)


class FixtureConfig(CollisionObjectConfig):
    group: Literal["fixture"] = "fixture"


class ClearanceResult(BaseModel):
    ok: bool
    minimum_clearance: Optional[float] = None
    error: Optional[AlgorithmError] = None


class IKBackendConfig(BaseModel):
    backend: Literal["auto", "DLS", "LM", "OPTIMIZATION", "SQP", "ANALYTICAL", "EAIK", "PINOCCHIO"] = "auto"


class IKConstraintConfig(BaseModel):
    position_tolerance: float = 1e-4
    orientation_tolerance: float = 1e-3
    singularity_threshold: float = 1e8


class PlannerConfig(BaseModel):
    planner: Literal["direct", "cartesian_linear", "rrt", "rrt_connect", "collision_aware"] = "collision_aware"
    max_joint_step: float = 0.1
    max_iterations: int = 1000
    timeout: float = 5.0


class PathConstraintConfig(BaseModel):
    minimum_clearance: float = 0.0
    singularity_threshold: float = 1e8


class PathPlanningRequest(BaseModel):
    start: List[float]
    goal: List[float]
    lower_limits: Optional[List[float]] = None
    upper_limits: Optional[List[float]] = None
    planner: PlannerConfig = Field(default_factory=PlannerConfig)


class PathPlanningResult(BaseModel):
    success: bool
    q_waypoints: List[List[float]] = Field(default_factory=list)
    planner_used: str = ""
    rejection_reason: str = "OK"
    debug_info: Dict[str, Any] = Field(default_factory=dict)


class TrajectoryOptions(BaseModel):
    method: Literal["cubic", "quintic", "trapezoidal"] = "trapezoidal"
    velocity_limits: List[float] = Field(default_factory=list)
    acceleration_limits: List[float] = Field(default_factory=list)
    sample_count: int = 101


class TrajectoryRequest(BaseModel):
    q_waypoints: List[List[float]]
    options: TrajectoryOptions = Field(default_factory=TrajectoryOptions)


class TrajectoryResult(BaseModel):
    success: bool
    trajectory: Dict[str, Any] = Field(default_factory=dict)
    rejection_reason: str = "OK"


class TrajectoryValidationRequest(BaseModel):
    trajectory: Dict[str, Any]
    joint_limits: Optional[List[List[float]]] = None
    velocity_limits: Optional[List[float]] = None
    acceleration_limits: Optional[List[float]] = None


class TrajectoryValidationResult(BaseModel):
    success: bool
    failed_waypoint_index: Optional[int] = None
    rejection_reason: str = "OK"


class MotionType(str, Enum):
    JOINT = "JOINT"
    LINEAR = "LINEAR"
    COLLISION_AWARE = "COLLISION_AWARE"


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


class ExtractOptions(BaseModel):
    directions: List[str] = Field(default_factory=lambda: ["+Z", "+X", "-X", "+Y", "-Y"])


class MoveJRequest(BaseModel):
    motion_request: Any


class MoveLRequest(BaseModel):
    motion_request: Any


class MotionSegment(BaseModel):
    name: str
    request: Any


class MotionSequence(BaseModel):
    segments: List[MotionSegment] = Field(default_factory=list)


class MotionSequenceResult(BaseModel):
    success: bool
    segments: List[Any] = Field(default_factory=list)
    rejection_reason: str = "OK"


class PickSequenceRequest(BaseModel):
    sequence: Any


class PickSequenceResult(MotionSequenceResult):
    pass
