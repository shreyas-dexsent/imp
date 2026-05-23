# interfaces

Public API contracts: Pydantic schemas, error codes, result types, and the UI API surface.

---

## `error_codes.py`

### `ErrorCode` (str Enum)

Canonical error identifiers used in `AlgorithmError` and `APIResult`. All string values match the enum name.

| Category | Codes |
|---|---|
| Generic | `OK`, `INVALID_REQUEST`, `NOT_IMPLEMENTED` |
| Geometry | `INVALID_TRANSFORM`, `FRAME_NOT_FOUND`, `INVALID_FRAME_CHAIN`, `INVALID_QUATERNION`, `INVALID_ROTATION_MATRIX`, `UNSUPPORTED_ASSET_FORMAT`, `EMPTY_MESH` |
| Collision | `COLLISION_BACKEND_UNAVAILABLE`, `COLLISION_DETECTED`, `CLEARANCE_TOO_LOW` |
| Kinematics | `IK_FAILED`, `IK_BACKEND_UNAVAILABLE`, `IK_DISCONTINUITY`, `JOINT_LIMIT_VIOLATION`, `SINGULARITY_RISK` |
| Limits | `VELOCITY_LIMIT_VIOLATION`, `ACCELERATION_LIMIT_VIOLATION`, `JERK_LIMIT_VIOLATION` |
| Planning | `PATH_PLANNING_FAILED`, `RRT_FAILED`, `PRM_FAILED`, `PATH_REPAIR_FAILED` |
| Trajectory | `TRAJECTORY_GENERATION_FAILED`, `TRAJECTORY_VALIDATION_FAILED` |
| Model | `UNSUPPORTED_ROBOT_MODEL`, `UNSUPPORTED_IK_BACKEND`, `BACKEND_UNAVAILABLE`, `TOLERANCE_NOT_MET` |

### `not_implemented(message) → ErrorInfo`

Convenience constructor: `ErrorInfo(error_code=NOT_IMPLEMENTED, error_message=message)`.

---

## `result_types.py`

### `APIResult`

Generic result wrapper. Provides:
- `APIResult.fail(error_code, message)` — factory for failure results.
- `ok: bool`, `error_code: ErrorCode`, `error_message: str`.

### `ErrorInfo`

Lightweight struct: `error_code: ErrorCode`, `error_message: str`.

---

## `schemas.py`

Pydantic models defining the full data contract between planner components. Key schemas:

### `Transform3D`
```
parent_frame: str
child_frame: str
matrix: List[List[float]]  # 4×4 row-major
```

### `AlgorithmError`
```
code: str
message: str
details: Optional[dict]
```

### `KinematicChainConfig`
```
base_frame: str
tip_frame: str
joints: List[JointConfig]
tcp: Optional[TCPConfig]
```

### `JointConfig`
```
name: str
joint_type: Literal["revolute", "prismatic", "fixed"]
parent_frame: str
child_frame: str
origin: Transform3D
axis: List[float]   # 3-vector, unit axis of motion
lower: float        # rad or m
upper: float
```

### `FKRequest / FKResult`
- Request: `chain: KinematicChainConfig`, `joint_positions: Dict[str, float]`, `target_frame: Optional[str]`
- Result: `ok: bool`, `transforms: Dict[str, Transform3D]`, `error: Optional[AlgorithmError]`

### `IKRequest`
- `chain`, `target: Transform3D`, `seed: Dict[str, float]`
- `max_iterations: int`, `position_tolerance: float`, `orientation_tolerance: float`
- `damping: float`, `singularity_threshold: float`

### `IKResult`
- `ok: bool`, `joint_positions: Dict[str, float]`
- `iterations: int`, `position_error: float`, `orientation_error: float`
- `reason: str`, `error: Optional[AlgorithmError]`

### `JacobianRequest / JacobianResult`
- Request: `chain`, `joint_positions`, `frame_id: Optional[str]`
- Result: `ok`, `jacobian: List[List[float]]` (6×n), `condition_number: Optional[float]`

### `CollisionCheckResult`
```
collision: bool
colliding_pairs: List[List[str]]
contacts: List[dict]   # {position, normal, penetration_depth}
ok: bool
error: Optional[AlgorithmError]
```

### `DistanceQueryResult`
```
object_a: str
object_b: str
distance: Optional[float]
nearest_point_a: Optional[List[float]]
nearest_point_b: Optional[List[float]]
in_collision: bool
```

### `GraspCandidate`
```
grasp_id: str
score: float
grasp_transform: Transform3D   # T_object_gripper
```

### `GraspFeasibilityRequest / GraspFeasibilityResult`
- Request: `candidate: GraspCandidate`, `ik_request: Optional[IKRequest]`
- Result: `grasp_id`, `feasible: bool`, `ik: IKResult`, `collision: CollisionCheckResult`, `rejection_reasons: List[AlgorithmError]`

### `ObjectAssetConfig`
```
object_id: str
mesh_path: str
frame_id: str
scale: float = 1.0
point_cloud_mode: str = "convex_hull"
```

### `CollisionMatrix`
```
default_action: str = "check"
rules: List[CollisionPairRule]
```

### `CollisionPairRule`
```
object_a: str
object_b: str
action: Literal["check", "allow", "ignore"]
```

---

## `ui_api.py`

UI-facing API surface. Provides thin typed wrappers over the planning and kinematics modules, returning structured `APIResult` objects suitable for JSON serialisation to the frontend. Acts as the boundary between the React/UI layer and the robot engine internals.
