# motion

High-level motion planning API: frame offsets, approach/retreat patterns, joint-space and linear moves, trajectory generation, time-parameterisation, and validation.

---

## `motion_request.py`

Key request/options dataclasses:

| Class | Key fields |
|---|---|
| `MotionRequest` | `motion_type`, `target_frame`, `chain`, `seed`, `ik_options`, `approach`, `retreat`, `trajectory_options` |
| `ApproachOptions` | `enabled`, `distance`, `axis`, `direction`, `reference_frame`, `motion_type` |
| `RetreatOptions` | Same as ApproachOptions |
| `TrajectoryOptions` | `max_joint_velocity`, `max_joint_acceleration`, `time_step` |
| `TrajectoryValidationRequest` | `trajectory`, `chain`, `ik_options`, `collision_options`, `trajectory_options` |
| `MotionType` | `JOINT`, `LINEAR` |

---

## `motion_result.py`

| Class | Key fields |
|---|---|
| `MotionSegmentResult` | `success`, `trajectory: JointTrajectory`, `rejection_reason`, `failed_stage` |
| `MotionSequenceResult` | `success`, `segments: List[MotionSegmentResult]`, `failed_stage`, `debug_info` |
| `JointTrajectory` | `joint_names`, `positions`, `times`, `velocities`, `accelerations` |
| `TrajectoryValidationResult` | `success`, `minimum_clearance`, `max_joint_motion`, `failed_stage`, `failed_waypoint_index`, `rejection_reason` |
| `MotionRejectionReason` | Enum: `INVALID_DISTANCE`, `IK_FAILED`, `COLLISION_DETECTED`, `SINGULARITY_RISK`, `JOINT_LIMIT_VIOLATION`, `VELOCITY_LIMIT_VIOLATION`, `ACCELERATION_LIMIT_VIOLATION`, `CLEARANCE_TOO_LOW`, `UNSUPPORTED_MOTION_TYPE` |

---

## `frame_offset.py`

### `offset_transform(request: FrameOffsetRequest) → Transform3D`

Computes a pre-grasp or post-grasp offset frame:
```
T_offset = T_target @ translate(axis * distance)
```
where `axis` is a unit vector (`+z`, `-z`, `+x`, etc.) in either the `target` or a specified `reference_frame`. The resulting child frame is named `output_child_frame`.

### `compute_offset_frame(T_target, axis, distance, reference_frame) → np.ndarray`

Core matrix computation. If `reference_frame` is world: translate in world space. If `reference_frame` matches target: translate in target's local frame.

---

## `approach_retreat.py`

### `plan_approach_to_frame(request) → MotionSequenceResult`

1. Validates `approach.distance > 0`.
2. Computes `T_approach = offset_transform(target, axis, -distance)`.
3. Plans a 2-segment sequence: `[approach_segment, target_segment]`.
4. `approach_segment` uses `approach.motion_type` (default: `JOINT`); `target_segment` uses the request's own type.

### `plan_retreat_from_frame(request) → MotionSequenceResult`

Mirror of approach: computes `T_retreat = offset_transform(target, axis, +distance)`, plans `[target_segment, retreat_segment]`.

---

## `joint_motion.py`

### `plan_joint_move_to_frame(request) → MotionSegmentResult`

1. Runs IK for `request.target_frame` with the provided seed.
2. Calls `CollisionAwarePlanner` (or `JointDirectPlanner` if no collision world) to plan a joint-space path from `q_seed` to `q_target`.
3. Removes duplicate waypoints.
4. Generates a time-parameterised `JointTrajectory` via `trajectory_generator`.

---

## `linear_motion.py`

### `plan_linear_move_to_frame(request) → MotionSegmentResult`

1. Samples Cartesian frames along the straight-line path from current TCP to target using `sample_cartesian_path`.
2. Solves IK at each frame (`CartesianLinearPlanner`), checking for IK continuity (`max_joint_step` threshold) at each step.
3. Validates the resulting path for collisions.
4. Time-parameterises.

---

## `path_planner.py`

### `plan_motion(request) → MotionSegmentResult`

Dispatcher:
- `MotionType.JOINT` → `plan_joint_move_to_frame`
- `MotionType.LINEAR` → `plan_linear_move_to_frame`
- Unknown → failure with `UNSUPPORTED_MOTION_TYPE`

### `export_robot_trajectory(request) → dict`

Serialises a `JointTrajectory` to a JSON-safe dict via `model_dump`.

---

## `motion_sequence.py`

### `plan_motion_sequence(sequence: MotionSequence) → MotionSequenceResult`

Iterates segments in order. Each segment is planned via `plan_motion`. On the first failure, returns immediately with the failed segment's reason and stage. On full success, returns all segment results.

---

## `path_smoothing.py`

### `remove_duplicate_joint_waypoints(chain, waypoints, tolerance=1e-12)`

Removes consecutive waypoints where `‖q_i - q_{i-1}‖ ≤ tolerance`.

---

## `time_parameterization.py`

### `time_parameterize_joint_path(joint_names, positions, max_velocity, time_step) → JointTrajectory`

Assigns timestamps to waypoints using the velocity-based rule:
```
Δt_i = max(time_step, ‖Δq_i‖_∞ / max_velocity)
```

Velocities: `v_i = (q_{i+1} - q_i) / Δt_i` (forward finite difference)
Accelerations: `a_i = (v_{i+1} - v_i) / Δt_i` (forward finite difference)

---

## `trajectory_generator.py`

### `generate_joint_trajectory(chain, waypoints, options) → JointTrajectory`

Converts `Dict[str, float]` waypoints to ordered joint position lists, then calls `time_parameterize_joint_path`.

---

## `trajectory_validator.py`

### `validate_trajectory(request: TrajectoryValidationRequest) → TrajectoryValidationResult`

Full per-waypoint validation pipeline:

1. **Joint limits**: `chain.violates_limits(q)` at each waypoint.
2. **Singularity**: `condition_number(J) > singularity_threshold`.
3. **Collision / clearance**: AABB distance from TCP point to all world objects.
4. **Velocity limits**: `max(|v|) ≤ max_joint_velocity + 1e-9`.
5. **Acceleration limits**: `max(|a|) ≤ max_joint_acceleration + 1e-9`.

Returns first failure with `failed_stage`, `failed_waypoint_index`, and `rejection_reason`.

#### Clearance check detail

Approximates TCP as a point and computes AABB distance:
```
clearance_i = aabb_distance(tcp_point_bounds, obj.world_aabb()) - tcp_clearance_radius
```
Fails if `clearance_i ≤ 0` (collision) or `clearance_i < minimum_clearance` (too close).
