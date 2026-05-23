# collision

Collision detection pipeline: geometry loading, broadphase culling, narrowphase Coal queries, distance computation, path-level checking, and continuous collision validation.

---

## `collision_object.py`

### `CollisionObject` (dataclass)

| Field | Type | Description |
|---|---|---|
| `object_id` | `str` | Unique name |
| `geometry` | `CollisionGeometry` | Mesh or box geometry with Coal handle |
| `pose` | `Transform3D` | Current pose in world/robot frame |
| `group` | `str` | `"world"`, `"robot"`, `"attached_object"` |
| `enabled` | `bool` | Whether included in collision queries |
| `metadata` | `dict` | Grasp/attachment data |

#### `coal_object() → coal.CollisionObject | None`

Wraps `geometry.coal_geometry` with the current pose as a Coal transform:
`coal.CollisionObject(coal_geometry, matrix_to_coal_transform(matrix))`

#### `world_aabb() → np.ndarray (2×3)`

Transforms all 8 AABB corners through the world pose matrix and returns the axis-aligned bounding box in world space:
```
corners = all combinations of (x_min|x_max, y_min|y_max, z_min|z_max)
world_corners = (matrix @ corners.T).T[:, :3]
return (world_corners.min(0), world_corners.max(0))
```

---

## `collision_world.py`

### `CollisionWorld`

Container for all `CollisionObject` instances and the `CollisionMatrix`.

Key methods:

| Method | Description |
|---|---|
| `add_object(obj)` | Registers by `object_id` |
| `add_from_config(config)` | Builds geometry from `asset_path` (mesh) or `size_xyz` (box) |
| `update_pose(object_id, pose)` | Updates live pose |
| `attach_object(id, frame, grasp_tf)` | Marks object as attached; stores grasp transform in metadata |
| `detach_object(id)` | Clears attachment metadata |
| `active_pairs()` | Delegates to `CollisionMatrix.active_pairs(object_ids)` |
| `check_state(q)` | Updates robot state, calls `check_active_pairs` |
| `clone()` | Shallow copy (shared geometry, independent dict) |

---

## `collision_matrix.py`

### `CollisionMatrix`

Policy table: for any `(object_a, object_b)` pair, returns an action: `"check"`, `"allow"`, or `"ignore"`.

#### `action_for(a, b) → str`

Looks up `_rules[sorted(a,b)]`; falls back to `schema.default_action` (`"check"`).

#### `active_pairs(object_ids) → List[Tuple[str, str]]`

`combinations(sorted(ids), 2)` filtered to those where `action == "check"`.

#### `set_rule(rule) / add_rule(a, b, action)`

Upsert into the rules dict. Keys are always `tuple(sorted(a, b))`.

---

## `broadphase.py`

### `aabb_overlap(a_bounds, b_bounds, margin=0.0) → bool`

Separating-axis AABB test:
```
overlap = all(a.min ≤ b.max + margin) and all(b.min ≤ a.max + margin)
```

### `broadphase_candidates(objects, matrix, margin) → List[Tuple[str, str]]`

Returns pairs that pass both the `CollisionMatrix` filter and the AABB overlap test. Used to prune expensive narrowphase checks.

---

## `collision_checker.py`

### `check_pair(a, b) → CollisionCheckResult`

Runs Coal narrowphase if both objects have valid Coal geometry:
```python
req = coal.CollisionRequest()
res = coal.CollisionResult()
collided = coal.collide(coal_a, coal_b, req, res)
```
Returns contact positions, normals, and penetration depths when Coal exposes them for the shape pair. `aabb_distance` remains a separate broadphase helper.

### `check_active_pairs(world) → CollisionCheckResult`

Iterates `world.active_pairs()`, calls `check_pair` for each, accumulates all contacts.

### `check_scene(world) → CollisionCheckResult`

Alias for `check_active_pairs`.

---

## `distance_queries.py`

### `minimum_distance_pair(a, b) → DistanceQueryResult`

Coal distance query:
```python
req = coal.DistanceRequest()
res = coal.DistanceResult()
distance = coal.distance(coal_a, coal_b, req, res)
```
Returns `distance`, `nearest_point_a`, `nearest_point_b`, `in_collision` (distance ≤ 0).

AABB distance (`aabb_distance`) remains available for broadphase checks.

### `aabb_distance(bounds_a, bounds_b) → (float, point_a, point_b, bool)`

```
sep = max(0, max(b.min - a.max, a.min - b.max))   per axis
distance = ‖sep‖₂
colliding = sep == 0 on all axes
```

Nearest points are computed by clamping the other box's centre to this box's extents.

### `minimum_distances_active_pairs(world) → List[DistanceQueryResult]`

Calls `minimum_distance_pair` for every active pair in the collision world.

---

## `narrowphase.py`

### `pairwise_collision_query(a, b) → CollisionCheckResult`

Thin wrapper for `check_pair`.

### `pairwise_distance_query(a, b) → DistanceQueryResult`

Thin wrapper for `minimum_distance_pair`.

---

## `path_collision_checker.py`

### `PathCollisionResult` (dataclass)

| Field | Description |
|---|---|
| `success` | Path is collision-free |
| `collision` | Collision detected |
| `first_collision_waypoint` | Index into waypoint list |
| `first_collision_segment` | Segment index (between waypoints i and i+1) |
| `minimum_clearance` | Minimum clearance seen across all samples |
| `interpolation_samples` | Total state evaluations |

### `PathCollisionChecker`

Wraps either a `state_checker` callable `(q) → bool/dict` or a `CollisionWorld`.

#### `adaptive_subdivision(q0, q1, max_joint_delta) → List[np.ndarray]`

```
n = ceil(‖q1 - q0‖_∞ / max_joint_delta) + 1
```
Returns `n` linearly interpolated joint states.

#### `check_segment(q0, q1) → PathCollisionResult`

Conservative continuous collision: interpolates the segment with `adaptive_subdivision` and checks every sample. Not an exact swept-mesh test — guarantees detection when the step size is smaller than the object's minimum feature size relative to max joint motion.

#### `check_path(q_waypoints) → PathCollisionResult`

1. `check_waypoints`: visits each waypoint directly.
2. For each consecutive pair `(q_i, q_{i+1})`: calls `check_segment`.

Reports first failure with `first_collision_waypoint` or `first_collision_segment`.

---

## `continuous_collision.py`

### `conservative_continuous_collision(q0, q1, ..., resolution=0.05)`
### `conservative_continuous_path_collision(q_waypoints, ..., resolution=0.05)`

Top-level API for conservative swept-collision validation. Delegates to `PathCollisionChecker.check_segment` / `.check_path`. Resolution is the maximum joint step between interpolated samples.

---

## `clearance.py`

### `clearance_margin_for_state(q, checker) → float | None`

Returns `checker.check_state(q).minimum_clearance`.

### `clearance_margin_for_path(q_waypoints, checker) → float | None`

Returns `checker.check_path(q_waypoints).minimum_clearance`.

### `check_clearance_above_threshold(q_waypoints, checker, threshold) → PathCollisionResult`

Checks path and additionally marks `success=False` if `minimum_clearance < threshold`.

---

## `swept_volume.py`

### `swept_volume(*args) → APIResult`

Not implemented. Returns `NOT_IMPLEMENTED` error. Exact analytic swept-volume geometry would require computing the Minkowski sum of the link meshes along the joint trajectory.

### `conservative_swept_validation(q0, q1, ..., resolution=0.05)`

Delegates to `conservative_continuous_collision` — interpolated discrete check.

---

## `attached_object.py`

### `attach_object(world, object_id, link_frame, grasp_transform) → AttachedObject`

Stores `grasp_transform` and `attached_to` in the object's metadata. Sets group to `"attached_object"`.

### `update_attached_object_pose(world, object_id, parent_pose)`

Propagates parent frame motion to the attached object:
```
T_object_world = T_parent_world @ T_grasp
```

---

## `geometry_loader.py`

Factory dispatching `CollisionObjectConfig`, `ObjectAssetConfig`, `GripperConfig`, and `BinAssetConfig` to the appropriate `geometry_from_asset` or `box_geometry` constructor.
