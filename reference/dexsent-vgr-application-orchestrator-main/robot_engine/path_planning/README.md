# path_planning

Sampling-based and deterministic joint-space path planners, Cartesian linear planner, path repair, and shortcut smoothing.

---

## `planner_base.py`

### `PathRequest` (dataclass)

| Field | Default | Description |
|---|---|---|
| `start` | — | Start joint config (array-like) |
| `goal` | — | Goal joint config |
| `joint_limits` | `None` | `(lower, upper)` arrays |
| `state_validity_fn` | `None` | `(q) → bool`, `True` = collision-free |
| `max_joint_step` | `0.1` | Maximum joint displacement per step (rad/m) |
| `max_iterations` | `1000` | Maximum tree/graph nodes |
| `timeout` | `5.0` | Planning timeout (seconds) |
| `goal_bias` | `0.1` | Probability of sampling goal directly (RRT) |
| `debug_info` | `{}` | Planner-specific extras (IK functions, etc.) |

### `PathResult` (dataclass)

| Field | Description |
|---|---|
| `success` | Planning succeeded |
| `q_waypoints` | Joint-space path |
| `cartesian_waypoints` | Cartesian frames (Cartesian planner only) |
| `planner_used` | Name of planner that produced the result |
| `length` | Sum of `‖q_{i+1} - q_i‖` segment norms |
| `minimum_clearance` | Smallest clearance seen |
| `rejection_reason` | `"OK"` or error code string |

### `PlannerBase`

Abstract base. `validate_request` checks shape consistency. Subclasses override `plan(request) → PathResult`.

---

## `joint_direct_planner.py`

### `JointDirectPlanner`

Straight-line interpolation in joint space. No collision avoidance — used as a fast first attempt.

#### `plan(request) → PathResult`

```
n = ceil(‖goal - start‖_∞ / max_joint_step) + 1
path = [start + (goal - start) * t  for t in linspace(0, 1, n)]
```

Checks each waypoint with `state_validity_fn`. Returns `COLLISION_DETECTED` with `failed_waypoint_index` on first collision.

---

## `rrt.py`

### `RRTPlanner`

Rapidly-exploring Random Tree (RRT) — single tree grown from start.

#### `plan(request) → PathResult`

Main loop:
1. Sample: `q_rand = goal` with probability `goal_bias`, else `U(lower, upper)`.
2. Nearest: `q_near = argmin_i ‖nodes_i - q_rand‖`.
3. Steer: `q_new = q_near + direction/‖direction‖ · min(max_joint_step, ‖direction‖)`.
4. Edge valid? Interpolate `q_near → q_new` at `max_joint_step` resolution, check all samples.
5. If `‖q_new - goal‖ ≤ max_joint_step` and direct connection is valid: path found.

Path extraction: walk parent pointers from new node to root, reverse.

---

## `rrt_connect.py`

### `RRTConnectPlanner` (extends `RRTPlanner`)

Bidirectional RRT-Connect — two trees, one from start (`ta`), one from goal (`tb`).

#### `plan(request) → PathResult`

Each iteration:
1. Extend `ta` toward a random sample → `q_new_a`.
2. **Connect** `tb` toward `q_new_a`: repeatedly extend until stuck or within `max_joint_step`.
3. If both trees meet (`‖q_new_a - q_new_b‖ ≤ max_joint_step`): merge paths.
4. Swap `ta ↔ tb` (alternate extending).

Path merge:
```
path = path_from_ta_root_to_new + reversed(path_from_tb_root_to_new)
```

Significantly faster than single-tree RRT for typical robot workspaces.

---

## `birrt.py`

### `BiRRTPlanner`

Subclass of `RRTConnectPlanner` with `planner_name = "BIRRT"`. Identical algorithm, alternate name.

---

## `prm.py`

### `PRMPlanner`

Probabilistic Roadmap Method — samples a roadmap offline then queries with Dijkstra.

#### `plan(request) → PathResult`

**Build phase:**
1. Start and goal added as nodes 0 and 1.
2. Sample up to `max_samples` random valid configs in `[lower, upper]`.

**Connect phase:**
For each node, find `k_nearest` neighbours within `max_edge_length`:
- Edge valid? Interpolate and check all samples at `max_joint_step` resolution.
- Add bidirectional edge with Euclidean cost.

**Query phase:**
Dijkstra from node 0 to node 1:
```
dist[goal] = min sum of edge weights along path
```
Returns the waypoint sequence along the shortest path.

---

## `cartesian_linear_planner.py`

### `CartesianLinearPlanner`

Plans a straight Cartesian TCP path (MoveL).

#### `plan(request) → PathResult`

Requires `debug_info["ik_fn"]` or `debug_info["ik_request_factory"]`.

1. Sample Cartesian frames:

```
n = ceil(max(‖p1-p0‖/translation_step, ‖ω‖/rotation_step)) + 1
```

2. Verify straight-line: for each intermediate frame, measure perpendicular deviation from the `start→goal` line. Fails if deviation > `cartesian_line_tolerance`.

3. IK at each frame with seed continuity check:

```
if ‖q_new - q_prev‖_∞ > continuity_joint_step: IK_DISCONTINUITY
```

4. Collision check each solved config.

#### `_straight_line_error(frames) → float`

```
for each frame p:
    α = ((p - start) · direction) / (direction · direction)
    projected = start + α · direction
    error = max error, ‖p - projected‖
```

---

## `collision_aware_planner.py`

### `CollisionAwarePlanner`

Cascaded planner: try direct first, fall back to RRT-Connect + smoothing.

#### `plan(request) → PathResult`

1. `JointDirectPlanner` — if collision-free: return immediately.
2. `RRTConnectPlanner` — find a feasible path.
3. `shortcut_smooth_path` with 100 iterations.
4. Validate smoothed path.

Reports `direct_rejection_reason` and smoothing stats in `debug_info`.

---

## `shortcut_smoothing.py`

### `remove_redundant_waypoints(q_waypoints, tolerance=1e-9)`

Removes consecutive duplicates `‖q_i - q_{i-1}‖ ≤ tolerance`.

### `validate_shortcut(q_i, q_j, collision_checker) → bool`

Calls `collision_checker.check_segment(q_i, q_j).success`.

### `shortcut_smooth_path(q_waypoints, collision_checker, iterations=100) → (path, stats)`

Random shortcut smoothing:
```
for _ in range(iterations):
    i, j = random pair with j > i + 1
    if validate_shortcut(path[i], path[j]):
        path = path[:i+1] + path[j:]   # remove intermediate waypoints
```

Returns smoothed path and `{"accepted": n, "attempted": iterations}`.

---

## `path_repair.py`

### `find_first_colliding_segment(q_waypoints, collision_checker) → int | None`

Returns `result.first_collision_segment`.

### `local_repair_with_rrt(q_before, q_after, scene) → PathResult`

Calls `RRTConnectPlanner` on the sub-problem `q_before → q_after`.

### `splice_repaired_segment(original, repaired, start_index, end_index) → path`

```
result = original[:start+1] + repaired_middle + original[end:]
```

Strips boundary duplicates before splicing.

### `repair_path(q_waypoints, collision_checker, scene, smoothing_iterations=100) → (path, stats)`

Full repair pipeline:
1. Identify first colliding segment.
2. Local RRT repair of that segment.
3. Splice repaired segment back.
4. Shortcut smooth the full repaired path.
5. Validate final path.

---

## `constraints.py`

### `PathPlanningConstraints` (dataclass)

| Field | Default | Description |
|---|---|---|
| `joint_limits` | `None` | `(lower, upper)` |
| `singularity_threshold` | `1e8` | Max condition number |
| `minimum_clearance_threshold` | `0.0` | Min allowed clearance |
| `max_joint_step` | `0.1` | Step size for interpolation |
| `max_planning_time` | `5.0` | Timeout |
| `max_iterations` | `1000` | |
| `keep_gripper_upright` | `False` | Orientation constraint |
| `maintain_approach_direction` | `False` | Approach axis constraint |
| `avoid_bin_walls` | `False` | Extra clearance around bin |

---

## `planning_scene.py`

### `PlanningScene` (dataclass)

Container aggregating all planning context:
- `robot_model`, `collision_world`, `collision_matrix`
- `frame_graph`, `current_joint_state`, `attached_objects`
- `joint_limits`, `velocity_limits`, `acceleration_limits`
