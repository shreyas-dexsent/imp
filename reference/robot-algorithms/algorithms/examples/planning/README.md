# Path Planning

## 1. Description

Path planning produces a `Path` — an ordered sequence of joint configurations describing the geometry of a motion. **A path has no time, no velocity, no acceleration.** Time appears at the trajectory layer.

Three entry points:

| Function | Purpose | Returns |
|---|---|---|
| `plan_joint(model, scene, q_start, q_goal, ...)` | Joint-space search via OMPL or straight-line | `PathPlanResult` |
| `plan_cartesian(scene, robot_id, frame_id, T_start, T_goal, q_seed, ...)` | TCP straight-line with IK at each sample | `PathPlanResult` (with `cartesian_waypoints`) |
| `validate_path(model, scene, path, ...)` | Post-plan advisory checks | `PathValidationReport` |

A `PathPlanResult` with `status.SUCCESS` carries:

- `result.path.waypoints` — `(N, dof)` array in `model.active_joint_names` order.
- `result.path.cartesian_waypoints` — `(N, 4, 4)` for Cartesian paths, `None` otherwise.

Promises on success:

- `waypoints[0] == q_start` and `waypoints[-1] == q_goal` within tolerance.
- Every waypoint inside joint limits with the configured margin.
- Every segment collision-free at the planner's sampling resolution.

Not promised: optimal length, smooth velocities, validity under live state updates after the call.

## 2. Data Flow

```text
YAML (world)
        |
        v
WorldDescription -> CollisionModel -> Scene
                                       |
q_start, q_goal, options ------+       |
                               |       |
                               v       v
                       make_state_validity_fn(model, scene, ...)  -> q -> bool closure
                                       |
                               +-------+
                               |
                               v
                         plan_joint / plan_cartesian
                               |
        +----------------------+----------------------+
        v                                             v
   OMPLBackend (default)                       StraightLineBackend
   ompl.geometric.RRTConnect                   linear interp + per-sample validity
        |                                             |
        +----------------------+----------------------+
                               |
                               v
                       Path  -> validate_path -> PathValidationReport
                               |
                               v
                       (downstream: smoothing -> trajectory)
```

The state-validity closure combines:

- Joint limits with margin.
- Self-collision via `is_in_collision` against the scene's `CollisionModel`.
- Environment collision (everything in `Scene.object_poses`, including perception-added objects and attached objects).

A single closure is reused by every OMPL `isValid` callback, so the cost of validity checks is amortised across the search.

## 3. Usage

### Setup

```python
import numpy as np
from algorithms.descriptions import WorldDescription
from algorithms.kinematics import fk
from algorithms.planning import plan_joint, plan_cartesian, validate_path, PathStatus
from algorithms.resolved import CollisionModel, KinematicModel, Scene

world = WorldDescription.from_yaml("configs/worlds/my_world.yaml")
cm = CollisionModel.from_world(world)
scene = Scene.from_world(world, cm)
system = world.robot("arm").robot_system
model = KinematicModel.from_robot_system(system)
home = system.named_joint_state("home")
q_home = np.array([home[name] for name in model.active_joint_names])
```

### Joint-space plan

```python
result = plan_joint(model, scene, q_home, q_goal)
if result.status is PathStatus.SUCCESS:
    path = result.path
```

### Cartesian straight line

```python
T_goal = fk(scene, "arm", q_home, "robot_tcp").copy()
T_goal[0, 3] += 0.10
result = plan_cartesian(scene, "arm", "robot_tcp", None, T_goal, q_home)
```

The first `None` lets the planner derive `T_start` from FK on `q_seed`.

### Multi-robot composite planning

`plan_joint` accepts dicts for composite-state planning across all robots in the world simultaneously:

```python
result = plan_joint(
    model, scene,
    {"left_arm": q_left_start, "right_arm": q_right_start},
    {"left_arm": q_left_goal,  "right_arm": q_right_goal},
)
# result.path.waypoints is (N, left_dof + right_dof); composite=True in metadata.
```

Cross-robot collisions are checked automatically.

### PathStatus taxonomy

| Status | When |
|---|---|
| `SUCCESS` | Path returned and validated. |
| `INVALID_INPUT` | Wrong q shape, unknown backend, malformed target. |
| `START_OUT_OF_LIMITS` / `GOAL_OUT_OF_LIMITS` | Endpoint outside joint limits. |
| `START_IN_COLLISION` / `GOAL_IN_COLLISION` | Endpoint in collision with scene. |
| `NO_PATH_FOUND` | Planner ran but didn't connect. |
| `TIMEOUT` | OMPL hit time cap with only an approximate solution. |
| `MAX_ITERATIONS` | Iteration cap hit. |
| `IK_FAILED` | (Cartesian) IK couldn't reach a sampled pose. |
| `IK_DISCONTINUITY` | (Cartesian) IK flipped branches between samples. |
| `CARTESIAN_DEVIATION` | (Cartesian) sampled pose deviates from the straight line. |
| `NUMERICAL_FAILURE` | Unexpected exception. |
| `POST_PLAN_INVALID` | Reserved for post-plan rejection. |

### Performance (FR3, default options, N=50)

| Operation | Success | Median | p95 |
|---|---|---|---|
| `plan_joint` (OMPL RRTConnect), random reachable goal | 50 / 50 | 139 ms | 198 ms |
| `plan_joint` (direct line), small offset | 50 / 50 | 7 ms | 10 ms |
| `plan_cartesian`, 10 cm straight line | 50 / 50 | 101 ms | 115 ms |

Monte Carlo: 200 / 200 success at 27 ms median across random reachable goals.

### Defaults (`PlanOptions`)

| Field | Default | What it controls |
|---|---|---|
| `max_joint_step` | 0.05 rad | Edge / waypoint sampling granularity |
| `max_iterations` | 5000 | OMPL node cap |
| `max_time_ms` | 2000 | Wall-clock budget |
| `goal_bias` | 0.10 | OMPL RRT goal-sampling probability |
| `interpolation_waypoints` | 100 | Post-solve interpolation density |
| `cartesian_translation_step` | 0.005 m | Cartesian sampling step |
| `cartesian_rotation_step` | 0.02 rad | Cartesian sampling step |
| `cartesian_ik_continuity` | 0.5 rad | Max joint jump between consecutive Cartesian samples |
| `cartesian_line_tolerance` | 5e-4 m | Max deviation from start→goal line |
| `joint_margin` | 1e-3 rad | Validity margin inside joint limits |
| `random_seed` | 0 | OMPL determinism |
| `planner_name` | `"RRTConnect"` | OMPL planner choice |

## 4. Examples

| File | What it shows |
|---|---|
| `01_plan_joint_home_to_target.py` | Minimal joint-space plan. |
| `02_plan_cartesian_straight_line.py` | TCP straight-line with `cartesian_waypoints` populated. |
| `03_validate_path.py` | Valid path passes; spoiled path fails at the right waypoint. |
| `04_diagnose_planner_failure.py` | Four deliberate failure modes. |

For obstacle-avoidance and palletizing demonstrations see `examples/integration/`.

## 5. Common Errors

| Symptom | Cause | Fix |
|---|---|---|
| `START_IN_COLLISION` immediately | Home pose collides with the scene (often a YAML world that has the workpiece placed where the arm rests). | Check `is_in_collision(model, scene, q_home)`; either move start, remove the workpiece, or use a different home pose. |
| `NO_PATH_FOUND` after 2 seconds | Cluttered scene, narrow passage, or OMPL needs more time. | Raise `max_time_ms` and `max_iterations`; try `planner_name="BITstar"`; widen `max_joint_step`. |
| Cartesian `IK_DISCONTINUITY` | IK flipped to a different branch (elbow, wrist) between samples. | Tighten `cartesian_translation_step` so the next sample's seed is closer, or pre-resolve the branch with an explicit IK call. |
| Cartesian path doesn't deviate from line but plan fails | Likely an arm-link self-collision at an intermediate sample, not a TCP issue. | Inspect `validate_path` output; consider a joint-space `move_joint` instead. |
| Plan succeeds but `validate_path` rejects | Validator runs continuous-collision sweep at a finer resolution than the planner sampled. | Tighten `PlanOptions.max_joint_step` to match `PathValidationOptions.collision_step`. |
| OMPL warning about RNG seed = 0 | OMPL doesn't accept seed 0. | Set `random_seed=42` (any nonzero value). |

## 6. FAQs

**Q: Can the planner avoid obstacles?**
Yes. `plan_joint` with the default OMPL backend plans through `Scene.collision_model` automatically. See `examples/integration/01_obstacle_avoidance_a_to_b.py`.

**Q: Can `plan_cartesian` avoid obstacles?**
No — Cartesian planning is straight-line only. If the line is blocked, `plan_cartesian` returns `IK_FAILED` or `NO_PATH_FOUND`. The industrial pattern is to fall back to `plan_joint` for the blocked leg; see `examples/integration/02_linear_palletizing.py`.

**Q: How do I get a smooth trajectory from this path?**
Pipeline: `plan_joint` → `shortcut_smooth` → `spline_fit` → `time_parameterize`. The `move_joint` primitive (in `algorithms.primitives`) composes all of these in one call.

**Q: How is multi-robot composite planning different from per-robot planning?**
Composite planning plans over the concatenated configuration space `q = concat(q_robot_1, ..., q_robot_N)`. The composite path has shape `(N, dof_left + dof_right + ...)`. Cross-robot collision pairs are checked automatically. Per-robot planning treats each robot separately, frozen at its current state.

**Q: Which OMPL planner should I use?**
`"RRTConnect"` (default) is the locked production default for most tasks. `"RRTstar"` and `"BITstar"` give asymptotically optimal paths at the cost of higher iteration budget. `"PRM"` is good for repeated queries in the same scene. `"KPIECE1"` / `"LBKPIECE1"` can be better in narrow passages.

**Q: Is the state-validity check thread-safe?**
The closure captured by `make_state_validity_fn` is read-only over `model` and `scene` and allocates per-call scratch buffers internally. Safe for OMPL's threaded planners.
