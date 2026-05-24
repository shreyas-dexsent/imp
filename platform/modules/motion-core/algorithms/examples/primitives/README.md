# Motion Primitives

## 1. Description

Motion primitives are high-level "do this motion" functions. Each one composes the IK, planning, optimization, and trajectory layers into one call that returns a validated, controller-ready `Trajectory`.

Primitives are pure composition. They do not introduce new algorithms; they exist so common motion patterns become one-liners.

Five primitives ship:

| Function | Purpose | Algorithm |
|---|---|---|
| `move_joint(model, scene, q_goal, q_seed)` | Joint-space goto | plan_joint → shortcut → spline → time_parameterize → validate |
| `move_l(scene, robot_id, frame_id, T_goal, q_seed)` | Linear Cartesian | plan_cartesian → time_parameterize → validate (no smoothing) |
| `approach(scene, robot_id, frame_id, T_target, q_seed, *, distance, axis)` | Linear descent to a target | move_l from pre-approach pose to target |
| `retreat(scene, robot_id, frame_id, q_seed, *, distance, axis)` | Linear lift from current TCP | move_l from FK(q_seed) to FK(q_seed) + offset |
| `via_motion(model, scene, q_waypoints)` | Pass-through across N via-points | plan_joint per segment → concat → shortcut → spline → time_parameterize → validate |

All five return a `MoveResult` whose `trajectory` is the controller-ready output on success. Diagnostics carry the intermediate artifacts (`path`, `ik_result`, `plan_result`, `path_validation`, `trajectory_result`, `trajectory_validation`).

A `MoveResult.status == SUCCESS` guarantees:

- The trajectory starts at `q_seed` (or `T_start`) and ends at the requested target.
- Pass-through motion at every interior waypoint — no pauses within a segment.
- Joint limits, v / a / j envelopes, and dense-time collision validated.
- For `move_l` / `approach` / `retreat`: the TCP follows a straight line.

Not promised:

- Velocity continuity across primitive boundaries (each ends at rest). For continuous chained motion use `via_motion`.
- Optimal path length.
- Gripper actuation — primitives are about motion only.

## 2. Data Flow

```text
move_joint(model, scene, q_goal, q_seed)
        |
        v
plan_joint  -> Path (collision-free)
        |
        v
shortcut_smooth + spline_fit  -> Path (shorter, C^2)
        |
        v
validate_path  -> PathValidationReport
        |
        v
time_parameterize  -> Trajectory (pass-through, dt=1 ms)
        |
        v
validate_trajectory  -> TrajectoryValidationReport
        |
        v
MoveResult(status=SUCCESS, trajectory=..., intermediate artifacts attached)


move_l(scene, robot_id, frame_id, T_goal, q_seed)
        |
        v
plan_cartesian  -> Path with cartesian_waypoints
        |
        (smoothing / spline DISABLED — would deviate from the line)
        |
        v
validate_path + time_parameterize + validate_trajectory
        |
        v
MoveResult


approach / retreat
        |
        v
compute pre-approach pose / retreat pose via axis-distance offset
        |
        v
move_l from offset pose to target pose (or vice versa)


via_motion(model, scene, q_waypoints)
        |
        v
plan_joint per consecutive pair  -> N segment paths
        |
        v
concat (drop duplicates at junctions)  -> one big Path
        |
        v
shortcut + spline + validate + time_parameterize + validate
        |
        v
MoveResult with ONE Trajectory across every via-point (pass-through interior)
```

## 3. Usage

### Setup

```python
import numpy as np
from algorithms.descriptions import WorldDescription
from algorithms.primitives import move_joint, move_l, approach, retreat, via_motion
from algorithms.resolved import CollisionModel, KinematicModel, Scene

world = WorldDescription.from_yaml("configs/worlds/franka_robot_only_world.yaml")
cm = CollisionModel.from_world(world)
scene = Scene.from_world(world, cm)
system = world.robots[0].robot_system
model = KinematicModel.from_robot_system(system)
home = system.named_joint_state("home")
q_home = np.array([home[name] for name in model.active_joint_names])
```

### Joint-space goto

```python
result = move_joint(model, scene, q_goal, q_seed=q_home)
if result.status.name == "SUCCESS":
    trajectory = result.trajectory
```

### Linear Cartesian

```python
from algorithms.kinematics import fk

T_goal = fk(scene, "arm", q_home, "robot_tcp").copy()
T_goal[0, 3] += 0.10
result = move_l(scene, "arm", "robot_tcp", T_goal, q_seed=q_home)
```

### Approach / retreat (top-down grasp pattern)

```python
result = approach(
    scene, "arm", "robot_tcp", T_grasp, q_seed=q_pre_grasp,
    distance=0.05, axis="-z", reference="target",
)
# ... gripper closes ...
result = retreat(
    scene, "arm", "robot_tcp", q_seed=q_after_grasp,
    distance=0.05, axis="z", reference="tcp",
)
```

`pre_approach_pose(T_target, distance, axis)` computes the pre-approach pose for callers that need it explicitly.

### Pass-through across via-points

```python
result = via_motion(model, scene, [q_home, q_via_a, q_via_b, q_goal])
# ONE trajectory; robot does NOT stop at via_a or via_b.
```

### MoveStatus taxonomy

| Status | When |
|---|---|
| `SUCCESS` | Trajectory ready. |
| `INVALID_INPUT` | Wrong-shape input, bad axis name, < 2 via-points. |
| `IK_FAILED` | IK call inside the primitive failed (Cartesian primitives only). |
| `PLAN_FAILED` | `plan_joint` or `plan_cartesian` returned non-SUCCESS. |
| `OPTIMIZATION_FAILED` | `shortcut_smooth` or `spline_fit` raised. |
| `PATH_VALIDATION_FAILED` | `validate_path` rejected the smoothed path. |
| `TRAJECTORY_FAILED` | `time_parameterize` returned non-SUCCESS. |
| `TRAJECTORY_VALIDATION_FAILED` | `validate_trajectory` rejected the trajectory. |
| `NUMERICAL_FAILURE` | Unexpected exception. |

`diagnostics.stage` tells you which stage failed. Intermediate artifacts (`path`, `ik_result`, `plan_result`, `path_validation`, `trajectory_result`, `trajectory_validation`) are populated up to and including the failing stage — useful for debugging.

### `MoveOptions`

Composed knob bag wrapping every sub-stage's options:

```python
from algorithms.primitives import MoveOptions
from algorithms.planning.options import PlanOptions
from algorithms.trajectory.options import TimeParameterizationOptions

opts = MoveOptions(
    plan=PlanOptions(max_time_ms=4000),
    time_parameterize=TimeParameterizationOptions(v_scale=0.3, a_scale=0.3),
    smoothing_iterations=500,
)
result = move_joint(model, scene, q_goal, q_seed=q_home, options=opts)
```

Pipeline toggles: `smooth_path`, `spline_fit`, `validate_path`, `validate_trajectory`, `smoothing_iterations`, `spline_samples`.

## 4. Examples

| File | What it shows |
|---|---|
| `01_move_joint.py` | Minimal joint-space goto. |
| `02_move_l.py` | Linear Cartesian goto; final TCP error at 1e-12 m. |
| `03_approach_retreat.py` | Final descent + lift-off (the grasp pair). |
| `04_via_motion.py` | 4 via-points; velocity at quartiles confirms pass-through. |
| `05_bin_pick_sequence.py` | End-to-end 4-leg bin pick: home → pre-grasp → grasp → post-retreat → home. |

For obstacle-avoidance, palletizing, and chained-motion demonstrations see `examples/integration/`.

## 5. Common Errors

| Symptom | Cause | Fix |
|---|---|---|
| `approach` returns `IK_DISCONTINUITY` | Caller's `q_seed` is not at the pre-approach pose; the IK branch flipped at one sample. | Use `move_joint` (or directly compute IK) to get to the pre-approach pose first; pass that `q` as `q_seed` to `approach`. |
| Chaining `move_joint(A→B)` then `move_joint(B→C)` produces stop at B | Each primitive's trajectory ends at rest. | Use `via_motion([A, B, C])` for one continuous trajectory; the robot flows through B. |
| `move_l` returns `PLAN_FAILED` on a long Cartesian move | Path crosses self-collision configurations or hits an obstacle. | Try `move_joint` instead; `plan_cartesian` does NOT route around obstacles. |
| `via_motion` works but `move_joint` chain doesn't | Each `move_joint` plans independently; the second leg's start state may not match where the first leg ended (especially after IK rounding). | Use `via_motion` for chained motion; or thread the final `q` from each leg into the next. |
| `MoveResult.trajectory_validation` rejects a `move_l` output | TCP speed cap set tighter than the path's Cartesian speed implies. | Either drop `tcp_v_max` from `trajectory_validation` options or run `move_l` with `v_scale < 1`. |
| `MoveStatus.PLAN_FAILED` with `diagnostics.message` mentioning collision | Start or goal in collision; or path blocked. | Check `is_in_collision(model, scene, q_seed)` and the goal IK; widen `max_time_ms`; consider `via_motion` through an intermediate clear point. |

## 6. FAQs

**Q: Does `move_l` smooth the path?**
No. Smoothing would deviate the TCP from the requested straight line. `move_l` runs the Cartesian planner, skips the smoother, and goes straight to time parameterization + validation.

**Q: What's the difference between chaining `move_joint` and using `via_motion`?**
Chained `move_joint`s give N trajectories, each ending at rest. The robot pauses at every junction. `via_motion` gives ONE trajectory; the robot passes through every interior via-point without stopping. For most production workflows `via_motion` is what you want.

**Q: Can primitives do velocity continuity across the boundary between two segments?**
Inside one primitive, yes (pass-through). Across two separate primitive calls, no — each ends at rest. To get continuous velocity across what would otherwise be two primitive calls, build the composite Path yourself and call `time_parameterize` once with `start_velocity` / `end_velocity` as needed.

**Q: Why is the primitive layer a separate module instead of just being functions in `planning`?**
Primitives compose IK, planning, optimization, and trajectory generation. Putting them in any one of those packages would create a circular or otherwise confusing dependency. Keeping them as their own layer makes the dependency graph clean: descriptions → resolved → kinematics → collision → planning / optimization / trajectory → primitives.

**Q: Do primitives validate?**
Yes — both `validate_path` (post-smoothing) and `validate_trajectory` (post-parameterization) run by default. Disable via `MoveOptions.validate_path=False` / `validate_trajectory=False` if you have your own validation pipeline downstream.

**Q: How do I add a `move_c` (circular arc) primitive?**
The slot exists in the module layout. A `move_c` implementation would plan a TCP arc through three points (start, via, goal), sample Cartesian poses along the arc, solve IK at each, then parameterize. Not yet shipped; track via plan.md §8.
