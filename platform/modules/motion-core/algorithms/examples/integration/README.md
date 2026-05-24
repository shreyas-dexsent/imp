# Integration

## 1. Description

End-to-end demonstrations that wire several library layers together to solve a complete motion task. These examples are **not** about any one feature — they show how perception inputs, planning, optimization, time parameterization, and validation combine to deliver controller-ready trajectories for real industrial scenarios.

Three scenarios ship:

| Scenario | Layers exercised | What you learn |
|---|---|---|
| Obstacle avoidance from A to B | perception (`add_object`) → planning → optimization → trajectory → validation | The library produces a collision-free smooth trajectory between two configurations even with an obstacle on the direct line. |
| Linear palletizing with fallback | planning (`plan_cartesian`) + primitives (`move_l`, `move_joint`) + per-leg IK | Industrial pattern: try MoveL first; fall back to joint-space replanning when the line is blocked. |
| Chained smooth motion | primitives (`move_joint` chained vs `via_motion`) | Pass-through `via_motion` produces ONE smooth trajectory; chained `move_joint` calls produce multiple trajectories with brakes between them. |

If you only have time for one demo, run `01_obstacle_avoidance_a_to_b.py` — it answers the question "can I just give start + goal + a scene with obstacles and get a clean trajectory?" with a yes.

## 2. Data Flow

```text
Application
    |
    +-- Build scene at startup:
    |     world = WorldDescription.from_yaml("configs/worlds/...yaml")
    |     cm = CollisionModel.from_world(world)
    |     scene = Scene.from_world(world, cm)
    |
    +-- Perception (live, every tick):
    |     scene.add_object(id, collision=..., visual=..., pose=...)    # new
    |     scene.set_object_pose(id, T_new)                              # update
    |     scene.remove_object(id)                                       # gone
    |
    +-- Motion intent:
    |     result = move_joint(model, scene, q_goal, q_seed=q_current)
    |   OR
    |     result = move_l(scene, robot_id, frame_id, T_goal, q_seed=q_current)
    |   OR
    |     result = via_motion(model, scene, [q_0, q_1, ..., q_N])
    |
    +-- Execute:
    |     for t_i, q_i, qd_i, qdd_i in zip(*result.trajectory.sample(dt)):
    |         controller.send(q_i, qd_i, qdd_i)
    |
    +-- On failure: inspect result.diagnostics.stage + .message
        and result.path / .plan_result / .trajectory_result for the intermediate artefact
```

The single architectural contract these examples illustrate: **perception writes to Scene; the library reads from Scene; the controller adapter reads only from `trajectory.sample(dt)`**. Nothing in the library knows or cares about perception transports or controller drivers.

## 3. Usage

### Obstacle avoidance from A to B

```python
from algorithms import (
    CollisionModel, KinematicModel, Scene, WorldDescription, move_joint,
)
from algorithms.descriptions import BoxGeometrySpec
import numpy as np

world = WorldDescription.from_yaml("configs/worlds/franka_robot_only_world.yaml")
cm = CollisionModel.from_world(world); scene = Scene.from_world(world, cm)
system = world.robots[0].robot_system
model = KinematicModel.from_robot_system(system)
home = system.named_joint_state("home")
q_home = np.array([home[name] for name in model.active_joint_names])

# Perception adds the obstacle
T_obstacle = np.eye(4); T_obstacle[:3, 3] = [0.40, 0.0, 0.40]
scene.add_object(
    "obstacle_box",
    collision=BoxGeometrySpec(type="box", size=(0.10, 0.30, 0.30)),
    pose=T_obstacle,
)

# Plan + smooth + parameterize + validate, all in one call
q_goal = q_home.copy(); q_goal[0] = 1.57
result = move_joint(model, scene, q_goal, q_seed=q_home)
assert result.status.name == "SUCCESS"
trajectory = result.trajectory   # ready to stream to the controller
```

### Linear palletizing with fallback

```python
def deposit_to_cell(cell_xyz, q_current):
    T_above = pose(orient, [cell_xyz[0], cell_xyz[1], safe_z])
    T_cell  = pose(orient, cell_xyz)

    # 1. Reach safe-above (try linear first)
    r_above = move_l(scene, "arm", "robot_tcp", T_above, q_seed=q_current)
    if r_above.status is MoveStatus.SUCCESS:
        q_at_above = ik_at(T_above, q_current)
    else:
        # Linear blocked — fall back to joint-space replanning
        q_at_above = ik_local(model, "robot_tcp", T_above, q_current).q
        r_above = move_joint(model, scene, q_at_above, q_seed=q_current)
        if r_above.status is not MoveStatus.SUCCESS:
            return None

    # 2. Final linear descent
    r_descent = move_l(scene, "arm", "robot_tcp", T_cell, q_seed=q_at_above)
    # ... lift back to safe-above ...
```

This is the canonical industrial pattern. The application owns the fallback logic; the library exposes the primitives.

### Pass-through across via-points

```python
result = via_motion(model, scene, [q_home, q_via_a, q_via_b, q_goal])
trajectory = result.trajectory   # ONE trajectory, no pauses at via-points
```

`via_motion` is ~3x faster end-to-end than chaining three `move_joint` calls because each `move_joint` ends at rest.

## 4. Examples

| File | What it shows |
|---|---|
| `01_obstacle_avoidance_a_to_b.py` | A→B with a runtime-added box obstacle. Confirms every sample of the resulting trajectory is collision-free. |
| `02_linear_palletizing.py` | 4 pallet cells; `move_l` for each safe-above + descent + lift. Demonstrates the fail-and-skip pattern when a cell can't be reached. |
| `03_chained_smooth_motion.py` | Two patterns side by side: chained `move_joint` (3 trajectories, brakes between) vs `via_motion` (one trajectory, pass-through). |

## 5. Common Errors

| Symptom | Cause | Fix |
|---|---|---|
| `START_IN_COLLISION` on a scene that "should be clean" | YAML world has the home pose colliding with a workpiece. | Pick a clean baseline world (e.g., `franka_robot_only_world.yaml`) and add obstacles at runtime via `Scene.add_object`. |
| `move_l` fails on what looks like a clear straight line | Robot arm geometry intervenes mid-line even though the TCP doesn't. | Switch to `move_joint` for the blocked leg; `move_l` is line-only and IK-driven. |
| `via_motion` returns `PATH_VALIDATION_FAILED` | One inter-via segment plans through a near-collision configuration. | Drop the failing via-point; if you can't, plan the full path and inspect `result.path_validation.first_failure`. |
| Trajectory generated but the controller protective-stops | Trajectory velocity envelope at the model's published limits is too aggressive for live hardware. | Use `MoveOptions.time_parameterize.v_scale = 0.7` (or lower) for headroom. |
| Palletizing "above" succeeds but "descent" fails with `FINAL_COLLISION` | The descent line passes through a self-collision configuration. | Raise the safe-above height; reduce the descent distance. |
| Two robots in one scene; planning slows | Composite-state planning over both arms exponentially harder than per-robot. | If they don't share workspace, plan per-robot serially. Use composite planning only when arms can physically interact. |

## 6. FAQs

**Q: Can I get a collision-free trajectory if I just give start, goal, and a scene with obstacles?**
Yes. `move_joint(model, scene, q_goal, q_seed=q_start)` does the full pipeline (plan + smooth + parameterize + validate). The validator confirms every sample is collision-free before the trajectory is returned. See `01_obstacle_avoidance_a_to_b.py`.

**Q: Can I do palletizing with linear motion AND obstacle avoidance?**
The library's `plan_cartesian` (and therefore `move_l` / `approach` / `retreat`) is straight-line only. It does not route around obstacles. The industrial pattern is: try `move_l` first; if it fails, fall back to `move_joint` for that segment. The application owns the fallback policy. See `02_linear_palletizing.py`.

**Q: Why is `move_l` straight-line only? Why not a Cartesian planner that avoids obstacles?**
That's a fundamentally hard problem (constrained-Cartesian planning). The industrial answer is: place safe waypoints such that linear motion between them is free, and fall back to joint-space when blocked. This matches how Fanuc / KUKA / ABB pendants work; the operator (or vision) provides safe Cartesian poses.

**Q: Where does perception fit into this stack?**
Perception lives outside the library by design. The library exposes one entry point — `Scene.add_object(id, collision=..., visual=..., pose=...)` — that perception writes to. Once an object is in the scene, every collision query and planner sees it.

**Q: How do I chain multiple primitives without rest pauses between them?**
For joint-space chains: use `via_motion(model, scene, [q_0, q_1, ..., q_N])`. For mixed (some Cartesian, some joint) chains: build a single composite `Path` and run `time_parameterize` once. There's no library-level "chain trajectories" function (yet); the application composes.

**Q: What if I want to validate the trajectory more strictly than the default?**
Pass a custom `TrajectoryValidationOptions` via `MoveOptions.trajectory_validation`. Useful knobs: `tcp_v_max`, `tcp_omega_max`, `controller_dt`, tighter `numerical_slack`.
