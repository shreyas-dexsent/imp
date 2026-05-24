# Trajectory

## 1. Description

A `Trajectory` is geometry plus timing — a function `t → (q(t), qd(t), qdd(t))` over `[0, duration]`. Time parameterization is the only step that bridges a geometric `Path` to a `Trajectory`. Validation runs dense-time envelope and collision checks on the result.

The library promises **pass-through motion** by default: the robot does NOT pause at interior waypoints. Velocity at every interior waypoint is non-zero (Catmull-Rom finite difference, clipped to the v envelope). Only the start and end of the trajectory are at rest. Set `rest_to_rest=True` if you need legacy stop-at-each-waypoint behaviour.

Two top-level functions:

| Function | Purpose | Returns |
|---|---|---|
| `time_parameterize(path, model, *, options)` | Assign timing to a `Path`; produce a `Trajectory` | `TrajectoryResult` |
| `validate_trajectory(trajectory, model, scene, *, options)` | Dense-time validation | `TrajectoryValidationReport` |

Two backends:

| Backend | Algorithm | Dependency | When to pick |
|---|---|---|---|
| `"polynomial"` | Quintic spline with Catmull-Rom interior velocities. C^2 in space. | Pure Python | Always available. Default fallback. |
| `"ruckig"` | Per-segment Ruckig with non-zero target velocities chained from Catmull-Rom. Jerk-limited online motion. | `pip install ruckig` | Production default when installed. |

Dispatch via `TimeParameterizationOptions.backend`:

- `"auto"` (default) → Ruckig if importable, else polynomial.
- `"ruckig"` → force Ruckig.
- `"polynomial"` → force polynomial.

The Ruckig backend uses **per-segment local computation only**. It does NOT use Ruckig's `intermediate_positions` API (which routes through the Ruckig Community cloud server and is production-unsafe).

## 2. Data Flow

```text
Path (geometry only)
        |
        +-- model -- get v / a / j limits per joint
        |
        v
time_parameterize  (auto backend selection)
        |
        v
+----------------------+-----------------------+
|                                              |
v                                              v
PolynomialBackend                       RuckigBackend
quintic per segment                     per-segment Ruckig online update
Catmull-Rom interior velocities          target_velocity = Catmull-Rom
                                          start = previous segment's end (continuity)
|                                              |
+----------------------+-----------------------+
                       |
                       v
                Trajectory
                duration, joint_names, backend_used,
                positions, velocities, accelerations sampled at dt
                       |
                       v
                trajectory.at(t)  ->  one-off query
                trajectory.sample(dt)  ->  streaming for a controller
                       |
                       v
                validate_trajectory  ->  joint limits, v/a/j envelopes,
                                        dense-time collision, TCP speed,
                                        controller-rate compatibility
```

`Trajectory.at(t)` is linear interpolation between bracketing stored samples. At the construction `dt` (default 1 ms) the linear interp is indistinguishable from the underlying polynomial / Ruckig curve.

## 3. Usage

### Setup + minimum call

```python
import numpy as np
from algorithms.descriptions import WorldDescription
from algorithms.optimization import shortcut_smooth, spline_fit
from algorithms.planning import plan_joint
from algorithms.resolved import CollisionModel, KinematicModel, Scene
from algorithms.trajectory import (
    TimeParameterizationOptions, time_parameterize, validate_trajectory,
)

world = WorldDescription.from_yaml("configs/worlds/franka_robot_only_world.yaml")
cm = CollisionModel.from_world(world)
scene = Scene.from_world(world, cm)
system = world.robots[0].robot_system
model = KinematicModel.from_robot_system(system)
home = system.named_joint_state("home")
q_home = np.array([home[name] for name in model.active_joint_names])
q_goal = q_home.copy(); q_goal[0] += 0.8

planned = plan_joint(model, scene, q_home, q_goal)
smoothed, _ = shortcut_smooth(planned.path, model, scene, iterations=200)
splined = spline_fit(smoothed, samples=60)

result = time_parameterize(splined, model)
trajectory = result.trajectory
```

### Streaming for a controller

```python
times, q, qd, qdd = trajectory.sample(dt=0.001)   # 1 kHz
for t_i, q_i, qd_i, qdd_i in zip(times, q, qd, qdd):
    controller.send(t_i, q_i, qd_i, qdd_i)
```

`trajectory.at(t)` returns one `(q, qd, qdd)` for any `t ∈ [0, duration]`. Outside the domain it clamps to endpoints.

### TrajectoryStatus

| Status | When |
|---|---|
| `SUCCESS` | Trajectory returned. Endpoints pinned, envelopes respected. |
| `INVALID_INPUT` | Wrong path dof, unknown backend. |
| `LIMITS_INFEASIBLE` | Path needs higher v / a / j than the model permits. |
| `BACKEND_FAILURE` | Backend raised (e.g., missing import, Ruckig error). |
| `NUMERICAL_FAILURE` | Unexpected exception. |
| `NO_WAYPOINTS` | Path had fewer than 2 waypoints. |

### Validation

```python
report = validate_trajectory(
    trajectory, model, scene,
    options=TrajectoryValidationOptions(
        check_collision=True,
        tcp_v_max=0.25,
        tcp_frame_id="robot_tcp",
        controller_dt=0.001,
    ),
)
if not report.passed:
    t, reason = report.first_failure
    print(f"failed at t={t:.3f} s: {reason}")
```

The validator checks: joint position limits, velocity / acceleration / jerk envelopes, dense-time collision (every `validation_dt`), optional TCP linear / angular speed, optional controller-rate compatibility.

### Pass-through vs rest-to-rest

```python
# Default — robot does NOT pause at interior waypoints
result = time_parameterize(path, model)

# Legacy — robot brakes at every interior waypoint
result = time_parameterize(
    path, model,
    options=TimeParameterizationOptions(rest_to_rest=True),
)
```

### Defaults

`TimeParameterizationOptions`:

| Field | Default | Meaning |
|---|---|---|
| `backend` | `"auto"` | `"auto"`, `"ruckig"`, `"polynomial"` |
| `dt` | 0.001 | Output sample period (1 ms = 1 kHz controller) |
| `v_scale` | 1.0 | Multiplier on model velocity limits |
| `a_scale` | 1.0 | Multiplier on model acceleration limits |
| `j_scale` | 1.0 | Multiplier on model jerk limits (Ruckig only) |
| `start_velocity` / `end_velocity` | `None` | Non-zero boundary velocities for chained trajectories |
| `start_acceleration` / `end_acceleration` | `None` | Same for accelerations |
| `rest_to_rest` | `False` | Force stop-at-each-waypoint |
| `interior_velocity_scale` | 1.0 | Scales Catmull-Rom interior velocity |

`TrajectoryValidationOptions`:

| Field | Default | Meaning |
|---|---|---|
| `validation_dt` | 0.01 | Validator sampling step |
| `joint_margin` | 1e-3 | Joint-limit margin |
| `v_scale`, `a_scale`, `j_scale` | 1.0 | Envelope multipliers |
| `check_collision` | True | Dense-time collision check |
| `tcp_v_max`, `tcp_omega_max` | `None` | Optional TCP-speed caps |
| `tcp_frame_id` | `None` | Required when TCP bounds are set |
| `controller_dt` | `None` | Required output `dt` for the controller |
| `numerical_slack` | 1e-6 | Limit-check tolerance |

### Performance (measured, end-to-end including upstream planning)

| Operation | Success | Median |
|---|---|---|
| `time_parameterize` (polynomial, dt=0.001) | 20 / 20 | 711 ms |
| `time_parameterize` (ruckig, dt=0.001) | 20 / 20 | 379 ms |
| `validate_trajectory` (full, validation_dt=0.01) | 20 / 20 | 533 ms |

Pure `time_parameterize` cost (excluding upstream planning) is ~50–100 ms for a 3-second trajectory at 1 kHz sampling.

## 4. Examples

| File | What it shows |
|---|---|
| `01_time_parameterize_polynomial.py` | Minimal polynomial call; peak v/a numbers. |
| `02_pass_through_verified.py` | Interior velocities printed for both pass-through and rest-to-rest. |
| `03_stream_to_controller.py` | `sample(dt)` for 125 / 500 / 1000 Hz controllers. |
| `04_full_pipeline.py` | End-to-end: plan → smooth → spline → time_parameterize → validate. |

## 5. Common Errors

| Symptom | Cause | Fix |
|---|---|---|
| `LIMITS_INFEASIBLE` | Path needs accelerations the model can't deliver. | Run more spline_fit samples upstream (smaller dq per segment) or scale down `v_scale` / `a_scale`. |
| Trajectory pauses at interior waypoints | `rest_to_rest=True`. | Default is pass-through; check `options.rest_to_rest`. |
| `sample(dt)` returns one sample | Path produced a near-zero-duration trajectory (start = goal). | Skip trajectory generation when start equals goal. |
| Ruckig errors out with cloud-API mention | You set `intermediate_positions` via the Ruckig API directly. | The library never uses that API; if you're calling Ruckig outside the library, use chained per-segment calls instead. |
| `validate_trajectory` rejects a valid-looking trajectory at the controller-rate check | `options.controller_dt` smaller than the trajectory's stored `dt`. | Either re-parameterize with smaller `dt`, or relax `controller_dt`. |
| Trajectory respects v limits but the controller protective-stops on velocity | Velocity envelope was scaled (`v_scale=1.0`) up to the model maximum. | Use `v_scale=0.8` for headroom; the model's published limits assume ideal conditions. |

## 6. FAQs

**Q: Why does pass-through start and end at rest if it's pass-through?**
The endpoints are at rest because most callers want a complete motion they can hand to a controller. Interior waypoints pass through. To chain trajectories with non-zero endpoint velocity, pass `start_velocity` / `end_velocity` in `TimeParameterizationOptions`.

**Q: How accurate is `Trajectory.at(t)`?**
At construction `dt` (default 1 ms), linear interpolation between samples matches the underlying polynomial / Ruckig curve to within numerical precision. At larger `dt` the linear-interp error becomes visible; lower `dt` for fine-grained queries.

**Q: Can I get a trajectory without a planner?**
Yes — `time_parameterize` takes any valid `Path`. You can construct a `Path` directly from a list of via-points.

**Q: What's the difference between `polynomial` and `ruckig` for pass-through?**
Both use Catmull-Rom interior velocities. Polynomial fits a quintic per segment offline; Ruckig solves a jerk-limited online trajectory per segment. Ruckig respects the jerk envelope; polynomial doesn't model jerk. For high-speed industrial motion Ruckig is the production choice.

**Q: Is the dense-sample representation memory-heavy?**
A 3-second 1 kHz trajectory has 3000 samples × 3 fields (q, qd, qdd) × 8 floats × 8 bytes ≈ 0.5 MB. Negligible.
