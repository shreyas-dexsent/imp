# Path Optimization

## 1. Description

Path optimization runs **geometric** passes that take a `Path` and return a `Path`. No timing, no velocities — those live at the trajectory layer.

Two operations + one helper ship:

| Function | Purpose | Input → Output |
|---|---|---|
| `shortcut_smooth(path, model, scene, ...)` | Random shortcut smoothing with collision validity. | Path → Path with `ShortcutStats` |
| `spline_fit(path, *, order, samples)` | Quintic / cubic spline fit with Catmull-Rom interior velocities. | Path → Path |
| `remove_redundant_waypoints(path, *, tolerance)` | Drop consecutive duplicates from OMPL output. | Path → Path |

The typical sequence after planning:

```text
plan_joint -> remove_redundant_waypoints -> shortcut_smooth -> spline_fit -> time_parameterize
```

The motion primitive `move_joint` composes all of these internally; this layer exposes them for hand-tuning.

## 2. Data Flow

```text
Path (from plan_joint)
        |
        v
remove_redundant_waypoints  (linear; drops consecutive duplicates)
        |
        v
shortcut_smooth  (random pair, try direct connection, accept if collision-free)
        |       uses make_state_validity_fn(model, scene)
        v
Path (fewer waypoints, shorter)
        |
        v
spline_fit  (quintic / cubic; Catmull-Rom finite-difference interior velocities)
        |
        v
Path (denser, C^k smooth) -- ready for time_parameterize
```

Output is always a `Path`. Optimization is the geometric phase; time parameterization is the kinematic phase. Keeping them distinct is what allows each to be tested in isolation.

## 3. Usage

### Setup

```python
import numpy as np
from algorithms.descriptions import WorldDescription
from algorithms.optimization import remove_redundant_waypoints, shortcut_smooth, spline_fit
from algorithms.planning import plan_joint, validate_path
from algorithms.resolved import CollisionModel, KinematicModel, Scene

world = WorldDescription.from_yaml("configs/worlds/franka_robot_only_world.yaml")
cm = CollisionModel.from_world(world)
scene = Scene.from_world(world, cm)
system = world.robots[0].robot_system
model = KinematicModel.from_robot_system(system)
home = system.named_joint_state("home")
q_home = np.array([home[name] for name in model.active_joint_names])
q_goal = q_home.copy(); q_goal[0] += 0.8
```

### Standard pipeline

```python
planned = plan_joint(model, scene, q_home, q_goal)
deduped = remove_redundant_waypoints(planned.path)
smoothed, stats = shortcut_smooth(deduped, model, scene, iterations=200, random_seed=0)
splined = spline_fit(smoothed, order="quintic", samples=80)

# Confirm still valid
report = validate_path(model, scene, splined)
assert report.passed
```

### Shortcut details

```text
for iteration in 1..K:
    pick i, j uniformly at random with j > i + 1
    if every sample on q[i] -> q[j] (at max_joint_step resolution) is valid:
        path = path[:i+1] + path[j:]    # drop intermediate waypoints
```

`ShortcutStats` records:

| Field | Meaning |
|---|---|
| `attempted` | Total shortcut attempts |
| `accepted` | Shortcuts found valid and applied |
| `initial_waypoints` / `final_waypoints` | Path length before / after |
| `initial_length` / `final_length` | Sum of joint-space segment norms before / after |

### Spline-fit details

```text
1. Assign chord-length s_i to each waypoint.
2. Compute interior velocities via Catmull-Rom finite difference:
       v_i = (q_{i+1} - q_{i-1}) / (s_{i+1} - s_{i-1})
3. Per segment, solve a polynomial (cubic or quintic) with boundary conditions.
4. Sample uniformly in s at the requested density.
```

Cubic (order=4 coeffs): C^1 in space with v=0 at start and end.
Quintic (order=6 coeffs): C^2 in space with v=0 AND a=0 at start and end.

Quintic is the default because the trajectory layer parameterizes C^2 geometry more smoothly than C^1.

### Defaults

`shortcut_smooth`:

| Field | Default | Meaning |
|---|---|---|
| `iterations` | 100 | Shortcut attempts. 100–200 is typical. |
| `max_joint_step` | 0.05 rad | Sampling resolution along each candidate shortcut. |
| `joint_margin` | 1e-3 rad | Margin used by the validity check. |
| `random_seed` | 0 | RNG seed for reproducibility. |

`spline_fit`:

| Field | Default | Meaning |
|---|---|---|
| `order` | `"quintic"` | `"cubic"` or `"quintic"` |
| `samples` | 200 | Output waypoint count. |

## 4. Examples

| File | What it shows |
|---|---|
| `01_shortcut_smoothing.py` | 100-waypoint OMPL output collapses to 2 waypoints when the straight line is collision-free. |
| `02_spline_fit.py` | Quintic vs cubic; endpoint pinning verified. |
| `03_full_optimization_pipeline.py` | Plan → dedupe → shortcut → spline → validate. |

## 5. Common Errors

| Symptom | Cause | Fix |
|---|---|---|
| Shortcut accepts 0 / 200 | Every shortcut hits a collision; the planner already produced the best feasible geometry. | Useful signal — no improvement possible. Move on. |
| Spline-fit output rejected by `validate_path` | Spline pushed interior waypoints sideways into an obstacle. | Re-validate after spline_fit; if it fails, increase `samples` or skip the spline step. |
| Smoothed path is shorter but no smoother | You ran shortcut without spline_fit. | Add `spline_fit` after `shortcut_smooth` for C^2 geometry. |
| `shortcut_smooth` slow | Many shortcut attempts × dense per-shortcut validity sampling. | Drop `iterations` to 100; or `max_joint_step` to a larger value. |
| Spline endpoints drift slightly | Numerical noise; the spline coefficients aren't pinned by construction. | The library pins endpoints explicitly. If you see drift you're using a custom backend; switch to the bundled `spline_fit`. |

## 6. FAQs

**Q: Why not jam smoothing into the planner directly?**
Each phase owns one concern. The planner finds a path; the smoother improves the geometry; the trajectory layer assigns timing. Mixing concerns means harder debugging and tighter coupling. Industrial-grade libraries keep them separate.

**Q: Should I run spline_fit before or after shortcut_smooth?**
After. Quintic-fitting 100 OMPL waypoints is wasteful — most of them get smoothed away by shortcut. Shortcut first, spline second.

**Q: What's `interior_velocity_scale` for in the trajectory layer? Is it related?**
Yes. The Catmull-Rom velocities computed during spline_fit feed the same interior-velocity computation the trajectory layer uses for pass-through motion. The spline gives geometry; the trajectory layer assigns timing.

**Q: Can I run `shortcut_smooth` multiple times?**
Yes. It's idempotent on a fully-smoothed path. Chaining a few invocations with different seeds can help on hard problems.

**Q: Does the smoother preserve the start and goal exactly?**
Yes. The shortcut algorithm never removes waypoint 0 or waypoint N-1. Spline_fit pins both endpoints to the exact input values regardless of polynomial drift.
