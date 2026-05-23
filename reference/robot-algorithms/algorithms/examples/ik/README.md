# Inverse Kinematics

## 1. Description

Inverse kinematics (IK) finds a joint configuration `q` that places a chosen frame at a target pose `T`. The library exposes three top-level functions:

| Function | Use case | Returns |
|---|---|---|
| `ik_local(model, frame_id, T_target, q_seed, ...)` | Pose IK in the robot's base frame | `IKResult` |
| `ik(scene, robot_id, frame_id, T_target, q_seed, ...)` | Pose IK in the world frame | `IKResult` |
| `ik_velocity(model, frame_id, twist, q_current, dt=...)` | Cartesian velocity IK for servo loops | `qdot` (ndarray) |

On `IKStatus.SUCCESS` the returned `q` guarantees:

- The final pose meets `pos_tol` and `rot_tol`.
- `q` is inside joint position limits with the configured margin.
- The configuration is not near a singularity (singular-value floor + condition-number ceiling).
- Self- and environment-collision are checked when a `Scene` with a `CollisionModel` is supplied.

IK does **not** promise path collision-freeness, velocity / acceleration / jerk feasibility, or human-safe execution. Those belong to validity, planning, trajectory generation, and the runtime monitor.

Five backends ship:

| Backend | Algorithm | When to pick |
|---|---|---|
| `GenericConstrainedIK` (default) | Multi-start bounded NLS with analytical Jacobian | Generic; default for unknown robots |
| `DLSIK` (opt-in) | Damped least squares | Fast local refinement; debug |
| `OPWIK` (opt-in) | Closed-form OPW analytical | Robots with OPW structure and registered parameters |
| `SphericalWrist6RIK` (opt-in) | Closed-form spherical-wrist 6R | 6R arms with a spherical wrist |
| `QPVelocityIK` (servo only) | Bounded least-squares over qdot | 1 kHz Cartesian servoing |

Dispatch order:

1. Explicit `backend="opw" | "spherical_wrist_6r" | "dls"` → that backend.
2. A robot-specific analytical solver registered via `register_analytical(robot_id, BackendClass)` → that backend.
3. Otherwise → `GenericConstrainedIK`.

`ik_velocity` does not go through this dispatch — it always uses `QPVelocityIK`.

## 2. Data Flow

```text
YAML (robot system)            KinematicModel  ----+
                                                   |
T_target, q_seed, options, scene? ----------------+
                                                   |
                                                   v
                              dispatch.choose_backend
                                                   |
        +------------------+----------------+--------------+----------+
        v                  v                v              v          v
   OPWIK / Spherical / Registered      GenericConstrainedIK         DLSIK
        |                  |                |              |          |
        +------------------+----------------+--------------+----------+
                                                   |
                                                   v
                                            candidate q
                                                   |
                                                   v
                              IKValidation (5b — every check)
                                                   |
                                                   v
                                       IKResult(status, q, diagnostics)
```

Drake-style modular `IKProblem` lives at `algorithms.kinematics.ik.solve_problem`. Most callers do not need it; use it when you add custom constraints / costs (tool axis, RCM, minimum distance) beyond what `ik_local` ships by default.

## 3. Usage

### Setup

```python
import numpy as np
from algorithms.descriptions import RobotSystemDescription
from algorithms.kinematics import fk_local, ik_local
from algorithms.resolved import KinematicModel

system = RobotSystemDescription.from_yaml("configs/robots/franka_fr3_robot_only.yaml")
model = KinematicModel.from_robot_system(system)
home = system.named_joint_state("home")
q_home = np.array([home[name] for name in model.active_joint_names], dtype=float)

# Build a reachable target (in production it comes from perception or planning)
T_target = fk_local(model, q_home, "robot_tcp").copy()
T_target[0, 3] += 0.05

result = ik_local(model, "robot_tcp", T_target, q_seed=q_home)
if result.status.name == "SUCCESS":
    q = result.q
```

### IKStatus taxonomy

Every value reachable in `tests/test_ik_validator.py`:

| Status | When |
|---|---|
| `SUCCESS` | Valid `q` returned. All hard checks passed. |
| `INVALID_INPUT` | Bad target, unknown frame, wrong `q_seed` shape, analytical backend rejected the structure. |
| `UNREACHABLE` | Every seed explored; no `q` near the target. |
| `MAX_ITERATIONS` | Backend hit its iteration cap. |
| `TIMEOUT` | `options.max_time_ms` exhausted. |
| `JOINT_LIMIT_VIOLATION` | Best candidate inside the joint margin. |
| `POSE_ERROR_TOO_HIGH` | Solver locally optimal but `pos_tol` / `rot_tol` not met. |
| `SINGULARITY_RISK` | Final Jacobian below the singularity threshold. |
| `FINAL_COLLISION` | Scene supplied; final `q` self-collides or hits a world object. |
| `CONSTRAINT_VIOLATION` | Reserved for future nonlinear constraints. |
| `NO_VALID_CANDIDATE` | Backend produced no candidates (analytical with no branches). |
| `NUMERICAL_FAILURE` | Backend raised; details in `diagnostics.message`. |

### Registering a robot-specific analytical solver

```python
from algorithms.kinematics.ik import register_analytical

class MyUR5IK:
    name = "ur5_closed_form"
    def solve_branches(self, model, spec, q_seed):
        # compute closed-form q's for the target pose
        return (q_a, q_b, ...)

register_analytical("ur5", MyUR5IK)
# Subsequent ik_local(model, ...) where model.system.robot.id == "ur5"
# routes to MyUR5IK automatically.
```

### Drake-style modular path

```python
from algorithms.kinematics.ik import (
    IKProblem, PoseTarget, JointPositionBounds,
    SeedRegularization, JointCenteringCost, solve_problem,
)

problem = IKProblem()
problem.add_task(PoseTarget("robot_tcp", T_target))
problem.add_constraint(JointPositionBounds(q_min, q_max, margin=1e-3))
problem.add_cost(SeedRegularization(q_seed, weight=1e-6))
problem.add_cost(JointCenteringCost(weight=1e-7))

result = solve_problem(model, problem, q_seed)
```

### Performance (measured on FR3 from `q_home`)

| Operation | Success | Median |
|---|---|---|
| `ik_local` (multi-start default) | 96 % | 27 ms |
| `ik_local` with `multi_start=False` | 95 % | 21 ms |
| `ik_velocity` (6-D twist) | 100 % | 0.09 ms |
| Two-robot world | 93–98 % | 27–29 ms |

`ik_velocity` is the only one suitable for a 1 kHz servo loop.

## 4. Examples

| File | What it shows |
|---|---|
| `01_default_pose_ik.py` | The minimum useful call. |
| `02_multi_start.py` | Multi-start vs single-start side by side. |
| `03_analytical_opw.py` | Forcing the OPW backend (no parameters registered for FR3). |
| `04_validation_failure_modes.py` | Three deliberate failures: `INVALID_INPUT`, `SINGULARITY_RISK`, `FINAL_COLLISION`. |
| `05_with_constraints.py` | Modular `IKProblem` solved with `solve_problem`. |
| `06_qp_velocity_servo.py` | `ik_velocity` driving a 2-second Cartesian servo loop. |
| `07_register_analytical.py` | Registering a robot-specific analytical solver. |
| `08_collision_aware_ik.py` | Pose IK with collision validation. |
| `09_diagnosing_failures.py` | Reading `IKDiagnostics` to debug a failure. |

## 5. Common Errors

| Symptom | Cause | Fix |
|---|---|---|
| `POSE_ERROR_TOO_HIGH` from seemingly trivial targets | Soft-cost weights too high (`seed_regularization_weight`, `joint_centering_weight`). | Defaults are intentionally tiny (1e-6, 1e-7). If you increased them, drop back to the defaults. |
| `INVALID_INPUT` with message "OPWIK requires registered parameters" | Forced an analytical backend without registering robot parameters. | Either register parameters via `register_analytical(robot_id, ...)` or omit the `backend=` hint. |
| `FINAL_COLLISION` everywhere | Scene's home pose is in self-collision because adjacent-link pairs aren't allowed. | Declare adjacent pairs in robot YAML `collision.allowed_pairs`. |
| `UNREACHABLE` on a pose you can reach by hand | Target outside the robot's geometric reach or seed in the wrong basin. | Increase `num_random_seeds`; check `result.diagnostics.seed_reports`. |
| `ik_velocity` produces large qdot | Twist exceeds what the Jacobian can deliver at this `q`. | The QP clips to velocity / joint-step limits; large qdot near a singularity is the symptom — slow down or switch motion modes. |
| Validation rejects a `q` you generated externally | Pose tolerance, joint margin, or singularity threshold tighter than your candidate. | Call the public `validate(model, spec, q, options, scene=...)` to inspect which check failed. |

## 6. FAQs

**Q: How do I run IK in world coordinates?**
`ik(scene, robot_id, frame_id, T_world_target, q_seed)`. The world transform `T_world_base` is composed automatically.

**Q: What does `q_seed` do?**
It is the starting configuration for the local optimiser, and the default deterministic seed in multi-start. Pass `q_current` (the robot's current state) for branch continuity, or `q_home` when you just want a feasible IK.

**Q: How does multi-start work?**
The default seed list is `[q_current, q_last_success, q_home, q_center, q_nominal]` plus N random bounded seeds. The first seed that produces a candidate within pose tolerance short-circuits the loop. RNG is seeded by `options.random_seed` for reproducibility.

**Q: Why isn't OPW the default for FR3?**
The library does not ship FR3 OPW parameters out of the box; deriving them is robot-specific work. `GenericConstrainedIK` is the production default for unknown robots. Register OPW parameters per robot if you want the analytical speedup.

**Q: How do I add a tool-axis constraint?**
Use the modular path: build an `IKProblem`, add a custom constraint object (the `Constraint` protocol slot is reserved but the Recommended / Industrial constraint catalogue is not yet shipped — see `docs/plan.md` §6.3 for the roadmap).

**Q: Is `ik_velocity` safe at 1 kHz?**
Measured median 0.09 ms per call. Easily safe at 1 kHz. The library does not enforce safety; the servo loop's runtime monitor owns the live envelope.
