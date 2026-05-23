# Singularity Metrics

## 1. Description

Singularity metrics quantify how well-conditioned the Jacobian `J(q)` is. They answer: how much joint motion does a small Cartesian motion require? Near a singularity that ratio becomes unbounded.

Five scalars are exposed:

| Metric | Formula | Geometric meaning |
|---|---|---|
| Manipulability (Yoshikawa) | `sqrt(det(J Jᵀ))` | Volume of the velocity ellipsoid at `q`. 0 at a singularity. |
| Condition number | `σ_max / σ_min` | Ratio of largest to smallest singular value. ∞ at a singularity. |
| Inverse condition number | `σ_min / σ_max` | Same information, numerically stable. 0 at a singularity. |
| Minimum singular value | `σ_min` | Direct "distance to singularity". 0 at a singularity. |
| Rank | numerical SVD rank | Drops below 6 when the Jacobian loses a degree of freedom. |

Use the inverse condition number or minimum singular value as a continuous distance-to-singularity metric in planners and validators. Use rank for hard rank-deficiency checks.

## 2. Data Flow

```text
KinematicModel + q + frame_id
        |
        v
jacobian(model, q, frame_id, reference="world")  -> J  (6, n_active)
        |
        v
singularity_report(J)  -> SingularityMetrics(manipulability, ...)
```

All five metrics share one SVD; computing `singularity_report` is no more expensive than computing the most expensive metric.

## 3. Usage

### Setup

```python
import numpy as np
from algorithms.descriptions import RobotSystemDescription
from algorithms.kinematics import jacobian, singularity_report
from algorithms.resolved import KinematicModel

system = RobotSystemDescription.from_yaml("configs/robots/franka_fr3_robot_only.yaml")
model = KinematicModel.from_robot_system(system)
home = system.named_joint_state("home")
q = np.array([home[name] for name in model.active_joint_names], dtype=float)

J = jacobian(model, q, "robot_tcp")
report = singularity_report(J)
```

### Picking a threshold

| Use case | Recommended metric | Threshold guidance |
|---|---|---|
| IK accept / reject | `min_singular_value` | `>= 1e-4` (the IK validator's default) |
| Path validator advisory | `condition_number` | `<= 1000` (the path validator's default) |
| Trajectory smoothing cost | `manipulability` | maximise (no threshold, used as a soft cost) |

The IK and path validators expose `min_sigma_limit` and `condition_number_limit` knobs that map directly to these metrics.

### Performance

One SVD of a `6 × n_active` matrix per call. On FR3 (`n_active = 7..8`) that's a few microseconds. Cheap enough to evaluate at every trajectory sample.

## 4. Examples

| File | What it shows |
|---|---|
| `01_singularity_at_home.py` | Compute every metric at the home pose. |
| `02_near_wrist_singularity.py` | Sweep `q` toward a wrist singularity; watch `min_singular_value` drop. |
| `03_singularity_along_a_path.py` | Plot a metric along a planned path; flag the most-singular waypoint. |

## 5. Common Errors

| Symptom | Cause | Fix |
|---|---|---|
| `condition_number` is `inf` | `σ_min == 0`, true rank deficiency. | Use `min_singular_value` instead; treat `inf` cond as a hard "do not execute" signal. |
| Different metric values for the same `q` | You evaluated `J` in different reference frames. | The singular values are reference-frame invariant in principle but Pinocchio's local vs world rotation can shift them numerically. Pick one frame and stick with it (`"world"` is the default). |
| `min_singular_value` is small at a configuration you thought was safe | You are closer to a singularity than expected — check shoulder, elbow, and wrist explicitly. | Visualise the manipulability ellipsoid via SVD eigenvectors; the smallest singular value's direction is where you have lost dexterity. |

## 6. FAQs

**Q: Which metric should the planner use to penalise near-singular paths?**
`min_singular_value` for a hard bound (set `min_sigma_limit` in `PathValidationOptions`). `condition_number` for an advisory bound. Manipulability for an optimisation cost.

**Q: Why does the IK validator default to `min_sigma_limit = 1e-4` instead of a condition number?**
`min_singular_value` is bounded above by `σ_max` of the Jacobian; for FR3 that's around 1 m/rad. Setting `min_sigma_limit = 1e-4` means "reject configurations where the smallest controllable direction is below 0.1 mm per radian of joint motion". It is more interpretable than condition number for hard cutoffs.

**Q: Can a non-redundant 6-DOF arm be away from a singularity everywhere in its workspace?**
No. Every 6-DOF arm has at least the wrist singularity (last two axes aligned) somewhere in workspace. Choose tasks that avoid it.

**Q: What about 7-DOF arms like FR3?**
7-DOF gives one degree of redundancy; `J` is `6 × 7` and the singular values of `J Jᵀ` are still 6 in number. The redundancy lets the planner pick `q` that avoids singular configurations.
