# Jacobian

## 1. Description

The Jacobian `J(q)` of a frame relates joint velocities to the frame's spatial twist:

```text
twist_frame = J(q) @ qdot           # twist in chosen reference frame
   shape: (6, n_active_dof)         # 3 linear rows + 3 angular rows
```

`algorithms.kinematics.jacobian` returns a `(6, n)` matrix in a chosen reference frame. Reference frames are:

| Frame | What the rows mean |
|---|---|
| `"local"` | twist expressed in the frame's own coordinates |
| `"world"` (default) | linear and angular components in world axes, applied at the frame's origin |
| `"local_world_aligned"` | linear in world, angular in world (a hybrid useful for tool-velocity control) |

Companion functions in `algorithms.kinematics.singularity`:

| Function | Returns |
|---|---|
| `manipulability(J)` | Yoshikawa volume `sqrt(det(J Jᵀ))` |
| `condition_number(J)` | `σ_max / σ_min` |
| `inverse_condition_number(J)` | `σ_min / σ_max` (numerically nicer) |
| `min_singular_value(J)` | direct distance-to-singularity proxy |
| `singularity_report(J)` | all of the above plus the SVD rank in one call |

## 2. Data Flow

```text
YAML (robot system)
        |
        v
RobotSystemDescription
        |
        v
KinematicModel  --(active_joint_names, active_to_full, pin_model)-->
        |
        +--- jacobian(model, q, frame_id, reference=...) -> (6, n_active)
        |
        +--- singularity_report(J) -> SingularityMetrics
```

The Jacobian uses `pin.computeJointJacobian` + `pin.getFrameJacobian` internally, then folds the columns from `pin_model`'s full DOF down to active DOF via `J_full @ model.active_to_full` to hide mimic followers. A fresh `pin.Data` is allocated per call.

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

J = jacobian(model, q, "robot_tcp")                            # world reference (default)
J_local = jacobian(model, q, "robot_tcp", reference="local")
J_lwa = jacobian(model, q, "robot_tcp", reference="local_world_aligned")
```

Shape is always `(6, n_active)` where `n_active = len(model.active_joint_names)`.

### Singularity metrics

```python
report = singularity_report(J)
print(report.manipulability)             # Yoshikawa
print(report.condition_number)           # σ_max / σ_min
print(report.inverse_condition_number)   # σ_min / σ_max
print(report.min_singular_value)         # smallest σ
print(report.rank)                       # numerical rank
```

### Picking a reference frame

| You want… | Use |
|---|---|
| Tool-velocity control: linear in world, angular in tool | `"local_world_aligned"` |
| Visualising the Jacobian against world axes | `"world"` |
| Working with quantities expressed in the frame's own coordinates | `"local"` |

Defaulting to `"world"` matches MoveIt / OMPL conventions; switch deliberately when you need otherwise.

### Performance

Per call: one Pinocchio Jacobian evaluation (~tens of microseconds on FR3). `singularity_report` adds one SVD of a `6 × n_active` matrix.

## 4. Examples

| File | What it shows |
|---|---|
| `01_jacobian_at_tcp.py` | Minimal Jacobian at the TCP frame; verifies the linear part against finite-differencing FK. |
| `02_jacobian_reference_frames.py` | Same `q`, three reference frames; relates them via the rotation between TCP and world. |
| `03_jacobian_mimic_column_folding.py` | FR3 + Franka hand: Jacobian has `n_active = 8` columns, not 9; mimic finger is folded. |

## 5. Common Errors

| Symptom | Cause | Fix |
|---|---|---|
| `J.shape == (6, n_full)` and you expected active-only | You bypassed the `algorithms` wrapper and called Pinocchio's Jacobian directly. | Use `algorithms.kinematics.jacobian` so the column folding by `active_to_full` runs. |
| `min_singular_value` is suspiciously small | Robot is near a wrist / elbow / shoulder singularity at this `q`. | Either move away from the singular configuration or accept that fine motions are ill-conditioned there. |
| `condition_number` is huge but `manipulability > 0` | Ill-conditioned but not strictly rank-deficient. | Use `inverse_condition_number` for stable comparisons; `condition_number` overflows for nearly-singular `J`. |
| Reference-frame mismatch with another library | Pinocchio's defaults differ from MoveIt's. | Pass `reference=` explicitly; the three documented values are exhaustive. |

## 6. FAQs

**Q: What's the difference between `world` and `local_world_aligned`?**
`world`: both linear and angular parts of the twist are expressed in world axes and represent how the frame's origin moves in world coordinates.
`local_world_aligned`: linear part in world coordinates, angular part in world coordinates — but specifically the world-aligned twist at the frame's origin, useful when you want world-frame motion commands.

**Q: How do I get the Jacobian of just one chain, not the whole composite model?**
Build the Jacobian on the composite model, then slice the columns matching `chain.joints` from `model.active_joint_names`. Per-chain Jacobian helpers are intentionally not exposed at the top level; the full composite Jacobian is the canonical artifact.

**Q: Is the Jacobian's angular block the spatial twist or the body twist?**
World-frame Jacobian (`reference="world"`): the angular block returns the spatial twist `ω` in world coordinates. Local-frame Jacobian (`reference="local"`): body twist in the frame's own coordinates. Local-world-aligned: world coordinates but at the frame's origin.

**Q: How accurate is it?**
Analytical Pinocchio Jacobian — exact to machine precision. The IK solver uses this exact Jacobian; no finite differencing.
