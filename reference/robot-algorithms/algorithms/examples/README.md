# Examples

Run examples from the `algorithms/` directory:

```bash
python examples/fk/01_robot_only_fk.py
```

Each example is intentionally self-contained. It repeats the setup steps rather
than hiding them behind shared helper code:

1. Build a path to the YAML file.
2. Parse YAML into a description object.
3. Build a resolved `KinematicModel`, `CollisionModel`, or `Scene`.
4. Convert named joint states into NumPy `q` vectors in active-joint order.
5. Call the algorithm.
6. Print the result or diagnostic report.

## Guides

| Folder | What it covers |
| --- | --- |
| [fk](fk/README.md) | Local/world FK, TCP frames, grippers, multi-robot FK, batched frame lookup |
| [jacobian](jacobian/README.md) | Jacobians at TCPs and reference-frame behavior |
| [singularity](singularity/README.md) | Manipulability, condition number, and minimum singular value |
| [ik](ik/README.md) | Generic IK, DLS, QP velocity IK, validation failures, analytical backend registration |
| [collision](collision/README.md) | Collision models, ACMs, object queries, distance, clearance, edge checks |
| [planning](planning/README.md) | Joint-space and Cartesian planning plus path validation |
| [optimization](optimization/README.md) | Shortcut smoothing and spline fitting |
| [trajectory](trajectory/README.md) | Time-parameterization, sampling, dense trajectory validation |
| [primitives](primitives/README.md) | `move_joint`, `move_l`, approach/retreat, via motion, bin-pick style sequence |
| [integration](integration/README.md) | End-to-end examples that compose several modules |

## Recommended First Pass

```bash
python examples/fk/01_robot_only_fk.py
python examples/ik/01_default_pose_ik.py
python examples/collision/09_known_overlap_pair.py
python examples/planning/01_plan_joint_home_to_target.py
python examples/trajectory/01_time_parameterize_polynomial.py
python examples/primitives/01_move_joint.py
```

## q Vector Rule

All examples use this pattern:

```python
q = np.array([named_state[name] for name in model.active_joint_names], dtype=float)
```

That order is required by every algorithm in this package.
