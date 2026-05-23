# Documentation

`algorithms` is organized around a simple data flow:

```text
YAML descriptions
    -> typed description objects
    -> resolved Pinocchio/Coal models
    -> stateless algorithm calls
    -> typed results and diagnostics
```

The library is intentionally not middleware. It does not subscribe to topics,
open robot-driver sessions, stream commands, or own UI state. Applications
provide runtime inputs, call the algorithms, and decide how to execute or
visualize the outputs.

## Read These First

| Document | Purpose |
| --- | --- |
| [Architecture](architecture.md) | Current architecture, layer boundaries, runtime-state rules, and module responsibilities |
| [YAML schema](yaml_schema.md) | How to author robot-system and world-description YAML |
| [Plan](plan.md) | Long-form design record, methods inventory, build phases, and future decisions |
| [Examples](../examples/README.md) | Runnable scripts for every major module |

## Minimal FK Quickstart

```python
from pathlib import Path

import numpy as np

from algorithms.descriptions import RobotSystemDescription
from algorithms.kinematics import fk_local
from algorithms.resolved import KinematicModel

root = Path("/Users/shreyaskumar/bin-picking/algorithms")

system = RobotSystemDescription.from_yaml(
    root / "configs" / "robots" / "franka_fr3_robot_only.yaml"
)
model = KinematicModel.from_robot_system(system)

home = system.named_joint_state("home")
q = np.array([home[name] for name in model.active_joint_names], dtype=float)

T_base_tcp = fk_local(model, q, "robot_tcp")
```

## Runtime-State Rule

YAML is static. It describes physical facts: URDF paths, meshes, limits, TCPs,
mounts, world objects, and default poses.

Runtime state belongs in `Scene` or function arguments:

- current robot `q`
- live object poses
- attached objects
- dynamic collision allowances
- IK/planning/trajectory options

This is what keeps the same library usable from ROS2, direct robot APIs,
offline scripts, UI tools, and tests.

## Module Map

| Module | Responsibility |
| --- | --- |
| `descriptions` | Pydantic YAML models. Validation only. |
| `resolved` | Heavy resolved objects: `KinematicModel`, `CollisionModel`, `Scene`. |
| `kinematics` | FK, Jacobian, singularity, IK. |
| `collision` | Collision, distance, clearance, and sampled edge checks. |
| `planning` | Joint-space and Cartesian path planning. |
| `optimization` | Geometric path cleanup: shortcut and spline fitting. |
| `trajectory` | Time-parameterization and dense trajectory validation. |
| `primitives` | High-level motion commands composed from lower layers. |

## Verification

```bash
cd /Users/shreyaskumar/bin-picking/algorithms
conda activate robot-engine
python -m pytest tests/
```
