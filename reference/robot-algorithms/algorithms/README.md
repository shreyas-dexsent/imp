# algorithms

`algorithms` is a robotics algorithms library for industrial manipulation.
It loads static robot/world descriptions from YAML, resolves them into
Pinocchio and Coal models, and exposes stateless NumPy-based operations for:

- forward kinematics, Jacobians, singularity metrics, and inverse kinematics
- collision and distance queries against runtime scene state
- joint-space and Cartesian path planning
- geometric path optimization
- trajectory time-parameterization and validation
- reusable motion primitives such as `move_joint`, `move_l`, `approach`,
  `retreat`, and `via_motion`

The package is deliberately transport-agnostic. It does not own ROS nodes,
robot drivers, controller sessions, UI synchronization, bagging, or live
middleware monitoring. Application code owns those concerns and writes live
state into `Scene`.

## Install

```bash
cd /Users/shreyaskumar/bin-picking/algorithms
conda env create -f environment.yml
conda activate robot-engine
pip install -e .
```

If the conda environment already exists:

```bash
conda activate robot-engine
cd /Users/shreyaskumar/bin-picking/algorithms
pip install -e .
```

## Quick Test

```bash
python -m pytest tests/
python examples/fk/01_robot_only_fk.py
python examples/ik/01_default_pose_ik.py
python examples/planning/01_plan_joint_home_to_target.py
```

## First Example

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
print(T_base_tcp)
```

The important contract is the `q` order:

```python
model.active_joint_names
```

Every kinematics, collision, planning, trajectory, and primitive API expects
NumPy vectors in that active-joint order.

## Project Layout

```text
configs/                 Robot and world YAML descriptions
algorithms/descriptions  Pydantic schema layer; no URDF or algorithm work
algorithms/resolved      Resolved Pinocchio/Coal models and mutable Scene
algorithms/kinematics    FK, Jacobian, singularity, IK
algorithms/collision     Collision, distance, clearance, edge checks
algorithms/planning      Joint and Cartesian path planning
algorithms/optimization  Shortcut smoothing and spline fitting
algorithms/trajectory    Time-parameterization and trajectory validation
algorithms/primitives    Reusable high-level motion primitives
examples/                Runnable examples, organized by module
tests/                   Test suite
scripts/                 Optional benchmark scripts for local performance checks
docs/                    Architecture and schema documentation
```

## Documentation

- [docs/README.md](docs/README.md) — quickstart and library map
- [docs/architecture.md](docs/architecture.md) — locked architecture and data flow
- [docs/yaml_schema.md](docs/yaml_schema.md) — robot/world YAML authoring
- [docs/plan.md](docs/plan.md) — detailed build plan and design decisions
- [examples/README.md](examples/README.md) — runnable examples by topic

## What This Library Does Not Do

`algorithms` does not execute commands on hardware. It produces typed data:
poses, paths, trajectories, validation reports, and diagnostics. A separate
application layer should adapt those outputs to ROS actions, libfranka loops,
RTDE streams, EtherCAT, PLCs, or proprietary robot APIs.

## Optional Benchmark Scripts

The `scripts/` folder is intentionally kept. These are not production APIs;
they are local performance smoke tests:

```bash
python scripts/bench_ik.py
python scripts/bench_planning.py
python scripts/bench_trajectory.py
```

Use them after algorithm changes to catch obvious latency regressions.
