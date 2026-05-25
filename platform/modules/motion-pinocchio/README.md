# motion-pinocchio

**Kind:** Module &nbsp;|&nbsp; **Status:** phase 3 — FK (world-frame) + IK (base-frame) implemented + verified

Kinematics imp modules wrapping motion-core (spec §9):

- **`FkModule`** — builds a resolved `Scene` + `KinematicModel` from a
  `world.yaml` at `configure`, then on each `RobotState` calls
  `algorithms.kinematics.fk(scene, robot, q, frame_id)` and publishes the
  TCP pose **in the world frame**. Closes debt **D3** (Scene-fill seam for FK).
- **`IkModule`** — given a base-frame `PoseTarget` + a `RobotState` seed,
  solves with `ik_local` and publishes a `JointSolution`. Verified by
  IK round-trip (`FK(IK(target)) ≈ target`, pos/quat error < 1e-3).

## Run

```bash
# FK is world-frame now: pass --world, not --robot-system.
PYTHONPATH=sdk/py:modules/motion-core/algorithms:modules/motion-pinocchio \
  python -m imp_module_motion_pinocchio --module fk \
    --world modules/motion-core/algorithms/configs/worlds/franka_table_world.yaml \
    --world-robot arm
python modules/motion-pinocchio/examples/verify_fk.py \
  modules/motion-core/algorithms/configs/worlds/franka_table_world.yaml   # RESULT: OK

# IK still works against a robot_system.yaml (it solves in the local base frame).
python -m imp_module_motion_pinocchio --module ik \
  --robot-system modules/motion-core/algorithms/configs/robots/franka_fr3_robot_only.yaml
python modules/motion-pinocchio/examples/verify_ik.py \
  modules/motion-core/algorithms/configs/robots/franka_fr3_robot_only.yaml   # RESULT: OK
```

## Why FK takes a world

The world YAML carries each robot's `base_pose` (a `TransformSpec` from
the world frame to the robot's base). Without it, FK can only compose
into the robot's local base — which is exactly the gap PLAN.md called
out as D3. With `--world`, the resolved `Scene` carries that
composition; the operator's published `Pose6D` lives in the same frame
the perception pipeline uses, no off-band conversion required.

Wraps `robot-algorithms` kinematics via motion-core.
