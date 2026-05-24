# motion-pinocchio

**Kind:** Module &nbsp;|&nbsp; **Status:** phase 4 — FK + IK implemented + verified

Kinematics imp modules wrapping motion-core (spec §9):

- **`FkModule`** — builds `KinematicModel` from a `robot_system.yaml` at `configure`,
  then on each `RobotState` publishes the TCP `Pose6D` (`algorithms.kinematics.fk_local`).
- **`IkModule`** — given a `PoseTarget` + a `RobotState` seed, solves with
  `ik_local` and publishes a `JointSolution`. Verified by IK round-trip
  (`FK(IK(target)) ≈ target`, pos/quat error < 1e-3).

```bash
PYTHONPATH=sdk/py:modules/motion-core/algorithms:modules/motion-pinocchio \
  python -m imp_module_motion_pinocchio \
    --robot-system modules/motion-core/algorithms/configs/robots/franka_fr3_robot_only.yaml
# cross-check the over-the-bus FK against motion-core's direct FK:
python modules/motion-pinocchio/examples/verify_fk.py \
  modules/motion-core/algorithms/configs/robots/franka_fr3_robot_only.yaml   # RESULT: OK

# IK (add --module ik), then verify the round-trip:
python -m imp_module_motion_pinocchio --module ik \
  --robot-system modules/motion-core/algorithms/configs/robots/franka_fr3_robot_only.yaml
python modules/motion-pinocchio/examples/verify_ik.py \
  modules/motion-core/algorithms/configs/robots/franka_fr3_robot_only.yaml   # RESULT: OK
```

Wraps `robot-algorithms kinematics/` (via motion-core).
