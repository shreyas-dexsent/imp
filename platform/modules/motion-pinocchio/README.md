# motion-pinocchio

**Kind:** Module &nbsp;|&nbsp; **Status:** phase 3 — FK implemented + verified

Kinematics imp module wrapping motion-core. `FkModule` is a Compute-Runtime module
(spec §9): it builds the resolved `KinematicModel` from a `robot_system.yaml` at
`configure`, then on each `RobotState` computes the TCP pose via
`algorithms.kinematics.fk_local` and publishes a `Pose6D` in the robot base frame.
IK + Jacobian follow the same pattern (motion-core already provides them).

```bash
PYTHONPATH=sdk/py:modules/motion-core/algorithms:modules/motion-pinocchio \
  python -m imp_module_motion_pinocchio \
    --robot-system modules/motion-core/algorithms/configs/robots/franka_fr3_robot_only.yaml
# cross-check the over-the-bus FK against motion-core's direct FK:
python modules/motion-pinocchio/examples/verify_fk.py \
  modules/motion-core/algorithms/configs/robots/franka_fr3_robot_only.yaml   # RESULT: OK
```

Wraps `robot-algorithms kinematics/` (via motion-core).
