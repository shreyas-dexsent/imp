# robot-mujoco-ur5e

**Kind:** HAL driver &nbsp;|&nbsp; **Status:** phase 2 — implemented (sim)

MuJoCo UR5e simulation robot, on the Python HAL framework (`imp_sdk.hal`).
Publishes `RobotState` (q, dq, TCP pose, mode) at 125 Hz and subscribes
`MotionCommand` (joint target or trajectory), executing smooth kinematic motion.
IK / TCP targets are resolved by a motion module, not the HAL (spec §8/§9).

```bash
# deps: pip install -e sdk/py && pip install mujoco numpy
PYTHONPATH=sdk/py:hal/robot-mujoco-ur5e python -m imp_hal_robot_mujoco_ur5e --device ur5e
# then, in another shell:
imp topic echo 'imp/devstation/hal/ur5e/state'
python hal/robot-mujoco-ur5e/examples/send_joint_target.py
```

The bundled `assets/ur5e_mjcf/` is the MuJoCo Menagerie UR5e model (see its
LICENSE). Migrated from the VGR reference `robot_controller/adapters/mujoco_ur5e/`.
