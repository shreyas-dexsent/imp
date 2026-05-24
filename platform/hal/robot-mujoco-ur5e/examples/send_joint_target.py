"""Send a joint-target MotionCommand to a running UR5e HAL node, then print the
state stream so you can watch it move.

    python -m imp_hal_robot_mujoco_ur5e --device ur5e          # terminal 1
    python examples/send_joint_target.py                       # terminal 2
"""

import os
import sys
import time

from imp_sdk import Bus, QosClass, keyexpr
from imp_sdk.schemas import imp_pb2

STATION = os.environ.get("IMP_STATION", "devstation")
DEVICE = os.environ.get("IMP_DEVICE", "ur5e")
TARGET = [0.5, -1.2, 1.0, -1.4, -1.5, 0.3]


def main() -> None:
    with Bus.open() as bus:
        cmd_key = keyexpr.hal(STATION, DEVICE, "command")
        state_key = keyexpr.hal(STATION, DEVICE, "state")
        sub = bus.subscribe(state_key, imp_pb2.RobotState)
        # Wait for state to flow, then a beat more so this session has learned
        # the node's command subscription before we publish to it.
        sub.recv()
        time.sleep(1.0)

        cmd = imp_pb2.MotionCommand(
            header=imp_pb2.Header(schema="imp.MotionCommand/1"),
            motion_id="demo-1",
            kind="joint",
            q_target=TARGET,
        )
        bus.put(cmd_key, cmd, QosClass.COMMAND)
        print(f"sent joint target {TARGET}", flush=True)

        for _ in range(int(sys.argv[1]) if len(sys.argv) > 1 else 40):
            s = sub.recv()
            print(f"mode={s.mode:8s} q=[{', '.join(f'{x:+.3f}' for x in s.q)}]", flush=True)


if __name__ == "__main__":
    main()
