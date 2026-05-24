"""Run the MuJoCo UR5e HAL node:

    python -m imp_hal_robot_mujoco_ur5e --station devstation --device ur5e
"""

import argparse
import os

from imp_sdk import run_device

from .driver import MujocoUR5e


def main() -> None:
    ap = argparse.ArgumentParser(prog="imp-hal-robot-mujoco-ur5e")
    ap.add_argument("--station", default=os.environ.get("IMP_STATION", "devstation"))
    ap.add_argument("--device", default=os.environ.get("IMP_DEVICE", "ur5e"))
    ap.add_argument("--model", default=None, help="path to an MJCF scene (defaults to the bundled UR5e)")
    ap.add_argument("--state-hz", type=float, default=125.0)
    args = ap.parse_args()

    device = MujocoUR5e(model_path=args.model, state_hz=args.state_hz)
    run_device(device, args.station, args.device)


if __name__ == "__main__":
    main()
