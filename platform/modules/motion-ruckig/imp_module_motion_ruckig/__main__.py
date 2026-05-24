"""Run the trajectory module:

    python -m imp_module_motion_ruckig --robot-system .../configs/robots/franka_fr3_robot_only.yaml
"""

import argparse
import os

from imp_sdk import run_module

from .trajectory import TrajectoryModule


def main() -> None:
    ap = argparse.ArgumentParser(prog="imp-module-motion-ruckig")
    ap.add_argument("--station", default=os.environ.get("IMP_STATION", "devstation"))
    ap.add_argument("--robot", default=os.environ.get("IMP_DEVICE", "fr3"))
    ap.add_argument("--robot-system", required=True, help="path to a robot_system.yaml")
    args = ap.parse_args()

    run_module(TrajectoryModule(station=args.station, robot=args.robot,
                                robot_system_path=args.robot_system))


if __name__ == "__main__":
    main()
