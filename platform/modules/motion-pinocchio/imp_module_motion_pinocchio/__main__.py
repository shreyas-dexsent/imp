"""Run the FK module:

    python -m imp_module_motion_pinocchio \
        --robot-system .../configs/robots/franka_fr3_robot_only.yaml \
        --station devstation --robot fr3
"""

import argparse
import os

from imp_sdk import run_module

from .fk import FkModule
from .ik import IkModule


def main() -> None:
    ap = argparse.ArgumentParser(prog="imp-module-motion-pinocchio")
    ap.add_argument("--module", choices=["fk", "ik"], default="fk")
    ap.add_argument("--station", default=os.environ.get("IMP_STATION", "devstation"))
    ap.add_argument("--robot", default=os.environ.get("IMP_DEVICE", "fr3"))
    ap.add_argument("--robot-system", required=True, help="path to a robot_system.yaml")
    ap.add_argument("--chain", default="arm")
    ap.add_argument("--tcp-frame", default=None)
    args = ap.parse_args()

    cls = FkModule if args.module == "fk" else IkModule
    module = cls(
        station=args.station,
        robot=args.robot,
        robot_system_path=args.robot_system,
        chain=args.chain,
        tcp_frame=args.tcp_frame,
    )
    run_module(module)


if __name__ == "__main__":
    main()
