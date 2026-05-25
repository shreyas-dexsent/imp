"""Run the FK or IK module.

    # FK now publishes world-frame poses; it needs a world.yaml.
    python -m imp_module_motion_pinocchio --module fk \
        --world  .../configs/worlds/franka_table_world.yaml \
        --world-robot arm \
        --station devstation --robot fr3

    # IK is still local (base-frame target -> joint solution).
    python -m imp_module_motion_pinocchio --module ik \
        --robot-system .../configs/robots/franka_fr3_with_franka_hand.yaml \
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
    ap.add_argument("--world", default=None, help="path to a world.yaml (FK)")
    ap.add_argument("--world-robot", default="arm", help="robot id inside the world (FK)")
    ap.add_argument("--robot-system", default=None, help="path to a robot_system.yaml (IK)")
    ap.add_argument("--chain", default="arm")
    ap.add_argument("--tcp-frame", default=None)
    args = ap.parse_args()

    if args.module == "fk":
        if not args.world:
            ap.error("--module fk requires --world")
        module = FkModule(
            station=args.station,
            robot=args.robot,
            world_path=args.world,
            world_robot=args.world_robot,
            chain=args.chain,
            tcp_frame=args.tcp_frame,
        )
    else:
        if not args.robot_system:
            ap.error("--module ik requires --robot-system")
        module = IkModule(
            station=args.station,
            robot=args.robot,
            robot_system_path=args.robot_system,
            chain=args.chain,
            tcp_frame=args.tcp_frame,
        )

    run_module(module)


if __name__ == "__main__":
    main()
