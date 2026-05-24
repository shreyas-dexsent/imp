"""Run the collision module:

    python -m imp_module_motion_coal --world .../configs/worlds/franka_table_world.yaml
"""

import argparse
import os

from imp_sdk import run_module

from .collision import CollisionModule


def main() -> None:
    ap = argparse.ArgumentParser(prog="imp-module-motion-coal")
    ap.add_argument("--station", default=os.environ.get("IMP_STATION", "devstation"))
    ap.add_argument("--robot", default=os.environ.get("IMP_DEVICE", "fr3"))
    ap.add_argument("--world", required=True, help="path to a world.yaml")
    ap.add_argument("--world-robot", default="arm")
    args = ap.parse_args()

    module = CollisionModule(
        station=args.station,
        robot=args.robot,
        world_path=args.world,
        world_robot=args.world_robot,
    )
    run_module(module)


if __name__ == "__main__":
    main()
