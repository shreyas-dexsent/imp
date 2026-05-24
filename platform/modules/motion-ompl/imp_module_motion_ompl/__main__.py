"""Run the planning module:

    python -m imp_module_motion_ompl --world .../configs/worlds/franka_robot_only_world.yaml
"""

import argparse
import os

from imp_sdk import run_module

from .plan import PlanModule


def main() -> None:
    ap = argparse.ArgumentParser(prog="imp-module-motion-ompl")
    ap.add_argument("--station", default=os.environ.get("IMP_STATION", "devstation"))
    ap.add_argument("--robot", default=os.environ.get("IMP_DEVICE", "fr3"))
    ap.add_argument("--world", required=True, help="path to a world.yaml")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    run_module(PlanModule(station=args.station, robot=args.robot,
                          world_path=args.world, random_seed=args.seed))


if __name__ == "__main__":
    main()
