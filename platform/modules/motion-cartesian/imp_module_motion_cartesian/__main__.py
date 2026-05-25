"""Run the Cartesian planning module:

    python -m imp_module_motion_cartesian \
        --world .../configs/worlds/franka_robot_only_world.yaml
"""

import argparse
import os

from imp_sdk import run_module

from .plan import CartesianPlanModule


def main() -> None:
    ap = argparse.ArgumentParser(prog="imp-module-motion-cartesian")
    ap.add_argument("--station", default=os.environ.get("IMP_STATION", "devstation"))
    ap.add_argument("--robot", default=os.environ.get("IMP_DEVICE", "fr3"))
    ap.add_argument("--world", required=True, help="path to a world.yaml")
    ap.add_argument("--world-robot", default="arm")
    ap.add_argument("--chain", default="arm")
    ap.add_argument("--tcp-frame", default=None)
    ap.add_argument("--plan", default="cartesian")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    run_module(
        CartesianPlanModule(
            station=args.station,
            robot=args.robot,
            world_path=args.world,
            world_robot=args.world_robot,
            chain=args.chain,
            tcp_frame=args.tcp_frame,
            plan=args.plan,
            random_seed=args.seed,
        )
    )


if __name__ == "__main__":
    main()
