"""Run the path processor:

    python -m imp_module_motion_path_processor \
        --world .../configs/worlds/franka_robot_only_world.yaml
"""

import argparse
import os

from imp_sdk import run_module

from .process import PathProcessorModule


def main() -> None:
    ap = argparse.ArgumentParser(prog="imp-module-motion-path-processor")
    ap.add_argument("--station", default=os.environ.get("IMP_STATION", "devstation"))
    ap.add_argument("--robot", default=os.environ.get("IMP_DEVICE", "fr3"))
    ap.add_argument("--world", required=True, help="path to a world.yaml")
    ap.add_argument("--world-robot", default="arm")
    ap.add_argument("--plan-in", default="plan", help="upstream plan key (default: plan)")
    ap.add_argument("--plan-out", default="processed", help="downstream plan key (default: processed)")
    ap.add_argument("--shortcut-iters", type=int, default=100)
    ap.add_argument("--max-joint-step", type=float, default=0.05)
    ap.add_argument(
        "--spline-order",
        choices=["cubic", "quintic", "none"],
        default="quintic",
    )
    ap.add_argument("--spline-samples", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    run_module(
        PathProcessorModule(
            station=args.station,
            robot=args.robot,
            world_path=args.world,
            world_robot=args.world_robot,
            plan_in=args.plan_in,
            plan_out=args.plan_out,
            shortcut_iters=args.shortcut_iters,
            max_joint_step=args.max_joint_step,
            spline_order=None if args.spline_order == "none" else args.spline_order,
            spline_samples=args.spline_samples,
            random_seed=args.seed,
        )
    )


if __name__ == "__main__":
    main()
