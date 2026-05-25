"""Run spatial-transform:

  # eye-to-hand (static base->camera edge published once on tf)
  python -m imp_module_spatial_transform \
      --station devstation --pose-key imp/devstation/perc/s1/pose

  # eye-in-hand (camera mounted on the gripper)
  python -m imp_module_spatial_transform \
      --station devstation --pose-key imp/devstation/perc/s1/pose \
      --eye-in-hand --robot fr3 \
      --robot-system .../franka_fr3_with_franka_hand.yaml
"""

import argparse
import os

from imp_sdk import run_module

from .transform import TransformModule


def main() -> None:
    ap = argparse.ArgumentParser(prog="imp-module-spatial-transform")
    ap.add_argument("--station", default=os.environ.get("IMP_STATION", "devstation"))
    ap.add_argument("--pose-key", required=True, help="full keyexpr of the Pose6D input")
    ap.add_argument("--out-plan", default="transform")
    ap.add_argument("--base-frame", default="base")
    ap.add_argument("--eye-in-hand", action="store_true")
    ap.add_argument("--robot", default=None)
    ap.add_argument("--robot-system", default=None)
    ap.add_argument("--chain", default="arm")
    ap.add_argument("--tcp-frame", default=None)
    args = ap.parse_args()

    run_module(
        TransformModule(
            station=args.station,
            pose_key=args.pose_key,
            out_plan=args.out_plan,
            base_frame=args.base_frame,
            eye_in_hand=args.eye_in_hand,
            robot=args.robot,
            robot_system_path=args.robot_system,
            chain=args.chain,
            tcp_frame=args.tcp_frame,
        )
    )


if __name__ == "__main__":
    main()
