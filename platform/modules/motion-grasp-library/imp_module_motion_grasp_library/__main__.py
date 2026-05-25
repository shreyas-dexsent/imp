"""Run the grasp synthesize module:

    python -m imp_module_motion_grasp_library \
        --station devstation \
        --object-pose-key imp/devstation/perc/s1/world_pose \
        --grasps /path/to/grasps.json
"""

import argparse
import os

from imp_sdk import run_module

from .module import SynthesizeGraspsModule


def main() -> None:
    ap = argparse.ArgumentParser(prog="imp-module-motion-grasp-library")
    ap.add_argument("--station", default=os.environ.get("IMP_STATION", "devstation"))
    ap.add_argument("--object-pose-key", required=True)
    ap.add_argument("--grasps", required=True, help="path to a grasps.json catalogue")
    ap.add_argument("--plan", default="grasps")
    args = ap.parse_args()
    run_module(
        SynthesizeGraspsModule(
            station=args.station,
            object_pose_key=args.object_pose_key,
            grasps_path=args.grasps,
            plan=args.plan,
        )
    )


if __name__ == "__main__":
    main()
