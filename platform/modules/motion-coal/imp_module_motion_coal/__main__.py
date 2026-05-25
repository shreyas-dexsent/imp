"""Run the collision module.

    # static-scene mode (no perception input):
    python -m imp_module_motion_coal --world .../configs/worlds/franka_table_world.yaml

    # Scene-fill mode: route a perception Pose6D into Scene.set_object_pose
    python -m imp_module_motion_coal \
        --world .../configs/worlds/franka_table_world.yaml \
        --object-pose-key imp/devstation/perc/s1/pose \
        --object-id matka
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
    ap.add_argument(
        "--object-pose-key",
        default=None,
        help="keyexpr of a perception Pose6D (world-frame); enables Scene-fill",
    )
    ap.add_argument(
        "--object-id",
        default=None,
        help="Scene object id updated by the incoming pose (required with --object-pose-key)",
    )
    args = ap.parse_args()

    module = CollisionModule(
        station=args.station,
        robot=args.robot,
        world_path=args.world,
        world_robot=args.world_robot,
        object_pose_key=args.object_pose_key,
        object_id=args.object_id,
    )
    run_module(module)


if __name__ == "__main__":
    main()
