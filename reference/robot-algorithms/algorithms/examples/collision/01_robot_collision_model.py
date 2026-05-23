"""
Inspect robot collision geometry loaded from URDF.

Run from the algorithms directory:

    python examples/collision/01_robot_collision_model.py
"""
from __future__ import annotations

from pathlib import Path

from algorithms.descriptions import WorldDescription
from algorithms.resolved import CollisionModel


REPO_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    # 1. Load a world that contains one Franka arm and hand.
    world_path = REPO_ROOT / "configs" / "worlds" / "franka_table_world.yaml"
    world = WorldDescription.from_yaml(world_path)

    # 2. Resolve URDF collision geometry into a CollisionModel.
    collision_model = CollisionModel.from_world(world)

    # 3. Print every robot geometry object.
    # parentJoint is the Pinocchio joint index that drives this shape.
    for geometry_object in collision_model.robot_geom.geometryObjects:
        print(
            geometry_object.name,
            "parentJoint=",
            geometry_object.parentJoint,
            "parentJointName=",
            collision_model.object_parent_joint[geometry_object.name],
        )


if __name__ == "__main__":
    main()
