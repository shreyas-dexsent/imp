"""
Add perception-supplied objects into a live Scene.

This is the integration point for perception code. Perception decides
the geometry (a primitive, a hull, a mesh, an octree, ...); the library
wires it into Scene + CollisionModel so the planner can use it.

Run from the algorithms directory:

    python examples/collision/12_perception_runtime_add.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from algorithms.collision import is_in_collision
from algorithms.descriptions import (
    BoxGeometrySpec,
    CapsuleGeometrySpec,
    ConvexHullGeometrySpec,
    WorldDescription,
)
from algorithms.kinematics import fk_local
from algorithms.resolved import CollisionModel, KinematicModel, Scene

REPO_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    world = WorldDescription.from_yaml(
        REPO_ROOT / "configs" / "worlds" / "franka_table_world.yaml"
    )
    cm = CollisionModel.from_world(world)
    scene = Scene.from_world(world, cm)

    system = world.robot("arm").robot_system
    model = KinematicModel.from_robot_system(system)
    home = system.named_joint_state("home")
    q_home = np.array([home[name] for name in model.active_joint_names], dtype=float)

    print("Catalogue before perception adds:")
    print(f"  world objects: {[n for n in cm.object_names() if cm.object_owner.get(n) == 'world']}")

    # Perception delivers a safety zone as a capsule at runtime.
    safety_pose = np.eye(4)
    safety_pose[:3, 3] = [0.8, 0.0, 0.5]
    scene.add_object(
        "safety_zone",
        collision=CapsuleGeometrySpec(type="capsule", radius=0.10, length=0.40),
        pose=safety_pose,
    )

    # And a workpiece estimated as a convex block.
    block_verts = [
        (0.0, 0.0, 0.0), (0.08, 0.0, 0.0), (0.0, 0.08, 0.0), (0.0, 0.0, 0.05),
        (0.08, 0.08, 0.0), (0.08, 0.0, 0.05), (0.0, 0.08, 0.05), (0.08, 0.08, 0.05),
    ]
    block_pose = np.eye(4)
    block_pose[:3, 3] = [0.55, -0.1, 0.05]
    scene.add_object(
        "workpiece_est",
        collision=ConvexHullGeometrySpec(type="convex_hull", vertices=block_verts),
        pose=block_pose,
    )

    # Or a coarse obstacle approximated as a box.
    scene.add_object(
        "table_edge",
        collision=BoxGeometrySpec(type="box", size=(0.05, 1.0, 0.05)),
        pose=np.eye(4),
    )

    print("Catalogue after perception adds:")
    perception_objects = ["safety_zone", "workpiece_est", "table_edge"]
    for name in perception_objects:
        info = cm.shapes_for(name)[0]
        print(f"  {name:15s}  kind={info.kind:12s}  owner={info.owner}")

    # Run a collision query against the augmented scene.
    report = is_in_collision(model, scene, q_home)
    print(f"\nrobot at home, in_collision: {report.in_collision}")

    # Perception removes the safety zone (e.g., human left the cell).
    scene.remove_object("safety_zone")
    print(f"\nafter removing safety_zone, has_object: {cm.has_object('safety_zone')}")
    print(f"final object_poses ids: {sorted(scene.object_poses.keys())}")


if __name__ == "__main__":
    main()
