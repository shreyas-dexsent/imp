"""
Iterate the collision catalogue exactly as a UI would.

This shows the single-source-of-truth pattern: anything that visualises
the scene reads from `CollisionModel.shapes_for(...)`, which returns
the exact Coal objects the planner queries. If V-HACD produced 12
hulls, the UI iterates 12 entries. If perception shipped an octree,
the UI sees `kind="octree"` and can render the voxel boxes via
`coal.OcTree.toBoxes()`.

Run from the algorithms directory:

    python examples/collision/14_inspect_shapes_for_ui.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from algorithms.descriptions import (
    BoxGeometrySpec,
    CapsuleGeometrySpec,
    ConvexHullGeometrySpec,
    OctreeGeometrySpec,
    WorldDescription,
)
from algorithms.kinematics import fk
from algorithms.resolved import CollisionModel, Scene

REPO_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    world = WorldDescription.from_yaml(
        REPO_ROOT / "configs" / "worlds" / "franka_table_world.yaml"
    )
    cm = CollisionModel.from_world(world)
    scene = Scene.from_world(world, cm)

    # Inject one of each shape kind so the UI loop has something to do.
    scene.add_object(
        "rod", collision=CapsuleGeometrySpec(type="capsule", radius=0.05, length=0.3),
        pose=_translate([0.6, 0.0, 0.4]),
    )
    block_verts = [
        (0, 0, 0), (0.1, 0, 0), (0, 0.1, 0), (0, 0, 0.1),
        (0.1, 0.1, 0), (0.1, 0, 0.1), (0, 0.1, 0.1), (0.1, 0.1, 0.1),
    ]
    scene.add_object(
        "block",
        collision=ConvexHullGeometrySpec(type="convex_hull", vertices=block_verts),
        pose=_translate([0.5, 0.2, 0.05]),
    )
    rng = np.random.default_rng(0)
    pts = rng.uniform(-0.05, 0.05, (200, 3)).tolist()
    scene.add_object(
        "scan",
        collision=OctreeGeometrySpec(type="octree", points=pts, resolution=0.01),
        pose=_translate([0.4, -0.2, 0.1]),
    )

    # Pretend to be a UI. Loop every object, ask CollisionModel what
    # Coal actually holds, render accordingly.
    print(f"{'object':<25s}  {'kind':<14s}  {'world pose (xyz)':<25s}  detail")
    print("-" * 100)
    for name in cm.object_names():
        for info in cm.shapes_for(name):
            T_world_shape = _world_shape_pose(scene, info)
            xyz = T_world_shape[:3, 3]
            detail = _detail_string(info)
            print(f"{name:<25s}  {info.kind:<14s}  {xyz!s:<25s}  {detail}")


def _world_shape_pose(scene: Scene, info) -> np.ndarray:
    """Compose the world pose of a shape using the same path the
    collision query does.

    World objects: `Scene.object_poses[name] @ T_parent_shape`.
    Robot links: FK on the parent joint composed with `T_parent_shape`.
    UI code should do exactly this composition.
    """
    if info.owner == "world":
        T_world_obj = scene.object_poses.get(info.name, np.eye(4))
        return T_world_obj @ info.T_parent_shape
    # Robot link: the example world has a single robot named "arm".
    # In production, look up the robot id by parent-joint membership.
    robot_id = scene.world.robots[0].id
    q = scene.robot_states.get(robot_id)
    if q is None:
        # No live state yet; fall back to "home".
        system = scene.world.robot(robot_id).robot_system
        home = system.named_joint_state("home")
        from algorithms.resolved import KinematicModel
        model = KinematicModel.from_robot_system(system)
        q = np.array([home[name] for name in model.active_joint_names], dtype=float)
    # FK on the parent joint frame.
    return fk(scene, robot_id, q, info.parent_joint) @ info.T_parent_shape


def _detail_string(info) -> str:
    """Short renderer-facing string. A real UI would dispatch on `kind`
    to its mesh / primitive / voxel renderer here."""
    shape = info.coal_shape
    if info.kind == "box":
        return f"halfSide={np.asarray(shape.halfSide).tolist()}"
    if info.kind == "sphere":
        return f"radius={shape.radius:.3f}"
    if info.kind == "cylinder":
        return f"radius={shape.radius:.3f} halfLength={shape.halfLength:.3f}"
    if info.kind == "capsule":
        return f"radius={shape.radius:.3f} halfLength={shape.halfLength:.3f}"
    if info.kind == "convex_hull":
        return f"vertices={shape.num_points}"
    if info.kind == "octree":
        return f"resolution={shape.getResolution():.3f} occupied={shape.size()}"
    if info.kind == "height_field":
        return "height field grid"
    if info.kind == "mesh":
        return f"BVH triangle mesh"
    return "?"


def _translate(xyz):
    T = np.eye(4)
    T[:3, 3] = xyz
    return T


if __name__ == "__main__":
    main()
