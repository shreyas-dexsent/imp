"""
Wire a perception-supplied point cloud into the scene as a Coal OcTree.

This is the native answer for point-cloud inputs. Perception ships an
(N, 3) array of points plus a voxel resolution; the library hands it
to `coal.makeOctree` and the planner queries collisions directly. No
mesh reconstruction, no surface fitting, no algorithmic work inside the
library.

Run from the algorithms directory:

    python examples/collision/13_pointcloud_to_octree.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from algorithms.collision import min_distance
from algorithms.descriptions import OctreeGeometrySpec, WorldDescription
from algorithms.kinematics import fk_local
from algorithms.resolved import CollisionModel, KinematicModel, Scene

REPO_ROOT = Path(__file__).resolve().parents[2]


def fake_point_cloud():
    """Synthesise a small clustered point cloud near the robot TCP.

    A real perception pipeline would supply this after RGB-D fusion,
    background subtraction, downsampling, and outlier removal. The
    library does not care how the points were produced.
    """
    rng = np.random.default_rng(0)
    # Two clusters: a "bin wall" and a "workpiece blob".
    wall = rng.normal(loc=[0.6, 0.3, 0.15], scale=[0.005, 0.15, 0.10], size=(400, 3))
    blob = rng.normal(loc=[0.5, 0.05, 0.06], scale=0.03, size=(150, 3))
    return np.vstack([wall, blob])


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

    points = fake_point_cloud()
    print(f"perception cloud: {len(points)} points")

    # Resolution is the voxel edge length in metres. 1 cm is a common
    # default for industrial cells; coarser is faster but more
    # conservative.
    scene.add_object(
        "scene_cloud",
        collision=OctreeGeometrySpec(
            type="octree",
            points=points.tolist(),
            resolution=0.01,
        ),
        pose=np.eye(4),
    )

    info = cm.shapes_for("scene_cloud")[0]
    octree = info.coal_shape  # the actual coal.OcTree the planner uses
    print(f"registered as kind={info.kind!r}, resolution={octree.getResolution():.3f} m, "
          f"occupied nodes={octree.size()}")

    # Distance from the robot at home to the perception cloud.
    report = min_distance(model, scene, q_home)
    print(f"min distance robot-vs-world at home: {report.min_distance:.4f} m"
          f" between {report.pair}")


if __name__ == "__main__":
    main()
