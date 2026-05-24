"""
Run a robot-vs-world collision query.

Run from the algorithms directory:

    python examples/collision/05_world_collision.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from algorithms.collision import CollisionOptions, is_in_collision
from algorithms.descriptions import WorldDescription
from algorithms.resolved import CollisionModel, KinematicModel, Scene


REPO_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    # 1. Load world, collision model, and runtime scene.
    world = WorldDescription.from_yaml(
        REPO_ROOT / "configs" / "worlds" / "franka_table_world.yaml"
    )
    collision_model = CollisionModel.from_world(world)
    scene = Scene.from_world(world, collision_model)

    # 2. Move the object close to the robot base to make a world query visible.
    T_world_matka = scene.get_object_pose("matka").copy()
    T_world_matka[:3, 3] = np.array([0.0, 0.0, 0.10])
    scene.set_object_pose("matka", T_world_matka)

    # 3. Build q for the robot.
    system = world.robot("arm").robot_system
    model = KinematicModel.from_robot_system(system)
    home = system.named_joint_state("home")
    q = np.array([home[name] for name in model.active_joint_names], dtype=float)

    # 4. Run the query.
    report = is_in_collision(
        model,
        scene,
        q,
        options=CollisionOptions(stop_at_first_contact=False, collect_contacts=True),
    )

    print("in collision:", report.in_collision)
    for contact in report.contacts:
        if "matka" in contact.pair:
            print("world contact:", contact.pair, "penetration:", contact.penetration)


if __name__ == "__main__":
    main()
