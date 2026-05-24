"""
Run a robot self-collision query.

Run from the algorithms directory:

    python examples/collision/04_self_collision.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from algorithms.collision import CollisionOptions, is_in_collision
from algorithms.descriptions import WorldDescription
from algorithms.resolved import CollisionModel, KinematicModel, Scene


REPO_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    # 1. Load the world and build resolved collision state.
    world = WorldDescription.from_yaml(
        REPO_ROOT / "configs" / "worlds" / "franka_table_world.yaml"
    )
    collision_model = CollisionModel.from_world(world)
    scene = Scene.from_world(world, collision_model)

    # 2. Resolve the robot kinematic model.
    system = world.robot("arm").robot_system
    model = KinematicModel.from_robot_system(system)

    # 3. Build q in active-joint order.
    home = system.named_joint_state("home")
    q = np.array([home[name] for name in model.active_joint_names], dtype=float)

    # 4. Ask for contacts so the report names the first touching pair.
    report = is_in_collision(
        model,
        scene,
        q,
        options=CollisionOptions(stop_at_first_contact=True, collect_contacts=True),
    )

    print("in collision:", report.in_collision)
    print("checked pairs:", report.checked_pairs)
    print("skipped pairs:", report.skipped_pairs)
    for contact in report.contacts:
        print("contact pair:", contact.pair, "penetration:", contact.penetration)


if __name__ == "__main__":
    main()
