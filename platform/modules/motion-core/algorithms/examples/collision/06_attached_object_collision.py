"""
Attach an object and check collision while it follows a TCP.

Run from the algorithms directory:

    python examples/collision/06_attached_object_collision.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from algorithms.collision import CollisionOptions, is_in_collision
from algorithms.descriptions import WorldDescription
from algorithms.resolved import CollisionModel, KinematicModel, Scene


REPO_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    # 1. Build world, collision model, and scene.
    world = WorldDescription.from_yaml(
        REPO_ROOT / "configs" / "worlds" / "franka_table_world.yaml"
    )
    collision_model = CollisionModel.from_world(world)
    scene = Scene.from_world(world, collision_model)

    # 2. Resolve robot model and q.
    system = world.robot("arm").robot_system
    model = KinematicModel.from_robot_system(system)
    home = system.named_joint_state("home")
    q = np.array([home[name] for name in model.active_joint_names], dtype=float)

    # 3. Attach matka to the hand TCP with an identity local offset.
    scene.attach(
        "matka",
        "fr3_hand_tcp",
        np.eye(4),
        allow_collision_with=["fr3_hand_0"],
    )

    # 4. Query collisions. The object's pose now comes from FK on fr3_hand_tcp.
    report = is_in_collision(
        model,
        scene,
        q,
        options=CollisionOptions(stop_at_first_contact=False, collect_contacts=True),
    )

    print("in collision:", report.in_collision)
    print("attached objects:", sorted(scene.attached))
    for contact in report.contacts:
        if "matka" in contact.pair:
            print("attached-object contact:", contact.pair)


if __name__ == "__main__":
    main()
