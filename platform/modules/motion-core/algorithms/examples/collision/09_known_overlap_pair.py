"""
Known ground-truth example: two objects are intentionally overlapping.

Run from the algorithms directory:

    python examples/collision/09_known_overlap_pair.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from algorithms.collision import CollisionOptions, active_pairs, is_in_collision
from algorithms.descriptions import WorldDescription
from algorithms.resolved import CollisionModel, KinematicModel, Scene


REPO_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    # ------------------------------------------------------------------
    # 1. Load the deterministic playground world.
    # ------------------------------------------------------------------
    # This YAML contains sphere objects with known positions.
    # matka_overlap_a and matka_overlap_b both have radius 0.10 m.
    # Their centers are 0.15 m apart.
    # Expected signed distance = 0.15 - 0.10 - 0.10 = -0.05 m.
    world_path = REPO_ROOT / "configs" / "worlds" / "collision_playground_world.yaml"
    world = WorldDescription.from_yaml(world_path)

    # ------------------------------------------------------------------
    # 2. Build the resolved collision model and runtime scene.
    # ------------------------------------------------------------------
    # CollisionModel is static: it stores all collision shapes.
    collision_model = CollisionModel.from_world(world)

    # Scene is runtime: it stores live object poses and runtime collision allows.
    scene = Scene.from_world(world, collision_model)

    # ------------------------------------------------------------------
    # 3. Build the robot model and q vector.
    # ------------------------------------------------------------------
    # Collision queries currently take a robot model and q because the world
    # may contain robot collision geometry too.
    system = world.robot("arm").robot_system
    model = KinematicModel.from_robot_system(system)

    # Named states in YAML are dictionaries: joint name -> value.
    home = system.named_joint_state("home")

    # Collision queries need NumPy q in model.active_joint_names order.
    q = np.array([home[name] for name in model.active_joint_names], dtype=float)

    # ------------------------------------------------------------------
    # 4. Keep only the object pair we want to study.
    # ------------------------------------------------------------------
    # The full world has many pairs: robot link pairs, robot-object pairs,
    # and object-object pairs. For this tutorial we allow every pair except
    # the known overlapping matka pair.
    target_pair = tuple(sorted(("matka_overlap_a", "matka_overlap_b")))

    for pair in active_pairs(collision_model, scene):
        if pair != target_pair:
            scene.allow_collision(*pair)

    # ------------------------------------------------------------------
    # 5. Run the contact query.
    # ------------------------------------------------------------------
    report = is_in_collision(
        model,
        scene,
        q,
        options=CollisionOptions(
            stop_at_first_contact=True,
            collect_contacts=True,
        ),
    )

    # ------------------------------------------------------------------
    # 6. Read the result.
    # ------------------------------------------------------------------
    print("target pair:", target_pair)
    print("expected collision:", True)
    print("measured collision:", report.in_collision)
    print("checked pairs:", report.checked_pairs)
    print("contact count:", len(report.contacts))

    if report.contacts:
        contact = report.contacts[0]
        print("contact pair:", contact.pair)
        print("penetration depth:", round(contact.penetration, 6))


if __name__ == "__main__":
    main()
