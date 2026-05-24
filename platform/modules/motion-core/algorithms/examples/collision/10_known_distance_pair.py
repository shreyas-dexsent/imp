"""
Known ground-truth example: two objects are separated by a known distance.

Run from the algorithms directory:

    python examples/collision/10_known_distance_pair.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from algorithms.collision import active_pairs, min_distance
from algorithms.descriptions import WorldDescription
from algorithms.resolved import CollisionModel, KinematicModel, Scene


REPO_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    # ------------------------------------------------------------------
    # 1. Load the deterministic playground world.
    # ------------------------------------------------------------------
    # matka_gap_a and matka_gap_b both have radius 0.10 m.
    # Their centers are 0.35 m apart.
    # Expected signed distance = 0.35 - 0.10 - 0.10 = +0.15 m.
    world_path = REPO_ROOT / "configs" / "worlds" / "collision_playground_world.yaml"
    world = WorldDescription.from_yaml(world_path)

    # ------------------------------------------------------------------
    # 2. Build resolved/static and runtime objects.
    # ------------------------------------------------------------------
    collision_model = CollisionModel.from_world(world)
    scene = Scene.from_world(world, collision_model)

    # ------------------------------------------------------------------
    # 3. Build the robot model and q vector.
    # ------------------------------------------------------------------
    system = world.robot("arm").robot_system
    model = KinematicModel.from_robot_system(system)
    home = system.named_joint_state("home")
    q = np.array([home[name] for name in model.active_joint_names], dtype=float)

    # ------------------------------------------------------------------
    # 4. Isolate the known-distance pair.
    # ------------------------------------------------------------------
    target_pair = tuple(sorted(("matka_gap_a", "matka_gap_b")))

    for pair in active_pairs(collision_model, scene):
        if pair != target_pair:
            scene.allow_collision(*pair)

    # ------------------------------------------------------------------
    # 5. Run minimum-distance query.
    # ------------------------------------------------------------------
    report = min_distance(model, scene, q)

    # ------------------------------------------------------------------
    # 6. Read the result.
    # ------------------------------------------------------------------
    print("target pair:", target_pair)
    print("expected signed distance:", 0.15)
    print("measured signed distance:", round(report.min_distance, 6))
    print("closest pair:", report.pair)
    print("nearest point on first object:", report.nearest_points[0].round(4))
    print("nearest point on second object:", report.nearest_points[1].round(4))


if __name__ == "__main__":
    main()
