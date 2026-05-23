"""
Known ground-truth example: two objects are inside a clearance threshold.

Run from the algorithms directory:

    python examples/collision/11_known_clearance_pair.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from algorithms.collision import active_pairs, clearance
from algorithms.descriptions import WorldDescription
from algorithms.resolved import CollisionModel, KinematicModel, Scene


REPO_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    # ------------------------------------------------------------------
    # 1. Load the deterministic playground world.
    # ------------------------------------------------------------------
    # matka_clearance_a and matka_clearance_b both have radius 0.10 m.
    # Their centers are 0.23 m apart.
    # Expected signed distance = 0.23 - 0.10 - 0.10 = +0.03 m.
    # Since 0.03 < 0.05, this pair should be reported below threshold.
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
    # 4. Isolate the pair we care about.
    # ------------------------------------------------------------------
    target_pair = tuple(sorted(("matka_clearance_a", "matka_clearance_b")))

    for pair in active_pairs(collision_model, scene):
        if pair != target_pair:
            scene.allow_collision(*pair)

    # ------------------------------------------------------------------
    # 5. Run clearance query.
    # ------------------------------------------------------------------
    threshold = 0.05
    report = clearance(model, scene, q, threshold=threshold)

    # ------------------------------------------------------------------
    # 6. Read the result.
    # ------------------------------------------------------------------
    print("target pair:", target_pair)
    print("expected signed distance:", 0.03)
    print("clearance threshold:", threshold)
    print("reported clearance:", round(report.clearance, 6))
    print("pairs below threshold:", report.pairs_below_threshold)
    print("is target below threshold:", target_pair in report.pairs_below_threshold)


if __name__ == "__main__":
    main()
