"""
Compute minimum distance and clearance.

Run from the algorithms directory:

    python examples/collision/07_min_distance_and_clearance.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from algorithms.collision import clearance, min_distance
from algorithms.descriptions import WorldDescription
from algorithms.resolved import CollisionModel, KinematicModel, Scene


REPO_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    # 1. Load world and resolved models.
    world = WorldDescription.from_yaml(
        REPO_ROOT / "configs" / "worlds" / "franka_table_world.yaml"
    )
    collision_model = CollisionModel.from_world(world)
    scene = Scene.from_world(world, collision_model)

    # 2. Build active q in model.active_joint_names order.
    system = world.robot("arm").robot_system
    model = KinematicModel.from_robot_system(system)
    home = system.named_joint_state("home")
    q = np.array([home[name] for name in model.active_joint_names], dtype=float)

    # 3. Compute the closest active pair.
    distance_report = min_distance(model, scene, q)
    print("minimum distance:", distance_report.min_distance)
    print("closest pair:", distance_report.pair)
    print("checked pairs:", distance_report.checked_pairs)

    # 4. Compute all pairs below a threshold.
    clearance_report = clearance(model, scene, q, threshold=0.05)
    print("clearance clipped at threshold:", clearance_report.clearance)
    print("pairs below threshold:", len(clearance_report.pairs_below_threshold))


if __name__ == "__main__":
    main()
