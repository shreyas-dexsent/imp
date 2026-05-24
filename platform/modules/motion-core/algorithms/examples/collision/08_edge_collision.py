"""
Check a sampled joint-space edge for collision.

Run from the algorithms directory:

    python examples/collision/08_edge_collision.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from algorithms.collision import EdgeCollisionOptions, check_edge_collision
from algorithms.descriptions import WorldDescription
from algorithms.resolved import CollisionModel, KinematicModel, Scene


REPO_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    # 1. Load world, collision model, and scene.
    world = WorldDescription.from_yaml(
        REPO_ROOT / "configs" / "worlds" / "franka_table_world.yaml"
    )
    collision_model = CollisionModel.from_world(world)
    scene = Scene.from_world(world, collision_model)

    # 2. Resolve robot and build a start q.
    system = world.robot("arm").robot_system
    model = KinematicModel.from_robot_system(system)
    home = system.named_joint_state("home")
    q_start = np.array([home[name] for name in model.active_joint_names], dtype=float)

    # 3. Build a nearby goal q.
    q_goal = q_start.copy()
    q_goal[0] += 0.10

    # 4. Check the edge using sampled interpolation.
    report = check_edge_collision(
        model,
        scene,
        q_start,
        q_goal,
        options=EdgeCollisionOptions(max_joint_step=0.02),
    )

    print("edge in collision:", report.in_collision)
    print("first collision alpha:", report.first_collision_alpha)
    print("checked states:", report.checked_states)


if __name__ == "__main__":
    main()
