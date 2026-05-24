"""
Plan a collision-free joint-space path with the default OMPL backend.

Run from the algorithms directory:

    python examples/planning/01_plan_joint_home_to_target.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from algorithms.descriptions import WorldDescription
from algorithms.planning import PathStatus, plan_joint
from algorithms.resolved import CollisionModel, KinematicModel, Scene

REPO_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    world = WorldDescription.from_yaml(
        REPO_ROOT / "configs" / "worlds" / "franka_robot_only_world.yaml"
    )
    cm = CollisionModel.from_world(world)
    scene = Scene.from_world(world, cm)

    system = world.robots[0].robot_system
    model = KinematicModel.from_robot_system(system)
    home = system.named_joint_state("home")
    q_home = np.array([home[name] for name in model.active_joint_names], dtype=float)

    # Pick a goal: 45 deg in joint 0, lift link 4 a bit. In production
    # the goal comes from IK on a perception target.
    q_goal = q_home.copy()
    q_goal[0] += 0.8
    q_goal[2] -= 0.3

    result = plan_joint(model, scene, q_home, q_goal)
    print(f"status  : {result.status.name}")
    print(f"planner : {result.planner_used}")
    print(f"elapsed : {result.elapsed_ms:.1f} ms")
    if result.path is not None:
        print(f"waypoints: {result.path.num_waypoints}")
        print(f"length  : {result.path.length():.3f} rad (sum of segment norms)")


if __name__ == "__main__":
    main()
