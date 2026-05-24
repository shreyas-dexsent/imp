"""
Shortcut smoothing on an OMPL path.

Demonstrates:

* OMPL paths zigzag because each iteration extends by `max_joint_step`.
* `shortcut_smooth` removes the zigzags while keeping the path
  collision-free.
* The smoothed path is shorter and has fewer waypoints.

Run from the algorithms directory:

    python examples/optimization/01_shortcut_smoothing.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from algorithms.descriptions import WorldDescription
from algorithms.optimization import shortcut_smooth
from algorithms.planning import PathStatus, plan_joint, validate_path
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
    q_goal = q_home.copy()
    q_goal[0] += 0.8
    q_goal[2] -= 0.3

    result = plan_joint(model, scene, q_home, q_goal)
    assert result.status is PathStatus.SUCCESS
    print(f"raw OMPL path : {result.path.num_waypoints} waypoints, "
          f"length = {result.path.length():.3f} rad")

    smoothed, stats = shortcut_smooth(
        result.path, model, scene, iterations=200, random_seed=0,
    )
    print(f"smoothed path : {smoothed.num_waypoints} waypoints, "
          f"length = {smoothed.length():.3f} rad")
    print(f"shortcut stats: accepted {stats.accepted}/{stats.attempted} attempts")

    # Confirm the smoothed path is still valid.
    report = validate_path(model, scene, smoothed)
    print(f"validate smoothed: passed = {report.passed}")


if __name__ == "__main__":
    main()
