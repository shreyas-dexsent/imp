"""
End-to-end optimization: plan -> dedupe -> shortcut -> spline -> validate.

This is the canonical pre-trajectory pipeline. Each step is a separate
function that takes a Path and returns a Path. Timing comes later, at
the trajectory layer.

Run from the algorithms directory:

    python examples/optimization/03_full_optimization_pipeline.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from algorithms.descriptions import WorldDescription
from algorithms.optimization import (
    remove_redundant_waypoints,
    shortcut_smooth,
    spline_fit,
)
from algorithms.planning import plan_joint, validate_path
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

    # 1. Plan
    result = plan_joint(model, scene, q_home, q_goal)
    print(f"1. plan      : {result.path.num_waypoints:3d} waypoints, "
          f"length {result.path.length():.3f} rad")

    # 2. Dedupe (trim OMPL output)
    deduped = remove_redundant_waypoints(result.path)
    print(f"2. dedupe    : {deduped.num_waypoints:3d} waypoints, "
          f"length {deduped.length():.3f} rad")

    # 3. Shortcut smooth (geometric)
    smoothed, stats = shortcut_smooth(
        deduped, model, scene, iterations=200, random_seed=0,
    )
    print(f"3. shortcut  : {smoothed.num_waypoints:3d} waypoints, "
          f"length {smoothed.length():.3f} rad "
          f"({stats.accepted}/{stats.attempted} shortcuts accepted)")

    # 4. Spline fit (C^2 geometry)
    splined = spline_fit(smoothed, order="quintic", samples=80)
    print(f"4. spline    : {splined.num_waypoints:3d} waypoints, "
          f"length {splined.length():.3f} rad")

    # 5. Validate
    report = validate_path(model, scene, splined)
    print(f"5. validate  : passed = {report.passed}, "
          f"checks = {[c.name for c in report.checks]}")


if __name__ == "__main__":
    main()
