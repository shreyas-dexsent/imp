"""
Run a path through the validator and inspect the report.

Two paths are produced:

* A valid one (plan_joint output) — passes every check.
* A deliberately broken one (joint-limit violation inserted) — fails
  at a known waypoint index.

Run from the algorithms directory:

    python examples/planning/03_validate_path.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from algorithms.descriptions import WorldDescription
from algorithms.planning import Path as PlannedPath
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

    # 1. Plan a real path and validate it.
    q_goal = q_home.copy()
    q_goal[0] += 0.5
    result = plan_joint(model, scene, q_home, q_goal, backend="direct")
    assert result.path is not None
    report = validate_path(model, scene, result.path)
    print("Valid path:")
    print(f"  passed       : {report.passed}")
    print(f"  checks run   : {[c.name for c in report.checks]}")
    print(f"  first_failure: {report.first_failure}")

    # 2. Spoil the path: replace a middle waypoint with one outside joint limits.
    _, upper = model.active_position_limits()
    spoiled_wps = result.path.waypoints.copy()
    spoiled_wps[len(spoiled_wps) // 2] = upper + 1.0
    spoiled = PlannedPath(
        waypoints=spoiled_wps,
        joint_names=result.path.joint_names,
    )
    bad_report = validate_path(model, scene, spoiled)
    print()
    print("Spoiled path (joint-limit violation at the middle waypoint):")
    print(f"  passed       : {bad_report.passed}")
    print(f"  first_failure: {bad_report.first_failure}")


if __name__ == "__main__":
    main()
