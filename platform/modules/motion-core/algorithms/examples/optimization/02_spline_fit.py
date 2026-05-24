"""
Quintic spline fit through a path's waypoints.

Demonstrates:

* The fit takes any Path and returns a denser Path with smooth
  C^2 geometry.
* Output is still a Path (geometry only) — no velocities, no
  timestamps.
* Endpoints are pinned exactly; interior waypoints are reshaped to
  satisfy continuity.

Run from the algorithms directory:

    python examples/optimization/02_spline_fit.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from algorithms.descriptions import WorldDescription
from algorithms.optimization import shortcut_smooth, spline_fit
from algorithms.planning import plan_joint
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

    planned = plan_joint(model, scene, q_home, q_goal)
    smoothed, _ = shortcut_smooth(planned.path, model, scene, iterations=200)

    quintic = spline_fit(smoothed, order="quintic", samples=100)
    cubic = spline_fit(smoothed, order="cubic", samples=100)

    print(f"shortcut: {smoothed.num_waypoints} waypoints, length {smoothed.length():.3f} rad")
    print(f"quintic : {quintic.num_waypoints} waypoints, length {quintic.length():.3f} rad")
    print(f"cubic   : {cubic.num_waypoints} waypoints, length {cubic.length():.3f} rad")

    # Endpoints are preserved exactly across all of them.
    np.testing.assert_allclose(quintic.waypoints[0], q_home, atol=1e-12)
    np.testing.assert_allclose(quintic.waypoints[-1], q_goal, atol=1e-6)
    print("endpoints preserved across the optimization passes")


if __name__ == "__main__":
    main()
