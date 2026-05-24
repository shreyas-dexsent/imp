"""
Verify pass-through behaviour: robot does NOT stop at interior waypoints.

Plots velocity at every waypoint of the original path: zero at start
and end, non-zero everywhere in between. With `rest_to_rest=True` the
behaviour reverts to stop-at-each-waypoint for comparison.

Run from the algorithms directory:

    python examples/trajectory/02_pass_through_verified.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from algorithms.descriptions import WorldDescription
from algorithms.optimization import shortcut_smooth, spline_fit
from algorithms.planning import plan_joint
from algorithms.resolved import CollisionModel, KinematicModel, Scene
from algorithms.trajectory import TimeParameterizationOptions, time_parameterize

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
    splined = spline_fit(smoothed, samples=10)
    print(f"path has {splined.num_waypoints} waypoints (start, 8 interior, end)")

    for label, rest in [("pass-through (default)", False), ("rest_to_rest=True", True)]:
        result = time_parameterize(
            splined, model,
            options=TimeParameterizationOptions(
                backend="polynomial", dt=0.001, rest_to_rest=rest,
            ),
        )
        traj = result.trajectory
        # Each waypoint sits at chord-length scaling along the duration.
        times = np.linspace(0.0, traj.duration, splined.num_waypoints)
        print(f"\n{label} (duration {traj.duration:.3f} s):")
        print(f"  {'t (s)':>8}  {'qd_norm (rad/s)':>16}")
        for i, t in enumerate(times):
            _, qd, _ = traj.at(float(t))
            tag = "START" if i == 0 else ("END" if i == len(times) - 1 else "INTERIOR")
            print(f"  {float(t):8.3f}  {float(np.linalg.norm(qd)):16.4f}  {tag}")


if __name__ == "__main__":
    main()
