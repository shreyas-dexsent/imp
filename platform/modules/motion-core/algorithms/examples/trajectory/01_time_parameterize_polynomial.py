"""
Generate a smooth pass-through trajectory with the polynomial backend.

The polynomial backend is pure Python and always available. It uses
Catmull-Rom finite-difference velocities at interior waypoints, so the
robot moves smoothly through them without stopping.

Run from the algorithms directory:

    python examples/trajectory/01_time_parameterize_polynomial.py
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
    splined = spline_fit(smoothed, samples=50)

    result = time_parameterize(
        splined, model,
        options=TimeParameterizationOptions(backend="polynomial", dt=0.001),
    )
    traj = result.trajectory
    print(f"backend  : {traj.backend_used}")
    print(f"duration : {traj.duration:.3f} s")
    print(f"samples  : {traj.num_samples}  (dt = 1 ms)")
    print(f"peak |qd|: {np.max(np.abs(traj.velocities)):.3f} rad/s")
    print(f"peak |qdd|: {np.max(np.abs(traj.accelerations)):.3f} rad/s^2")


if __name__ == "__main__":
    main()
