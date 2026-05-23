"""
End-to-end motion pipeline:

  plan_joint -> shortcut_smooth -> spline_fit
              -> time_parameterize -> validate_trajectory

This is the canonical motion-pipeline call sequence. Each step is a
separate function, taking the previous step's output. Failures at any
stage surface with an enumerated status the caller can branch on.

Run from the algorithms directory:

    python examples/trajectory/04_full_pipeline.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from algorithms.descriptions import WorldDescription
from algorithms.optimization import shortcut_smooth, spline_fit
from algorithms.planning import PathStatus, plan_joint, validate_path
from algorithms.resolved import CollisionModel, KinematicModel, Scene
from algorithms.trajectory import (
    TimeParameterizationOptions,
    TrajectoryStatus,
    TrajectoryValidationOptions,
    time_parameterize,
    validate_trajectory,
)

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

    print("1. plan_joint")
    planned = plan_joint(model, scene, q_home, q_goal)
    assert planned.status is PathStatus.SUCCESS
    print(f"   -> {planned.path.num_waypoints} waypoints, length {planned.path.length():.3f} rad")

    print("2. shortcut_smooth")
    smoothed, stats = shortcut_smooth(planned.path, model, scene, iterations=200)
    print(f"   -> {smoothed.num_waypoints} waypoints, length {smoothed.length():.3f} rad "
          f"({stats.accepted}/{stats.attempted} shortcuts)")

    print("3. spline_fit")
    splined = spline_fit(smoothed, samples=80)
    print(f"   -> {splined.num_waypoints} waypoints, length {splined.length():.3f} rad")

    print("4. validate_path (advisory)")
    path_report = validate_path(model, scene, splined)
    print(f"   -> passed = {path_report.passed}")

    print("5. time_parameterize (auto backend, pass-through, dt=1 ms)")
    traj_result = time_parameterize(
        splined, model,
        options=TimeParameterizationOptions(backend="auto", dt=0.001),
    )
    assert traj_result.status is TrajectoryStatus.SUCCESS
    traj = traj_result.trajectory
    print(f"   -> backend={traj.backend_used}  duration={traj.duration:.3f} s  samples={traj.num_samples}")

    print("6. validate_trajectory")
    traj_report = validate_trajectory(
        traj, model, scene,
        options=TrajectoryValidationOptions(check_collision=True),
    )
    print(f"   -> passed = {traj_report.passed}")

    print()
    print(f"controller sample at t={traj.duration:.3f}s (endpoint): "
          f"q = {traj.at(traj.duration)[0]}")


if __name__ == "__main__":
    main()
