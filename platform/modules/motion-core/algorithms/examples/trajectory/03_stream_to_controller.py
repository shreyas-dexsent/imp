"""
Sample a trajectory at a fixed controller tick.

`Trajectory.sample(dt)` returns aligned (times, q, qd, qdd) arrays
sized for streaming at the controller's rate. A 1 kHz controller
consumes the 1-ms dt directly; a 125 Hz controller resamples without
re-running the parameterizer.

Run from the algorithms directory:

    python examples/trajectory/03_stream_to_controller.py
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

    planned = plan_joint(model, scene, q_home, q_goal)
    smoothed, _ = shortcut_smooth(planned.path, model, scene, iterations=100)
    splined = spline_fit(smoothed, samples=40)
    result = time_parameterize(
        splined, model,
        options=TimeParameterizationOptions(backend="polynomial", dt=0.001),
    )
    traj = result.trajectory
    print(f"native trajectory : duration {traj.duration:.3f} s at dt=1 ms "
          f"({traj.num_samples} samples)")

    # Resample at common controller rates.
    for rate_hz in [125, 500, 1000]:
        dt_ctrl = 1.0 / rate_hz
        times, q, qd, qdd = traj.sample(dt_ctrl)
        print(f"  {rate_hz:>5} Hz controller stream: {len(times):>5} samples, dt={dt_ctrl*1000:.2f} ms")

    # One-off query
    t_sample = traj.duration * 0.5
    q_mid, qd_mid, qdd_mid = traj.at(float(t_sample))
    print(f"\nat t={t_sample:.3f}s mid-trajectory: |q|={np.linalg.norm(q_mid):.3f}  "
          f"|qd|={np.linalg.norm(qd_mid):.3f}  |qdd|={np.linalg.norm(qdd_mid):.3f}")


if __name__ == "__main__":
    main()
