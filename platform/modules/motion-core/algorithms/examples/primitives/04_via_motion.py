"""
via_motion: smooth motion through a sequence of via-points.

The robot moves from home through two intermediate joint configurations
to a final goal. Velocity at every interior via-point is non-zero — the
robot flows through them without stopping.

Run from the algorithms directory:

    python examples/primitives/04_via_motion.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from algorithms.descriptions import WorldDescription
from algorithms.primitives import via_motion
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

    via_a = q_home.copy()
    via_a[0] += 0.3
    via_b = q_home.copy()
    via_b[0] += 0.6
    via_b[2] -= 0.2
    goal = q_home.copy()
    goal[0] += 0.8

    print(f"4 via-points: home -> via_a -> via_b -> goal")
    result = via_motion(model, scene, [q_home, via_a, via_b, goal])
    print(f"status   : {result.status.name}")
    print(f"elapsed  : {result.elapsed_ms:.1f} ms")
    if result.trajectory is not None:
        traj = result.trajectory
        print(f"duration : {traj.duration:.3f} s")
        print(f"samples  : {traj.num_samples}")

        # Spot-check pass-through behaviour at evenly spaced times.
        print("\nVelocity norm at quarter intervals (non-zero = pass-through):")
        for frac in [0.0, 0.25, 0.5, 0.75, 1.0]:
            t = frac * traj.duration
            _, qd, _ = traj.at(float(t))
            tag = ["START", "Q1", "MID", "Q3", "END"][int(frac * 4)]
            print(f"  t={t:5.2f}s ({tag:>5s})  |qd|={float(np.linalg.norm(qd)):.3f} rad/s")


if __name__ == "__main__":
    main()
