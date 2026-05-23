"""
move_joint: joint-space goto with smooth pass-through motion.

The simplest primitive. Plans collision-free, smooths, splines, and
time-parameterizes in one call. The returned trajectory is ready to
stream to a controller.

Run from the algorithms directory:

    python examples/primitives/01_move_joint.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from algorithms.descriptions import WorldDescription
from algorithms.primitives import move_joint
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

    result = move_joint(model, scene, q_goal, q_seed=q_home)
    print(f"status   : {result.status.name}")
    print(f"primitive: {result.primitive_used}")
    print(f"elapsed  : {result.elapsed_ms:.1f} ms")
    if result.trajectory is not None:
        traj = result.trajectory
        print(f"backend  : {traj.backend_used}")
        print(f"duration : {traj.duration:.3f} s")
        print(f"samples  : {traj.num_samples}")
        print(f"peak |qd|: {np.max(np.abs(traj.velocities)):.3f} rad/s")


if __name__ == "__main__":
    main()
