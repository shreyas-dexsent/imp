"""
move_l: linear Cartesian goto (MoveL).

TCP follows a straight line from FK(q_seed) to T_goal. No smoothing,
no spline fit — both would deviate from the line. Pass-through
trajectory by default.

Run from the algorithms directory:

    python examples/primitives/02_move_l.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from algorithms.descriptions import WorldDescription
from algorithms.kinematics import fk
from algorithms.primitives import move_l
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

    T_start = fk(scene, "arm", q_home, "robot_tcp")
    T_goal = T_start.copy()
    T_goal[0, 3] += 0.10
    print(f"TCP at start: {T_start[:3, 3]}")
    print(f"TCP at goal : {T_goal[:3, 3]}")

    result = move_l(scene, "arm", "robot_tcp", T_goal, q_seed=q_home)
    print(f"status   : {result.status.name}")
    print(f"elapsed  : {result.elapsed_ms:.1f} ms")
    if result.trajectory is not None:
        print(f"duration : {result.trajectory.duration:.3f} s")
        print(f"samples  : {result.trajectory.num_samples}")
        final_tcp = result.path.cartesian_waypoints[-1][:3, 3]
        print(f"final TCP error: {np.linalg.norm(final_tcp - T_goal[:3, 3]):.2e} m")


if __name__ == "__main__":
    main()
