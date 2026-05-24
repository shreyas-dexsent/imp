"""
Cartesian straight-line path: TCP moves 10 cm in +x.

Run from the algorithms directory:

    python examples/planning/02_plan_cartesian_straight_line.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from algorithms.descriptions import WorldDescription
from algorithms.kinematics import fk_local
from algorithms.planning import PathStatus, plan_cartesian
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

    # Define the TCP path: from current TCP pose, move 10 cm in world +x.
    T_start = fk_local(model, q_home, "robot_tcp")
    T_goal = T_start.copy()
    T_goal[0, 3] += 0.10

    result = plan_cartesian(scene, "arm", "robot_tcp", T_start, T_goal, q_home)
    print(f"status   : {result.status.name}")
    print(f"planner  : {result.planner_used}")
    print(f"elapsed  : {result.elapsed_ms:.1f} ms")
    if result.path is not None:
        print(f"samples  : {result.path.num_waypoints}")
        print(f"frame_id : {result.path.metadata['frame_id']}")
        print(f"robot_id : {result.path.metadata['robot_id']}")
        # cartesian_waypoints is populated (only for cartesian paths)
        assert result.path.cartesian_waypoints is not None
        deviation = np.linalg.norm(
            result.path.cartesian_waypoints[-1][:3, 3] - T_goal[:3, 3]
        )
        print(f"final TCP error: {deviation:.6e} m")


if __name__ == "__main__":
    main()
