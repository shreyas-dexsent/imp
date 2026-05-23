"""
approach + retreat: the bin-picking grasp/lift pattern.

This example simulates the final two legs of a top-down grasp:

1. `approach`: linear descent from a 5 cm pre-grasp pose down to the
   grasp pose along the target's local -z axis.
2. `retreat`: linear lift 5 cm along the local +z axis from the grasp
   pose.

Application code usually does a joint-space `move_joint` first to get
the arm to the pre-approach pose; here we cheat by using q_home as the
seed for both calls.

Run from the algorithms directory:

    python examples/primitives/03_approach_retreat.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from algorithms.descriptions import WorldDescription
from algorithms.kinematics import fk
from algorithms.primitives import approach, pre_approach_pose, retreat
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

    T_grasp = fk(scene, "arm", q_home, "robot_tcp")
    T_pre = pre_approach_pose(T_grasp, distance=0.05, axis="-z", reference="target")
    print(f"target  TCP: {T_grasp[:3, 3]}")
    print(f"pre-app TCP: {T_pre[:3, 3]}  (5 cm above target)")

    print("\nLeg 1: approach (linear descent from pre-approach to grasp)")
    r_approach = approach(
        scene, "arm", "robot_tcp", T_grasp, q_seed=q_home,
        distance=0.05, axis="-z", reference="target",
    )
    print(f"  status   : {r_approach.status.name}")
    print(f"  elapsed  : {r_approach.elapsed_ms:.1f} ms")
    if r_approach.trajectory is not None:
        print(f"  duration : {r_approach.trajectory.duration:.3f} s")

    print("\nLeg 2: retreat (linear lift 5 cm along +z)")
    r_retreat = retreat(
        scene, "arm", "robot_tcp", q_seed=q_home,
        distance=0.05, axis="z", reference="tcp",
    )
    print(f"  status   : {r_retreat.status.name}")
    print(f"  elapsed  : {r_retreat.elapsed_ms:.1f} ms")
    if r_retreat.trajectory is not None:
        print(f"  duration : {r_retreat.trajectory.duration:.3f} s")


if __name__ == "__main__":
    main()
