"""
Validation failure modes.

Three calls, three different IKStatus values:

* INVALID_INPUT    - q_seed has the wrong shape
* SINGULARITY_RISK - threshold is set deliberately strict
* FINAL_COLLISION  - validated against a scene that intersects the robot

Run from the algorithms directory:

    python examples/ik/04_validation_failure_modes.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from algorithms.descriptions import RobotSystemDescription, WorldDescription
from algorithms.kinematics import fk_local, ik_local
from algorithms.kinematics.ik import IKOptions
from algorithms.resolved import CollisionModel, KinematicModel, Scene

REPO_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    system = RobotSystemDescription.from_yaml(
        REPO_ROOT / "configs" / "robots" / "franka_fr3_robot_only.yaml"
    )
    model = KinematicModel.from_robot_system(system)
    home = system.named_joint_state("home")
    q_home = np.array([home[name] for name in model.active_joint_names], dtype=float)
    T_target = fk_local(model, q_home, "robot_tcp")

    # Wrong q shape: 7 instead of 7. (Drop one element.)
    wrong_shape = ik_local(model, "robot_tcp", T_target, q_home[:-1])
    print(f"wrong q shape       : {wrong_shape.status.name}")

    # Deliberately strict singularity threshold rejects the home pose.
    singular = ik_local(
        model, "robot_tcp", T_target, q_home,
        options=IKOptions(min_sigma_limit=1e9),
    )
    print(f"strict singularity  : {singular.status.name}")

    # Collision validation rejects the home pose against an intersecting world.
    world = WorldDescription.from_yaml(
        REPO_ROOT / "configs" / "worlds" / "franka_table_world.yaml"
    )
    collision_model = CollisionModel.from_world(world)
    scene = Scene.from_world(world, collision_model)
    hand_model = KinematicModel.from_robot_system(world.robot("arm").robot_system)
    hand_home = world.robot("arm").robot_system.named_joint_state("home")
    q_hand = np.array(
        [hand_home[name] for name in hand_model.active_joint_names], dtype=float,
    )
    T_hand = fk_local(hand_model, q_hand, "fr3_hand_tcp")

    collision = ik_local(
        hand_model, "fr3_hand_tcp", T_hand, q_hand,
        scene=scene,
        options=IKOptions(reject_singular=False),
    )
    print(f"final collision     : {collision.status.name}")


if __name__ == "__main__":
    main()
