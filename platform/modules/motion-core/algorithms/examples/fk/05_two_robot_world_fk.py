"""Forward kinematics for two robot instances in one world.

Both world robots reference the same robot-system YAML. FK is evaluated per
robot id, and the world base pose makes their world-frame TCP poses different.

Run from the algorithms directory:

    python examples/fk/05_two_robot_world_fk.py
"""
from __future__ import annotations
from pathlib import Path
import numpy as np

# WorldDescription parses the multi-robot world YAML.
from algorithms.descriptions import WorldDescription

# fk computes a world-frame transform for a selected robot id.
from algorithms.kinematics import fk

# KinematicModel resolves robot systems; Scene stores runtime/world state.
from algorithms.resolved import KinematicModel, Scene


REPO_ROOT = Path(__file__).resolve().parents[2]


def active_q(model: KinematicModel) -> np.ndarray:
    # Read the home state from the model's originating description.
    home = model.system.named_joint_state("home")

    # Convert the joint-name dict into model.active_joint_names order.
    return np.array([home[name] for name in model.active_joint_names], dtype=float)


def main() -> None:
    # 1. Load a world with left_arm and right_arm.
    world_path = REPO_ROOT / "configs" / "worlds" / "two_franka_table_world.yaml"

    # Parse the world and its referenced robot-system YAMLs.
    world = WorldDescription.from_yaml(world_path)

    # Create the runtime scene container.
    scene = Scene.from_world(world)

    # 2. Build one model per robot instance. The cache reuses identical systems.
    # left_arm is one robot instance in the world.
    left_system = world.robot("left_arm").robot_system

    # right_arm is another robot instance in the same world.
    right_system = world.robot("right_arm").robot_system

    # These calls may return the same cached model if the systems are identical.
    left_model = KinematicModel.from_robot_system(left_system)
    right_model = KinematicModel.from_robot_system(right_system)

    # 3. Build q for each model.
    q_left = active_q(left_model)
    q_right = active_q(right_model)

    # 4. Query the same local frame on each robot instance.
    # fk uses robot_id to choose the correct world base pose.
    T_world_left_tcp = fk(scene, "left_arm", q_left, "fr3_hand_tcp")
    T_world_right_tcp = fk(scene, "right_arm", q_right, "fr3_hand_tcp")

    # Print the left robot's TCP pose in the shared world frame.
    print("world -> left_arm fr3_hand_tcp:")
    print(T_world_left_tcp.round(4))
    print()

    # Print the right robot's TCP pose in the shared world frame.
    print("world -> right_arm fr3_hand_tcp:")
    print(T_world_right_tcp.round(4))


if __name__ == "__main__":
    main()
