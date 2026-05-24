"""
World-frame forward kinematics.

`fk_local` returns robot-base-relative transforms. `fk` composes the local FK
result with the robot instance's base pose from the world YAML.

Run from the algorithms directory:

    python examples/fk/04_world_frame_fk.py
"""
from __future__ import annotations
from pathlib import Path
import numpy as np

# WorldDescription parses a world YAML that references robot-system YAML files.
from algorithms.descriptions import WorldDescription

# fk_local gives base-frame FK; fk gives world-frame FK through Scene.
from algorithms.kinematics import fk, fk_local

# KinematicModel resolves the robot, Scene stores runtime/world state.
from algorithms.resolved import KinematicModel, Scene


REPO_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    # 1. Load a world YAML. The world references the robot-system YAML.
    world_path = REPO_ROOT / "configs" / "worlds" / "franka_table_world.yaml"

    # Parse the world and sub-load its referenced robot-system YAML.
    world = WorldDescription.from_yaml(world_path)

    # 2. Create the runtime scene container from the world description.
    # Scene stores world robot placements and runtime object/robot state.
    scene = Scene.from_world(world)

    # 3. Get the robot instance and build its model.
    # "arm" is the robot id declared in the world YAML.
    world_robot = world.robot("arm")

    # Each world robot references one RobotSystemDescription.
    system = world_robot.robot_system

    # Build or fetch the cached KinematicModel for this robot system.
    model = KinematicModel.from_robot_system(system)

    # 4. Build q in active-joint order.
    # Named states are stored by joint name.
    home = system.named_joint_state("home")

    # FK expects q in model.active_joint_names order.
    q = np.array([home[name] for name in model.active_joint_names], dtype=float)

    # 5. Compare robot-base FK and world FK.
    # This transform is relative to the robot's base frame.
    T_base_tcp = fk_local(model, q, "fr3_hand_tcp")

    # This transform includes the world robot base_pose from Scene/world YAML.
    T_world_tcp = fk(scene, "arm", q, "fr3_hand_tcp")

    # Print the robot-base-relative result.
    print("base -> fr3_hand_tcp:")
    print(T_base_tcp.round(4))
    print()

    # Print the world-relative result.
    print("world -> fr3_hand_tcp:")
    print(T_world_tcp.round(4))


if __name__ == "__main__":
    main()
