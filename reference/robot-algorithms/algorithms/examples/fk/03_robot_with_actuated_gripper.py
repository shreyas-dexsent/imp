"""
Forward kinematics for a robot with an actuated gripper.

The FR3 robot URDF and Franka hand URDF are composed into one Pinocchio model.
The user-facing q vector contains one active finger joint; the mimic follower
is expanded internally.

Run from the algorithms directory:

    python examples/fk/03_robot_with_actuated_gripper.py
"""
from __future__ import annotations
from pathlib import Path
import numpy as np

# RobotSystemDescription parses the robot + gripper YAML.
from algorithms.descriptions import RobotSystemDescription

# fk_local computes poses in the robot base frame.
from algorithms.kinematics import fk_local

# KinematicModel composes the robot URDF and gripper URDF.
from algorithms.resolved import KinematicModel


REPO_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    # 1. Load a robot-system YAML that includes a gripper.
    # This description references both the FR3 URDF and Franka hand URDF.
    system_path = (
        REPO_ROOT / "configs" / "robots" / "franka_fr3_with_franka_hand.yaml"
    )

    # Parse YAML into a typed description object.
    system = RobotSystemDescription.from_yaml(system_path)

    # 2. Build the composed robot + gripper model.
    # The gripper is mounted onto the robot using the YAML gripper.mount transform.
    model = KinematicModel.from_robot_system(system)

    # 3. Build q in active-joint order.
    # The home pose is a dict keyed by joint name.
    home = system.named_joint_state("home")

    # The active joint order excludes mimic followers.
    q = np.array([home[name] for name in model.active_joint_names], dtype=float)

    # 4. Change the gripper opening by editing the active finger driver.
    # fr3_finger_joint2 is a mimic follower, so the user does not set it.
    finger_index = model.active_joint_names.index("fr3_finger_joint1")

    # Set the active finger joint to 3 cm opening.
    q[finger_index] = 0.03

    # 5. Query gripper frames.
    # fr3_hand is the gripper root frame.
    T_base_hand = fk_local(model, q, "fr3_hand")

    # Finger links move because q contains the active finger driver.
    T_base_left_finger = fk_local(model, q, "fr3_leftfinger")
    T_base_right_finger = fk_local(model, q, "fr3_rightfinger")

    # fr3_hand_tcp is the end-effector TCP frame.
    T_base_tcp = fk_local(model, q, "fr3_hand_tcp")

    # Show the q order so the user knows where the finger joint lives.
    print("active joint order:")
    print(model.active_joint_names)
    print()

    # Print the gripper root pose.
    print("base -> fr3_hand:")
    print(T_base_hand.round(4))
    print()

    # Print the left finger pose.
    print("base -> fr3_leftfinger:")
    print(T_base_left_finger.round(4))
    print()

    # Print the right finger pose.
    print("base -> fr3_rightfinger:")
    print(T_base_right_finger.round(4))
    print()

    # Print the TCP pose.
    print("base -> fr3_hand_tcp:")
    print(T_base_tcp.round(4))


if __name__ == "__main__":
    main()
