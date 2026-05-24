"""
Show how mimic joints affect Jacobian columns.

Run from the algorithms directory:

    python examples/jacobian/03_jacobian_mimic_column_folding.py
"""
from __future__ import annotations
from pathlib import Path
import numpy as np

# RobotSystemDescription parses the robot + actuated hand YAML.
from algorithms.descriptions import RobotSystemDescription

# jacobian returns columns in active-joint order.
from algorithms.kinematics import jacobian

# KinematicModel resolves mimic joints into active and full joint maps.
from algorithms.resolved import KinematicModel


REPO_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    # 1. Load the Franka arm with the actuated Franka hand.
    system_path = (
        REPO_ROOT / "configs" / "robots" / "franka_fr3_with_franka_hand.yaml"
    )

    # Parse YAML into the robot-system description.
    system = RobotSystemDescription.from_yaml(system_path)

    # Resolve the robot + gripper URDF into one kinematic model.
    model = KinematicModel.from_robot_system(system)

    # 2. Build q in active-joint order.
    # The active vector has one gripper driver joint, not both finger joints.
    home = system.named_joint_state("home")
    q = np.array([home[name] for name in model.active_joint_names], dtype=float)

    # 3. Compute a TCP Jacobian.
    J = jacobian(model, q, "fr3_hand_tcp")

    # Pinocchio still knows the full model DOF.
    print(f"full model nq from Pinocchio: {model.pin_model.nq}")

    # full_joint_names includes mimic followers such as fr3_finger_joint2.
    print("full joint names:")
    print(model.full_joint_names)
    print()

    # active_joint_names is the user-facing q order.
    print("active joint names:")
    print(model.active_joint_names)
    print()

    # The Jacobian uses active columns, so FR3+hand is 8 columns, not 9.
    print(f"Jacobian shape: {J.shape}")
    print("fr3_finger_joint2 is folded into the driver column:")
    print("fr3_finger_joint2" not in model.active_joint_names)


if __name__ == "__main__":
    main()
