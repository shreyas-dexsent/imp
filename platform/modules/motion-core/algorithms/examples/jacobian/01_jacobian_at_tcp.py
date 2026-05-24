"""
Jacobian at a TCP frame.

Run from the algorithms directory:

    python examples/jacobian/01_jacobian_at_tcp.py
"""
from __future__ import annotations
from pathlib import Path
import numpy as np

# RobotSystemDescription parses the robot + gripper YAML.
from algorithms.descriptions import RobotSystemDescription

# jacobian computes a 6 x active_dof geometric Jacobian.
from algorithms.kinematics import jacobian

# KinematicModel resolves YAML + URDF into a reusable Pinocchio model.
from algorithms.resolved import KinematicModel


REPO_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    # 1. Load and resolve the robot + gripper system.
    system_path = (
        REPO_ROOT / "configs" / "robots" / "franka_fr3_with_franka_hand.yaml"
    )

    # Parse the robot-system YAML.
    system = RobotSystemDescription.from_yaml(system_path)

    # Build the resolved kinematic model.
    model = KinematicModel.from_robot_system(system)

    # 2. Build q in active-joint order.
    # Read the YAML named state.
    home = system.named_joint_state("home")

    # Convert joint-name dict to NumPy q.
    q = np.array([home[name] for name in model.active_joint_names], dtype=float)

    # 3. Compute the geometric Jacobian at the TCP.
    # Columns are in model.active_joint_names order.
    J = jacobian(model, q, "fr3_hand_tcp")

    # Print the column order first, otherwise the matrix is ambiguous.
    print("active joint order:")
    print(model.active_joint_names)
    print()

    # Print the Jacobian shape and values.
    print(f"Jacobian shape: {J.shape}")
    print(J.round(4))


if __name__ == "__main__":
    main()
