"""
Compare singularity metrics for two robot configurations.

Run from the algorithms directory:

    python examples/singularity/01_singularity_metrics.py
"""
from __future__ import annotations
from pathlib import Path
import numpy as np

# RobotSystemDescription parses the robot + gripper YAML.
from algorithms.descriptions import RobotSystemDescription

# singularity_report summarizes one Jacobian using one SVD.
from algorithms.kinematics import jacobian, singularity_report

# KinematicModel resolves YAML + URDF into a reusable Pinocchio model.
from algorithms.resolved import KinematicModel


REPO_ROOT = Path(__file__).resolve().parents[2]


def print_singularity_report(
    *,
    model: KinematicModel,
    q: np.ndarray,
    frame_id: str,
    label: str,
) -> None:
    # Compute the frame Jacobian for this q.
    J = jacobian(model, q, frame_id)

    # Compute all common singularity metrics from one SVD.
    metrics = singularity_report(J)

    # Print a small labeled block so two configurations are easy to compare.
    print(label)
    print("-" * len(label))

    # Print every metric by name.
    for name, value in metrics.items():
        print(f"{name}: {value}")

    # Add a blank line between reports.
    print()


def main() -> None:
    # 1. Load and resolve the robot + gripper system.
    system_path = (
        REPO_ROOT / "configs" / "robots" / "franka_fr3_with_franka_hand.yaml"
    )

    # Parse the YAML description.
    system = RobotSystemDescription.from_yaml(system_path)

    # Build the resolved kinematic model once.
    model = KinematicModel.from_robot_system(system)

    # 2. Build q in active-joint order.
    # Read the home joint state from YAML.
    home = system.named_joint_state("home")

    # Convert the named joint-state dict to model.active_joint_names order.
    q_home = np.array([home[name] for name in model.active_joint_names], dtype=float)

    # Use the zero posture as an intentionally poor conditioning example.
    # For this FR3 model, the TCP Jacobian loses rank at this posture.
    q_near_singular = np.zeros_like(q_home)

    # Keep the gripper opening comparable between both examples.
    finger_joint = "fr3_finger_joint1"
    if finger_joint in model.active_joint_names:
        finger_index = model.active_joint_names.index(finger_joint)
        q_near_singular[finger_index] = q_home[finger_index]

    # 3. Compare the TCP singularity metrics at both configurations.
    frame_id = "fr3_hand_tcp"

    print_singularity_report(
        model=model,
        q=q_home,
        frame_id=frame_id,
        label="Home configuration",
    )

    print_singularity_report(
        model=model,
        q=q_near_singular,
        frame_id=frame_id,
        label="Near-singular zero-arm configuration",
    )


if __name__ == "__main__":
    main()
