"""
Compare Jacobian reference-frame conventions.

Run from the algorithms directory:

    python examples/jacobian/02_jacobian_reference_frames.py
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
    # 1. Load the robot + gripper description.
    system_path = (
        REPO_ROOT / "configs" / "robots" / "franka_fr3_with_franka_hand.yaml"
    )

    # Parse the YAML into a typed description.
    system = RobotSystemDescription.from_yaml(system_path)

    # Resolve YAML + URDF into one Pinocchio-backed model.
    model = KinematicModel.from_robot_system(system)

    # 2. Build q in the same active-joint order used by FK.
    home = system.named_joint_state("home")
    q = np.array([home[name] for name in model.active_joint_names], dtype=float)

    # 3. Choose a resolved model frame.
    # This TCP comes from YAML and is injected into the model during resolution.
    frame_id = "fr3_hand_tcp"

    # LOCAL expresses the twist in the queried frame's own axes.
    J_local = jacobian(model, q, frame_id, reference="local")

    # WORLD expresses translation and rotation in the model/world axes.
    J_world = jacobian(model, q, frame_id, reference="world")

    # LOCAL_WORLD_ALIGNED uses the frame origin but world-aligned axes.
    J_lwa = jacobian(model, q, frame_id, reference="local_world_aligned")

    # Print the q/J column order first.
    print("active joint order:")
    print(model.active_joint_names)
    print()

    # Print the three conventions side by side.
    print("LOCAL Jacobian:")
    print(J_local.round(4))
    print()

    print("WORLD Jacobian:")
    print(J_world.round(4))
    print()

    print("LOCAL_WORLD_ALIGNED Jacobian:")
    print(J_lwa.round(4))


if __name__ == "__main__":
    main()
