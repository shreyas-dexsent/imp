"""
Robot-only forward kinematics.

Run from the algorithms directory:

    python examples/fk/01_robot_only_fk.py
"""
from __future__ import annotations
from pathlib import Path
import numpy as np

# RobotSystemDescription parses the robot YAML.
from algorithms.descriptions import RobotSystemDescription

# fk_local computes frame poses relative to the robot base frame.
from algorithms.kinematics import fk_local

# KinematicModel resolves YAML + URDF into a reusable Pinocchio model.
from algorithms.resolved import KinematicModel


REPO_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    # 1. Load the robot description YAML.
    # This YAML points to the manufacturer FR3 URDF and declares chains/TCPs.
    system_path = REPO_ROOT / "configs" / "robots" / "franka_fr3_robot_only.yaml"

    # Parse YAML into a typed Pydantic object. No URDF is loaded yet.
    system = RobotSystemDescription.from_yaml(system_path)

    # 2. Resolve the YAML + URDF into a Pinocchio-backed kinematic model.
    # This is the heavy step. Do this once and reuse the model.
    model = KinematicModel.from_robot_system(system)

    # 3. Convert a named joint state dict into the active NumPy q order.
    # YAML stores named states as {joint_name: value}.
    home = system.named_joint_state("home")

    # Algorithms expect q in model.active_joint_names order.
    q = np.array([home[name] for name in model.active_joint_names], dtype=float)

    # 4. Compute FK for a resolved model frame, expressed in the robot base frame.
    # The frame may come from URDF or from YAML TCP injection.
    T_base_link8 = fk_local(model, q, "fr3_link8")

    # Show the q ordering so the printed transform is easy to reproduce.
    print("active joint order:")
    print(model.active_joint_names)
    print()

    # Print the 4x4 homogeneous transform from base to fr3_link8.
    print("base -> fr3_link8:")
    print(T_base_link8.round(4))


if __name__ == "__main__":
    main()
