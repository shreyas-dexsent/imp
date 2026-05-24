"""
Forward kinematics for a YAML-defined TCP.

The `robot_tcp` frame is declared in the robot YAML. It does not need to be
present in the manufacturer URDF; KinematicModel injects it as a fixed frame.

Run from the algorithms directory:

    python examples/fk/02_robot_with_static_tcp.py
"""
from __future__ import annotations
from pathlib import Path
import numpy as np

# RobotSystemDescription parses robot-system YAML.
from algorithms.descriptions import RobotSystemDescription

# fk_local returns robot-base-relative frame poses.
from algorithms.kinematics import fk_local

# KinematicModel injects YAML TCP frames into the resolved Pinocchio model.
from algorithms.resolved import KinematicModel


REPO_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    # 1. Load the robot-only system. It declares a TCP offset from fr3_link8.
    system_path = REPO_ROOT / "configs" / "robots" / "franka_fr3_robot_only.yaml"

    # Parse YAML. At this point robot_tcp is just static description data.
    system = RobotSystemDescription.from_yaml(system_path)

    # 2. Build the resolved kinematic model. This injects YAML TCP frames.
    # After this, robot_tcp can be queried like a normal Pinocchio frame.
    model = KinematicModel.from_robot_system(system)

    # 3. Build q in model.active_joint_names order.
    # The named state is stored as a dict keyed by joint name.
    home = system.named_joint_state("home")

    # Convert the dict into the exact NumPy order expected by the model.
    q = np.array([home[name] for name in model.active_joint_names], dtype=float)

    # 4. Query both the parent link and the YAML TCP.
    # fr3_link8 comes from the URDF.
    T_base_link8 = fk_local(model, q, "fr3_link8")

    # robot_tcp comes from YAML and was injected into the model.
    T_base_tcp = fk_local(model, q, "robot_tcp")

    # 5. Verify the relative TCP offset from YAML.
    # base->link8 inverted and composed with base->tcp gives link8->tcp.
    T_link8_tcp = np.linalg.inv(T_base_link8) @ T_base_tcp

    # Print the TCP pose in the robot base frame.
    print("base -> robot_tcp:")
    print(T_base_tcp.round(4))
    print()

    # Print the local TCP offset relative to its parent link.
    print("fr3_link8 -> robot_tcp:")
    print(T_link8_tcp.round(4))


if __name__ == "__main__":
    main()
