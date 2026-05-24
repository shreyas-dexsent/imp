"""
Batch forward kinematics for many frames.

`fk_local_many` runs one Pinocchio FK pass and returns poses for several
frames. Use this when a planner, collision checker, or UI needs many frames
from the same q.

Run from the algorithms directory:

    python examples/fk/06_batch_fk_many_frames.py
"""
from __future__ import annotations
from pathlib import Path
import numpy as np

# RobotSystemDescription parses the robot + gripper YAML.
from algorithms.descriptions import RobotSystemDescription

# fk_local_many computes several frame poses using one FK pass.
from algorithms.kinematics import fk_local_many

# KinematicModel resolves YAML + URDF into a reusable model.
from algorithms.resolved import KinematicModel


REPO_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    # 1. Load and resolve a robot-system YAML.
    system_path = (
        REPO_ROOT / "configs" / "robots" / "franka_fr3_with_franka_hand.yaml"
    )

    # Parse the YAML.
    system = RobotSystemDescription.from_yaml(system_path)

    # Build or fetch the cached KinematicModel.
    model = KinematicModel.from_robot_system(system)

    # 2. Build q in active-joint order.
    # Named states are joint-name dictionaries.
    home = system.named_joint_state("home")

    # Algorithms expect the model's active NumPy q order.
    q = np.array([home[name] for name in model.active_joint_names], dtype=float)

    # 3. Ask for multiple frames in one FK pass.
    # This is more efficient than calling fk_local once per frame.
    frame_ids = ["fr3_link4", "fr3_link8", "fr3_hand", "fr3_hand_tcp"]

    # The result is a dict: frame_id -> 4x4 transform.
    transforms = fk_local_many(model, q, frame_ids)

    # Print every requested frame transform.
    for frame_id, transform in transforms.items():
        print(f"base -> {frame_id}:")
        print(transform.round(4))
        print()


if __name__ == "__main__":
    main()
