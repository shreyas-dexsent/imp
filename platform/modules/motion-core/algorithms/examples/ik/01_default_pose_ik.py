"""
Default pose IK with the ergonomic top-level entrypoint.

Run from the algorithms directory:

    python examples/ik/01_default_pose_ik.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from algorithms.descriptions import RobotSystemDescription
from algorithms.kinematics import fk_local, ik_local
from algorithms.resolved import KinematicModel

REPO_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    system = RobotSystemDescription.from_yaml(
        REPO_ROOT / "configs" / "robots" / "franka_fr3_robot_only.yaml"
    )
    model = KinematicModel.from_robot_system(system)

    home = system.named_joint_state("home")
    q_home = np.array([home[name] for name in model.active_joint_names], dtype=float)

    # Pick a reachable target. In production this comes from perception,
    # planning, or the user. Here we offset the home TCP by 5 cm in x.
    T_target = fk_local(model, q_home, "robot_tcp").copy()
    T_target[0, 3] += 0.05

    # One call. q_seed is required; q_home is the typical default.
    result = ik_local(model, "robot_tcp", T_target, q_seed=q_home)

    print("status :", result.status.name)
    print("backend:", result.backend_used)
    print("pos_err:", result.pose_error[0])
    print("rot_err:", result.pose_error[1])
    print("q      :", result.q)


if __name__ == "__main__":
    main()
