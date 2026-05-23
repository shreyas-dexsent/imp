"""
Multi-start vs single-start IK.

Run from the algorithms directory:

    python examples/ik/02_multi_start.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from algorithms.descriptions import RobotSystemDescription
from algorithms.kinematics import fk_local, ik_local
from algorithms.kinematics.ik import IKOptions
from algorithms.resolved import KinematicModel

REPO_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    system = RobotSystemDescription.from_yaml(
        REPO_ROOT / "configs" / "robots" / "franka_fr3_robot_only.yaml"
    )
    model = KinematicModel.from_robot_system(system)

    home = system.named_joint_state("home")
    q_home = np.array([home[name] for name in model.active_joint_names], dtype=float)

    # Use FK from home as a known reachable target.
    T_target = fk_local(model, q_home, "robot_tcp")

    # Intentionally poor seed near the centre of the joint limits.
    q_min, q_max = model.active_position_limits()
    q_center = 0.5 * (q_min + q_max)

    # One seed only.
    single = ik_local(
        model,
        "robot_tcp",
        T_target,
        q_center,
        options=IKOptions(multi_start=False, max_time_ms=200, reject_singular=False),
    )

    # Deterministic named seeds + 4 bounded random seeds.
    multi = ik_local(
        model,
        "robot_tcp",
        T_target,
        q_center,
        options=IKOptions(
            multi_start=True,
            num_random_seeds=4,
            max_time_ms=500,
            reject_singular=False,
        ),
    )

    print(f"single-start: status={single.status.name}  pos_err={single.pose_error[0]:.2e}")
    print(f"multi-start : status={multi.status.name}  pos_err={multi.pose_error[0]:.2e}  "
          f"candidates={len(multi.candidates)}")


if __name__ == "__main__":
    main()
