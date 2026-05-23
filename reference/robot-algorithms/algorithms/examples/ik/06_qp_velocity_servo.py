"""
Cartesian velocity IK in a servo loop.

ik_velocity returns qdot, not an IKResult. There is no validator and
no failure status: live safety belongs to the controller / runtime
monitor that consumes qdot.

Run from the algorithms directory:

    python examples/ik/06_qp_velocity_servo.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from algorithms.descriptions import RobotSystemDescription
from algorithms.kinematics import ik_velocity
from algorithms.resolved import KinematicModel

REPO_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    system = RobotSystemDescription.from_yaml(
        REPO_ROOT / "configs" / "robots" / "franka_fr3_robot_only.yaml"
    )
    model = KinematicModel.from_robot_system(system)

    home = system.named_joint_state("home")
    q = np.array([home[name] for name in model.active_joint_names], dtype=float)

    # Twist convention: [vx, vy, vz, wx, wy, wz]. 1 cm/s in +x at the TCP.
    target_twist = np.array([0.01, 0.0, 0.0, 0.0, 0.0, 0.0])
    dt = 0.01

    # 200 steps at 100 Hz simulates a 2-second servo move.
    for _ in range(200):
        qdot = ik_velocity(model, "robot_tcp", target_twist, q, dt=dt)
        q = q + qdot * dt

    print("final q after 200 steps of x-velocity servoing:")
    print(q)


if __name__ == "__main__":
    main()
