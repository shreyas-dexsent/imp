"""
Force the analytical OPW backend.

The shipped FR3 description does not declare OPW parameters, so this
example shows the structured "unsupported backend" response. See
07_register_analytical.py for how to wire up a robot-specific
analytical solver.

Run from the algorithms directory:

    python examples/ik/03_analytical_opw.py
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
    T_target = fk_local(model, q_home, "robot_tcp")

    result = ik_local(model, "robot_tcp", T_target, q_home, backend="opw")

    print(f"status : {result.status.name}")
    print(f"backend: {result.backend_used}")
    print(f"message: {result.diagnostics.message}")


if __name__ == "__main__":
    main()
