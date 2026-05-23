"""
Drake-style modular IK problem.

Use `solve_problem` instead of `ik_local` when you need to compose
custom constraints or costs beyond the default pose-IK setup. The
default `ik_local` call already builds this same problem internally
with PoseTarget + JointPositionBounds + SeedRegularization +
JointCenteringCost; this example shows how to do it by hand so you can
add tool-axis, minimum-distance, RCM, or custom costs later.

Run from the algorithms directory:

    python examples/ik/05_with_constraints.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from algorithms.descriptions import RobotSystemDescription
from algorithms.kinematics import fk_local
from algorithms.kinematics.ik import (
    IKOptions,
    IKProblem,
    JointCenteringCost,
    JointPositionBounds,
    PoseTarget,
    SeedRegularization,
    solve_problem,
)
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

    # Build the problem incrementally.
    problem = IKProblem()
    problem.add_task(PoseTarget("robot_tcp", T_target))

    q_min, q_max = model.active_position_limits()
    problem.add_constraint(JointPositionBounds(q_min, q_max, margin=1e-3))

    # Soft costs influence ranking/optimisation only. They never replace
    # the hard validation checks that run after the solve.
    problem.add_cost(SeedRegularization(q_home, weight=1e-6))
    problem.add_cost(JointCenteringCost(weight=1e-7))

    result = solve_problem(model, problem, q_home, options=IKOptions(reject_singular=False))

    print(f"status : {result.status.name}")
    print(f"backend: {result.backend_used}")
    print(f"pos_err: {result.pose_error[0]:.2e}")
    print(f"q      : {result.q}")


if __name__ == "__main__":
    main()
