"""
Diagnosing IK failures.

Every IKResult carries an `IKDiagnostics` payload with one entry per
seed attempted, plus one ValidationReport per candidate that survived
to validation. This script walks through how to read it when something
goes wrong, and shows the standalone `validate()` function for
checking an externally-produced q.

Run from the algorithms directory:

    python examples/ik/09_diagnosing_failures.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from algorithms.descriptions import RobotSystemDescription
from algorithms.kinematics import fk_local, ik_local
from algorithms.kinematics.ik import (
    IKOptions,
    IKProblem,
    JointPositionBounds,
    PoseTarget,
    validate,
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

    # Reach far enough that the default budget can't converge.
    T_target = fk_local(model, q_home, "robot_tcp").copy()
    T_target[0, 3] += 1.5  # outside FR3's reach

    result = ik_local(
        model, "robot_tcp", T_target, q_home,
        options=IKOptions(max_time_ms=80, max_iters=20, num_random_seeds=2),
    )

    print(f"status  : {result.status.name}")
    print(f"backend : {result.backend_used}")
    print(f"pos_err : {result.pose_error[0]}")
    print(f"rot_err : {result.pose_error[1]}")
    print(f"elapsed : {result.elapsed_ms:.1f} ms")
    print()
    print(f"seeds attempted: {len(result.diagnostics.seed_reports)}")
    for i, sr in enumerate(result.diagnostics.seed_reports):
        pe = sr.get("pose_error", (float("inf"), float("inf")))
        print(f"  seed[{i}]: scipy_status={sr.get('status'):>3}  "
              f"pos_err={pe[0]:.2e}  rot_err={pe[1]:.2e}  msg={sr.get('message', '')[:60]}")

    print()
    print(f"backend_statuses recorded: {[s.name for s in result.diagnostics.backend_statuses]}")
    print(f"diagnostic message       : {result.diagnostics.message}")

    # Validator can be used standalone, e.g. on a q produced outside the
    # library. Same acceptance rules as `ik_local`. Useful when wiring
    # in a third-party solver.
    print()
    print("Standalone validate() on q_home against the home pose:")
    home_pose = fk_local(model, q_home, "robot_tcp")
    problem = IKProblem()
    problem.add_task(PoseTarget("robot_tcp", home_pose))
    q_min, q_max = model.active_position_limits()
    problem.add_constraint(JointPositionBounds(q_min, q_max))
    report = validate(model, problem.freeze(), q_home, IKOptions(reject_singular=False))
    print(f"  status : {report.status.name}")
    print(f"  pos_err: {report.pose_error[0]:.2e}")
    print(f"  checks : {[c['name'] for c in report.checks]}")


if __name__ == "__main__":
    main()
