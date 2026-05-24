"""
Register a robot-specific analytical IK backend.

The analytical-IK registry maps a robot id (the YAML `id:` field) to a
solver class. When the dispatch sees a registered solver for the
current robot, it picks the analytical backend automatically and skips
the generic NLS path. Common production cases:

* An OPW arm shipping with known a1/a2/b/c1..c4 parameters.
* A spherical-wrist 6R with a vendor-supplied closed-form solution.
* Any custom 6R/7R for which you have written branch enumeration.

This example registers a *toy* solver that always returns the seed.
The point is to show the registration mechanics, not to be a useful
solver. Replace the body of `solve_branches` with your closed-form
solution.

Run from the algorithms directory:

    python examples/ik/07_register_analytical.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from algorithms.descriptions import RobotSystemDescription
from algorithms.kinematics import fk_local, ik_local
from algorithms.kinematics.ik import register_analytical
from algorithms.kinematics.ik.backends.analytical.registry import clear as clear_registry
from algorithms.resolved import KinematicModel

REPO_ROOT = Path(__file__).resolve().parents[2]


class IdentitySeedAnalyticalIK:
    """Trivial analytical solver: it returns the seed as the only branch.

    A real implementation would compute the closed-form q for the
    target pose. The contract is: take `(model, spec, q_seed)`, return
    a tuple of candidate q arrays. The framework validates every
    branch against pose tolerance, joint limits, singularity, and
    collision before any one is returned as the solution.
    """

    name = "identity_seed"

    def solve_branches(self, model, spec, q_seed):
        return (np.asarray(q_seed, dtype=float),)


def main() -> None:
    system = RobotSystemDescription.from_yaml(
        REPO_ROOT / "configs" / "robots" / "franka_fr3_robot_only.yaml"
    )
    model = KinematicModel.from_robot_system(system)
    home = system.named_joint_state("home")
    q_home = np.array([home[name] for name in model.active_joint_names], dtype=float)
    T_target = fk_local(model, q_home, "robot_tcp")

    print(f"Robot id: {system.robot.id!r}")

    # Without registration: dispatch falls back to the generic NLS backend.
    no_reg = ik_local(model, "robot_tcp", T_target, q_home)
    print(f"  no registration  -> backend={no_reg.backend_used:8s}  status={no_reg.status.name}")

    # Register the toy analytical solver for this robot id.
    register_analytical(system.robot.id, IdentitySeedAnalyticalIK)
    try:
        with_reg = ik_local(model, "robot_tcp", T_target, q_home)
        print(f"  registered toy   -> backend={with_reg.backend_used:8s}  status={with_reg.status.name}")
    finally:
        clear_registry()

    # After the registry is cleared, dispatch reverts to generic.
    after_clear = ik_local(model, "robot_tcp", T_target, q_home)
    print(f"  after clear      -> backend={after_clear.backend_used:8s}  status={after_clear.status.name}")


if __name__ == "__main__":
    main()
