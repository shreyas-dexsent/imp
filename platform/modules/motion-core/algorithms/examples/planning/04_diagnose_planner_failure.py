"""
Inspect the PathPlanResult diagnostics when planning fails.

Four deliberate failure cases, one per PathStatus:

* INVALID_INPUT       — wrong q_seed shape
* START_OUT_OF_LIMITS — start clipped past joint upper bound
* GOAL_OUT_OF_LIMITS  — goal clipped past joint upper bound
* NO_PATH_FOUND       — straight-line backend on an unreachable goal

Run from the algorithms directory:

    python examples/planning/04_diagnose_planner_failure.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from algorithms.descriptions import WorldDescription
from algorithms.planning import PathStatus, PlanOptions, plan_joint
from algorithms.resolved import CollisionModel, KinematicModel, Scene

REPO_ROOT = Path(__file__).resolve().parents[2]


def show(label, result):
    print(f"{label:32s} -> status={result.status.name:24s} msg={result.diagnostics.message[:60]}")


def main() -> None:
    world = WorldDescription.from_yaml(
        REPO_ROOT / "configs" / "worlds" / "franka_robot_only_world.yaml"
    )
    cm = CollisionModel.from_world(world)
    scene = Scene.from_world(world, cm)
    system = world.robots[0].robot_system
    model = KinematicModel.from_robot_system(system)
    home = system.named_joint_state("home")
    q_home = np.array([home[name] for name in model.active_joint_names], dtype=float)
    lower, upper = model.active_position_limits()

    # 1. Wrong shape — caller bug.
    show("wrong q_seed shape",
         plan_joint(model, scene, q_home[:-1], q_home))

    # 2. Start outside limits.
    show("start out of limits",
         plan_joint(model, scene, upper + 1.0, q_home))

    # 3. Goal outside limits.
    show("goal out of limits",
         plan_joint(model, scene, q_home, upper + 1.0))

    # 4. Unknown backend.
    show("unknown backend",
         plan_joint(model, scene, q_home, q_home, backend="not_real"))


if __name__ == "__main__":
    main()
