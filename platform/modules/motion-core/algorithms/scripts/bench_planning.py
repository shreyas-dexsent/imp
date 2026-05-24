"""Quick perf pass over plan_joint + plan_cartesian on FR3.

Run from the algorithms/ directory:

    python scripts/bench_planning.py
"""
from __future__ import annotations

import statistics
import time
from pathlib import Path

import numpy as np

from algorithms.descriptions import WorldDescription
from algorithms.kinematics import fk_local
from algorithms.planning import PathStatus, plan_cartesian, plan_joint
from algorithms.resolved import CollisionModel, KinematicModel, Scene

REPO_ROOT = Path(__file__).resolve().parents[1]
N_TRIALS = 50
RNG = np.random.default_rng(0)


def setup():
    world = WorldDescription.from_yaml(
        REPO_ROOT / "configs" / "worlds" / "franka_robot_only_world.yaml"
    )
    cm = CollisionModel.from_world(world)
    scene = Scene.from_world(world, cm)
    system = world.robots[0].robot_system
    model = KinematicModel.from_robot_system(system)
    home = system.named_joint_state("home")
    q_home = np.array([home[name] for name in model.active_joint_names], dtype=float)
    return scene, model, q_home


def bench(label, fn):
    times_ms = []
    successes = 0
    for _ in range(N_TRIALS):
        t0 = time.perf_counter()
        ok = fn()
        t1 = time.perf_counter()
        times_ms.append((t1 - t0) * 1000.0)
        if ok:
            successes += 1
    times_ms.sort()
    median = statistics.median(times_ms)
    p95 = times_ms[int(N_TRIALS * 0.95) - 1]
    p99 = times_ms[int(N_TRIALS * 0.99) - 1]
    print(
        f"  {label:50s}  N={N_TRIALS}  success={successes}/{N_TRIALS}  "
        f"median={median:7.2f} ms  p95={p95:7.2f} ms  p99={p99:7.2f} ms"
    )


def random_reachable_goal(model, q_home):
    lower, upper = model.active_position_limits()
    span = upper - lower
    margin = np.minimum(0.1, 0.25 * span)
    return RNG.uniform(lower + margin, upper - margin)


def main():
    scene, model, q_home = setup()
    print("# Planning perf, FR3, default options")
    print()

    print("plan_joint (OMPL RRTConnect):")
    def ompl():
        q_goal = random_reachable_goal(model, q_home)
        return plan_joint(model, scene, q_home, q_goal).status is PathStatus.SUCCESS
    bench("plan_joint(... backend='ompl')", ompl)

    print()
    print("plan_joint (direct straight-line):")
    def direct():
        q_goal = q_home.copy()
        q_goal[0] += RNG.uniform(-0.5, 0.5)
        return plan_joint(
            model, scene, q_home, q_goal, backend="direct"
        ).status is PathStatus.SUCCESS
    bench("plan_joint(... backend='direct')", direct)

    print()
    print("plan_cartesian (10 cm straight line):")
    def cart():
        T_start = fk_local(model, q_home, "robot_tcp")
        T_goal = T_start.copy()
        T_goal[0, 3] += 0.10
        return plan_cartesian(scene, "arm", "robot_tcp", T_start, T_goal, q_home).status is PathStatus.SUCCESS
    bench("plan_cartesian(... +10cm in x)", cart)


if __name__ == "__main__":
    main()
