"""Quick performance pass over the IK backends on FR3.

Run from the algorithms/ directory:

    python scripts/bench_ik.py
"""
from __future__ import annotations

import statistics
import time
from pathlib import Path

import numpy as np

from algorithms.descriptions import RobotSystemDescription, WorldDescription
from algorithms.kinematics import fk_local
from algorithms.kinematics import ik_local, ik_velocity
from algorithms.kinematics.ik import IKOptions
from algorithms.resolved import KinematicModel, Scene

REPO_ROOT = Path(__file__).resolve().parents[1]
N_TRIALS = 100
RNG = np.random.default_rng(0)


def load_fr3():
    system = RobotSystemDescription.from_yaml(
        REPO_ROOT / "configs" / "robots" / "franka_fr3_robot_only.yaml"
    )
    model = KinematicModel.from_robot_system(system)
    home = system.named_joint_state("home")
    q_home = np.array([home[name] for name in model.active_joint_names], dtype=float)
    return system, model, q_home


def jitter(q_home, scale=0.2):
    q_min, q_max = (
        np.asarray(KinematicModel.from_robot_system.__self__ if False else None),
        None,
    )
    # Simpler: just jitter and clip later.
    return q_home + RNG.uniform(-scale, scale, size=q_home.shape)


def random_reachable_target(model, q_home, frame_id):
    q_min, q_max = model.active_position_limits()
    span = q_max - q_min
    margin = np.minimum(0.1, 0.25 * span)
    q_rand = RNG.uniform(q_min + margin, q_max - margin)
    return fk_local(model, q_rand, frame_id), q_rand


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
        f"  {label:40s}  N={N_TRIALS}  success={successes}/{N_TRIALS}  "
        f"median={median:7.3f} ms  p95={p95:7.3f} ms  p99={p99:7.3f} ms"
    )


def main():
    system, model, q_home = load_fr3()

    print("# IK perf, FR3, default options unless noted")
    print()

    # 1. Generic solve, single seed = q_home
    print("Generic constrained IK (default options):")

    def single():
        T, _ = random_reachable_target(model, q_home, "robot_tcp")
        result = ik_local(model, "robot_tcp", T, q_home)
        return result.status.name == "SUCCESS"

    bench("ik_local(model, frame, T, q_home)", single)

    # 2. Generic, multi_start disabled (single-seed only)
    print()
    print("Generic constrained IK (multi_start=False):")
    opts_single = IKOptions(multi_start=False)

    def single_no_multi():
        T, _ = random_reachable_target(model, q_home, "robot_tcp")
        result = ik_local(model, "robot_tcp", T, q_home, options=opts_single)
        return result.status.name == "SUCCESS"

    bench("ik_local(... multi_start=False)", single_no_multi)

    # 3. QP velocity IK
    print()
    print("QP velocity IK:")
    twist = np.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.0])

    def qp():
        qdot = ik_velocity(model, "robot_tcp", twist, q_home, dt=0.01)
        return qdot.shape == q_home.shape

    bench("ik_velocity(... 6D twist)", qp)

    # 4. Two-robot world — single-robot scaling sanity check
    print()
    print("Two-robot world (per-robot solve, base poses differ):")
    world_path = REPO_ROOT / "configs" / "worlds" / "two_franka_table_world.yaml"
    if world_path.exists():
        world = WorldDescription.from_yaml(world_path)
        Scene.from_world(world)
        left_system = world.robot("left_arm").robot_system
        left_model = KinematicModel.from_robot_system(left_system)
        right_system = world.robot("right_arm").robot_system
        right_model = KinematicModel.from_robot_system(right_system)
        left_home_state = left_system.named_joint_state("home")
        left_home = np.array(
            [left_home_state[name] for name in left_model.active_joint_names],
            dtype=float,
        )
        right_home_state = right_system.named_joint_state("home")
        right_home = np.array(
            [right_home_state[name] for name in right_model.active_joint_names],
            dtype=float,
        )

        def two_robot_left():
            T, _ = random_reachable_target(left_model, left_home, "robot_tcp")
            r = ik_local(left_model, "robot_tcp", T, left_home)
            return r.status.name == "SUCCESS"

        def two_robot_right():
            T, _ = random_reachable_target(right_model, right_home, "robot_tcp")
            r = ik_local(right_model, "robot_tcp", T, right_home)
            return r.status.name == "SUCCESS"

        bench("ik_local(left_arm ...)", two_robot_left)
        bench("ik_local(right_arm ...)", two_robot_right)
    else:
        print(f"  (skipped; {world_path} not found)")


if __name__ == "__main__":
    main()
