"""Quick perf pass over the trajectory layer on FR3.

Run from the algorithms/ directory:

    python scripts/bench_trajectory.py
"""
from __future__ import annotations

import statistics
import time
from pathlib import Path

import numpy as np

from algorithms.descriptions import WorldDescription
from algorithms.optimization import shortcut_smooth, spline_fit
from algorithms.planning import plan_joint
from algorithms.resolved import CollisionModel, KinematicModel, Scene
from algorithms.trajectory import (
    TimeParameterizationOptions,
    TrajectoryStatus,
    TrajectoryValidationOptions,
    time_parameterize,
    validate_trajectory,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
N_TRIALS = 20
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


def random_planned_path(scene, model, q_home):
    lower, upper = model.active_position_limits()
    span = upper - lower
    margin = np.minimum(0.1, 0.25 * span)
    q_goal = RNG.uniform(lower + margin, upper - margin)
    planned = plan_joint(model, scene, q_home, q_goal)
    if planned.path is None:
        return None
    smoothed, _ = shortcut_smooth(planned.path, model, scene, iterations=100)
    return spline_fit(smoothed, samples=40)


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
    if not times_ms:
        print(f"  {label:50s}  no samples")
        return
    times_ms.sort()
    median = statistics.median(times_ms)
    p95 = times_ms[max(0, int(N_TRIALS * 0.95) - 1)]
    p99 = times_ms[max(0, int(N_TRIALS * 0.99) - 1)]
    print(
        f"  {label:50s}  N={N_TRIALS}  success={successes}/{N_TRIALS}  "
        f"median={median:7.2f} ms  p95={p95:7.2f} ms  p99={p99:7.2f} ms"
    )


def main():
    scene, model, q_home = setup()
    print("# Trajectory perf, FR3, default options unless noted")
    print()

    print("time_parameterize (polynomial backend, default dt=0.001):")
    def poly():
        path = random_planned_path(scene, model, q_home)
        if path is None:
            return False
        return time_parameterize(
            path, model,
            options=TimeParameterizationOptions(backend="polynomial", dt=0.001),
        ).status is TrajectoryStatus.SUCCESS
    bench("time_parameterize(backend='polynomial', dt=0.001)", poly)

    print()
    print("time_parameterize (ruckig backend, dt=0.001):")
    def ruck():
        path = random_planned_path(scene, model, q_home)
        if path is None:
            return False
        return time_parameterize(
            path, model,
            options=TimeParameterizationOptions(backend="ruckig", dt=0.001),
        ).status is TrajectoryStatus.SUCCESS
    bench("time_parameterize(backend='ruckig', dt=0.001)", ruck)

    print()
    print("validate_trajectory (full check at dt=0.01):")
    path = random_planned_path(scene, model, q_home)
    if path is not None:
        traj_result = time_parameterize(
            path, model,
            options=TimeParameterizationOptions(backend="polynomial", dt=0.001),
        )
        traj = traj_result.trajectory
        def val():
            return validate_trajectory(
                traj, model, scene,
                options=TrajectoryValidationOptions(check_collision=True, validation_dt=0.01),
            ).passed
        bench("validate_trajectory(... check_collision=True)", val)


if __name__ == "__main__":
    main()
