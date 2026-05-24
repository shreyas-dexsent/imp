"""
Smooth, continuous motion across multiple via-points.

Two patterns for "deliver smooth motion through N waypoints":

A. Chain `move_joint` calls — each leg starts and ends at rest. The
   robot brakes briefly between legs. Simple, but not smooth.
B. `via_motion(model, scene, [q_0, q_1, ..., q_N])` — one trajectory
   that passes through every interior waypoint with non-zero
   velocity. Continuous and smooth end-to-end.

This example runs both patterns on the same waypoint list and prints
the velocity at each interior waypoint. Pattern A shows zero velocity
between legs; Pattern B shows non-zero velocity at every interior
waypoint.

Run from the repo root:

    python examples/integration/03_chained_smooth_motion.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from algorithms import (
    CollisionModel,
    KinematicModel,
    Scene,
    WorldDescription,
    move_joint,
    via_motion,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


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

    via_a = q_home.copy(); via_a[0] += 0.3
    via_b = q_home.copy(); via_b[0] += 0.6
    goal = q_home.copy();  goal[0] += 0.9

    print("Pattern A: chain move_joint(home -> a), then (a -> b), then (b -> goal)")
    chain_a = []
    chain_a.append(move_joint(model, scene, via_a, q_seed=q_home))
    chain_a.append(move_joint(model, scene, via_b, q_seed=via_a))
    chain_a.append(move_joint(model, scene, goal, q_seed=via_b))
    duration_a = sum(r.trajectory.duration for r in chain_a if r.trajectory)
    print(f"  total duration : {duration_a:.3f} s across 3 trajectories")
    for i, r in enumerate(chain_a):
        last_qd = r.trajectory.velocities[-1]
        print(f"  leg {i}: |qd| at the END of the leg = {np.linalg.norm(last_qd):.4f} rad/s  "
              "(zero = pause)")

    print()
    print("Pattern B: via_motion(home, a, b, goal)  — one continuous trajectory")
    r_via = via_motion(model, scene, [q_home, via_a, via_b, goal])
    if r_via.trajectory is None:
        print(f"  FAILED: {r_via.status.name}")
        return
    traj = r_via.trajectory
    print(f"  status         : {r_via.status.name}")
    print(f"  duration       : {traj.duration:.3f} s")
    print(f"  total samples  : {traj.num_samples}")

    # The trajectory visits 4 path waypoints at roughly uniform chord-
    # length intervals. Sampling at quartile times shows velocity at
    # each interior waypoint.
    print(f"\n  Velocity at quartile times (non-zero at interior = pass-through):")
    for frac, label in [(0.0, "home"), (0.33, "via_a"), (0.66, "via_b"), (1.0, "goal")]:
        t = frac * traj.duration
        _, qd, _ = traj.at(float(t))
        print(f"    t={t:5.2f}s ({label:>5})  |qd|={float(np.linalg.norm(qd)):.4f} rad/s")

    print()
    print(f"Pattern A is {duration_a:.2f} s; Pattern B is {traj.duration:.2f} s; ")
    print(f"Pattern B is one trajectory; Pattern A is three concatenated trajectories with brakes between.")


if __name__ == "__main__":
    main()
