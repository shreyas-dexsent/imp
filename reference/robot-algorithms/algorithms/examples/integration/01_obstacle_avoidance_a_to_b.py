"""
End-to-end: plan + smooth + parameterize a collision-free trajectory
that goes around an obstacle, then verify every sample is collision-free.

The scenario:
* FR3 at home.
* A box obstacle is added to the scene between the robot's current
  reach and the goal.
* `move_joint` produces one smooth, validated trajectory from the
  home configuration to a joint goal whose direct line would
  collide.

This is the answer to "can I get a collision-free trajectory if I
just give start, goal, and an obstacle in between?" — yes, via
`move_joint` (which composes plan_joint + shortcut_smooth +
spline_fit + time_parameterize + validators).

Run from the repo root:

    python examples/integration/01_obstacle_avoidance_a_to_b.py
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
)
from algorithms.collision import is_in_collision
from algorithms.descriptions import BoxGeometrySpec

REPO_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    # 1. Load a no-obstacle world and add a box at runtime via perception API.
    world = WorldDescription.from_yaml(
        REPO_ROOT / "configs" / "worlds" / "franka_robot_only_world.yaml"
    )
    cm = CollisionModel.from_world(world)
    scene = Scene.from_world(world, cm)
    system = world.robots[0].robot_system
    model = KinematicModel.from_robot_system(system)
    home = system.named_joint_state("home")
    q_home = np.array([home[name] for name in model.active_joint_names], dtype=float)

    T_obstacle = np.eye(4)
    T_obstacle[:3, 3] = [0.40, 0.0, 0.40]
    scene.add_object(
        "obstacle_box",
        collision=BoxGeometrySpec(type="box", size=(0.10, 0.30, 0.30)),
        pose=T_obstacle,
    )
    print(f"obstacle placed at world {T_obstacle[:3, 3]}")

    # 2. Pick a joint goal that requires going around (rotate base 90 deg).
    q_goal = q_home.copy()
    q_goal[0] = 1.57

    # 3. Solve in one call. move_joint handles plan + smooth + spline +
    #    time_parameterize + validate_path + validate_trajectory.
    result = move_joint(model, scene, q_goal, q_seed=q_home)
    print(f"status     : {result.status.name}")
    print(f"primitive  : {result.primitive_used}")
    print(f"elapsed    : {result.elapsed_ms:.1f} ms")
    if result.trajectory is None:
        print("FAILED — see diagnostics:")
        print(f"  stage  : {result.diagnostics.stage}")
        print(f"  message: {result.diagnostics.message}")
        return
    print(f"duration   : {result.trajectory.duration:.3f} s")
    print(f"samples    : {result.trajectory.num_samples}  (dt = 1 ms)")
    print(f"backend    : {result.trajectory.backend_used}")

    # 4. Verify no sample collides, end to end. The validator already
    #    did this, but doing it again here proves the controller would
    #    not hit the box if it streamed the trajectory.
    qs = result.trajectory.positions
    colliding = sum(
        1 for q in qs if is_in_collision(model, scene, q).in_collision
    )
    print(f"colliding samples: {colliding} / {len(qs)}")


if __name__ == "__main__":
    main()
