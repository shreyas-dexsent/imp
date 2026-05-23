"""
Collision-aware IK in a world frame.

When `ik_local` is given a Scene that has a built CollisionModel, the
validator runs self-collision and environment-collision checks at the
final q. A candidate that satisfies pose tolerance but penetrates a
world object is rejected with `IKStatus.FINAL_COLLISION`.

`ik(scene, robot_id, ...)` is the same call but with the world base
pose composed in automatically; the target is in world coordinates.

Run from the algorithms directory:

    python examples/ik/08_collision_aware_ik.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from algorithms.descriptions import WorldDescription
from algorithms.kinematics import fk, ik, ik_local
from algorithms.kinematics.ik import IKOptions
from algorithms.resolved import CollisionModel, KinematicModel, Scene

REPO_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    world = WorldDescription.from_yaml(
        REPO_ROOT / "configs" / "worlds" / "franka_table_world.yaml"
    )
    collision_model = CollisionModel.from_world(world)
    scene = Scene.from_world(world, collision_model)

    robot_id = "arm"
    system = world.robot(robot_id).robot_system
    model = KinematicModel.from_robot_system(system)
    home = system.named_joint_state("home")
    q_home = np.array([home[name] for name in model.active_joint_names], dtype=float)

    # In this scene, the home configuration intersects the table, so the
    # home TCP pose is already a colliding goal. Two queries demonstrate
    # validation on vs off, then we show that `ik` and `ik_local` produce
    # the same status for the same target up to the world->base composition.

    T_world_home = fk(scene, robot_id, q_home, "fr3_hand_tcp")
    opts = IKOptions(reject_singular=False)

    with_collision = ik(
        scene, robot_id, "fr3_hand_tcp", T_world_home, q_home,
        options=opts,
    )
    print(f"collision validation on  : {with_collision.status.name}")

    without_collision = ik(
        scene, robot_id, "fr3_hand_tcp", T_world_home, q_home,
        options=opts,
        validate_collision=False,
    )
    print(f"collision validation off : {without_collision.status.name}")

    # The same query via the base-frame entrypoint. Caller composes
    # base_pose by hand. Demonstrates that `ik(scene, ...)` is doing
    # exactly this composition under the hood.
    base_pose = world.robot(robot_id).base_pose
    T_world_base = base_pose.as_matrix() if base_pose is not None else np.eye(4)
    T_base_home = np.linalg.inv(T_world_base) @ T_world_home
    local_equiv = ik_local(
        model, "fr3_hand_tcp", T_base_home, q_home,
        scene=scene,
        options=opts,
    )
    print(f"  via ik_local equivalent: {local_equiv.status.name}")


if __name__ == "__main__":
    main()
