"""
End-to-end bin-pick sequence using motion primitives.

Combines move_joint -> approach -> retreat -> move_joint into a single
4-segment motion plan. Each leg returns its own Trajectory; a real
application would either concatenate them or play them through the
controller adapter in order.

The example does NOT actually run a gripper — the grasp / release
happens between leg 2 and leg 3 in real life. Here we just show the
motion planning side.

Run from the algorithms directory:

    python examples/primitives/05_bin_pick_sequence.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from algorithms.descriptions import WorldDescription
from algorithms.kinematics import fk
from algorithms.kinematics.ik import ik_local
from algorithms.primitives import (
    MoveStatus, approach, move_joint, pre_approach_pose, retreat,
)
from algorithms.resolved import CollisionModel, KinematicModel, Scene

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

    # Pretend perception reported a workpiece TCP pose 10 cm in +x of home.
    T_grasp = fk(scene, "arm", q_home, "robot_tcp").copy()
    T_grasp[0, 3] += 0.10
    T_pre_grasp = pre_approach_pose(T_grasp, distance=0.05, axis="-z", reference="target")

    # Solve IK for the pre-grasp pose (this is the joint-space target
    # for the first move_joint leg).
    pre_ik = ik_local(model, "robot_tcp", T_pre_grasp, q_home)
    if pre_ik.status.name != "SUCCESS":
        print(f"FAILED: IK on pre-grasp pose returned {pre_ik.status.name}")
        return
    q_pre_grasp = pre_ik.q

    legs = []

    # Leg 1: move_joint home -> pre-grasp
    r1 = move_joint(model, scene, q_pre_grasp, q_seed=q_home)
    print(f"Leg 1 move_joint  : {r1.status.name}  duration={r1.trajectory.duration if r1.trajectory else None}")
    legs.append(r1)

    # Leg 2: approach pre-grasp -> grasp
    r2 = approach(
        scene, "arm", "robot_tcp", T_grasp, q_seed=q_pre_grasp,
        distance=0.05, axis="-z", reference="target",
    )
    print(f"Leg 2 approach    : {r2.status.name}  duration={r2.trajectory.duration if r2.trajectory else None}")
    legs.append(r2)

    # [Gripper closes here in real application]

    # Solve IK on the grasp pose to know where we ended up.
    grasp_ik = ik_local(model, "robot_tcp", T_grasp, q_pre_grasp)
    q_grasp = grasp_ik.q if grasp_ik.status.name == "SUCCESS" else q_pre_grasp

    # Leg 3: retreat lift 5 cm
    r3 = retreat(
        scene, "arm", "robot_tcp", q_seed=q_grasp,
        distance=0.05, axis="z", reference="tcp",
    )
    print(f"Leg 3 retreat     : {r3.status.name}  duration={r3.trajectory.duration if r3.trajectory else None}")
    legs.append(r3)

    # Leg 4: back to home
    # The retreat ended at FK(q_grasp) + 5 cm in TCP z. Resolve via IK.
    if r3.path is not None and r3.path.cartesian_waypoints is not None:
        T_post_retreat = r3.path.cartesian_waypoints[-1]
        post_ik = ik_local(model, "robot_tcp", T_post_retreat, q_grasp)
        q_post_retreat = post_ik.q if post_ik.status.name == "SUCCESS" else q_grasp
    else:
        q_post_retreat = q_grasp
    r4 = move_joint(model, scene, q_home, q_seed=q_post_retreat)
    print(f"Leg 4 move_joint  : {r4.status.name}  duration={r4.trajectory.duration if r4.trajectory else None}")
    legs.append(r4)

    total = sum(
        leg.trajectory.duration for leg in legs
        if leg.status is MoveStatus.SUCCESS and leg.trajectory is not None
    )
    successes = sum(1 for leg in legs if leg.status is MoveStatus.SUCCESS)
    print(f"\nSummary: {successes}/{len(legs)} legs succeeded; total motion time {total:.2f} s")


if __name__ == "__main__":
    main()
