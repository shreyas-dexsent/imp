"""
Palletizing with linear motion + obstacle-aware fallback.

The user case: deposit a sequence of objects onto a 2x2 pallet
grid. Each deposit position must be reached via a straight Cartesian
line, but if the line is blocked the application falls back to a
joint-space replanned move.

Pattern:

    for each pallet cell:
        try move_l(safe_above_cell)
            if line blocked -> move_joint(safe_above_cell) as fallback
        move_l(cell)       # final descent
        move_l(safe_above_cell)  # lift off

For demonstration the gripper is not actually closed/opened; only
the motion is shown.

Run from the repo root:

    python examples/integration/02_linear_palletizing.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from algorithms import (
    CollisionModel,
    KinematicModel,
    MoveStatus,
    Scene,
    WorldDescription,
    move_joint,
    move_l,
)
from algorithms.descriptions import BoxGeometrySpec
from algorithms.kinematics import fk
from algorithms.kinematics.ik import ik_local

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

    # Pallet cells laid out on a small grid in front of the robot,
    # at reachable heights for the FR3 from its home pose.
    cell_positions = [
        [0.40, -0.10, 0.30],
        [0.40,  0.10, 0.30],
        [0.50, -0.10, 0.30],
        [0.50,  0.10, 0.30],
    ]
    safe_z = 0.55  # 25 cm above each cell

    # A side wall outside the descent corridor — its only purpose is
    # to make at least one inter-cell horizontal move impossible by
    # straight line, exercising the joint-space fallback below. Sized
    # and placed so it never intersects any cell descent.
    scene.add_object(
        "side_wall",
        collision=BoxGeometrySpec(type="box", size=(0.02, 0.04, 0.10)),
        pose=_translation([0.45, 0.0, 0.60]),
    )

    # The TCP orientation we want to maintain: gripper pointing down.
    T_orient = fk(scene, "arm", q_home, "robot_tcp")[:3, :3]

    q_current = q_home
    summary = []

    for i, cell in enumerate(cell_positions):
        T_above = _pose(T_orient, [cell[0], cell[1], safe_z])
        T_cell = _pose(T_orient, cell)

        # Approach the safe-above pose.
        r_above = move_l(scene, "arm", "robot_tcp", T_above, q_seed=q_current)
        if r_above.status is MoveStatus.SUCCESS:
            label, q_current = "move_l", _ik_q_at(model, scene, "robot_tcp", T_above, q_current)
        else:
            # Linear path blocked — fall back to joint-space replanning.
            ik_above = ik_local(model, "robot_tcp", T_above, q_current)
            if ik_above.status.name != "SUCCESS":
                print(f"cell {i}: IK at safe-above failed; skipping cell")
                continue
            r_above = move_joint(model, scene, ik_above.q, q_seed=q_current)
            if r_above.status is not MoveStatus.SUCCESS:
                print(f"cell {i}: joint fallback failed too; skipping cell")
                continue
            label, q_current = "move_joint (fallback)", ik_above.q

        # Final descent to the cell — linear by requirement.
        r_descent = move_l(scene, "arm", "robot_tcp", T_cell, q_seed=q_current)
        if r_descent.status is not MoveStatus.SUCCESS:
            print(f"cell {i}: descent failed: {r_descent.status.name}")
            continue
        q_at_cell = _ik_q_at(model, scene, "robot_tcp", T_cell, q_current)

        # Lift back to safe-above.
        r_lift = move_l(scene, "arm", "robot_tcp", T_above, q_seed=q_at_cell)
        if r_lift.status is not MoveStatus.SUCCESS:
            print(f"cell {i}: lift failed: {r_lift.status.name}")
            continue
        q_current = _ik_q_at(model, scene, "robot_tcp", T_above, q_at_cell)

        total = r_above.trajectory.duration + r_descent.trajectory.duration + r_lift.trajectory.duration
        summary.append((i, label, total))
        print(f"cell {i}: above={label:22s}  total={total:.2f} s")

    print()
    print(f"Pallet cells reached: {len(summary)} / {len(cell_positions)}")
    print(f"Total motion time   : {sum(s[2] for s in summary):.2f} s")


def _pose(R: np.ndarray, p) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = p
    return T


def _translation(p) -> np.ndarray:
    T = np.eye(4)
    T[:3, 3] = p
    return T


def _ik_q_at(model, scene, frame_id, T_target, q_seed) -> np.ndarray:
    """Resolve the joint configuration at the requested TCP pose."""
    result = ik_local(model, frame_id, T_target, q_seed, scene=scene)
    if result.status.name != "SUCCESS":
        # Fall back to the seed if IK fails (should be rare given the
        # whole leg was already planned).
        return q_seed
    return result.q


if __name__ == "__main__":
    main()
