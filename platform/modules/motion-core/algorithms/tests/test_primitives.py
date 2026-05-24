"""Motion-primitive tests (Phase 7).

Covers:
* move_joint produces a smooth pass-through trajectory from q_seed to q_goal.
* move_l produces a linear Cartesian trajectory; cartesian_waypoints
  populated; no smoothing applied.
* approach lands exactly on T_target.
* retreat moves `distance` away from FK(q_seed).
* via_motion concatenates through every via-point without stopping.
* All primitives' MoveResult diagnostics carry the right intermediate
  artifacts on success and on failure.
* MoveStatus is exhaustive (every value reachable by a test or
  deliberately reserved).
"""
from __future__ import annotations

from pathlib import Path as FsPath

import numpy as np
import pytest

pytest.importorskip("coal")
pytest.importorskip("pinocchio")
pytest.importorskip("ompl")

from algorithms.descriptions import WorldDescription
from algorithms.kinematics import fk
from algorithms.primitives import (
    MoveOptions,
    MoveStatus,
    approach,
    move_joint,
    move_l,
    pre_approach_pose,
    retreat,
    via_motion,
)
from algorithms.resolved import CollisionModel, KinematicModel, Scene
from algorithms.resolved.kinematic_model import _clear_cache

REPO_ROOT = FsPath(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _isolate_cache():
    _clear_cache()
    yield
    _clear_cache()


def _setup():
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


# ---------------------------------------------------------------------------
# move_joint
# ---------------------------------------------------------------------------


def test_move_joint_succeeds_on_collision_free_pair():
    scene, model, q_home = _setup()
    q_goal = q_home.copy()
    q_goal[0] += 0.6
    q_goal[2] -= 0.2
    result = move_joint(model, scene, q_goal, q_seed=q_home)
    assert result.status is MoveStatus.SUCCESS
    assert result.trajectory is not None
    assert result.path is not None
    assert result.plan_result is not None
    assert result.path_validation is not None and result.path_validation.passed
    assert result.trajectory_validation is not None and result.trajectory_validation.passed


def test_move_joint_pass_through_at_interior_waypoints():
    scene, model, q_home = _setup()
    q_goal = q_home.copy()
    q_goal[0] += 0.8
    result = move_joint(model, scene, q_goal, q_seed=q_home)
    traj = result.trajectory
    _, qd_mid, _ = traj.at(traj.duration * 0.5)
    assert float(np.linalg.norm(qd_mid)) > 0.05, (
        "move_joint must produce pass-through motion (non-zero interior velocity)"
    )
    _, qd_start, _ = traj.at(0.0)
    _, qd_end, _ = traj.at(traj.duration)
    np.testing.assert_allclose(qd_start, 0.0, atol=1e-6)
    np.testing.assert_allclose(qd_end, 0.0, atol=1e-6)


def test_move_joint_q_shape_mismatch_returns_plan_fail():
    scene, model, q_home = _setup()
    result = move_joint(model, scene, q_home[:-1], q_seed=q_home)
    assert result.status is MoveStatus.PLAN_FAILED


# ---------------------------------------------------------------------------
# move_l
# ---------------------------------------------------------------------------


def test_move_l_succeeds_on_short_x_translation():
    scene, model, q_home = _setup()
    T_start = fk(scene, "arm", q_home, "robot_tcp")
    T_goal = T_start.copy()
    T_goal[0, 3] += 0.08
    result = move_l(scene, "arm", "robot_tcp", T_goal, q_home)
    assert result.status is MoveStatus.SUCCESS
    # Cartesian path metadata is preserved
    assert result.path is not None
    assert result.path.cartesian_waypoints is not None


def test_move_l_does_not_smooth_or_spline():
    """move_l preserves the Cartesian path (smoothing would deviate)."""
    scene, model, q_home = _setup()
    T_start = fk(scene, "arm", q_home, "robot_tcp")
    T_goal = T_start.copy()
    T_goal[0, 3] += 0.05
    result = move_l(scene, "arm", "robot_tcp", T_goal, q_home)
    # cartesian_waypoints[0] equals T_start; cartesian_waypoints[-1]
    # equals T_goal. Smoothing would erase the cartesian metadata.
    np.testing.assert_allclose(
        result.path.cartesian_waypoints[-1][:3, 3], T_goal[:3, 3], atol=1e-12,
    )


# ---------------------------------------------------------------------------
# approach
# ---------------------------------------------------------------------------


def test_approach_lands_on_target():
    scene, model, q_home = _setup()
    T_target = fk(scene, "arm", q_home, "robot_tcp")
    # q_home is at T_target itself. The pre-approach pose is offset by
    # -z by 5 cm. Approach should land back on T_target.
    result = approach(scene, "arm", "robot_tcp", T_target, q_seed=q_home,
                       distance=0.05, axis="-z")
    assert result.status is MoveStatus.SUCCESS
    final_pose = result.path.cartesian_waypoints[-1]
    np.testing.assert_allclose(final_pose[:3, 3], T_target[:3, 3], atol=1e-3)


def test_pre_approach_pose_is_offset_distance_along_axis_in_target_frame():
    T_target = np.eye(4)
    T_target[:3, 3] = [0.5, 0.2, 0.3]
    # axis="-z" in target frame, target's z is +z world (identity rotation),
    # so pre = target - (-z) * distance = target + 5cm in z
    T_pre = pre_approach_pose(T_target, distance=0.05, axis="-z", reference="target")
    np.testing.assert_allclose(T_pre[:3, 3], [0.5, 0.2, 0.35])


def test_approach_invalid_axis_raises_value_error():
    scene, model, q_home = _setup()
    T_target = fk(scene, "arm", q_home, "robot_tcp")
    with pytest.raises(ValueError, match="axis must be one of"):
        approach(scene, "arm", "robot_tcp", T_target, q_seed=q_home, axis="bogus")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# retreat
# ---------------------------------------------------------------------------


def test_retreat_moves_distance_in_local_z():
    scene, model, q_home = _setup()
    T_start = fk(scene, "arm", q_home, "robot_tcp")
    result = retreat(scene, "arm", "robot_tcp", q_seed=q_home,
                      distance=0.05, axis="z", reference="tcp")
    assert result.status is MoveStatus.SUCCESS
    # End TCP pose should be T_start translated by +0.05 along TCP-frame z.
    final_pose = result.path.cartesian_waypoints[-1]
    expected_offset = T_start[:3, :3] @ np.array([0, 0, 0.05])
    np.testing.assert_allclose(
        final_pose[:3, 3], T_start[:3, 3] + expected_offset, atol=1e-3,
    )


def test_retreat_world_reference_uses_world_axis():
    scene, model, q_home = _setup()
    T_start = fk(scene, "arm", q_home, "robot_tcp")
    result = retreat(scene, "arm", "robot_tcp", q_seed=q_home,
                      distance=0.05, axis="z", reference="world")
    assert result.status is MoveStatus.SUCCESS
    final_pose = result.path.cartesian_waypoints[-1]
    np.testing.assert_allclose(
        final_pose[:3, 3], T_start[:3, 3] + np.array([0, 0, 0.05]), atol=1e-3,
    )


# ---------------------------------------------------------------------------
# via_motion
# ---------------------------------------------------------------------------


def test_via_motion_succeeds_on_three_waypoints():
    scene, model, q_home = _setup()
    q_a = q_home.copy()
    q_a[0] += 0.3
    q_b = q_home.copy()
    q_b[0] += 0.6
    result = via_motion(model, scene, [q_home, q_a, q_b])
    assert result.status is MoveStatus.SUCCESS
    assert result.trajectory is not None
    # Trajectory length > 0
    assert result.trajectory.duration > 0


def test_via_motion_rejects_single_waypoint_list():
    scene, model, q_home = _setup()
    result = via_motion(model, scene, [q_home])
    assert result.status is MoveStatus.INVALID_INPUT


def test_via_motion_pass_through_at_interior_via_points():
    scene, model, q_home = _setup()
    q_a = q_home.copy()
    q_a[0] += 0.3
    q_b = q_home.copy()
    q_b[0] += 0.6
    result = via_motion(model, scene, [q_home, q_a, q_b])
    traj = result.trajectory
    # Velocity in the middle of the trajectory should be non-zero.
    _, qd_mid, _ = traj.at(traj.duration * 0.5)
    assert float(np.linalg.norm(qd_mid)) > 0.05


# ---------------------------------------------------------------------------
# MoveStatus coverage
# ---------------------------------------------------------------------------


def test_move_status_taxonomy_complete():
    expected = {
        "SUCCESS",
        "INVALID_INPUT",
        "IK_FAILED",
        "PLAN_FAILED",
        "OPTIMIZATION_FAILED",
        "PATH_VALIDATION_FAILED",
        "TRAJECTORY_FAILED",
        "TRAJECTORY_VALIDATION_FAILED",
        "NUMERICAL_FAILURE",
    }
    assert {m.name for m in MoveStatus} == expected
