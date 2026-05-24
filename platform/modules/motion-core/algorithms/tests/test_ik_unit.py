"""Unit tests for kinematics.ik."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pinocchio")

from algorithms.descriptions import RobotSystemDescription, WorldDescription
from algorithms.kinematics import fk_local, ik_local, ik_velocity
from algorithms.kinematics.ik import IKOptions, IKStatus
from algorithms.resolved import CollisionModel, KinematicModel, Scene
from algorithms.resolved.kinematic_model import _clear_cache


REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _isolate_cache():
    _clear_cache()
    yield
    _clear_cache()


def _robot_only_model():
    system = RobotSystemDescription.from_yaml(
        REPO_ROOT / "configs" / "robots" / "franka_fr3_robot_only.yaml"
    )
    model = KinematicModel.from_robot_system(system)
    home = system.named_joint_state("home")
    q = np.array([home[name] for name in model.active_joint_names], dtype=float)
    return system, model, q


def test_default_ik_solves_fk_generated_target():
    _, model, q = _robot_only_model()
    target_pose = fk_local(model, q, "robot_tcp")

    result = ik_local(model, "robot_tcp", target_pose, q)

    assert result.status is IKStatus.SUCCESS
    assert result.q is not None
    np.testing.assert_allclose(
        fk_local(model, result.q, "robot_tcp"),
        target_pose,
        atol=1e-5,
    )


def test_generic_ik_solves_small_offset_target():
    _, model, q = _robot_only_model()
    target_pose = fk_local(model, q, "robot_tcp").copy()
    target_pose[0, 3] += 0.01

    result = ik_local(
        model,
        "robot_tcp",
        target_pose,
        q,
        options=IKOptions(max_time_ms=1000, num_random_seeds=1, reject_singular=False),
    )

    assert result.status is IKStatus.SUCCESS
    assert result.pose_error[0] <= 1e-4


def test_wrong_seed_shape_returns_invalid_input():
    _, model, q = _robot_only_model()
    target_pose = fk_local(model, q, "robot_tcp")

    result = ik_local(model, "robot_tcp", target_pose, q[:-1])

    assert result.status is IKStatus.INVALID_INPUT
    assert result.q is None


def test_unknown_frame_returns_invalid_input():
    _, model, q = _robot_only_model()
    target_pose = fk_local(model, q, "robot_tcp")

    result = ik_local(model, "no_such_frame", target_pose, q)

    assert result.status is IKStatus.INVALID_INPUT


def test_explicit_analytical_backend_without_registration_is_invalid_input():
    _, model, q = _robot_only_model()
    target_pose = fk_local(model, q, "robot_tcp")

    result = ik_local(model, "robot_tcp", target_pose, q, backend="opw")

    assert result.status is IKStatus.INVALID_INPUT
    assert "OPWIK requires" in result.diagnostics.message


def test_qp_velocity_returns_qdot_shape():
    _, model, q = _robot_only_model()

    qdot = ik_velocity(model, "robot_tcp", np.array([0.01, 0, 0, 0, 0, 0]), q)

    assert qdot.shape == q.shape
    assert np.all(np.isfinite(qdot))


def test_collision_validation_can_reject_final_q():
    world = WorldDescription.from_yaml(
        REPO_ROOT / "configs" / "worlds" / "franka_table_world.yaml"
    )
    collision_model = CollisionModel.from_world(world)
    scene = Scene.from_world(world, collision_model)
    system = world.robot("arm").robot_system
    model = KinematicModel.from_robot_system(system)
    home = system.named_joint_state("home")
    q = np.array([home[name] for name in model.active_joint_names], dtype=float)
    target_pose = fk_local(model, q, "fr3_hand_tcp")

    result = ik_local(
        model,
        "fr3_hand_tcp",
        target_pose,
        q,
        scene=scene,
        options=IKOptions(reject_singular=False),
    )

    assert result.status is IKStatus.FINAL_COLLISION
