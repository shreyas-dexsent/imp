"""Boundary-style tests for kinematics.ik."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pinocchio")

from algorithms.descriptions import RobotSystemDescription
from algorithms.kinematics import fk_local, ik_local
from algorithms.kinematics.ik import IKOptions, IKStatus
from algorithms.resolved import KinematicModel
from algorithms.resolved.kinematic_model import _clear_cache


REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _isolate_cache():
    _clear_cache()
    yield
    _clear_cache()


@pytest.fixture()
def robot_model_and_q():
    system = RobotSystemDescription.from_yaml(
        REPO_ROOT / "configs" / "robots" / "franka_fr3_robot_only.yaml"
    )
    model = KinematicModel.from_robot_system(system)
    home = system.named_joint_state("home")
    q = np.array([home[name] for name in model.active_joint_names], dtype=float)
    return model, q


def test_unreachable_target_is_reported(robot_model_and_q):
    model, q = robot_model_and_q
    T_target = fk_local(model, q, "robot_tcp").copy()
    T_target[0, 3] += 2.0

    result = ik_local(
        model,
        "robot_tcp",
        T_target,
        q,
        options=IKOptions(
            max_time_ms=250,
            max_iters=50,
            multi_start=False,
            reject_singular=False,
        ),
    )

    assert result.status is IKStatus.UNREACHABLE
    assert result.q is None


def test_joint_limit_boundary_seed_can_recover_to_valid_solution(robot_model_and_q):
    model, q = robot_model_and_q
    T_target = fk_local(model, q, "robot_tcp")
    q_min, _ = model.active_position_limits()

    result = ik_local(
        model,
        "robot_tcp",
        T_target,
        q_min,
        options=IKOptions(
            max_time_ms=50,
            multi_start=False,
            joint_margin=1e-2,
            reject_singular=False,
        ),
    )

    q_min_margin = q_min + 1e-2
    assert result.status is IKStatus.SUCCESS
    assert result.q is not None
    assert np.all(result.q >= q_min_margin)


def test_orientation_wraparound_target_at_seed_pose_succeeds(robot_model_and_q):
    model, q = robot_model_and_q
    T_target = fk_local(model, q, "robot_tcp").copy()
    T_target[:3, :3] = T_target[:3, :3] @ np.eye(3)

    result = ik_local(model, "robot_tcp", T_target, q)

    assert result.status is IKStatus.SUCCESS
