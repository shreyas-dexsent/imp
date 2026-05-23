"""Regression-test scaffold for kinematics.ik.

Every deployment IK bug should add one row to REGRESSION_CASES with:
target pose, seed, robot model id/version, tool frame, expected status or q,
and the reason the case exists.
"""
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


REGRESSION_CASES = (
    {
        "name": "fr3_home_robot_tcp_exact",
        "robot_yaml": "franka_fr3_robot_only.yaml",
        "frame_id": "robot_tcp",
        "source_pose": "home",
        "seed_pose": "home",
        "expected_status": IKStatus.SUCCESS,
        "reason": "bootstrap exact-target regression case",
    },
)


@pytest.fixture(autouse=True)
def _isolate_cache():
    _clear_cache()
    yield
    _clear_cache()


@pytest.mark.parametrize("case", REGRESSION_CASES, ids=lambda case: case["name"])
def test_ik_regression_cases(case):
    system = RobotSystemDescription.from_yaml(
        REPO_ROOT / "configs" / "robots" / case["robot_yaml"]
    )
    model = KinematicModel.from_robot_system(system)
    source_pose = system.named_joint_state(case["source_pose"])
    seed_pose = system.named_joint_state(case["seed_pose"])
    q_source = np.array(
        [source_pose[name] for name in model.active_joint_names],
        dtype=float,
    )
    q_seed = np.array([seed_pose[name] for name in model.active_joint_names], dtype=float)
    T_target = fk_local(model, q_source, case["frame_id"])

    result = ik_local(
        model,
        case["frame_id"],
        T_target,
        q_seed,
        options=IKOptions(reject_singular=False),
    )

    assert result.status is case["expected_status"]
