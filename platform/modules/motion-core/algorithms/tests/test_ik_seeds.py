"""Seed-generation tests for kinematics.ik."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pinocchio")

from algorithms.descriptions import RobotSystemDescription
from algorithms.kinematics.ik import IKOptions
from algorithms.kinematics.ik.seeds import generate_seeds
from algorithms.resolved import KinematicModel
from algorithms.resolved.kinematic_model import _clear_cache


REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _isolate_cache():
    _clear_cache()
    yield
    _clear_cache()


def _model_and_home_q():
    system = RobotSystemDescription.from_yaml(
        REPO_ROOT / "configs" / "robots" / "franka_fr3_robot_only.yaml"
    )
    model = KinematicModel.from_robot_system(system)
    home = system.named_joint_state("home")
    q_home = np.array([home[name] for name in model.active_joint_names], dtype=float)
    return model, q_home


def test_generate_seeds_is_deterministic():
    model, q_home = _model_and_home_q()
    options = IKOptions(num_random_seeds=4, random_seed=123)

    first = generate_seeds(model, q_home, options, q_home=q_home)
    second = generate_seeds(model, q_home, options, q_home=q_home)

    assert len(first) == len(second)
    for a, b in zip(first, second):
        np.testing.assert_allclose(a, b)


def test_generate_seeds_clips_user_seeds_to_position_limits():
    model, q_home = _model_and_home_q()
    q_min, q_max = model.active_position_limits()
    q_bad = q_max + 10.0

    seeds = generate_seeds(
        model,
        q_bad,
        IKOptions(multi_start=False),
        q_home=q_home,
    )

    assert np.all(seeds[0] <= q_max)
    assert np.all(seeds[0] >= q_min)
