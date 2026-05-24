"""Monte Carlo workspace tests for kinematics.ik.

These tests exercise the actual NLS solver, not the early-out path. The
seed is intentionally **not** the q that produced the target — that would
short-circuit the optimiser on the first iteration of `generic.solve`.
Instead each call seeds from `q_home`, the realistic scenario for an
industrial caller who does not yet know the answer.

Set `PYTEST_IK_MONTECARLO_N` to a larger number for longer local sweeps.
"""
from __future__ import annotations

import csv
import os
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pinocchio")

from algorithms.descriptions import RobotSystemDescription
from algorithms.kinematics import fk_local, ik_local
from algorithms.kinematics.ik import IKStatus
from algorithms.resolved import KinematicModel
from algorithms.resolved.kinematic_model import _clear_cache


REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _isolate_cache():
    _clear_cache()
    yield
    _clear_cache()


def _home_q(model, system):
    home = system.named_joint_state("home")
    return np.array([home[name] for name in model.active_joint_names], dtype=float)


def test_fk_ik_fk_round_trip_from_home_seed():
    """The realistic case: target is reachable, seed is q_home (no cheating)."""
    system = RobotSystemDescription.from_yaml(
        REPO_ROOT / "configs" / "robots" / "franka_fr3_robot_only.yaml"
    )
    model = KinematicModel.from_robot_system(system)
    q_min, q_max = model.active_position_limits()
    q_home = _home_q(model, system)
    rng = np.random.default_rng(7)
    sample_count = int(os.environ.get("PYTEST_IK_MONTECARLO_N", "50"))
    rows = []
    successes = 0

    for sample_index in range(sample_count):
        q_random = rng.uniform(q_min + 1e-3, q_max - 1e-3)
        T_target = fk_local(model, q_random, "robot_tcp")
        result = ik_local(model, "robot_tcp", T_target, q_home)
        success = result.status is IKStatus.SUCCESS
        successes += int(success)
        rows.append(
            {
                "sample": sample_index,
                "status": result.status.value,
                "position_error": result.pose_error[0],
                "orientation_error": result.pose_error[1],
                "elapsed_ms": result.elapsed_ms,
            }
        )

    artifact_dir = REPO_ROOT / "tests" / "artifacts"
    artifact_dir.mkdir(exist_ok=True)
    with (artifact_dir / "ik_montecarlo_latest.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    success_rate = successes / sample_count
    assert success_rate >= 0.95, (
        f"Generic IK success rate from q_home seed is {success_rate:.0%} "
        f"({successes}/{sample_count}); acceptance bar is >=95%."
    )


def test_fk_ik_fk_round_trip_from_warm_seed():
    """Sanity check: when the seed is already near the target, the solver
    succeeds trivially via the seed early-out path."""
    system = RobotSystemDescription.from_yaml(
        REPO_ROOT / "configs" / "robots" / "franka_fr3_robot_only.yaml"
    )
    model = KinematicModel.from_robot_system(system)
    q_min, q_max = model.active_position_limits()
    rng = np.random.default_rng(7)
    sample_count = int(os.environ.get("PYTEST_IK_MONTECARLO_N", "25"))
    successes = 0

    # Sample well inside the joint margin so the warm seed isn't rejected
    # by the validator's hard bound check.
    margin = 0.05
    for _ in range(sample_count):
        q_random = rng.uniform(q_min + margin, q_max - margin)
        T_target = fk_local(model, q_random, "robot_tcp")
        # Warm seed: the producing q. Short-circuits on the first iteration.
        result = ik_local(model, "robot_tcp", T_target, q_random)
        if result.status is IKStatus.SUCCESS:
            successes += 1

    assert successes == sample_count, (
        f"Warm-seed path is broken: {successes}/{sample_count} succeeded."
    )
