"""Validation tests for kinematics.ik.

Exercises every reachable IKStatus value so the plan's "every status
reachable in tests" acceptance criterion is provably met.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pinocchio")

from algorithms.descriptions import RobotSystemDescription
from algorithms.kinematics import fk_local, ik_local
from algorithms.kinematics.ik import IKOptions, IKProblem, IKStatus, PoseTarget
from algorithms.kinematics.ik.backends.analytical import registry as analytical_registry
from algorithms.kinematics.ik.constraints import JointPositionBounds
from algorithms.kinematics.ik.validator import validate
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


def _problem_for_target(target: PoseTarget, model: KinematicModel) -> IKProblem:
    problem = IKProblem()
    problem.add_task(target)
    q_min, q_max = model.active_position_limits()
    problem.add_constraint(JointPositionBounds(q_min, q_max))
    return problem


def test_validator_accepts_exact_fk_target(robot_model_and_q):
    model, q = robot_model_and_q
    T_target = fk_local(model, q, "robot_tcp")
    problem = _problem_for_target(PoseTarget("robot_tcp", T_target), model)

    report = validate(model, problem.freeze(), q, IKOptions())

    assert report.status is IKStatus.SUCCESS
    assert report.success


def test_validator_rejects_nonfinite_q(robot_model_and_q):
    model, q = robot_model_and_q
    T_target = fk_local(model, q, "robot_tcp")
    problem = _problem_for_target(PoseTarget("robot_tcp", T_target), model)
    q_bad = q.copy()
    q_bad[0] = np.nan

    report = validate(model, problem.freeze(), q_bad, IKOptions())

    assert report.status is IKStatus.NUMERICAL_FAILURE


def test_validator_rejects_joint_limit_violation(robot_model_and_q):
    model, q = robot_model_and_q
    T_target = fk_local(model, q, "robot_tcp")
    problem = _problem_for_target(PoseTarget("robot_tcp", T_target), model)
    q_min, _ = model.active_position_limits()

    report = validate(model, problem.freeze(), q_min, IKOptions(joint_margin=1e-3))

    assert report.status is IKStatus.JOINT_LIMIT_VIOLATION


def test_validator_rejects_pose_error(robot_model_and_q):
    model, q = robot_model_and_q
    T_target = fk_local(model, q, "robot_tcp").copy()
    T_target[0, 3] += 0.05
    problem = _problem_for_target(PoseTarget("robot_tcp", T_target), model)

    report = validate(
        model,
        problem.freeze(),
        q,
        IKOptions(reject_singular=False),
    )

    assert report.status is IKStatus.POSE_ERROR_TOO_HIGH


def test_validator_rejects_singularity_risk_when_threshold_is_strict(robot_model_and_q):
    model, q = robot_model_and_q
    T_target = fk_local(model, q, "robot_tcp")
    problem = _problem_for_target(PoseTarget("robot_tcp", T_target), model)

    report = validate(
        model,
        problem.freeze(),
        q,
        IKOptions(min_sigma_limit=1e9),
    )

    assert report.status is IKStatus.SINGULARITY_RISK


def test_validator_rejects_invalid_target_rotation(robot_model_and_q):
    model, q = robot_model_and_q
    T_target = fk_local(model, q, "robot_tcp").copy()
    T_target[0, 0] = 2.0
    problem = _problem_for_target(PoseTarget("robot_tcp", T_target), model)

    report = validate(model, problem.freeze(), q, IKOptions())

    assert report.status is IKStatus.INVALID_INPUT


# ---------------------------------------------------------------------------
# Solver-level statuses propagated from backends
# ---------------------------------------------------------------------------


def test_solver_returns_max_iterations(robot_model_and_q):
    """DLS backend hits max_iters before reaching a far target."""
    model, q = robot_model_and_q
    T_target = fk_local(model, q, "robot_tcp").copy()
    T_target[0, 3] += 0.3  # 30 cm offset, far from seed

    result = ik_local(model, "robot_tcp", T_target, q, backend="dls",
                      options=IKOptions(max_iters=1))

    assert result.status is IKStatus.MAX_ITERATIONS


def test_solver_returns_timeout(robot_model_and_q):
    """DLS backend bails when max_time_ms is effectively zero."""
    model, q = robot_model_and_q
    T_target = fk_local(model, q, "robot_tcp").copy()
    T_target[0, 3] += 0.3

    result = ik_local(
        model,
        "robot_tcp",
        T_target,
        q,
        backend="dls",
        options=IKOptions(max_iters=200, max_time_ms=1e-9),
    )

    assert result.status is IKStatus.TIMEOUT


class _EmptyBranchAnalyticalIK:
    """Stub analytical backend that produces no branches."""

    name = "empty_analytical_stub"

    def solve_branches(self, model, spec, q_seed):
        return ()


def test_solver_returns_no_valid_candidate_when_analytical_yields_no_branches(
    robot_model_and_q,
):
    """Registered analytical backend returning zero branches propagates as
    NO_VALID_CANDIDATE."""
    model, q = robot_model_and_q
    T_target = fk_local(model, q, "robot_tcp")

    analytical_registry.register(model.system.robot.id, _EmptyBranchAnalyticalIK)
    try:
        result = ik_local(model, "robot_tcp", T_target, q)
    finally:
        analytical_registry.clear()

    assert result.status is IKStatus.NO_VALID_CANDIDATE


def test_constraint_violation_is_reserved_for_future_use():
    """`CONSTRAINT_VIOLATION` exists in the taxonomy but is unreachable in v1.

    Only constraint v1 ships is `JointPositionBounds`, which surfaces as
    `JOINT_LIMIT_VIOLATION`. The status is reserved for nonlinear
    user-added constraints (tool axis, RCM, minimum distance) that will
    land alongside the Recommended/Industrial constraint modules.
    """
    assert IKStatus.CONSTRAINT_VIOLATION.value == "constraint_violation"
    assert IKStatus.CONSTRAINT_VIOLATION in set(IKStatus)
