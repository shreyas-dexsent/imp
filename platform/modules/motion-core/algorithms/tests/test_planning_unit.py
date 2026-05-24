"""Unit tests for Path data type + result envelope + options."""
from __future__ import annotations

import numpy as np
import pytest

from algorithms.planning import (
    Path,
    PathDiagnostics,
    PathPlanResult,
    PathStatus,
    PathValidationOptions,
    PlanOptions,
)


def test_path_requires_at_least_two_waypoints():
    with pytest.raises(ValueError, match="at least 2 waypoints"):
        Path(waypoints=np.zeros((1, 7)), joint_names=("j1",) * 7)


def test_path_rejects_dof_joint_names_mismatch():
    with pytest.raises(ValueError, match="dof"):
        Path(waypoints=np.zeros((3, 7)), joint_names=("j1", "j2"))


def test_path_rejects_non_2d_waypoints():
    with pytest.raises(ValueError, match="2D ndarray"):
        Path(waypoints=np.zeros(7), joint_names=("j1",) * 7)


def test_path_cartesian_waypoints_shape_validated():
    wps = np.zeros((3, 7))
    cart = np.zeros((2, 4, 4))  # wrong N
    with pytest.raises(ValueError, match="cartesian_waypoints"):
        Path(waypoints=wps, joint_names=("j",) * 7, cartesian_waypoints=cart)


def test_path_length_is_sum_of_segment_norms():
    wps = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])
    p = Path(waypoints=wps, joint_names=("a", "b"))
    assert p.length() == pytest.approx(2.0)


def test_path_dof_num_waypoints_num_segments():
    wps = np.zeros((5, 3))
    p = Path(waypoints=wps, joint_names=("a", "b", "c"))
    assert p.dof == 3
    assert p.num_waypoints == 5
    assert p.num_segments == 4


def test_path_status_enum_has_every_documented_value():
    expected = {
        "SUCCESS", "INVALID_INPUT",
        "START_IN_COLLISION", "GOAL_IN_COLLISION",
        "START_OUT_OF_LIMITS", "GOAL_OUT_OF_LIMITS",
        "NO_PATH_FOUND", "TIMEOUT", "MAX_ITERATIONS", "NUMERICAL_FAILURE",
        "IK_FAILED", "IK_DISCONTINUITY", "CARTESIAN_DEVIATION",
        "POST_PLAN_INVALID",
    }
    actual = {member.name for member in PathStatus}
    assert actual == expected


def test_path_plan_result_success_flag():
    p = Path(waypoints=np.zeros((2, 7)), joint_names=("j",) * 7)
    ok = PathPlanResult(
        status=PathStatus.SUCCESS, path=p, planner_used="x",
        iterations=1, elapsed_ms=1.0, diagnostics=PathDiagnostics(),
    )
    assert ok.success
    fail = PathPlanResult(
        status=PathStatus.NO_PATH_FOUND, path=None, planner_used="x",
        iterations=0, elapsed_ms=1.0, diagnostics=PathDiagnostics(),
    )
    assert not fail.success


def test_plan_options_defaults_are_frozen():
    opts = PlanOptions()
    with pytest.raises(Exception):  # frozen dataclass
        opts.max_iterations = 10000  # type: ignore[misc]


def test_path_validation_options_defaults_are_frozen():
    opts = PathValidationOptions()
    with pytest.raises(Exception):
        opts.joint_margin = 5.0  # type: ignore[misc]
