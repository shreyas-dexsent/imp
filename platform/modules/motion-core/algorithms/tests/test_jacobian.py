"""Tests for kinematics.jacobian.

Verifies:
* Shape matches (6, active_dof) regardless of how many full DOF the
  underlying Pinocchio model has (mimic folding via active_to_full).
* Reference-frame conventions LOCAL / WORLD / LOCAL_WORLD_ALIGNED differ
  predictably.
* Numerical Jacobian (finite-difference twist) matches analytical Jacobian
  to small tolerance.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pinocchio")

from algorithms.descriptions import RobotSystemDescription
from algorithms.kinematics.fk import fk_local
from algorithms.kinematics.jacobian import jacobian
from algorithms.resolved import KinematicModel
from algorithms.resolved.kinematic_model import _clear_cache


REPO_ROOT = Path(__file__).resolve().parents[1]


def franka_with_hand_path() -> Path:
    return REPO_ROOT / "configs" / "robots" / "franka_fr3_with_franka_hand.yaml"


def franka_robot_only_path() -> Path:
    return REPO_ROOT / "configs" / "robots" / "franka_fr3_robot_only.yaml"


@pytest.fixture(autouse=True)
def _isolate_cache():
    _clear_cache()
    yield
    _clear_cache()


def _home_q(model: KinematicModel) -> np.ndarray:
    home = model.system.named_joint_state("home")
    return np.array([home[name] for name in model.active_joint_names])


# ---------------------------------------------------------------------------
# Shape & basic correctness
# ---------------------------------------------------------------------------


def test_jacobian_shape_is_6_by_active_dof():
    system = RobotSystemDescription.from_yaml(franka_with_hand_path())
    model = KinematicModel.from_robot_system(system)
    q = _home_q(model)

    J = jacobian(model, q, "fr3_hand_tcp")
    assert J.shape == (6, len(model.active_joint_names))
    # Pinocchio model has 9 full DOF; we must NOT return 9 columns.
    assert model.pin_model.nq == 9
    assert J.shape[1] == 8


def test_jacobian_accepts_yaml_injected_tcp_frame():
    system = RobotSystemDescription.from_yaml(franka_robot_only_path())
    model = KinematicModel.from_robot_system(system)
    q = _home_q(model)

    J = jacobian(model, q, "robot_tcp")

    assert J.shape == (6, len(model.active_joint_names))


def test_jacobian_unknown_frame_raises():
    system = RobotSystemDescription.from_yaml(franka_with_hand_path())
    model = KinematicModel.from_robot_system(system)
    q = _home_q(model)

    with pytest.raises(KeyError, match="resolved kinematic model"):
        jacobian(model, q, "no_such_frame")


def test_jacobian_wrong_q_shape_raises():
    system = RobotSystemDescription.from_yaml(franka_with_hand_path())
    model = KinematicModel.from_robot_system(system)
    q = _home_q(model)

    with pytest.raises(ValueError, match="q_active has shape"):
        jacobian(model, q[:-1], "fr3_hand_tcp")


def test_jacobian_unknown_reference_raises():
    system = RobotSystemDescription.from_yaml(franka_with_hand_path())
    model = KinematicModel.from_robot_system(system)
    q = _home_q(model)

    with pytest.raises(ValueError, match="unknown reference"):
        jacobian(model, q, "fr3_link8", reference="wrong")


# ---------------------------------------------------------------------------
# Reference-frame conventions
# ---------------------------------------------------------------------------


def test_reference_frames_produce_different_jacobians_at_non_identity_pose():
    """At a pose where the body frame is rotated wrt world, LOCAL / WORLD /
    LOCAL_WORLD_ALIGNED must each give a different Jacobian. (Home config of
    FR3 has fr3_link8 rotated relative to base.)"""
    system = RobotSystemDescription.from_yaml(franka_with_hand_path())
    model = KinematicModel.from_robot_system(system)
    q = _home_q(model)

    J_local = jacobian(model, q, "fr3_link8", reference="local")
    J_world = jacobian(model, q, "fr3_link8", reference="world")
    J_lwa = jacobian(model, q, "fr3_link8", reference="local_world_aligned")

    assert not np.allclose(J_local, J_world)
    assert not np.allclose(J_local, J_lwa)
    # WORLD and LOCAL_WORLD_ALIGNED differ in linear-velocity rows only.
    assert not np.allclose(J_world, J_lwa)


def test_reference_frames_agree_at_identity_orientation():
    """At a config where the queried frame's orientation matches base/world,
    LOCAL and LOCAL_WORLD_ALIGNED should agree (orientation part identical).
    The base_frame itself is always at identity, so its rotational Jacobian
    columns are zero in all reference frames."""
    system = RobotSystemDescription.from_yaml(franka_with_hand_path())
    model = KinematicModel.from_robot_system(system)
    q = _home_q(model)

    J_local = jacobian(model, q, system.robot.base_frame, reference="local")
    J_lwa = jacobian(model, q, system.robot.base_frame, reference="local_world_aligned")
    np.testing.assert_allclose(J_local, J_lwa, atol=1e-10)


# ---------------------------------------------------------------------------
# Finite-difference cross-check (the real correctness test)
# ---------------------------------------------------------------------------


def test_analytical_jacobian_matches_finite_difference_local_world_aligned():
    """Numerical twist from FK at q+delta should agree with J @ deltaq.

    Uses LOCAL_WORLD_ALIGNED: rotation rows give axis-angle in world axes,
    translation rows give position difference in world (= base) frame.
    """
    system = RobotSystemDescription.from_yaml(franka_with_hand_path())
    model = KinematicModel.from_robot_system(system)
    q0 = _home_q(model)
    frame = "fr3_hand_tcp"

    J = jacobian(model, q0, frame, reference="local_world_aligned")
    T0 = fk_local(model, q0, frame)

    rng = np.random.default_rng(0)
    dq = rng.normal(scale=1e-5, size=q0.shape)

    T1 = fk_local(model, q0 + dq, frame)
    dp = T1[:3, 3] - T0[:3, 3]
    dR = T1[:3, :3] @ T0[:3, :3].T - np.eye(3)
    dw = np.array([dR[2, 1] - dR[1, 2], dR[0, 2] - dR[2, 0], dR[1, 0] - dR[0, 1]]) / 2.0

    twist_numerical = np.concatenate([dp, dw])
    twist_analytical = J @ dq

    np.testing.assert_allclose(twist_numerical, twist_analytical, atol=1e-7)
