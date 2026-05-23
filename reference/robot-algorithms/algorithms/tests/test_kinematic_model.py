"""Tests for resolved.KinematicModel.

The suite covers composed Pinocchio models, mimic-joint expansion, chain
ordering, limit resolution, and cache invalidation when source URDF files
change.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pinocchio")

from algorithms.descriptions import RobotSystemDescription
from algorithms.resolved import KinematicModel
from algorithms.resolved.kinematic_model import (
    MimicRelation,
    _CACHE,
    _clear_cache,
    parse_mimic_relations,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def franka_with_hand_path() -> Path:
    return REPO_ROOT / "configs" / "robots" / "franka_fr3_with_franka_hand.yaml"


def franka_only_path() -> Path:
    return REPO_ROOT / "configs" / "robots" / "franka_fr3_robot_only.yaml"


@pytest.fixture(autouse=True)
def _isolate_cache():
    _clear_cache()
    yield
    _clear_cache()


# ---------------------------------------------------------------------------
# Mimic parsing
# ---------------------------------------------------------------------------


def test_mimic_parser_finds_franka_hand_relation():
    system = RobotSystemDescription.from_yaml(franka_with_hand_path())
    gripper_urdf = system.resolve_path(system.gripper.urdf_path)

    relations = parse_mimic_relations(gripper_urdf)
    assert relations == [
        MimicRelation(
            follower="fr3_finger_joint2",
            driver="fr3_finger_joint1",
            multiplier=1.0,
            offset=0.0,
        )
    ]


def test_mimic_parser_returns_empty_for_robot_with_no_mimic():
    system = RobotSystemDescription.from_yaml(franka_only_path())
    robot_urdf = system.resolve_path(system.robot.urdf_path)
    assert parse_mimic_relations(robot_urdf) == []


# ---------------------------------------------------------------------------
# Composed model + mimic expansion
# ---------------------------------------------------------------------------


def test_composed_model_has_full_robot_plus_gripper_dof():
    system = RobotSystemDescription.from_yaml(franka_with_hand_path())
    model = KinematicModel.from_robot_system(system)

    # 7 arm + 2 finger joints in the underlying Pinocchio model.
    assert model.pin_model.nq == 9
    assert len(model.full_joint_names) == 9
    assert "fr3_finger_joint1" in model.full_joint_names
    assert "fr3_finger_joint2" in model.full_joint_names


def test_active_joints_exclude_mimic_followers():
    system = RobotSystemDescription.from_yaml(franka_with_hand_path())
    model = KinematicModel.from_robot_system(system)

    # 8 active = 7 arm + 1 active finger; the mimic follower is excluded.
    assert len(model.active_joint_names) == 8
    assert "fr3_finger_joint1" in model.active_joint_names
    assert "fr3_finger_joint2" not in model.active_joint_names


def test_robot_only_system_has_no_gripper_dof():
    system = RobotSystemDescription.from_yaml(franka_only_path())
    model = KinematicModel.from_robot_system(system)

    assert model.pin_model.nq == 7
    assert len(model.active_joint_names) == 7
    assert len(model.full_joint_names) == 7


def test_expand_propagates_mimic_value():
    system = RobotSystemDescription.from_yaml(franka_with_hand_path())
    model = KinematicModel.from_robot_system(system)

    q_active = np.zeros(len(model.active_joint_names))
    finger_idx = model.active_joint_names.index("fr3_finger_joint1")
    q_active[finger_idx] = 0.03

    q_full = model.expand(q_active)
    assert q_full.shape == (9,)

    # The two finger joints share the value because mimic multiplier=1, offset=0.
    follower_q_idx = model.pin_model.joints[
        model.pin_model.getJointId("fr3_finger_joint2")
    ].idx_q
    driver_q_idx = model.pin_model.joints[
        model.pin_model.getJointId("fr3_finger_joint1")
    ].idx_q
    assert q_full[driver_q_idx] == 0.03
    assert q_full[follower_q_idx] == 0.03


def test_expand_rejects_wrong_shape():
    system = RobotSystemDescription.from_yaml(franka_with_hand_path())
    model = KinematicModel.from_robot_system(system)

    with pytest.raises(ValueError, match="expected"):
        model.expand(np.zeros(7))


# ---------------------------------------------------------------------------
# Chain indices
# ---------------------------------------------------------------------------


def test_chain_indices_map_into_active_q():
    system = RobotSystemDescription.from_yaml(franka_with_hand_path())
    model = KinematicModel.from_robot_system(system)

    arm_idx = model.chain_indices("arm")
    arm_with_gripper_idx = model.chain_indices("arm_with_gripper")

    assert arm_idx.shape == (7,)
    assert arm_with_gripper_idx.shape == (8,)
    # arm chain joints should be the first 7 active joints (per declaration order).
    assert arm_idx.tolist() == [
        model.active_joint_names.index(j)
        for j in system.chain("arm").joints
    ]


def test_chain_dof_matches_declared_joint_count():
    system = RobotSystemDescription.from_yaml(franka_with_hand_path())
    model = KinematicModel.from_robot_system(system)

    assert model.chain_dof("arm") == 7
    assert model.chain_dof("arm_with_gripper") == 8


# ---------------------------------------------------------------------------
# YAML TCP frames
# ---------------------------------------------------------------------------


def test_yaml_tcp_frame_is_added_when_missing_from_urdf():
    system = RobotSystemDescription.from_yaml(franka_only_path())
    model = KinematicModel.from_robot_system(system)

    assert model.pin_model.existFrame("fr3_link8")
    assert model.pin_model.existFrame("robot_tcp")

    parent_id = model.pin_model.getFrameId("fr3_link8")
    tcp_id = model.pin_model.getFrameId("robot_tcp")
    tcp_frame = model.pin_model.frames[tcp_id]

    assert tcp_frame.parentFrame == parent_id
    parent_frame = model.pin_model.frames[parent_id]
    relative = parent_frame.placement.inverse() * tcp_frame.placement
    np.testing.assert_allclose(
        np.asarray(relative.translation, dtype=float),
        np.array([0.0, 0.0, 0.12]),
    )


def test_yaml_tcp_frame_that_already_exists_in_urdf_is_preserved():
    system = RobotSystemDescription.from_yaml(franka_with_hand_path())
    model = KinematicModel.from_robot_system(system)

    assert model.pin_model.existFrame("fr3_hand_tcp")


# ---------------------------------------------------------------------------
# Limits API
# ---------------------------------------------------------------------------


def test_position_limits_are_urdf_defaults():
    system = RobotSystemDescription.from_yaml(franka_with_hand_path())
    model = KinematicModel.from_robot_system(system)

    lower, upper = model.position_limits("arm")
    # FR3 joint1 limits (per URDF): -2.9007..2.9007 rad
    assert lower.shape == (7,)
    assert upper.shape == (7,)
    assert np.isclose(lower[0], -2.9007, atol=1e-4)
    assert np.isclose(upper[0], 2.9007, atol=1e-4)


def test_position_limits_are_overridden_by_yaml():
    system = RobotSystemDescription.from_yaml(franka_with_hand_path())
    model = KinematicModel.from_robot_system(system)

    lower, upper = model.position_limits("arm_with_gripper")
    # The last entry (finger1) is overridden in YAML to [0.0, 0.04]; URDF would be wider.
    assert np.isclose(lower[-1], 0.0)
    assert np.isclose(upper[-1], 0.04)


def test_velocity_limits_default_from_urdf():
    system = RobotSystemDescription.from_yaml(franka_with_hand_path())
    model = KinematicModel.from_robot_system(system)

    v = model.velocity_limits("arm")
    assert v.shape == (7,)
    # FR3 joint1 velocity limit: 2.62 rad/s.
    assert np.isclose(v[0], 2.62, atol=1e-3)


def test_acceleration_and_jerk_are_yaml_sourced():
    system = RobotSystemDescription.from_yaml(franka_with_hand_path())
    model = KinematicModel.from_robot_system(system)

    a = model.acceleration_limits("arm")
    j = model.jerk_limits("arm")
    assert a.shape == (7,)
    assert j.shape == (7,)
    # The YAML sets joint1 accel=15.0 jerk=7500.0
    assert np.isclose(a[0], 15.0)
    assert np.isclose(j[0], 7500.0)


def test_effort_limits_default_from_urdf():
    system = RobotSystemDescription.from_yaml(franka_with_hand_path())
    model = KinematicModel.from_robot_system(system)

    e = model.effort_limits("arm")
    # FR3 joints 1-4 effort: 87 Nm; 5-7: 12 Nm.
    assert np.allclose(e[:4], 87.0)
    assert np.allclose(e[4:], 12.0)


def test_build_raises_when_chain_joint_lacks_accel_or_jerk(tmp_path: Path):
    """A chain joint without YAML accel+jerk should fail at build time."""
    # Copy the robot-only YAML and strip its joint_limits to trigger the error.
    src = franka_only_path()
    dst = tmp_path / "robot.yaml"
    text = src.read_text()
    # Replace the populated joint_limits with an empty dict.
    text = text.replace(
        "joint_limits:\n"
        "    fr3_joint1: { acceleration: 15.0, jerk: 7500.0 }\n"
        "    fr3_joint2: { acceleration: 7.5,  jerk: 3750.0 }\n"
        "    fr3_joint3: { acceleration: 10.0, jerk: 5000.0 }\n"
        "    fr3_joint4: { acceleration: 12.5, jerk: 6250.0 }\n"
        "    fr3_joint5: { acceleration: 15.0, jerk: 7500.0 }\n"
        "    fr3_joint6: { acceleration: 20.0, jerk: 10000.0 }\n"
        "    fr3_joint7: { acceleration: 20.0, jerk: 10000.0 }",
        "joint_limits: {}",
    )
    # The URDF reference is relative; make it absolute so it still resolves.
    absolute_urdf = src.parent / "../../../assets/robots/franka_fr3/urdf/franka_fr3.urdf"
    text = text.replace(
        "../../../assets/robots/franka_fr3/urdf/franka_fr3.urdf",
        str(absolute_urdf.resolve()),
    )
    dst.write_text(text)

    system = RobotSystemDescription.from_yaml(dst)
    with pytest.raises(ValueError, match="acceleration and jerk are required"):
        KinematicModel.from_robot_system(system)


# ---------------------------------------------------------------------------
# Cache behavior
# ---------------------------------------------------------------------------


def test_cache_returns_same_instance_for_same_inputs():
    system_a = RobotSystemDescription.from_yaml(franka_with_hand_path())
    system_b = RobotSystemDescription.from_yaml(franka_with_hand_path())

    model_a = KinematicModel.from_robot_system(system_a)
    model_b = KinematicModel.from_robot_system(system_b)

    assert model_a is model_b


def test_cache_invalidates_when_urdf_mtime_changes(tmp_path: Path):
    """Touching the URDF file should produce a different cached model."""
    # Make a self-contained copy of robot-only YAML + URDF.
    urdf_src = REPO_ROOT.parent / "assets" / "robots" / "franka_fr3" / "urdf" / "franka_fr3.urdf"
    urdf_dst = tmp_path / "franka_fr3.urdf"
    shutil.copy(urdf_src, urdf_dst)

    yaml_text = (
        "schema: dexsent.algorithms.robot_system\n"
        "version: 2\n"
        "id: cache_test\n"
        "name: cache test\n"
        "robot:\n"
        "  id: r\n"
        f"  urdf_path: {urdf_dst.name}\n"
        "  package_dirs: []\n"
        "  base_frame: base\n"
        "  joint_limits:\n"
        "    fr3_joint1: { acceleration: 1.0, jerk: 1.0 }\n"
        "    fr3_joint2: { acceleration: 1.0, jerk: 1.0 }\n"
        "    fr3_joint3: { acceleration: 1.0, jerk: 1.0 }\n"
        "    fr3_joint4: { acceleration: 1.0, jerk: 1.0 }\n"
        "    fr3_joint5: { acceleration: 1.0, jerk: 1.0 }\n"
        "    fr3_joint6: { acceleration: 1.0, jerk: 1.0 }\n"
        "    fr3_joint7: { acceleration: 1.0, jerk: 1.0 }\n"
        "  collision:\n"
        "    enabled: true\n"
        "    source: urdf\n"
        "kinematic_chains:\n"
        "  - id: arm\n"
        "    base_frame: base\n"
        "    tip_frame: fr3_link8\n"
        "    joints: [fr3_joint1, fr3_joint2, fr3_joint3, fr3_joint4, fr3_joint5, fr3_joint6, fr3_joint7]\n"
    )
    yaml_path = tmp_path / "robot.yaml"
    yaml_path.write_text(yaml_text)

    system_v1 = RobotSystemDescription.from_yaml(yaml_path)
    model_v1 = KinematicModel.from_robot_system(system_v1)
    assert len(_CACHE) == 1

    # Bump the URDF mtime.
    new_mtime = urdf_dst.stat().st_mtime + 10
    import os
    os.utime(urdf_dst, (new_mtime, new_mtime))

    system_v2 = RobotSystemDescription.from_yaml(yaml_path)
    model_v2 = KinematicModel.from_robot_system(system_v2)

    assert model_v1 is not model_v2
    assert len(_CACHE) == 2
