"""Tests for kinematics.fk: local FK, world FK, and batch variants.

The suite covers composed Pinocchio models, mimic-joint expansion, FK as the
runtime source of robot-link pose data, and the stateless operation contract
that prevents shared Pinocchio scratch data from leaking across robots.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pinocchio")

from algorithms.descriptions import RobotSystemDescription, WorldDescription
from algorithms.kinematics.fk import fk, fk_local, fk_local_many, fk_many
from algorithms.resolved import KinematicModel, Scene
from algorithms.resolved.kinematic_model import _clear_cache


REPO_ROOT = Path(__file__).resolve().parents[1]


def franka_with_hand_path() -> Path:
    return REPO_ROOT / "configs" / "robots" / "franka_fr3_with_franka_hand.yaml"


def franka_only_path() -> Path:
    return REPO_ROOT / "configs" / "robots" / "franka_fr3_robot_only.yaml"


def single_world_path() -> Path:
    return REPO_ROOT / "configs" / "worlds" / "franka_table_world.yaml"


def two_robot_world_path() -> Path:
    return REPO_ROOT / "configs" / "worlds" / "two_franka_table_world.yaml"


@pytest.fixture(autouse=True)
def _isolate_cache():
    _clear_cache()
    yield
    _clear_cache()


def _home_q(model: KinematicModel) -> np.ndarray:
    home = model.system.named_joint_state("home")
    return np.array([home[name] for name in model.active_joint_names])


# ---------------------------------------------------------------------------
# Local FK
# ---------------------------------------------------------------------------


def test_fk_local_returns_4x4_in_base_frame():
    system = RobotSystemDescription.from_yaml(franka_with_hand_path())
    model = KinematicModel.from_robot_system(system)
    q = _home_q(model)

    T = fk_local(model, q, "fr3_link8")
    assert T.shape == (4, 4)
    assert np.allclose(T[3], [0.0, 0.0, 0.0, 1.0])

    # Rotation block is orthonormal.
    R = T[:3, :3]
    np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-10)


def test_fk_local_base_frame_is_identity_for_q_irrelevant_base():
    """The robot's declared base_frame should always be identity in the
    base-frame-relative output, regardless of q."""
    system = RobotSystemDescription.from_yaml(franka_with_hand_path())
    model = KinematicModel.from_robot_system(system)

    for q_seed in (np.zeros(8), _home_q(model), np.full(8, 0.1)):
        T_base = fk_local(model, q_seed, system.robot.base_frame)
        np.testing.assert_allclose(T_base, np.eye(4), atol=1e-10)


def test_fk_local_gripper_mount_offset_matches_yaml():
    """fr3_hand should sit exactly at gripper.mount above fr3_link8, in link8's frame."""
    system = RobotSystemDescription.from_yaml(franka_with_hand_path())
    model = KinematicModel.from_robot_system(system)
    q = _home_q(model)

    T_link8 = fk_local(model, q, "fr3_link8")
    T_hand = fk_local(model, q, "fr3_hand")

    T_link8_to_hand = np.linalg.inv(T_link8) @ T_hand
    expected_mount = system.gripper.mount.as_matrix()
    np.testing.assert_allclose(T_link8_to_hand, expected_mount, atol=1e-10)


def test_fk_local_resolves_yaml_only_robot_tcp():
    system = RobotSystemDescription.from_yaml(franka_only_path())
    model = KinematicModel.from_robot_system(system)
    q = _home_q(model)

    T_link8 = fk_local(model, q, "fr3_link8")
    T_tcp = fk_local(model, q, "robot_tcp")
    expected = system.tcp("robot_tcp").transform.as_matrix()

    np.testing.assert_allclose(np.linalg.inv(T_link8) @ T_tcp, expected, atol=1e-10)


def test_fk_local_propagates_mimic_finger():
    """Setting fr3_finger_joint1 should move fr3_finger_joint2's child link by the same amount."""
    system = RobotSystemDescription.from_yaml(franka_with_hand_path())
    model = KinematicModel.from_robot_system(system)

    q = _home_q(model)
    finger_idx = model.active_joint_names.index("fr3_finger_joint1")
    q[finger_idx] = 0.03

    T_left = fk_local(model, q, "fr3_leftfinger")
    T_right = fk_local(model, q, "fr3_rightfinger")
    # Both fingers should symmetrically open by the same amount (mimic = 1*driver).
    # Their offsets along the open axis from fr3_hand should be equal in magnitude.
    T_hand = fk_local(model, q, "fr3_hand")
    left_offset = np.linalg.norm((T_left - T_hand)[:3, 3])
    right_offset = np.linalg.norm((T_right - T_hand)[:3, 3])
    np.testing.assert_allclose(left_offset, right_offset, atol=1e-10)


def test_fk_local_unknown_frame_raises():
    system = RobotSystemDescription.from_yaml(franka_with_hand_path())
    model = KinematicModel.from_robot_system(system)

    with pytest.raises(KeyError, match="resolved kinematic model"):
        fk_local(model, _home_q(model), "no_such_frame")


# ---------------------------------------------------------------------------
# World FK
# ---------------------------------------------------------------------------


def test_fk_world_equals_base_pose_times_local():
    world = WorldDescription.from_yaml(single_world_path())
    scene = Scene.from_world(world)
    model = KinematicModel.from_robot_system(world.robot("arm").robot_system)
    q = _home_q(model)

    T_world = fk(scene, "arm", q, "fr3_hand_tcp")
    T_base_to_frame = fk_local(model, q, "fr3_hand_tcp")
    T_world_base = world.robot("arm").base_pose.as_matrix()

    np.testing.assert_allclose(T_world, T_world_base @ T_base_to_frame, atol=1e-12)


def test_fk_world_independent_per_robot_in_multi_world():
    """Two FR3s at different base placements must produce different world poses
    for the same chain and q. Critical test for per-call Pinocchio data."""
    world = WorldDescription.from_yaml(two_robot_world_path())
    scene = Scene.from_world(world)
    model = KinematicModel.from_robot_system(world.robot("left_arm").robot_system)
    q = _home_q(model)

    T_left = fk(scene, "left_arm", q, "fr3_hand_tcp")
    T_right = fk(scene, "right_arm", q, "fr3_hand_tcp")

    # Same robot system, same q: identical pose in each robot's BASE frame.
    # Different base_pose in the world: different WORLD poses.
    np.testing.assert_allclose(T_left[:3, 3] - T_right[:3, 3],
                                np.array([-1.2, 0.0, 0.0]), atol=1e-10)


def test_fk_world_no_base_pose_means_identity():
    """A WorldRobotDescription with no base_pose is at world origin."""
    system_path = franka_with_hand_path()
    yaml = (
        "schema: dexsent.algorithms.world\n"
        "version: 2\n"
        "id: w\n"
        "world_frame: world\n"
        f"robots:\n"
        f"  - id: arm\n"
        f"    robot_system: {system_path}\n"
        f"    namespace: null\n"
        "objects: []\n"
        "collision_matrix:\n"
        "  default_action: check\n"
        "  rules: []\n"
    )

    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        f.write(yaml)
        path = Path(f.name)
    try:
        world = WorldDescription.from_yaml(path)
        scene = Scene.from_world(world)
        model = KinematicModel.from_robot_system(world.robot("arm").robot_system)
        q = _home_q(model)

        T_world = fk(scene, "arm", q, "fr3_hand_tcp")
        T_local = fk_local(model, q, "fr3_hand_tcp")
        np.testing.assert_allclose(T_world, T_local, atol=1e-12)
    finally:
        path.unlink()


def test_fk_unknown_robot_raises():
    world = WorldDescription.from_yaml(single_world_path())
    scene = Scene.from_world(world)
    with pytest.raises(KeyError, match="robot not found"):
        fk(scene, "nonexistent", np.zeros(8), "fr3_link8")


# ---------------------------------------------------------------------------
# Batch variants
# ---------------------------------------------------------------------------


def test_fk_local_many_matches_singleton():
    system = RobotSystemDescription.from_yaml(franka_with_hand_path())
    model = KinematicModel.from_robot_system(system)
    q = _home_q(model)

    frames = ["fr3_link0", "fr3_link4", "fr3_link8", "fr3_hand", "fr3_hand_tcp"]
    batch = fk_local_many(model, q, frames)

    for f in frames:
        np.testing.assert_allclose(batch[f], fk_local(model, q, f), atol=1e-12)


def test_fk_many_matches_singleton():
    world = WorldDescription.from_yaml(single_world_path())
    scene = Scene.from_world(world)
    model = KinematicModel.from_robot_system(world.robot("arm").robot_system)
    q = _home_q(model)

    frames = ["fr3_link0", "fr3_link8", "fr3_hand_tcp"]
    batch = fk_many(scene, "arm", q, frames)

    for f in frames:
        np.testing.assert_allclose(batch[f], fk(scene, "arm", q, f), atol=1e-12)


# ---------------------------------------------------------------------------
# Fresh pin.Data per call (no cross-robot corruption)
# ---------------------------------------------------------------------------


def test_two_robots_sharing_cached_model_dont_corrupt_each_other():
    """Both left_arm and right_arm use the same cached KinematicModel; if
    we accidentally shared one pin.Data, interleaved calls would leak state
    across them. Run an interleaved sequence and verify each result is the
    same as the standalone call."""
    world = WorldDescription.from_yaml(two_robot_world_path())
    scene = Scene.from_world(world)
    model = KinematicModel.from_robot_system(world.robot("left_arm").robot_system)
    q1 = _home_q(model)
    q2 = q1 + 0.1

    # Interleaved
    T_left_q1_first  = fk(scene, "left_arm",  q1, "fr3_hand_tcp")
    T_right_q2       = fk(scene, "right_arm", q2, "fr3_hand_tcp")
    T_left_q1_second = fk(scene, "left_arm",  q1, "fr3_hand_tcp")

    # The two left-arm calls must be identical regardless of the right-arm call between them.
    np.testing.assert_allclose(T_left_q1_first, T_left_q1_second, atol=1e-12)

    # And they must NOT equal the right-arm result (different base_pose).
    assert not np.allclose(T_left_q1_first, T_right_q2)
