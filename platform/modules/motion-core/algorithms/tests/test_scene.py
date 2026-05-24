"""Tests for resolved.Scene - mutable runtime state + dynamic ACM."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pinocchio")
pytest.importorskip("coal")

from algorithms.descriptions import WorldDescription
from algorithms.resolved import CollisionModel, Scene


REPO_ROOT = Path(__file__).resolve().parents[1]


def world_path() -> Path:
    return REPO_ROOT / "configs" / "worlds" / "franka_table_world.yaml"


def _identity() -> np.ndarray:
    return np.eye(4)


def _xlate(x: float = 0.0, y: float = 0.0, z: float = 0.0) -> np.ndarray:
    T = np.eye(4)
    T[:3, 3] = [x, y, z]
    return T


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------


def test_from_world_initialises_object_poses_from_yaml():
    world = WorldDescription.from_yaml(world_path())
    scene = Scene.from_world(world)

    assert "matka" in scene.object_poses
    expected = world.object_pose("matka").as_matrix()
    np.testing.assert_array_equal(scene.object_poses["matka"], expected)


def test_from_world_with_collision_model_links_acm():
    world = WorldDescription.from_yaml(world_path())
    cm = CollisionModel.from_world(world)
    scene = Scene.from_world(world, collision_model=cm)

    assert scene.collision_model is cm


# ---------------------------------------------------------------------------
# Object pose updates
# ---------------------------------------------------------------------------


def test_set_object_pose_updates_state():
    world = WorldDescription.from_yaml(world_path())
    scene = Scene.from_world(world)

    new_pose = _xlate(z=0.3)
    scene.set_object_pose("matka", new_pose)

    np.testing.assert_array_equal(scene.get_object_pose("matka"), new_pose)


def test_set_object_pose_rejects_wrong_shape():
    world = WorldDescription.from_yaml(world_path())
    scene = Scene.from_world(world)

    with pytest.raises(ValueError, match=r"\(4,4\)"):
        scene.set_object_pose("matka", np.eye(3))


def test_set_object_pose_rejects_attached_object():
    world = WorldDescription.from_yaml(world_path())
    scene = Scene.from_world(world)

    scene.attach("matka", parent_frame="fr3_hand_tcp", T_parent_obj=_identity())
    with pytest.raises(ValueError, match="attached"):
        scene.set_object_pose("matka", _identity())


# ---------------------------------------------------------------------------
# Robot state
# ---------------------------------------------------------------------------


def test_set_and_get_robot_state_roundtrip():
    world = WorldDescription.from_yaml(world_path())
    scene = Scene.from_world(world)

    q = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785, 0.01])
    scene.set_robot_state("arm", q)
    np.testing.assert_array_equal(scene.get_robot_state("arm"), q)


def test_set_robot_state_rejects_non_1d():
    world = WorldDescription.from_yaml(world_path())
    scene = Scene.from_world(world)

    with pytest.raises(ValueError, match="1-D"):
        scene.set_robot_state("arm", np.zeros((3, 3)))


# ---------------------------------------------------------------------------
# Attach / detach lifecycle
# ---------------------------------------------------------------------------


def test_attach_removes_object_from_object_poses():
    world = WorldDescription.from_yaml(world_path())
    scene = Scene.from_world(world)

    assert "matka" in scene.object_poses
    scene.attach("matka", parent_frame="fr3_hand_tcp", T_parent_obj=_identity())
    assert "matka" not in scene.object_poses
    assert "matka" in scene.attached
    assert scene.attached["matka"].parent_frame == "fr3_hand_tcp"


def test_attach_auto_allows_collision_with_listed_partners():
    world = WorldDescription.from_yaml(world_path())
    scene = Scene.from_world(world)

    scene.attach(
        "matka",
        parent_frame="fr3_hand_tcp",
        T_parent_obj=_identity(),
        allow_collision_with=["fr3_leftfinger_0", "fr3_rightfinger_0"],
    )

    assert scene.is_pair_allowed("matka", "fr3_leftfinger_0")
    assert scene.is_pair_allowed("matka", "fr3_rightfinger_0")


def test_attach_rejects_double_attach():
    world = WorldDescription.from_yaml(world_path())
    scene = Scene.from_world(world)

    scene.attach("matka", parent_frame="fr3_hand_tcp", T_parent_obj=_identity())
    with pytest.raises(ValueError, match="already attached"):
        scene.attach("matka", parent_frame="fr3_hand_tcp", T_parent_obj=_identity())


def test_detach_returns_object_to_world_pose_and_revokes_allowances():
    world = WorldDescription.from_yaml(world_path())
    scene = Scene.from_world(world)

    scene.attach(
        "matka",
        parent_frame="fr3_hand_tcp",
        T_parent_obj=_identity(),
        allow_collision_with=["fr3_leftfinger_0"],
    )
    assert scene.is_pair_allowed("matka", "fr3_leftfinger_0")

    drop_pose = _xlate(x=0.6)
    scene.detach("matka", T_world_obj=drop_pose)

    assert "matka" not in scene.attached
    np.testing.assert_array_equal(scene.get_object_pose("matka"), drop_pose)
    # The dynamic allowance for matka<->finger should be gone.
    assert not scene.is_pair_allowed("matka", "fr3_leftfinger_0")


def test_detach_rejects_when_not_attached():
    world = WorldDescription.from_yaml(world_path())
    scene = Scene.from_world(world)

    with pytest.raises(KeyError, match="not attached"):
        scene.detach("matka", T_world_obj=_identity())


# ---------------------------------------------------------------------------
# Dynamic ACM
# ---------------------------------------------------------------------------


def test_allow_collision_is_direction_agnostic():
    world = WorldDescription.from_yaml(world_path())
    scene = Scene.from_world(world)

    scene.allow_collision("matka", "fr3_link7_0", reason="contact phase")

    assert scene.is_pair_allowed("matka", "fr3_link7_0")
    assert scene.is_pair_allowed("fr3_link7_0", "matka")


def test_disallow_collision_overrides_static_allow(tmp_path: Path):
    yaml_text = (
        "schema: dexsent.algorithms.world\n"
        "version: 2\n"
        "id: w\n"
        "world_frame: world\n"
        "robots:\n"
        "  - id: arm\n"
        "    robot_system: " + str((REPO_ROOT / "configs" / "robots" / "franka_fr3_with_franka_hand.yaml").resolve()) + "\n"
        "    namespace: null\n"
        "objects: []\n"
        "collision_matrix:\n"
        "  default_action: check\n"
        "  rules:\n"
        "    - { a: floor, b: wall, action: allow }\n"
    )
    p = tmp_path / "world.yaml"
    p.write_text(yaml_text)

    world = WorldDescription.from_yaml(p)
    cm = CollisionModel.from_world(world)
    scene = Scene.from_world(world, collision_model=cm)

    # Static says allow.
    assert scene.is_pair_allowed("floor", "wall")

    # Dynamic override.
    scene.disallow_collision("floor", "wall")
    assert not scene.is_pair_allowed("floor", "wall")


def test_known_object_ids_combines_placed_and_attached():
    world = WorldDescription.from_yaml(world_path())
    scene = Scene.from_world(world)

    assert scene.known_object_ids() == frozenset({"matka"})
    scene.attach("matka", parent_frame="fr3_hand_tcp", T_parent_obj=_identity())
    # Still tracked, just in a different bucket.
    assert scene.known_object_ids() == frozenset({"matka"})
