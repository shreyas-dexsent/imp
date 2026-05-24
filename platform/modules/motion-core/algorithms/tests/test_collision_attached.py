"""Tests for attached-object collision helpers."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pinocchio")

from algorithms.collision.attached_object import attached_object_world_pose
from algorithms.descriptions import WorldDescription
from algorithms.kinematics import fk_local
from algorithms.resolved import KinematicModel, Scene
from algorithms.resolved.scene import AttachedObject


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_attached_object_pose_is_parent_fk_times_local_offset():
    world = WorldDescription.from_yaml(
        REPO_ROOT / "configs" / "worlds" / "franka_table_world.yaml"
    )
    system = world.robot("arm").robot_system
    model = KinematicModel.from_robot_system(system)
    home = system.named_joint_state("home")
    q = np.array([home[name] for name in model.active_joint_names], dtype=float)

    T_parent_obj = np.eye(4)
    T_parent_obj[0, 3] = 0.10
    attached = AttachedObject(
        object_id="matka",
        parent_frame="fr3_hand_tcp",
        T_parent_obj=T_parent_obj,
    )

    expected = fk_local(model, q, "fr3_hand_tcp") @ T_parent_obj

    np.testing.assert_allclose(
        attached_object_world_pose(model, q, attached),
        expected,
    )


def test_scene_attach_and_detach_bump_overlay_version():
    world = WorldDescription.from_yaml(
        REPO_ROOT / "configs" / "worlds" / "franka_table_world.yaml"
    )
    scene = Scene.from_world(world)
    start_version = scene.collision_overlay.version

    scene.attach("matka", "fr3_hand_tcp", np.eye(4), allow_collision_with=["finger"])
    attached_version = scene.collision_overlay.version
    scene.detach("matka", np.eye(4))

    assert attached_version > start_version
    assert scene.collision_overlay.version > attached_version
