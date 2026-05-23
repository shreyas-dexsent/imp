"""Tests for discrete collision checking."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("coal")
pytest.importorskip("pinocchio")

from algorithms.collision import CollisionOptions, ContactReport, is_in_collision
from algorithms.descriptions import WorldDescription
from algorithms.resolved import CollisionModel, KinematicModel, Scene


REPO_ROOT = Path(__file__).resolve().parents[1]


def _setup():
    world = WorldDescription.from_yaml(
        REPO_ROOT / "configs" / "worlds" / "franka_table_world.yaml"
    )
    cm = CollisionModel.from_world(world)
    scene = Scene.from_world(world, cm)
    system = world.robot("arm").robot_system
    model = KinematicModel.from_robot_system(system)
    home = system.named_joint_state("home")
    q = np.array([home[name] for name in model.active_joint_names], dtype=float)
    return model, scene, q


def test_is_in_collision_returns_contact_report():
    model, scene, q = _setup()

    report = is_in_collision(model, scene, q)

    assert isinstance(report, ContactReport)
    assert isinstance(report.in_collision, bool)
    assert report.checked_pairs >= 1


def test_is_in_collision_can_collect_contacts():
    model, scene, q = _setup()

    report = is_in_collision(
        model,
        scene,
        q,
        options=CollisionOptions(stop_at_first_contact=True, collect_contacts=True),
    )

    assert report.in_collision
    assert len(report.contacts) >= 1
    assert report.contacts[0].penetration >= 0.0


def test_query_requires_collision_model_on_scene():
    model, scene, q = _setup()
    scene.collision_model = None

    with pytest.raises(ValueError, match="collision_model"):
        is_in_collision(model, scene, q)
