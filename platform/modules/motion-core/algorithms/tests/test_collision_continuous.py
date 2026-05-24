"""Tests for sampled edge collision checking."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("coal")
pytest.importorskip("pinocchio")

from algorithms.collision import (
    EdgeCollisionOptions,
    EdgeCollisionReport,
    check_edge_collision,
)
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


def test_edge_collision_returns_typed_report():
    model, scene, q = _setup()

    report = check_edge_collision(model, scene, q, q)

    assert isinstance(report, EdgeCollisionReport)
    assert report.checked_states == 1


def test_edge_collision_honours_max_joint_step():
    model, scene, q_a = _setup()
    q_b = q_a.copy()
    q_b[0] += 0.10

    report = check_edge_collision(
        model,
        scene,
        q_a,
        q_b,
        options=EdgeCollisionOptions(max_joint_step=0.02, include_endpoints=False),
    )

    assert report.checked_states >= 1
