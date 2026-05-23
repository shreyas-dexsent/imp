"""Tests for collision distance queries."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("coal")
pytest.importorskip("pinocchio")

from algorithms.collision import ClearanceReport, DistanceReport, clearance, min_distance
from algorithms.collision.pairs import active_pairs
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


def _playground_setup():
    world = WorldDescription.from_yaml(
        REPO_ROOT / "configs" / "worlds" / "collision_playground_world.yaml"
    )
    cm = CollisionModel.from_world(world)
    scene = Scene.from_world(world, cm)
    system = world.robot("arm").robot_system
    model = KinematicModel.from_robot_system(system)
    home = system.named_joint_state("home")
    q = np.array([home[name] for name in model.active_joint_names], dtype=float)
    return model, scene, cm, q


def _isolate_pair(scene: Scene, cm: CollisionModel, target: tuple[str, str]) -> None:
    target = tuple(sorted(target))
    for pair in active_pairs(cm, scene):
        if pair != target:
            scene.allow_collision(*pair)


def test_min_distance_returns_typed_report():
    model, scene, q = _setup()

    report = min_distance(model, scene, q)

    assert isinstance(report, DistanceReport)
    assert report.pair is not None
    assert report.nearest_points is not None
    assert report.checked_pairs >= 1


def test_clearance_returns_pairs_below_threshold():
    model, scene, q = _setup()

    report = clearance(model, scene, q, threshold=0.05)

    assert isinstance(report, ClearanceReport)
    assert report.checked_pairs >= 1
    assert len(report.pairs_below_threshold) >= 1


def test_playground_known_sphere_distances():
    cases = [
        (("matka_overlap_a", "matka_overlap_b"), -0.05),
        (("matka_gap_a", "matka_gap_b"), 0.15),
        (("matka_touch_a", "matka_touch_b"), 0.00),
        (("matka_clearance_a", "matka_clearance_b"), 0.03),
    ]

    for pair, expected in cases:
        model, scene, cm, q = _playground_setup()
        _isolate_pair(scene, cm, pair)

        report = min_distance(model, scene, q)

        assert report.pair == tuple(sorted(pair))
        assert report.min_distance == pytest.approx(expected, abs=1e-6)
