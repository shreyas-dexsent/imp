"""Tests for active collision-pair materialisation."""
from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("coal")
pytest.importorskip("pinocchio")

from algorithms.collision.pairs import _clear_cache, active_pairs
from algorithms.descriptions import WorldDescription
from algorithms.resolved import CollisionModel, Scene


REPO_ROOT = Path(__file__).resolve().parents[1]


def franka_table_world_path() -> Path:
    return REPO_ROOT / "configs" / "worlds" / "franka_table_world.yaml"


@pytest.fixture(autouse=True)
def _isolate_pair_cache():
    _clear_cache()
    yield
    _clear_cache()


def _scene_and_model():
    world = WorldDescription.from_yaml(franka_table_world_path())
    cm = CollisionModel.from_world(world)
    scene = Scene.from_world(world, cm)
    return scene, cm


def test_dynamic_overlay_allowance_excludes_pair():
    scene, cm = _scene_and_model()
    pair = active_pairs(cm, scene)[0]

    scene.allow_collision(*pair)

    assert pair not in active_pairs(cm, scene)


def test_dynamic_overlay_disallow_adds_back_static_allowed_pair():
    scene, cm = _scene_and_model()
    pair = active_pairs(cm, scene)[0]
    cm.static_allowed_pairs = frozenset({pair})
    _clear_cache()

    assert pair not in active_pairs(cm, scene)

    scene.disallow_collision(*pair)

    assert pair in active_pairs(cm, scene)


def test_overlay_version_invalidates_pair_cache():
    scene, cm = _scene_and_model()
    first = active_pairs(cm, scene)
    pair = first[0]

    scene.allow_collision(*pair)
    second = active_pairs(cm, scene)

    assert scene.collision_overlay.version > 0
    assert pair not in second


def test_chain_filter_reduces_materialised_pairs():
    scene, cm = _scene_and_model()

    all_pairs = active_pairs(cm, scene)
    arm_pairs = active_pairs(cm, scene, chain_id="arm")

    assert len(arm_pairs) < len(all_pairs)
    assert len(arm_pairs) > 0
