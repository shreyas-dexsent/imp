"""Tests for collision geometry loading and processing."""
from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("coal")
pytest.importorskip("pinocchio")

from algorithms.descriptions import WorldDescription
from algorithms.resolved import CollisionModel
from algorithms.resolved.geometry_cache import cache_key


REPO_ROOT = Path(__file__).resolve().parents[1]


def franka_table_world_path() -> Path:
    return REPO_ROOT / "configs" / "worlds" / "franka_table_world.yaml"


def test_world_mesh_loads_as_real_mesh_not_placeholder_box():
    world = WorldDescription.from_yaml(franka_table_world_path())
    cm = CollisionModel.from_world(world)

    geom = cm.world_geom.geometryObjects[0].geometry

    assert type(geom).__name__ != "Box"


def test_geometry_cache_key_includes_scale():
    mesh_path = (
        REPO_ROOT.parent / "assets" / "objects" / "matka" / "visual" / "model.obj"
    )

    key_a = cache_key(mesh_path, {"type": "convex_decomposition"}, scale=(1, 1, 1))
    key_b = cache_key(mesh_path, {"type": "convex_decomposition"}, scale=(2, 1, 1))

    assert key_a != key_b


def test_raw_mesh_fallback_builds_without_processing():
    world = WorldDescription.from_yaml(franka_table_world_path())
    world.objects[0].collision.processing = None

    cm = CollisionModel.from_world(world)

    assert len(cm.world_geom.geometryObjects) == 1
