"""Tests for resolved.geometry_cache.

The cache is content-addressed: identical (mesh_bytes, processing_config) ->
identical key -> cached result is reused without re-running compute_fn.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from algorithms.descriptions import ConvexDecompositionSpec
from algorithms.resolved import cache_key, get_or_compute


@pytest.fixture
def isolated_cache(tmp_path, monkeypatch):
    """Point the cache at a tmp_path for the duration of a test."""
    monkeypatch.setenv("ROBOT_ENGINE_GEOM_CACHE", str(tmp_path))
    yield tmp_path


def _write_mesh(tmp_path: Path, contents: bytes = b"binary-stl-like-payload") -> Path:
    p = tmp_path / "mesh.stl"
    p.write_bytes(contents)
    return p


# ---------------------------------------------------------------------------
# Keying
# ---------------------------------------------------------------------------


def test_cache_key_is_stable_for_same_inputs(isolated_cache):
    mesh = _write_mesh(isolated_cache)
    cfg = ConvexDecompositionSpec(type="convex_decomposition", max_hulls=8)

    k1 = cache_key(mesh, cfg)
    k2 = cache_key(mesh, cfg)
    assert k1 == k2
    assert len(k1) == 64  # sha256 hex


def test_cache_key_changes_when_mesh_bytes_change(isolated_cache):
    a = _write_mesh(isolated_cache, b"payload-A")
    b = _write_mesh(isolated_cache / "subdir", b"payload-B") if False else (isolated_cache / "b.stl")
    b.write_bytes(b"payload-B")
    cfg = ConvexDecompositionSpec(type="convex_decomposition", max_hulls=8)

    assert cache_key(a, cfg) != cache_key(b, cfg)


def test_cache_key_changes_when_processing_changes(isolated_cache):
    mesh = _write_mesh(isolated_cache)
    cfg_8 = ConvexDecompositionSpec(type="convex_decomposition", max_hulls=8)
    cfg_16 = ConvexDecompositionSpec(type="convex_decomposition", max_hulls=16)

    assert cache_key(mesh, cfg_8) != cache_key(mesh, cfg_16)


def test_cache_key_accepts_none_processing(isolated_cache):
    mesh = _write_mesh(isolated_cache)
    k = cache_key(mesh, None)
    assert len(k) == 64


# ---------------------------------------------------------------------------
# Hit / miss
# ---------------------------------------------------------------------------


def test_compute_runs_on_miss_then_hits_on_second_call(isolated_cache):
    mesh = _write_mesh(isolated_cache)
    cfg = ConvexDecompositionSpec(type="convex_decomposition", max_hulls=4)

    call_count = {"n": 0}

    def compute(path, config):
        call_count["n"] += 1
        return {"hulls": ["A", "B"], "from": str(path), "max_hulls": config.max_hulls}

    v1 = get_or_compute(mesh, cfg, compute)
    v2 = get_or_compute(mesh, cfg, compute)

    assert call_count["n"] == 1
    assert v1 == v2
    assert v1["max_hulls"] == 4


def test_mesh_change_invalidates_cache(isolated_cache):
    mesh = _write_mesh(isolated_cache, b"first")
    cfg = ConvexDecompositionSpec(type="convex_decomposition", max_hulls=4)

    call_count = {"n": 0}

    def compute(path, config):
        call_count["n"] += 1
        return path.read_bytes()

    v1 = get_or_compute(mesh, cfg, compute)
    mesh.write_bytes(b"second")
    v2 = get_or_compute(mesh, cfg, compute)

    assert v1 == b"first"
    assert v2 == b"second"
    assert call_count["n"] == 2
