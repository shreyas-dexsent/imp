# Copyright (c) 2026 DexSent Robotics Pvt Ltd.
# All Rights Reserved.
#
# SPDX-License-Identifier: LicenseRef-Proprietary
# See LICENSE file for details.
#
# Author: Shreyas <shreyas.kumar@dexsentrobotics.com>
# Proprietary and confidential.

"""Content-addressed disk cache for processed geometry.

V-HACD (and similar) convex decomposition is expensive (seconds to
minutes per mesh) and deterministic for a given `(mesh, processing
config)` pair. This module caches the post-processed result keyed on a
sha256 of the inputs, so the heavy work runs once per pair on a given
machine and reuses the result on every subsequent call.

The cache value is opaque to this module - callers pass a `compute_fn`
that returns a pickle-serialisable result.

Cache location: `~/.cache/dexsent/algorithms/geom/`, or the path in
the `ROBOT_ENGINE_GEOM_CACHE` environment variable when set (the test
suite uses this for isolation).
"""
from __future__ import annotations

import hashlib
import json
import os
import pickle
from pathlib import Path
from typing import Any, Callable

from pydantic import BaseModel


DEFAULT_CACHE_DIR = Path.home() / ".cache" / "dexsent" / "algorithms" / "geom"


def _cache_dir() -> Path:
    """Return the active cache directory, creating it if needed."""
    override = os.environ.get("ROBOT_ENGINE_GEOM_CACHE")
    path = Path(override) if override else DEFAULT_CACHE_DIR
    path.mkdir(parents=True, exist_ok=True)
    return path


def _hash_bytes(*parts: bytes) -> str:
    """Compute the hex sha256 over a sequence of byte blobs."""
    h = hashlib.sha256()
    for part in parts:
        h.update(part)
    return h.hexdigest()


def cache_key(mesh_path: Path, processing_config: Any, *, scale: Any = None) -> str:
    """Build the cache key for one `(mesh, processing_config)` pair.

    The key is the hex sha256 of the mesh file bytes concatenated with a
    deterministic JSON encoding of the processing configuration. Identical
    inputs always produce identical keys; changing either input produces
    a different key, naturally invalidating the cache.

    Parameters
    ----------
    mesh_path : Path
        Path to the source mesh file.
    processing_config : Any
        Processing configuration. May be a pydantic model (model_dump is
        used), `None`, or any value JSON-serialisable with `default=str`.
    """
    mesh_bytes = Path(mesh_path).read_bytes()

    if isinstance(processing_config, BaseModel):
        config_dict = processing_config.model_dump(mode="json")
    elif processing_config is None:
        config_dict = None
    else:
        config_dict = processing_config

    payload = {
        "processing": config_dict,
        "scale": scale,
    }
    config_bytes = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return _hash_bytes(mesh_bytes, config_bytes)


def get_or_compute(
    mesh_path: Path,
    processing_config: Any,
    compute_fn: Callable[[Path, Any], Any],
    *,
    scale: Any = None,
) -> Any:
    """Return the cached result for `(mesh, config)`, or compute and cache it.

    `compute_fn(mesh_path, processing_config)` must return a
    pickle-serialisable value. The result is stored under
    `<cache_dir>/<key>.pkl` atomically.
    """
    key = cache_key(mesh_path, processing_config, scale=scale)
    cache_path = _cache_dir() / f"{key}.pkl"

    if cache_path.exists():
        with cache_path.open("rb") as f:
            return pickle.load(f)

    value = compute_fn(mesh_path, processing_config)

    tmp_path = cache_path.with_suffix(".pkl.tmp")
    with tmp_path.open("wb") as f:
        pickle.dump(value, f)
    tmp_path.replace(cache_path)

    return value


def clear_cache() -> None:
    """Remove every entry from the active cache directory. Intended for tests."""
    cache_dir = _cache_dir()
    for entry in cache_dir.iterdir():
        if entry.is_file():
            entry.unlink()
