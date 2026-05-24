# Copyright (c) 2026 DexSent Robotics Pvt Ltd.
# All Rights Reserved.
#
# SPDX-License-Identifier: LicenseRef-Proprietary
# See LICENSE file for details.
#
# Author: Shreyas <shreyas.kumar@dexsentrobotics.com>
# Proprietary and confidential.

"""Active collision-pair materialisation."""
from __future__ import annotations

from itertools import combinations
from typing import Dict, List, Tuple

from algorithms.resolved.collision_model import CollisionModel, _canonical_pair
from algorithms.resolved.scene import Scene


_PAIR_CACHE: Dict[tuple[int, int, str | None, int], List[Tuple[str, str]]] = {}


def active_pairs(
    collision_model: CollisionModel,
    scene: Scene,
    *,
    chain_id: str | None = None,
) -> list[tuple[str, str]]:
    """Return materialised geometry-name pairs that should be checked.

    Static allowed pairs, dynamic overlay allowances, and optional
    chain filtering are applied before the result is cached. The cache
    key includes `scene.collision_overlay.version`, so runtime ACM
    mutations invalidate the materialised list automatically.
    """
    key = (id(collision_model), id(scene), chain_id, scene.collision_overlay.version)
    cached = _PAIR_CACHE.get(key)
    if cached is not None:
        return list(cached)

    names = collision_model.object_names()
    chain_names = (
        collision_model.chain_geometry_names.get(chain_id)
        if chain_id is not None
        else None
    )

    pairs: list[tuple[str, str]] = []
    for a, b in combinations(names, 2):
        pair = _canonical_pair(a, b)

        if chain_names is not None and a not in chain_names and b not in chain_names:
            continue

        if pair in scene.collision_overlay.disallowed:
            pairs.append(pair)
            continue

        if pair in scene.collision_overlay.allowed:
            continue

        if pair in collision_model.static_allowed_pairs:
            continue

        pairs.append(pair)

    _PAIR_CACHE[key] = list(pairs)
    return pairs


def _clear_cache() -> None:
    """Clear the pair materialisation cache. Intended for tests."""
    _PAIR_CACHE.clear()
