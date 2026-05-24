# Copyright (c) 2026 DexSent Robotics Pvt Ltd.
# All Rights Reserved.
#
# SPDX-License-Identifier: LicenseRef-Proprietary
# See LICENSE file for details.
#
# Author: Shreyas <shreyas.kumar@dexsentrobotics.com>
# Proprietary and confidential.

"""Minimum-distance and clearance collision queries."""
from __future__ import annotations

import numpy as np

from algorithms.collision._runtime import geometry_entries
from algorithms.collision.options import CollisionOptions
from algorithms.collision.pairs import active_pairs
from algorithms.collision.types import ClearanceReport, DistanceReport
from algorithms.resolved.kinematic_model import KinematicModel
from algorithms.resolved.scene import Scene


def min_distance(
    model: KinematicModel,
    scene: Scene,
    q: np.ndarray,
    *,
    chain_id: str | None = None,
    options: CollisionOptions | None = None,
) -> DistanceReport:
    """Return the minimum signed distance over active geometry pairs."""
    import coal

    if scene.collision_model is None:
        raise ValueError("scene.collision_model is required for collision queries")

    opts = options or CollisionOptions()
    if opts.broadphase != "naive":
        raise ValueError(f"unsupported broadphase: {opts.broadphase!r}")

    entries = geometry_entries(model, scene, q)
    pairs = active_pairs(scene.collision_model, scene, chain_id=chain_id)

    request = coal.DistanceRequest()
    best_distance = float("inf")
    best_pair: tuple[str, str] | None = None
    best_points: tuple[np.ndarray, np.ndarray] | None = None
    checked_pairs = 0

    for pair in pairs:
        a, b = pair
        if a not in entries or b not in entries:
            continue

        checked_pairs += 1
        result = coal.DistanceResult()
        distance = float(
            coal.distance(
                entries[a].geometry,
                entries[a].placement,
                entries[b].geometry,
                entries[b].placement,
                request,
                result,
            )
        )

        if distance < best_distance:
            best_distance = distance
            best_pair = pair
            best_points = (
                np.asarray(result.getNearestPoint1(), dtype=float),
                np.asarray(result.getNearestPoint2(), dtype=float),
            )

    return DistanceReport(
        min_distance=best_distance,
        pair=best_pair,
        nearest_points=best_points,
        checked_pairs=checked_pairs,
    )


def clearance(
    model: KinematicModel,
    scene: Scene,
    q: np.ndarray,
    threshold: float,
    *,
    chain_id: str | None = None,
    options: CollisionOptions | None = None,
) -> ClearanceReport:
    """Return minimum distance clipped at `threshold` plus close pairs."""
    import coal

    if scene.collision_model is None:
        raise ValueError("scene.collision_model is required for collision queries")

    opts = options or CollisionOptions()
    if opts.broadphase != "naive":
        raise ValueError(f"unsupported broadphase: {opts.broadphase!r}")

    entries = geometry_entries(model, scene, q)
    pairs = active_pairs(scene.collision_model, scene, chain_id=chain_id)

    request = coal.DistanceRequest()
    min_seen = float("inf")
    below: list[tuple[str, str]] = []
    checked_pairs = 0

    for pair in pairs:
        a, b = pair
        if a not in entries or b not in entries:
            continue

        checked_pairs += 1
        result = coal.DistanceResult()
        distance = float(
            coal.distance(
                entries[a].geometry,
                entries[a].placement,
                entries[b].geometry,
                entries[b].placement,
                request,
                result,
            )
        )

        min_seen = min(min_seen, distance)
        if distance < threshold:
            below.append(pair)

    return ClearanceReport(
        clearance=min(min_seen, float(threshold)),
        pairs_below_threshold=below,
        checked_pairs=checked_pairs,
    )
