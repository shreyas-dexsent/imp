# Copyright (c) 2026 DexSent Robotics Pvt Ltd.
# All Rights Reserved.
#
# SPDX-License-Identifier: LicenseRef-Proprietary
# See LICENSE file for details.
#
# Author: Shreyas <shreyas.kumar@dexsentrobotics.com>
# Proprietary and confidential.

"""Discrete collision checks."""
from __future__ import annotations

import numpy as np

from algorithms.collision._runtime import (
    geometry_entries,
    skipped_pair_count,
)
from algorithms.collision.options import CollisionOptions
from algorithms.collision.pairs import active_pairs
from algorithms.collision.types import Contact, ContactReport
from algorithms.resolved.kinematic_model import KinematicModel
from algorithms.resolved.scene import Scene


def is_in_collision(
    model: KinematicModel,
    scene: Scene,
    q: np.ndarray,
    *,
    chain_id: str | None = None,
    options: CollisionOptions | None = None,
) -> ContactReport:
    """Check whether the active geometry pairs collide at `q`."""
    import coal

    if scene.collision_model is None:
        raise ValueError("scene.collision_model is required for collision queries")

    opts = options or CollisionOptions()
    if opts.broadphase != "naive":
        raise ValueError(f"unsupported broadphase: {opts.broadphase!r}")

    entries = geometry_entries(model, scene, q)
    pairs = active_pairs(scene.collision_model, scene, chain_id=chain_id)
    skipped_pairs = skipped_pair_count(scene.collision_model, len(pairs))

    contacts: list[Contact] = []
    checked_pairs = 0
    found_collision = False

    request = coal.CollisionRequest()
    request.enable_contact = opts.collect_contacts
    request.num_max_contacts = 16 if opts.collect_contacts else 1

    for pair in pairs:
        a, b = pair
        if a not in entries or b not in entries:
            continue

        checked_pairs += 1
        result = coal.CollisionResult()
        coal.collide(
            entries[a].geometry,
            entries[a].placement,
            entries[b].geometry,
            entries[b].placement,
            request,
            result,
        )

        if not result.isCollision():
            continue

        found_collision = True
        if opts.collect_contacts:
            contacts.extend(_contacts_from_result(pair, result))

        if opts.stop_at_first_contact:
            return ContactReport(
                in_collision=True,
                contacts=contacts,
                checked_pairs=checked_pairs,
                skipped_pairs=skipped_pairs,
            )

    return ContactReport(
        in_collision=found_collision,
        contacts=contacts,
        checked_pairs=checked_pairs,
        skipped_pairs=skipped_pairs,
    )


def _contacts_from_result(
    pair: tuple[str, str],
    result,
) -> list[Contact]:
    contacts: list[Contact] = []
    for idx in range(result.numContacts()):
        raw = result.getContact(idx)
        point_a = np.asarray(raw.getNearestPoint1(), dtype=float)
        point_b = np.asarray(raw.getNearestPoint2(), dtype=float)
        normal = np.asarray(raw.normal, dtype=float)
        norm = float(np.linalg.norm(normal))
        if norm > 1e-12:
            normal = normal / norm
        contacts.append(
            Contact(
                pair=pair,
                point_a=point_a,
                point_b=point_b,
                normal=normal,
                penetration=max(0.0, -float(raw.penetration_depth)),
            )
        )
    return contacts
