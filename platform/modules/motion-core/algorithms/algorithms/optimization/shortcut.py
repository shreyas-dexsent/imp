# Copyright (c) 2026 DexSent Robotics Pvt Ltd.
# All Rights Reserved.
#
# SPDX-License-Identifier: LicenseRef-Proprietary
# See LICENSE file for details.
#
# Author: Shreyas <shreyas.kumar@dexsentrobotics.com>
# Proprietary and confidential.

"""Random shortcut smoothing.

OMPL paths zigzag because each iteration only extends by
`max_joint_step`. Shortcut smoothing iteratively replaces stretches of
the path with direct joint-space connections, accepting only those
that stay collision-free. The result has fewer waypoints and lower
total joint-space length.

Geometric pass. Input: Path. Output: Path. Time parameterization is
the trajectory layer's concern.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from algorithms.planning.path import Path
from algorithms.planning.state_validity import make_state_validity_fn
from algorithms.resolved.kinematic_model import KinematicModel
from algorithms.resolved.scene import Scene


@dataclass(frozen=True)
class ShortcutStats:
    """Diagnostic record returned alongside the smoothed path."""

    attempted: int
    accepted: int
    initial_waypoints: int
    final_waypoints: int
    initial_length: float
    final_length: float


def shortcut_smooth(
    path: Path,
    model: KinematicModel,
    scene: Scene,
    *,
    iterations: int = 100,
    max_joint_step: float = 0.05,
    joint_margin: float = 1e-3,
    random_seed: int = 0,
) -> tuple[Path, ShortcutStats]:
    """Apply random shortcut smoothing to a joint-space path.

    On each iteration pick a random pair ``(i, j)`` with ``j > i + 1``,
    test the direct segment ``waypoints[i] -> waypoints[j]`` at
    ``max_joint_step`` resolution against the state validity function,
    and if every sample is valid, drop the intermediate waypoints
    ``waypoints[i+1..j]``.

    Returns the (possibly shorter) path and a `ShortcutStats` record
    describing how the smoothing went. The path metadata is preserved
    minus any planner-specific fields the smoother doesn't carry over;
    `cartesian_waypoints` is dropped because shortcut smoothing
    invalidates the Cartesian geometry.

    Parameters
    ----------
    path
        Input path. Must have ``num_waypoints >= 2``.
    model
        Resolved kinematic model used by the state validity function.
    scene
        Scene the path was planned against; collision is checked here.
    iterations
        Number of shortcut attempts. Larger values produce smoother
        paths but cost more validation work.
    max_joint_step
        Edge sampling resolution. Same semantics as the planner's
        ``PlanOptions.max_joint_step``.
    joint_margin
        Joint-limit margin used by the validity check.
    random_seed
        RNG seed for reproducibility.
    """
    if path.num_waypoints < 3:
        # Nothing to shortcut.
        return path, ShortcutStats(
            attempted=0,
            accepted=0,
            initial_waypoints=path.num_waypoints,
            final_waypoints=path.num_waypoints,
            initial_length=path.length(),
            final_length=path.length(),
        )

    validity_fn = make_state_validity_fn(model, scene, margin=joint_margin)
    waypoints = [path.waypoints[i].copy() for i in range(path.num_waypoints)]
    initial_length = path.length()
    rng = np.random.default_rng(random_seed)

    attempted = 0
    accepted = 0
    for _ in range(iterations):
        if len(waypoints) < 3:
            break
        i = int(rng.integers(0, len(waypoints) - 2))
        j = int(rng.integers(i + 2, len(waypoints)))
        attempted += 1
        if _segment_is_valid(waypoints[i], waypoints[j], max_joint_step, validity_fn):
            del waypoints[i + 1 : j]
            accepted += 1

    smoothed_array = np.asarray(waypoints, dtype=float)
    smoothed_path = Path(
        waypoints=smoothed_array,
        joint_names=path.joint_names,
        cartesian_waypoints=None,
        metadata={**path.metadata, "smoothed_by": "shortcut"},
    )
    return smoothed_path, ShortcutStats(
        attempted=attempted,
        accepted=accepted,
        initial_waypoints=path.num_waypoints,
        final_waypoints=smoothed_path.num_waypoints,
        initial_length=initial_length,
        final_length=smoothed_path.length(),
    )


def remove_redundant_waypoints(path: Path, *, tolerance: float = 1e-9) -> Path:
    """Drop consecutive waypoints separated by less than ``tolerance``.

    OMPL often emits tightly-clustered waypoints after interpolation.
    This pass removes them in linear time without changing geometry.
    """
    if path.num_waypoints < 2:
        return path
    kept = [path.waypoints[0]]
    for i in range(1, path.num_waypoints):
        if float(np.linalg.norm(path.waypoints[i] - kept[-1])) > tolerance:
            kept.append(path.waypoints[i])
    if len(kept) < 2:
        # If everything collapsed, keep the first and last to satisfy the
        # Path invariant of N >= 2.
        kept = [path.waypoints[0], path.waypoints[-1]]

    new_waypoints = np.asarray(kept, dtype=float)
    return Path(
        waypoints=new_waypoints,
        joint_names=path.joint_names,
        cartesian_waypoints=None,
        metadata={**path.metadata, "deduped": True},
    )


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _segment_is_valid(qa, qb, max_joint_step, validity_fn) -> bool:
    """Return True iff every sample along the linear segment qa->qb is valid."""
    delta = qb - qa
    span = float(np.max(np.abs(delta)))
    n_steps = max(1, int(math.ceil(span / max(max_joint_step, 1e-9))))
    for k in range(1, n_steps):
        alpha = k / n_steps
        if not validity_fn((1.0 - alpha) * qa + alpha * qb):
            return False
    # Endpoints are already known valid (they came from a valid path).
    return True
