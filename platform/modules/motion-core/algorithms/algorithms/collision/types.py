# Copyright (c) 2026 DexSent Robotics Pvt Ltd.
# All Rights Reserved.
#
# SPDX-License-Identifier: LicenseRef-Proprietary
# See LICENSE file for details.
#
# Author: Shreyas <shreyas.kumar@dexsentrobotics.com>
# Proprietary and confidential.

"""Typed reports returned by collision query operations."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Contact:
    """One contact between two collision geometry objects."""

    pair: tuple[str, str]
    point_a: np.ndarray
    point_b: np.ndarray
    normal: np.ndarray
    penetration: float


@dataclass(frozen=True)
class ContactReport:
    """Result of a discrete collision query."""

    in_collision: bool
    contacts: list[Contact]
    checked_pairs: int
    skipped_pairs: int


@dataclass(frozen=True)
class DistanceReport:
    """Minimum-distance query result."""

    min_distance: float
    pair: tuple[str, str] | None
    nearest_points: tuple[np.ndarray, np.ndarray] | None
    checked_pairs: int


@dataclass(frozen=True)
class ClearanceReport:
    """Clearance query result for a fixed distance threshold."""

    clearance: float
    pairs_below_threshold: list[tuple[str, str]]
    checked_pairs: int


@dataclass(frozen=True)
class EdgeCollisionReport:
    """Sampled edge-collision query result."""

    in_collision: bool
    first_collision_alpha: float | None
    contact_report: ContactReport | None
    checked_states: int
