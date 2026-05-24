# Copyright (c) 2026 DexSent Robotics Pvt Ltd.
# All Rights Reserved.
#
# SPDX-License-Identifier: LicenseRef-Proprietary
# See LICENSE file for details.
#
# Author: Shreyas <shreyas.kumar@dexsentrobotics.com>
# Proprietary and confidential.

"""Low-level collision query operations."""

from algorithms.collision.attached_object import attached_object_world_pose
from algorithms.collision.checker import is_in_collision
from algorithms.collision.continuous import check_edge_collision
from algorithms.collision.distance import clearance, min_distance
from algorithms.collision.options import CollisionOptions, EdgeCollisionOptions
from algorithms.collision.pairs import active_pairs
from algorithms.collision.types import (
    ClearanceReport,
    Contact,
    ContactReport,
    DistanceReport,
    EdgeCollisionReport,
)

__all__ = [
    "CollisionOptions",
    "EdgeCollisionOptions",
    "Contact",
    "ContactReport",
    "DistanceReport",
    "ClearanceReport",
    "EdgeCollisionReport",
    "active_pairs",
    "attached_object_world_pose",
    "is_in_collision",
    "min_distance",
    "clearance",
    "check_edge_collision",
]
