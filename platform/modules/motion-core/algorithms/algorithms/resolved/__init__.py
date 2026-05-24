# Copyright (c) 2026 DexSent Robotics Pvt Ltd.
# All Rights Reserved.
#
# SPDX-License-Identifier: LicenseRef-Proprietary
# See LICENSE file for details.
#
# Author: Shreyas <shreyas.kumar@dexsentrobotics.com>
# Proprietary and confidential.

"""Layer 2 - heavy resolved objects built from descriptions.

This package bridges the typed YAML descriptions (Layer 1) and the
algorithm operations (Layer 3+). The three primary objects are:

* `KinematicModel` - composed Pinocchio model with mimic expansion,
  chain slicing, and the limits API.
* `CollisionModel` - Coal collision shapes (wrapped by Pinocchio's
  `GeometryModel`) plus the static allowed-pair set.
* `Scene` - mutable runtime state: live object poses, attached objects,
  robot configurations, dynamic ACM overlay.

`geometry_cache` provides a content-addressed disk cache for expensive
mesh-processing operations.

Operations (`algorithms.kinematics`, future `collision/`, `planning/`,
etc.) consume these objects and never re-read URDFs or YAML.
"""

from algorithms.resolved.collision_model import CollisionModel
from algorithms.resolved.geometry_cache import (
    DEFAULT_CACHE_DIR,
    cache_key,
    clear_cache,
    get_or_compute,
)
from algorithms.resolved.kinematic_model import (
    JointLimits,
    KinematicModel,
    MimicRelation,
    parse_mimic_relations,
)
from algorithms.resolved.scene import (
    AttachedObject,
    CollisionOverlay,
    Scene,
)

__all__ = [
    # Kinematic model
    "JointLimits",
    "KinematicModel",
    "MimicRelation",
    "parse_mimic_relations",
    # Collision
    "CollisionModel",
    # Scene
    "AttachedObject",
    "CollisionOverlay",
    "Scene",
    # Geometry cache
    "DEFAULT_CACHE_DIR",
    "cache_key",
    "clear_cache",
    "get_or_compute",
]
