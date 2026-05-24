# Copyright (c) 2026 DexSent Robotics Pvt Ltd.
# All Rights Reserved.
#
# SPDX-License-Identifier: LicenseRef-Proprietary
# See LICENSE file for details.
#
# Author: Shreyas <shreyas.kumar@dexsentrobotics.com>
# Proprietary and confidential.

"""Layer 1 - pydantic schema for YAML descriptions of robots and worlds.

This package parses YAML into typed, validated objects. It contains no
computation, no URDF reading, no Pinocchio / Coal interaction. Those
responsibilities belong to the resolution layer (`algorithms.resolved`).

Public surface:

* `RobotSystemDescription` - one robot plus an optional gripper.
* `WorldDescription` - robots placed in a world plus world objects.
* The sub-models referenced by the two roots (transforms, geometry
  variants, collision specs, joint limits, kinematic chains, etc.).
"""

from algorithms.descriptions.robot_system import (
    AllowedLinkPairSpec,
    CollisionSpec,
    GripperDescription,
    JointLimitsSpec,
    KinematicChainSpec,
    NamedJointStateSpec,
    RobotDescription,
    RobotSystemDescription,
    TcpDescription,
)
from algorithms.descriptions.transforms import TransformSpec
from algorithms.descriptions.world import (
    BoxGeometrySpec,
    CapsuleGeometrySpec,
    CollisionGeometrySpec,
    CollisionMatrixSpec,
    ConvexDecompositionSpec,
    ConvexHullGeometrySpec,
    CylinderGeometrySpec,
    GeometrySpec,
    HeightFieldGeometrySpec,
    MeshDataGeometrySpec,
    MeshGeometrySpec,
    OctreeGeometrySpec,
    SphereGeometrySpec,
    VisualSpec,
    WorldCollisionRuleSpec,
    WorldDescription,
    WorldObjectDescription,
    WorldRobotDescription,
)

__all__ = [
    # Transforms
    "TransformSpec",
    # Robot system
    "AllowedLinkPairSpec",
    "CollisionSpec",
    "GripperDescription",
    "JointLimitsSpec",
    "KinematicChainSpec",
    "NamedJointStateSpec",
    "RobotDescription",
    "RobotSystemDescription",
    "TcpDescription",
    # World
    "BoxGeometrySpec",
    "CapsuleGeometrySpec",
    "CollisionGeometrySpec",
    "CollisionMatrixSpec",
    "ConvexDecompositionSpec",
    "ConvexHullGeometrySpec",
    "CylinderGeometrySpec",
    "GeometrySpec",
    "HeightFieldGeometrySpec",
    "MeshDataGeometrySpec",
    "MeshGeometrySpec",
    "OctreeGeometrySpec",
    "SphereGeometrySpec",
    "VisualSpec",
    "WorldCollisionRuleSpec",
    "WorldDescription",
    "WorldObjectDescription",
    "WorldRobotDescription",
]
