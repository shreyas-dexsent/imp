# Copyright (c) 2026 DexSent Robotics Pvt Ltd.
# All Rights Reserved.
#
# SPDX-License-Identifier: LicenseRef-Proprietary
# See LICENSE file for details.
#
# Author: Shreyas <shreyas.kumar@dexsentrobotics.com>
# Proprietary and confidential.

"""Pydantic schema for a world: robots placed in it and objects in it.

A `WorldDescription` is the YAML-facing typed representation of a robot
cell. It is immutable after load. Runtime state (live object poses,
attached objects, dynamic collision allowances) lives in a `Scene`,
not here.

The world-level collision matrix records only STATIC physical-fact
allowances (e.g. a bin permanently resting on a table). Task-driven
dynamic allowances (e.g. an end-effector permitted to touch a workpiece
during a grasp phase) belong to the runtime Scene's collision overlay.
"""
from __future__ import annotations

import re
from copy import deepcopy
from pathlib import Path
from typing import Annotated, Dict, List, Literal, Optional, Set, Tuple, Union

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

from algorithms.descriptions.robot_system import RobotSystemDescription
from algorithms.descriptions.transforms import TransformSpec


# Namespace tokens must be lowercase alphanumeric/underscore, starting with a
# letter. Enforced for multi-robot worlds (see WorldDescription.from_yaml).
_NAMESPACE_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")


# ---------------------------------------------------------------------------
# Geometry specifications
# ---------------------------------------------------------------------------


class MeshGeometrySpec(BaseModel):
    """A triangle mesh loaded from a file at runtime.

    Parameters
    ----------
    type : Literal["mesh"]
        Discriminator field.
    path : str
        Absolute or YAML-relative path to the mesh file. OBJ is loaded
        through Coal's native MeshLoader; other formats (STL, PLY, DAE,
        GLTF, 3MF, OFF) go through a trimesh fallback that yields the
        same `coal.BVHModelOBBRSS` representation.
    scale : tuple[float, float, float]
        Per-axis scale applied to mesh vertices at load time.

    Notes
    -----
    BVH mesh shapes are **surface** representations in Coal. A robot link
    fully inside a closed mesh reports positive distance to the nearest
    triangle. For convex objects that need solid behaviour, prefer
    :class:`ConvexHullGeometrySpec`.
    """

    type: Literal["mesh"]
    path: str
    scale: Tuple[float, float, float] = (1.0, 1.0, 1.0)

    model_config = ConfigDict(extra="forbid")


class MeshDataGeometrySpec(BaseModel):
    """A triangle mesh supplied as in-memory vertex and face arrays.

    Use this when perception (or any runtime producer) already holds the
    mesh in memory. Avoids the temp-file roundtrip required by
    :class:`MeshGeometrySpec`. Builds the same `coal.BVHModelOBBRSS`
    surface representation.

    Parameters
    ----------
    vertices : list[tuple[float, float, float]]
        Mesh vertices. Order is preserved and consumed by `faces` as
        triangle indices.
    faces : list[tuple[int, int, int]]
        Triangle vertex-index triplets. Indices must be in
        ``[0, len(vertices))``.
    """

    type: Literal["mesh_data"]
    vertices: List[Tuple[float, float, float]]
    faces: List[Tuple[int, int, int]]

    model_config = ConfigDict(extra="forbid")

    @field_validator("vertices")
    @classmethod
    def _vertices_nonempty(cls, value):
        if not value:
            raise ValueError("vertices must be non-empty")
        return value

    @field_validator("faces")
    @classmethod
    def _faces_nonempty(cls, value):
        if not value:
            raise ValueError("faces must be non-empty")
        return value


class BoxGeometrySpec(BaseModel):
    """An axis-aligned box defined by its three side lengths."""

    type: Literal["box"]
    size: Tuple[float, float, float]

    model_config = ConfigDict(extra="forbid")


class SphereGeometrySpec(BaseModel):
    """A sphere defined by its radius."""

    type: Literal["sphere"]
    radius: float

    model_config = ConfigDict(extra="forbid")


class CylinderGeometrySpec(BaseModel):
    """A cylinder defined by its radius and length along the local z axis."""

    type: Literal["cylinder"]
    radius: float
    length: float

    model_config = ConfigDict(extra="forbid")


class CapsuleGeometrySpec(BaseModel):
    """A capsule (cylinder with hemispherical caps) along the local z axis.

    Standard primitive for elongated arm-like obstacles. Solid GJK-friendly
    type with closed-form contact math.
    """

    type: Literal["capsule"]
    radius: float
    length: float

    model_config = ConfigDict(extra="forbid")


class ConvexHullGeometrySpec(BaseModel):
    """A convex polytope built from the convex hull of a vertex cloud.

    Use this when perception has already convex-hulled an object (or
    when the object is known-convex CAD). Backed by `coal.Convex`, which
    is **solid** (interior is in-collision) and faster than a triangle
    BVH of the same vertices.

    Parameters
    ----------
    vertices : list[tuple[float, float, float]]
        Vertices the hull is built from. Coal computes the hull
        internally via qhull; redundant interior points are ignored.
    """

    type: Literal["convex_hull"]
    vertices: List[Tuple[float, float, float]]

    model_config = ConfigDict(extra="forbid")

    @field_validator("vertices")
    @classmethod
    def _vertices_min_4(cls, value):
        if len(value) < 4:
            raise ValueError(
                "convex_hull requires at least 4 non-coplanar vertices"
            )
        return value


class OctreeGeometrySpec(BaseModel):
    """A sparse-occupancy voxel tree built from a point cloud.

    The native Coal answer for point-cloud inputs. Perception ships an
    (N, 3) point cloud and a voxel resolution; collision queries run
    directly against the resulting `coal.OcTree`. No mesh reconstruction
    or surface fitting happens inside the library — perception is
    responsible for filtering and downsampling before the call.

    Parameters
    ----------
    points : list[tuple[float, float, float]]
        Cloud points in the object's local frame.
    resolution : float
        Voxel edge length in metres. Smaller values give a finer
        representation at higher query cost.
    """

    type: Literal["octree"]
    points: List[Tuple[float, float, float]]
    resolution: float = 0.01

    model_config = ConfigDict(extra="forbid")

    @field_validator("resolution")
    @classmethod
    def _resolution_positive(cls, value):
        if value <= 0:
            raise ValueError("resolution must be > 0")
        return value

    @field_validator("points")
    @classmethod
    def _points_nonempty(cls, value):
        if not value:
            raise ValueError("points must be non-empty")
        return value


class HeightFieldGeometrySpec(BaseModel):
    """A regular grid of height samples in the local XY plane.

    Useful when perception delivers depth-map-like data (top-down depth
    camera over a workspace, table surface estimation). Backed by
    `coal.HeightFieldOBBRSS`.

    Parameters
    ----------
    x_size, y_size : float
        Physical extent of the grid along the local X and Y axes.
    heights : list[list[float]]
        2D matrix of height samples. ``heights[i][j]`` is the height at
        grid cell ``(i, j)`` over the XY plane.
    min_height : float
        Floor height used as the lower bound of the field's bounding
        volume; samples below this value are clipped.
    """

    type: Literal["height_field"]
    x_size: float
    y_size: float
    heights: List[List[float]]
    min_height: float = 0.0

    model_config = ConfigDict(extra="forbid")

    @field_validator("heights")
    @classmethod
    def _heights_rectangular(cls, value):
        if not value or not value[0]:
            raise ValueError("heights must be a non-empty 2D matrix")
        width = len(value[0])
        if any(len(row) != width for row in value):
            raise ValueError("heights rows must all have the same length")
        return value


# Discriminated union over the geometry variants. Pydantic dispatches on the
# `type` field, producing clean validation errors when an unknown variant is
# supplied.
GeometrySpec = Annotated[
    Union[
        MeshGeometrySpec,
        MeshDataGeometrySpec,
        BoxGeometrySpec,
        SphereGeometrySpec,
        CylinderGeometrySpec,
        CapsuleGeometrySpec,
        ConvexHullGeometrySpec,
        OctreeGeometrySpec,
        HeightFieldGeometrySpec,
    ],
    Field(discriminator="type"),
]


class ConvexDecompositionSpec(BaseModel):
    """V-HACD convex decomposition configuration for a concave collision mesh.

    Concave collision queries are slow and numerically fragile; V-HACD
    approximates a concave mesh by a set of convex hulls that can be
    checked efficiently with GJK/EPA.
    """

    type: Literal["convex_decomposition"]
    max_hulls: int = 16

    model_config = ConfigDict(extra="forbid")


# Discriminated union for processing instructions. Only one variant exists
# today; the union shape makes future processing modes (voxelization,
# remeshing) additive without schema breakage.
ProcessingSpec = Annotated[
    Union[ConvexDecompositionSpec],
    Field(discriminator="type"),
]


# ---------------------------------------------------------------------------
# Visual / collision geometry wrappers
# ---------------------------------------------------------------------------


class VisualSpec(BaseModel):
    """Visual geometry for a world object.

    The `origin` field is an optional offset from the object's local frame
    to this geometry's frame, matching the URDF `<visual><origin/></visual>`
    convention. Default is identity (geometry sits at the object's local
    origin). Effective placement at runtime is `world_T_object @ origin`.
    """

    enabled: bool = True
    geometry: GeometrySpec
    origin: Optional[TransformSpec] = None

    model_config = ConfigDict(extra="forbid")


class CollisionGeometrySpec(BaseModel):
    """Collision geometry for a world object.

    Separate from the visual geometry because production assets typically
    use a high-poly visual mesh paired with a simplified collision mesh.
    The `origin` field handles the common case where convex decomposition
    or hand-authored simplification produces a collision mesh centered
    differently than the visual mesh.
    """

    enabled: bool = True
    geometry: GeometrySpec
    origin: Optional[TransformSpec] = None
    processing: Optional[ProcessingSpec] = None

    model_config = ConfigDict(extra="forbid")


# ---------------------------------------------------------------------------
# World object
# ---------------------------------------------------------------------------


# Allowed types in YAML. The `"attached"` state is intentionally absent -
# it is reachable only at runtime via `Scene.attach`, never via YAML.
WorldObjectType = Literal["workpiece", "obstacle", "fixture", "bin"]


class WorldObjectDescription(BaseModel):
    """One object placed in the world.

    The `pose` field is the description-time default. At runtime the live
    pose lives in `Scene.object_poses` (and is the source of truth for
    every algorithm). Updates flow through `Scene.set_object_pose` and
    never modify this description.
    """

    id: str
    type: WorldObjectType
    pose: TransformSpec
    visual: Optional[VisualSpec] = None
    collision: Optional[CollisionGeometrySpec] = None

    model_config = ConfigDict(extra="forbid")


# ---------------------------------------------------------------------------
# World robot
# ---------------------------------------------------------------------------


class WorldRobotDescription(BaseModel):
    """One robot system instance placed in a world.

    A world may contain multiple robots. In that case every instance must
    have a non-null unique namespace; the namespace is prepended to frame
    and geometry names to keep multi-robot worlds collision-free in their
    naming.
    """

    id: str
    robot_system_path: str = Field(alias="robot_system")
    namespace: Optional[str] = None
    base_pose: Optional[TransformSpec] = None

    # Populated by WorldDescription.from_yaml after sub-loading the
    # referenced robot system YAML. Excluded from model serialization.
    robot_system_description: Optional[RobotSystemDescription] = Field(
        default=None, exclude=True
    )

    model_config = ConfigDict(
        populate_by_name=True, extra="forbid", arbitrary_types_allowed=True
    )

    @field_validator("namespace")
    @classmethod
    def _validate_namespace_format(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return value
        if not _NAMESPACE_PATTERN.fullmatch(value):
            raise ValueError(
                f"namespace must match {_NAMESPACE_PATTERN.pattern!r}; got {value!r}"
            )
        return value

    @property
    def robot_system(self) -> RobotSystemDescription:
        """Return the resolved robot-system description for this instance."""
        if self.robot_system_description is None:
            raise ValueError(
                f"world robot {self.id!r} has not resolved its robot_system"
            )
        return self.robot_system_description

    def frame_name(self, frame: str) -> str:
        """Apply this robot's namespace to a frame name.

        For an un-namespaced single-robot world the name is returned
        unchanged. For namespaced multi-robot worlds, the namespace is
        prepended with a forward slash.
        """
        return f"{self.namespace}/{frame}" if self.namespace else frame


# ---------------------------------------------------------------------------
# Static collision matrix
# ---------------------------------------------------------------------------


CollisionAction = Literal["check", "allow"]


class WorldCollisionRuleSpec(BaseModel):
    """One rule in the world-wide allowed collision matrix.

    Covers pairs across world objects and robot links (e.g. a bin
    permanently resting on a table, or robot A's base permanently
    abutting robot B's base). Within-system link pairs use
    `AllowedLinkPairSpec` in the robot system schema instead.
    """

    a: str
    b: str
    action: CollisionAction = "check"
    reason: Optional[str] = None

    model_config = ConfigDict(extra="forbid")


class CollisionMatrixSpec(BaseModel):
    """The static, world-level allowed collision matrix.

    Dynamic, task-driven allowances (end-effector touching a workpiece
    during grasp; attached objects) live in the runtime Scene, not here.
    """

    default_action: CollisionAction = "check"
    rules: List[WorldCollisionRuleSpec] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid")


# ---------------------------------------------------------------------------
# Top-level world
# ---------------------------------------------------------------------------


class WorldDescription(BaseModel):
    """A robot cell description: robots placed plus objects in the scene."""

    schema_name: Literal["dexsent.algorithms.world"] = Field(
        "dexsent.algorithms.world", alias="schema"
    )
    version: Literal[2] = 2
    id: str
    name: str = "untitled world"
    world_frame: str = "world"

    robots: List[WorldRobotDescription] = Field(default_factory=list)
    objects: List[WorldObjectDescription] = Field(default_factory=list)
    collision_matrix: CollisionMatrixSpec = Field(default_factory=CollisionMatrixSpec)

    source_path: Optional[Path] = Field(default=None, exclude=True)

    model_config = ConfigDict(
        populate_by_name=True, extra="forbid", arbitrary_types_allowed=True
    )

    @classmethod
    def from_yaml(cls, path: str | Path) -> "WorldDescription":
        """Parse and validate a world YAML file.

        Sub-loads the referenced robot-system YAMLs for each world robot,
        then validates multi-robot namespacing and id uniqueness.
        """
        path = Path(path).resolve()
        with path.open("r", encoding="utf-8") as handle:
            description = cls.model_validate(yaml.safe_load(handle) or {})
        description.source_path = path

        for world_robot in description.robots:
            world_robot.robot_system_description = RobotSystemDescription.from_yaml(
                description.resolve_path(world_robot.robot_system_path)
            )

        description._validate_namespaces()
        description._validate_unique_ids()
        return description

    def resolve_path(self, path: str) -> Path:
        """Resolve a relative path against the YAML's source directory."""
        raw = Path(path)
        if raw.is_absolute():
            return raw
        if self.source_path is None:
            return raw.resolve()
        return (self.source_path.parent / raw).resolve()

    def robot(self, robot_id: str) -> WorldRobotDescription:
        """Look up a world robot by id.

        Raises
        ------
        KeyError
            If no world robot with that id exists.
        """
        for robot in self.robots:
            if robot.id == robot_id:
                return robot
        raise KeyError(f"world robot not found: {robot_id}")

    def object_pose(self, object_id: str) -> TransformSpec:
        """Return a deep copy of a world object's description-time pose.

        Note: the runtime pose lives in `Scene.object_poses` and may
        differ. This method returns the *initial* pose declared in YAML.
        """
        for obj in self.objects:
            if obj.id == object_id:
                return deepcopy(obj.pose)
        raise KeyError(f"world object not found: {object_id}")

    def _validate_namespaces(self) -> None:
        """Multi-robot worlds require non-null unique namespaces."""
        if len(self.robots) <= 1:
            return

        nulls = [r.id for r in self.robots if r.namespace is None]
        if nulls:
            raise ValueError(
                "multi-robot worlds require a non-null namespace on every robot; "
                f"missing on: {nulls}"
            )

        seen: Dict[str, str] = {}
        duplicates: List[str] = []
        for robot in self.robots:
            existing = seen.setdefault(robot.namespace, robot.id)
            if existing != robot.id:
                duplicates.append(robot.namespace)
        if duplicates:
            raise ValueError(
                f"duplicate world robot namespaces: {sorted(set(duplicates))}"
            )

    def _validate_unique_ids(self) -> None:
        """Robot ids must be unique; object ids must be unique."""
        robot_ids = [robot.id for robot in self.robots]
        object_ids = [obj.id for obj in self.objects]
        for label, ids in (("robot", robot_ids), ("object", object_ids)):
            duplicates = sorted({i for i in ids if ids.count(i) > 1})
            if duplicates:
                raise ValueError(f"duplicate world {label} ids: {duplicates}")
