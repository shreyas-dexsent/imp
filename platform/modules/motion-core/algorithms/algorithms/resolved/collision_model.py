# Copyright (c) 2026 DexSent Robotics Pvt Ltd.
# All Rights Reserved.
#
# SPDX-License-Identifier: LicenseRef-Proprietary
# See LICENSE file for details.
#
# Author: Shreyas <shreyas.kumar@dexsentrobotics.com>
# Proprietary and confidential.

"""Resolved world-level collision representation.

A `CollisionModel` is the static collision catalogue for a world. It
holds Coal collision shapes (wrapped by Pinocchio's `GeometryObject` /
`GeometryModel` for joint-frame rigging) and the description-time
allowed-pair set.

Two catalogues live side by side:

* `robot_geom` - collision objects whose world pose is computed by FK
  on a robot joint frame.
* `world_geom` - collision objects whose world pose comes from the
  runtime `Scene` (free-standing objects) or from FK on a parent
  frame (attached objects).

Both use `pin.GeometryModel` purely as a catalogue type; the FK machinery
is not invoked for world objects. Collision operations populate per-query
geometry placements from these sources before invoking Coal.

The static allowed-pair set covers within-robot allowances (e.g.
fingertip <-> palm) and world-level allowances (e.g. bin <-> table).
Task-driven dynamic allowances belong to the runtime Scene overlay,
not here.

Mesh world objects are loaded eagerly through Coal's `MeshLoader`.
Optional processing instructions are applied before loading when a mesh
declares them; failures fall back to the raw mesh so the model can still
be built for inspection and query.
"""
from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

import numpy as np

from algorithms.descriptions import (
    BoxGeometrySpec,
    CapsuleGeometrySpec,
    CollisionGeometrySpec,
    ConvexDecompositionSpec,
    ConvexHullGeometrySpec,
    CylinderGeometrySpec,
    HeightFieldGeometrySpec,
    MeshDataGeometrySpec,
    MeshGeometrySpec,
    OctreeGeometrySpec,
    RobotSystemDescription,
    SphereGeometrySpec,
    WorldDescription,
    WorldObjectDescription,
)
from algorithms.resolved import geometry_cache


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _namespaced(namespace: Optional[str], name: str) -> str:
    """Prepend a robot namespace to a name, or return the name unchanged."""
    return f"{namespace}/{name}" if namespace else name


def _se3_from_matrix(matrix: np.ndarray):
    """Convert a 4x4 NumPy matrix to a `pin.SE3`."""
    import pinocchio as pin

    se3 = pin.SE3.Identity()
    se3.rotation = np.asarray(matrix[:3, :3], dtype=float)
    se3.translation = np.asarray(matrix[:3, 3], dtype=float)
    return se3


def _se3_identity():
    """Return an identity `pin.SE3`."""
    import pinocchio as pin

    return pin.SE3.Identity()


def _coal_shape_for_primitive(geometry):
    """Build a Coal shape for any non-file, non-mesh-data geometry spec.

    Covers primitives (Box / Sphere / Cylinder / Capsule), convex hulls,
    octrees, and height fields. All are built from inline data on the
    spec; none require disk I/O.
    """
    import coal

    if isinstance(geometry, BoxGeometrySpec):
        sx, sy, sz = geometry.size
        return coal.Box(float(sx), float(sy), float(sz))
    if isinstance(geometry, SphereGeometrySpec):
        return coal.Sphere(float(geometry.radius))
    if isinstance(geometry, CylinderGeometrySpec):
        return coal.Cylinder(float(geometry.radius), float(geometry.length))
    if isinstance(geometry, CapsuleGeometrySpec):
        return coal.Capsule(float(geometry.radius), float(geometry.length))
    if isinstance(geometry, ConvexHullGeometrySpec):
        return _coal_shape_for_convex_hull(geometry.vertices)
    if isinstance(geometry, OctreeGeometrySpec):
        return _coal_shape_for_octree(geometry.points, geometry.resolution)
    if isinstance(geometry, HeightFieldGeometrySpec):
        return _coal_shape_for_height_field(
            geometry.x_size, geometry.y_size, geometry.heights, geometry.min_height,
        )
    raise TypeError(f"unsupported geometry: {type(geometry).__name__}")


def _coal_shape_for_convex_hull(vertices) -> "object":
    """Build a `coal.Convex` from a vertex cloud.

    Solid representation: interior is in-collision. Faster GJK than the
    equivalent triangle BVH because Coal walks faces, not triangles.
    """
    import coal

    verts_vec = coal.StdVec_Vec3s()
    for v in vertices:
        verts_vec.append(np.asarray(v, dtype=float))
    # The "Qt" flag tells qhull to keep triangles, which Coal requires
    # when `triangulate=True` is passed.
    return coal.ConvexBase.convexHull(verts_vec, True, "Qt")


def _coal_shape_for_octree(points, resolution: float) -> "object":
    """Build a `coal.OcTree` from a point cloud + voxel resolution.

    Native Coal answer for point-cloud inputs. Perception has already
    filtered and downsampled the cloud; the library is a pass-through.
    """
    import coal

    points_array = np.asarray(points, dtype=float).reshape(-1, 3)
    return coal.makeOctree(points_array, float(resolution))


def _coal_shape_for_height_field(
    x_size: float,
    y_size: float,
    heights,
    min_height: float,
) -> "object":
    """Build a `coal.HeightFieldOBBRSS` from a 2D heights matrix."""
    import coal

    heights_array = np.asarray(heights, dtype=float)
    return coal.HeightFieldOBBRSS(
        float(x_size),
        float(y_size),
        heights_array,
        float(min_height),
    )


def _coal_shape_for_mesh_data(vertices, faces) -> "object":
    """Build a `coal.BVHModelOBBRSS` from in-memory vertices + faces.

    Surface representation. For perception outputs already in memory,
    avoids the temp-file roundtrip required by `_load_mesh_shape`.
    """
    import coal

    verts_vec = coal.StdVec_Vec3s()
    for v in vertices:
        verts_vec.append(np.asarray(v, dtype=float))
    tris_vec = coal.StdVec_Triangle()
    for f in faces:
        tris_vec.append(coal.Triangle(int(f[0]), int(f[1]), int(f[2])))
    bvh = coal.BVHModelOBBRSS()
    bvh.beginModel(len(faces), len(vertices))
    bvh.addSubModel(verts_vec, tris_vec)
    bvh.endModel()
    return bvh


def _load_mesh_shape(mesh_path: Path, scale: np.ndarray):
    """Load a mesh file into a Coal collision geometry.

    OBJ files go through Coal's native `MeshLoader`. Any other extension
    (STL, PLY, DAE, GLTF, 3MF, OFF, ...) is loaded via trimesh and the
    resulting vertices/faces are wrapped into a `BVHModelOBBRSS`. This
    gives the same surface representation regardless of source format.
    """
    import coal

    suffix = mesh_path.suffix.lower()
    if suffix == ".obj":
        return coal.MeshLoader().load(str(mesh_path), np.asarray(scale, dtype=float))

    return _load_mesh_via_trimesh(mesh_path, np.asarray(scale, dtype=float))


def _load_mesh_via_trimesh(mesh_path: Path, scale: np.ndarray):
    """Fallback loader for non-OBJ formats. Uses trimesh + in-memory BVH."""
    try:
        import trimesh
    except Exception as exc:  # pragma: no cover - trimesh is a hard dep today
        raise RuntimeError(
            f"loading {mesh_path.suffix} meshes requires trimesh; pip install trimesh"
        ) from exc

    mesh = trimesh.load(str(mesh_path), force="mesh")
    if mesh is None or getattr(mesh, "is_empty", True):
        raise ValueError(f"trimesh could not load mesh: {mesh_path}")
    vertices = np.asarray(mesh.vertices, dtype=float) * np.asarray(scale, dtype=float)
    faces = np.asarray(mesh.faces, dtype=int)
    return _coal_shape_for_mesh_data(vertices, faces)


def _coal_shape_for_mesh(mesh_path: Path, scale: np.ndarray, processing):
    """Load a mesh, optionally applying cached geometry processing first."""
    if isinstance(processing, ConvexDecompositionSpec):
        processed_bytes = geometry_cache.get_or_compute(
            mesh_path,
            processing,
            _compute_convex_decomposition_mesh_bytes,
            scale=tuple(float(v) for v in scale),
        )
        if processed_bytes:
            return _load_mesh_shape_from_bytes(processed_bytes)

    return _load_mesh_shape(mesh_path, scale)


def _load_mesh_shape_from_bytes(mesh_bytes: bytes):
    """Load an OBJ mesh byte payload through Coal's file-based MeshLoader API."""
    with tempfile.NamedTemporaryFile(suffix=".obj", delete=True) as handle:
        handle.write(mesh_bytes)
        handle.flush()
        return _load_mesh_shape(Path(handle.name), np.ones(3, dtype=float))


def _compute_convex_decomposition_mesh_bytes(
    mesh_path: Path,
    processing: ConvexDecompositionSpec,
) -> bytes:
    """Run V-HACD through trimesh and return a processed OBJ payload.

    The Coal Python API stores one collision geometry per GeometryObject.
    When V-HACD returns multiple hulls, they are concatenated into one
    processed triangle mesh for this phase. The cache boundary still
    isolates the expensive decomposition call, and future work can split
    the hulls into multiple GeometryObjects without changing YAML.
    """
    try:
        import trimesh
        import trimesh.decomposition
    except Exception:
        return b""

    try:
        mesh = trimesh.load(str(mesh_path), force="mesh")
        hulls = trimesh.decomposition.convex_decomposition(
            mesh,
            maxConvexHulls=int(processing.max_hulls),
        )
    except Exception:
        return b""

    if hulls is None:
        return b""
    if not isinstance(hulls, (list, tuple)):
        hulls = [hulls]
    hulls = [h for h in hulls if h is not None]
    if not hulls:
        return b""

    try:
        processed = trimesh.util.concatenate(hulls)
        exported = processed.export(file_type="obj")
    except Exception:
        return b""

    return exported.encode("utf-8") if isinstance(exported, str) else bytes(exported)


def _canonical_pair(a: str, b: str) -> Tuple[str, str]:
    """Order a pair so set lookup is direction-agnostic."""
    return (a, b) if a <= b else (b, a)


def _se3_to_matrix(se3) -> np.ndarray:
    """Convert a `pin.SE3` to a 4x4 NumPy matrix."""
    matrix = np.eye(4, dtype=float)
    matrix[:3, :3] = np.asarray(se3.rotation, dtype=float)
    matrix[:3, 3] = np.asarray(se3.translation, dtype=float)
    return matrix


def _shape_kind(coal_shape) -> str:
    """Return a short tag identifying which Coal shape category this is.

    Used by the UI accessor so callers can pick a renderer per kind
    without inspecting Coal types directly.
    """
    import coal

    if isinstance(coal_shape, coal.Box):
        return "box"
    if isinstance(coal_shape, coal.Sphere):
        return "sphere"
    if isinstance(coal_shape, coal.Cylinder):
        return "cylinder"
    if isinstance(coal_shape, coal.Capsule):
        return "capsule"
    if isinstance(coal_shape, coal.Cone):
        return "cone"
    if isinstance(coal_shape, coal.Ellipsoid):
        return "ellipsoid"
    if isinstance(coal_shape, coal.Halfspace):
        return "halfspace"
    if isinstance(coal_shape, coal.Plane):
        return "plane"
    if isinstance(coal_shape, coal.Convex):
        return "convex_hull"
    if isinstance(coal_shape, coal.OcTree):
        return "octree"
    if isinstance(coal_shape, (coal.HeightFieldOBBRSS, coal.HeightFieldAABB)):
        return "height_field"
    if isinstance(coal_shape, (coal.BVHModelOBBRSS, coal.BVHModelOBB, coal.BVHModelBase)):
        return "mesh"
    return "unknown"


@dataclass(frozen=True)
class ShapeInfo:
    """One geometry registered in the CollisionModel catalogue.

    `coal_shape` is the exact object Coal uses during GJK / distance
    queries. `T_parent_shape` is its placement relative to the parent
    joint (a robot link) or the universe joint (for world objects).
    The world pose is composed at runtime: for robot links, from FK on
    the parent joint; for world objects, from
    `Scene.object_poses[name] @ T_parent_shape`.

    `kind` is a short tag for UI dispatch (`"box"`, `"sphere"`, ...,
    `"convex_hull"`, `"octree"`, `"height_field"`, `"mesh"`).
    """

    name: str
    coal_shape: object
    owner: str           # "robot" or "world"
    parent_joint: str    # name of the parent Pinocchio joint
    T_parent_shape: np.ndarray
    kind: str


@dataclass
class _SyntheticWorldObject:
    """Adapter so `_build_world_object_geometry` can accept a runtime spec
    without needing a YAML-backed WorldObjectDescription."""

    object_id: str
    collision: "CollisionGeometrySpec"

    @property
    def id(self) -> str:
        return self.object_id


# ---------------------------------------------------------------------------
# CollisionModel
# ---------------------------------------------------------------------------


@dataclass
class CollisionModel:
    """Resolved world-level collision representation.

    Attributes
    ----------
    robot_geom : pinocchio.GeometryModel
        Collision catalogue for every robot link. One entry per
        namespaced link.
    world_geom : pinocchio.GeometryModel
        Collision catalogue for every world object. `parent_joint == 0`
        (universe) for all entries; world pose comes from the runtime
        Scene rather than from FK.
    geometry_index : dict[str, int]
        Name to index lookup. Helps callers identify objects without
        scanning the geometry-object lists.
    static_allowed_pairs : frozenset[tuple[str, str]]
        Pairs of geometry-object names allowed to touch per description.
        Within-robot allowances and world-level allow rules both land
        here. Pairs are stored as ordered tuples `(a, b)` with
        `a <= b` so set lookup is direction-agnostic.
    object_owner : dict[str, str]
        Map from geometry-object name to its owning catalogue
        (`"robot"` or `"world"`).
    object_parent_joint : dict[str, str]
        Map from geometry-object name to the Pinocchio parent joint name.
        World objects use `"universe"`.
    chain_geometry_names : dict[str, frozenset[str]]
        Geometry names that belong to each declared chain. Used by the
        collision pair materialiser for chain-scoped queries.
    """

    robot_geom: "object"
    world_geom: "object"
    geometry_index: Dict[str, int]
    static_allowed_pairs: FrozenSet[Tuple[str, str]]
    object_owner: Dict[str, str] = field(default_factory=dict)
    object_parent_joint: Dict[str, str] = field(default_factory=dict)
    chain_geometry_names: Dict[str, FrozenSet[str]] = field(default_factory=dict)
    # Per-robot geometry catalogues. The combined `robot_geom` above is
    # convenient for name lookups and pair management, but each
    # `GeometryObject` in it has a `parentJoint` index that is only
    # valid against its originating robot's `pin_model`. For runtime FK
    # placement in multi-robot worlds the collision pipeline iterates
    # `robot_geoms_by_id` and runs `updateGeometryPlacements` per robot
    # with the matching `KinematicModel.pin_model`.
    robot_geoms_by_id: Dict[str, "object"] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_world(cls, world: WorldDescription) -> "CollisionModel":
        """Build a CollisionModel from a fully-loaded WorldDescription.

        Iterates over each world robot to compose its collision geometry,
        namespacing every name. World objects are added separately under
        the universe joint. Static allowed pairs are accumulated from
        both within-robot allowances and the world's collision matrix.
        """
        import pinocchio as pin

        robot_geom = pin.GeometryModel()
        world_geom = pin.GeometryModel()
        geometry_index: Dict[str, int] = {}
        object_owner: Dict[str, str] = {}
        object_parent_joint: Dict[str, str] = {}
        chain_geometry_names: Dict[str, Set[str]] = {}
        allowed_pairs: Set[Tuple[str, str]] = set()
        robot_geoms_by_id: Dict[str, "object"] = {}

        # Each world robot contributes its composed collision geometry,
        # namespaced into the world-level catalogue. The sub_geom is also
        # kept on `robot_geoms_by_id` so the runtime can run
        # `updateGeometryPlacements` against the matching pin_model
        # without parent-joint index conflicts across robots.
        for world_robot in world.robots:
            system = world_robot.robot_system
            ns = world_robot.namespace

            (
                sub_geom,
                _,
                sub_allowed,
                sub_parent_joint,
                sub_chain_geometry,
            ) = _build_robot_geometry(
                system=system,
                namespace=ns,
                source_resolver=lambda p, s=system: s.resolve_path(p),
            )

            robot_geoms_by_id[world_robot.id] = sub_geom

            for obj in sub_geom.geometryObjects:
                idx = robot_geom.addGeometryObject(obj)
                geometry_index[obj.name] = idx
                object_owner[obj.name] = "robot"
                object_parent_joint[obj.name] = sub_parent_joint.get(
                    obj.name, "universe"
                )

            allowed_pairs.update(sub_allowed)
            for chain_id, names in sub_chain_geometry.items():
                chain_geometry_names.setdefault(chain_id, set()).update(names)

        # World objects: parented to the universe joint. Their pose comes
        # from Scene at runtime, not from FK.
        for obj in world.objects:
            if obj.collision is None or not obj.collision.enabled:
                continue
            geom_obj = _build_world_object_geometry(
                obj=obj,
                source_resolver=world.resolve_path,
            )
            idx = world_geom.addGeometryObject(geom_obj)
            geometry_index[geom_obj.name] = idx
            object_owner[geom_obj.name] = "world"
            object_parent_joint[geom_obj.name] = "universe"

        # World-level static allow rules.
        for rule in world.collision_matrix.rules:
            if rule.action == "allow":
                allowed_pairs.add(_canonical_pair(rule.a, rule.b))

        return cls(
            robot_geom=robot_geom,
            world_geom=world_geom,
            geometry_index=geometry_index,
            static_allowed_pairs=frozenset(allowed_pairs),
            object_owner=object_owner,
            object_parent_joint=object_parent_joint,
            chain_geometry_names={
                chain_id: frozenset(names)
                for chain_id, names in chain_geometry_names.items()
            },
            robot_geoms_by_id=robot_geoms_by_id,
        )

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def has_object(self, name: str) -> bool:
        """True if a geometry object with this name exists in either catalogue."""
        return name in self.geometry_index

    def is_statically_allowed(self, a: str, b: str) -> bool:
        """True if the (a, b) pair is in the static allowed-pair set."""
        return _canonical_pair(a, b) in self.static_allowed_pairs

    def object_names(self) -> List[str]:
        """Sorted list of all geometry-object names tracked by this model."""
        return sorted(self.geometry_index.keys())

    def shapes_for(self, object_id: str) -> List["ShapeInfo"]:
        """Return the actual Coal shape(s) registered for this object.

        This is the single-source-of-truth accessor for UI / inspection:
        the planner reads from the same `GeometryObject` entries this
        function returns. If V-HACD or another decomposition produces
        multiple geometry objects, they all appear here. The returned
        Coal shapes are the exact objects passed to GJK at query time.

        The returned `T_parent_shape` is the geometry's placement
        relative to its parent joint (robot link) or the universe joint
        (world objects). The UI composes the full world pose with the
        live `Scene.object_poses[object_id]` (world objects) or with FK
        on the parent joint (robot links).
        """
        info: List["ShapeInfo"] = []
        for owner, gmodel in (("robot", self.robot_geom), ("world", self.world_geom)):
            for go in gmodel.geometryObjects:
                if go.name != object_id:
                    continue
                info.append(
                    ShapeInfo(
                        name=go.name,
                        coal_shape=go.geometry,
                        owner=owner,
                        parent_joint=self.object_parent_joint.get(go.name, "universe"),
                        T_parent_shape=_se3_to_matrix(go.placement),
                        kind=_shape_kind(go.geometry),
                    )
                )
        return info

    # ------------------------------------------------------------------
    # Runtime mutation (perception integration)
    # ------------------------------------------------------------------

    def add_world_object(
        self,
        object_id: str,
        spec: "CollisionGeometrySpec",
        *,
        source_resolver=None,
    ) -> None:
        """Add a world-level geometry object at runtime.

        Used by `Scene.add_object` to inject perception-supplied objects
        into the live collision catalogue. The new object lives under
        the universe joint; its world pose is sourced from
        `Scene.object_poses[object_id]` at every query, identical to
        YAML-declared objects.

        Raises if an object with this id already exists. Use
        `remove_world_object` first to replace.
        """
        if object_id in self.geometry_index:
            raise ValueError(f"object {object_id!r} already exists in collision model")

        synthetic = _SyntheticWorldObject(object_id=object_id, collision=spec)
        resolver = source_resolver or (lambda p: Path(p))
        geom_obj = _build_world_object_geometry(
            obj=synthetic,
            source_resolver=resolver,
        )
        idx = self.world_geom.addGeometryObject(geom_obj)
        self.geometry_index[geom_obj.name] = idx
        self.object_owner[geom_obj.name] = "world"
        self.object_parent_joint[geom_obj.name] = "universe"

    def remove_world_object(self, object_id: str) -> None:
        """Remove a previously-added perception object from the catalogue.

        Only world-level objects can be removed; robot-link geometry is
        immutable post-build. Raises if `object_id` is unknown or is a
        robot link.
        """
        if object_id not in self.geometry_index:
            raise KeyError(f"object {object_id!r} not in collision model")
        if self.object_owner.get(object_id) != "world":
            raise ValueError(
                f"cannot remove robot-link geometry {object_id!r}; only world objects are mutable"
            )

        # Rebuild the world geom catalogue without this object. pin.GeometryModel
        # does not expose a stable in-place removal in this binding, so we
        # construct a fresh catalogue and re-index.
        import pinocchio as pin

        fresh = pin.GeometryModel()
        new_index: Dict[str, int] = {}
        for go in self.world_geom.geometryObjects:
            if go.name == object_id:
                continue
            new_idx = fresh.addGeometryObject(go)
            new_index[go.name] = new_idx

        # Preserve robot-side indices unchanged; only rewrite world entries.
        rebuilt_index = {
            name: idx for name, idx in self.geometry_index.items()
            if self.object_owner.get(name) != "world"
        }
        rebuilt_index.update(new_index)

        self.world_geom = fresh
        self.geometry_index = rebuilt_index
        self.object_owner.pop(object_id, None)
        self.object_parent_joint.pop(object_id, None)


# ---------------------------------------------------------------------------
# Robot geometry composition
# ---------------------------------------------------------------------------


def _build_robot_geometry(
    *,
    system: RobotSystemDescription,
    namespace: Optional[str],
    source_resolver,
):
    """Build the composed collision GeometryModel for one robot system.

    Returns `(geom_model, owned_names, allowed_pairs, object_parent_joint,
    chain_geometry_names)`.

    URDF parent directories are added to package_dirs automatically so
    relative mesh references inside the URDF resolve correctly.
    """
    import pinocchio as pin

    robot_urdf = source_resolver(system.robot.urdf_path)
    robot_pkgs = [str(robot_urdf.parent)] + [
        str(source_resolver(p)) for p in system.robot.package_dirs
    ]
    robot_model = pin.buildModelFromUrdf(str(robot_urdf))
    robot_geom = pin.buildGeomFromUrdf(
        robot_model,
        str(robot_urdf),
        pin.GeometryType.COLLISION,
        package_dirs=robot_pkgs,
    )

    if system.gripper is not None:
        gripper_urdf = source_resolver(system.gripper.urdf_path)
        gripper_pkgs = [str(gripper_urdf.parent)] + [
            str(source_resolver(p)) for p in system.gripper.package_dirs
        ]
        gripper_model = pin.buildModelFromUrdf(str(gripper_urdf))
        gripper_geom = pin.buildGeomFromUrdf(
            gripper_model,
            str(gripper_urdf),
            pin.GeometryType.COLLISION,
            package_dirs=gripper_pkgs,
        )

        parent_frame_name = system.gripper.mount.parent_frame
        if not robot_model.existFrame(parent_frame_name):
            raise ValueError(
                f"gripper mount parent frame {parent_frame_name!r} not in robot URDF"
            )
        parent_frame_id = robot_model.getFrameId(parent_frame_name)
        mount_se3 = _se3_from_matrix(system.gripper.mount.as_matrix())
        composed_model, composed_geom = pin.appendModel(
            robot_model,
            gripper_model,
            robot_geom,
            gripper_geom,
            parent_frame_id,
            mount_se3,
        )
    else:
        composed_model, composed_geom = robot_model, robot_geom

    disabled_links = set(system.robot.collision.disabled_links)
    if system.gripper is not None:
        disabled_links.update(system.gripper.collision.disabled_links)
    if disabled_links:
        composed_geom = _filter_disabled_geometry(composed_geom, disabled_links)

    # Namespace every geometry-object name so multi-robot worlds do not
    # produce name collisions.
    owned_names: Set[str] = set()
    parent_joint_by_object: Dict[str, str] = {}
    chain_geometry_names: Dict[str, Set[str]] = {
        chain.id: set() for chain in system.kinematic_chains
    }
    chain_joint_sets = {
        chain.id: set(chain.joints) for chain in system.kinematic_chains
    }

    for go in composed_geom.geometryObjects:
        parent_joint = (
            composed_model.names[go.parentJoint]
            if go.parentJoint < len(composed_model.names)
            else "universe"
        )
        go.name = _namespaced(namespace, go.name)
        owned_names.add(go.name)
        parent_joint_by_object[go.name] = parent_joint

        for chain_id, chain_joints in chain_joint_sets.items():
            if parent_joint in chain_joints:
                chain_geometry_names[chain_id].add(go.name)

    # Within-system allowed pairs are namespaced to match.
    allowed: Set[Tuple[str, str]] = set()
    for pair in system.robot.collision.allowed_pairs:
        allowed.add(
            _canonical_pair(
                _namespaced(namespace, pair.a),
                _namespaced(namespace, pair.b),
            )
        )
    if system.gripper is not None:
        for pair in system.gripper.collision.allowed_pairs:
            allowed.add(
                _canonical_pair(
                    _namespaced(namespace, pair.a),
                    _namespaced(namespace, pair.b),
                )
            )

    return composed_geom, owned_names, allowed, parent_joint_by_object, chain_geometry_names


def _filter_disabled_geometry(geom_model, disabled_links: Set[str]):
    """Return a GeometryModel without objects belonging to disabled links."""
    import pinocchio as pin

    filtered = pin.GeometryModel()
    for go in geom_model.geometryObjects:
        link_name = go.name.rsplit("_", 1)[0]
        if go.name in disabled_links or link_name in disabled_links:
            continue
        filtered.addGeometryObject(go)
    return filtered


# ---------------------------------------------------------------------------
# World object geometry
# ---------------------------------------------------------------------------


def _build_world_object_geometry(
    *,
    obj: WorldObjectDescription,
    source_resolver,
):
    """Build a single `pin.GeometryObject` for one world object.

    The placement encodes the per-geometry origin offset (defaulting to
    identity). The object's runtime world pose lives in the Scene; the
    collision query layer composes `Scene.object_poses[obj.id] @ origin`
    into the geometry data buffer at query time.
    """
    import coal
    import pinocchio as pin

    assert obj.collision is not None  # caller guards on `enabled`
    geom = obj.collision.geometry

    mesh_path = ""
    mesh_scale = np.array([1.0, 1.0, 1.0])

    if isinstance(geom, MeshGeometrySpec):
        resolved_mesh_path = Path(source_resolver(geom.path))
        mesh_path = str(resolved_mesh_path)
        mesh_scale = np.array(geom.scale, dtype=float)
        collision_geometry = _coal_shape_for_mesh(
            resolved_mesh_path,
            mesh_scale,
            obj.collision.processing,
        )
    elif isinstance(geom, MeshDataGeometrySpec):
        collision_geometry = _coal_shape_for_mesh_data(geom.vertices, geom.faces)
    else:
        collision_geometry = _coal_shape_for_primitive(geom)

    origin_se3 = (
        _se3_from_matrix(obj.collision.origin.as_matrix())
        if obj.collision.origin is not None
        else _se3_identity()
    )

    return pin.GeometryObject(
        obj.id,           # name
        0,                # parent_joint = universe (world-fixed)
        0,                # parent_frame = universe
        origin_se3,       # placement (origin offset; runtime pose comes from Scene)
        collision_geometry,
        mesh_path,
        mesh_scale,
    )
