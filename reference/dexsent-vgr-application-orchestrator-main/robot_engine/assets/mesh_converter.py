from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import trimesh

try:
    import coal
    _COAL_AVAILABLE = True
except Exception:
    coal = None
    _COAL_AVAILABLE = False


@dataclass
class MeshConversionOptions:
    simplify_faces: Optional[int] = None


def trimesh_to_coal_geometry(mesh: trimesh.Trimesh):
    """Convert a trimesh.Trimesh to a coal.BVHModelOBBRSS collision geometry."""
    if not _COAL_AVAILABLE:
        raise RuntimeError("coal is not installed. Run: pip install coal")
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int32)

    verts_vec = coal.StdVec_Vec3s()
    for v in vertices:
        verts_vec.append(v)

    tris_vec = coal.StdVec_Triangle()
    for f in faces:
        tris_vec.append(coal.Triangle(int(f[0]), int(f[1]), int(f[2])))

    bvh = coal.BVHModelOBBRSS()
    bvh.beginModel(len(faces), len(vertices))
    bvh.addSubModel(verts_vec, tris_vec)
    bvh.endModel()
    return bvh


def trimesh_to_solid_coal_geometry(mesh: trimesh.Trimesh):
    """
    Convert a mesh into solid collision geometry when Coal can represent it.

    Coal BVH triangle meshes behave like surfaces, so a robot link fully inside
    a closed mesh can report positive distance to the nearest triangle. For
    watertight convex obstacle meshes, use a Coal convex hull so the obstacle is
    treated as a solid volume. Non-convex meshes still fall back to BVH surface
    geometry and should be decomposed into convex parts or primitives upstream
    when solid occupancy is required.
    """
    if not _COAL_AVAILABLE:
        raise RuntimeError("coal is not installed. Run: pip install coal")

    if bool(getattr(mesh, "is_watertight", False)) and bool(getattr(mesh, "is_convex", False)):
        vertices = np.asarray(mesh.vertices, dtype=np.float64)
        verts_vec = coal.StdVec_Vec3s()
        for v in vertices:
            verts_vec.append(v)
        return coal.ConvexBase.convexHull(verts_vec, True, "Qt")

    return trimesh_to_coal_geometry(mesh)


def box_to_coal_geometry(size_xyz):
    """Create a coal.Box collision geometry."""
    if not _COAL_AVAILABLE:
        raise RuntimeError("coal is not installed.")
    s = np.asarray(size_xyz, dtype=float)
    return coal.Box(float(s[0]), float(s[1]), float(s[2]))


def sphere_to_coal_geometry(radius: float):
    if not _COAL_AVAILABLE:
        raise RuntimeError("coal is not installed.")
    return coal.Sphere(float(radius))


def cylinder_to_coal_geometry(radius: float, length: float):
    if not _COAL_AVAILABLE:
        raise RuntimeError("coal is not installed.")
    return coal.Cylinder(float(radius), float(length))


def capsule_to_coal_geometry(radius: float, length: float):
    if not _COAL_AVAILABLE:
        raise RuntimeError("coal is not installed.")
    return coal.Capsule(float(radius), float(length))


def matrix_to_coal_transform(matrix: np.ndarray):
    """Convert a 4x4 homogeneous transform to coal.Transform3s."""
    if not _COAL_AVAILABLE:
        raise RuntimeError("coal is not installed.")
    M = np.asarray(matrix, dtype=np.float64)
    R = M[:3, :3]
    t = M[:3, 3]
    T = coal.Transform3s()
    T.setRotation(R)
    T.setTranslation(t)
    return T


def se3_to_coal_transform(se3_obj):
    """Convert a pinocchio.SE3 to coal.Transform3s."""
    if not _COAL_AVAILABLE:
        raise RuntimeError("coal is not installed.")
    T = coal.Transform3s()
    T.setRotation(np.asarray(se3_obj.rotation, dtype=np.float64))
    T.setTranslation(np.asarray(se3_obj.translation, dtype=np.float64))
    return T


def coal_identity_transform():
    if not _COAL_AVAILABLE:
        raise RuntimeError("coal is not installed.")
    return coal.Transform3s.Identity()

