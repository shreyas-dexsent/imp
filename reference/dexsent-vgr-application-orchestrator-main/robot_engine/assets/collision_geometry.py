from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import trimesh

from robot_engine.assets.asset_loader import load_trimesh_asset
from robot_engine.assets.mesh_converter import (
    box_to_coal_geometry,
    trimesh_to_coal_geometry,
)
from robot_engine.interfaces.schemas import ObjectAssetConfig


@dataclass
class CollisionGeometry:
    geometry_id: str
    frame_id: str
    mesh: Optional[trimesh.Trimesh] = None
    size_xyz: Optional[np.ndarray] = None
    # Coal geometry object (BVHModelOBBRSS, Box, Sphere, etc.)
    coal_geometry: object = None
    coal_error: Optional[str] = None

    @property
    def aabb_bounds(self):
        if self.mesh is not None:
            return np.asarray(self.mesh.bounds, dtype=float)
        half = np.asarray(self.size_xyz, dtype=float) / 2.0
        return np.vstack([-half, half])


def geometry_from_asset(config: ObjectAssetConfig, simplify_faces: Optional[int] = None) -> CollisionGeometry:
    mesh = load_trimesh_asset(config, simplify_faces=simplify_faces)
    geom = CollisionGeometry(config.object_id, config.frame_id, mesh=mesh)
    try:
        geom.coal_geometry = trimesh_to_coal_geometry(mesh)
    except Exception as exc:
        geom.coal_geometry = None
        geom.coal_error = str(exc)
    return geom


def box_geometry(geometry_id: str, frame_id: str, size_xyz) -> CollisionGeometry:
    size = np.asarray(size_xyz, dtype=float)
    if size.shape != (3,) or not np.isfinite(size).all() or np.any(size <= 0):
        raise ValueError("size_xyz must be finite positive [x, y, z]")
    mesh = trimesh.creation.box(extents=size)
    geom = CollisionGeometry(geometry_id, frame_id, mesh=mesh, size_xyz=size)
    try:
        geom.coal_geometry = box_to_coal_geometry(size)
    except Exception as exc:
        geom.coal_geometry = None
        geom.coal_error = str(exc)
    return geom
