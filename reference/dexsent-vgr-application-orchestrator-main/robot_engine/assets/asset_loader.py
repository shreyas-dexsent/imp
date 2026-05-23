from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import trimesh

from robot_engine.interfaces.schemas import AlgorithmError, ObjectAssetConfig


class AssetLoadError(RuntimeError):
    def __init__(self, error: AlgorithmError):
        super().__init__(error.message)
        self.error = error


def load_trimesh_asset(config: ObjectAssetConfig, simplify_faces: Optional[int] = None) -> trimesh.Trimesh:
    path = Path(config.mesh_path)
    if not path.exists():
        raise AssetLoadError(AlgorithmError(code="UNSUPPORTED_FORMAT", message=f"Asset not found: {path}"))

    try:
        loaded = trimesh.load(path, force=None)
    except Exception as exc:
        raise AssetLoadError(AlgorithmError(code="UNSUPPORTED_FORMAT", message=str(exc), details={"path": str(path)})) from exc

    mesh = _as_mesh(loaded, config.point_cloud_mode)
    mesh.apply_scale(float(config.scale))
    validate_mesh(mesh, frame_id=config.frame_id)
    if simplify_faces and len(mesh.faces) > simplify_faces:
        mesh = simplify_mesh(mesh, simplify_faces)
        validate_mesh(mesh, frame_id=config.frame_id)
    return mesh


def _as_mesh(loaded, point_cloud_mode: str) -> trimesh.Trimesh:
    if isinstance(loaded, trimesh.Scene):
        meshes = [geom for geom in loaded.geometry.values() if isinstance(geom, trimesh.Trimesh)]
        if not meshes:
            raise AssetLoadError(AlgorithmError(code="EMPTY_MESH", message="Scene contains no mesh geometry."))
        return trimesh.util.concatenate(meshes)
    if isinstance(loaded, trimesh.PointCloud):
        if point_cloud_mode != "convex_hull":
            raise AssetLoadError(AlgorithmError(code="POINT_CLOUD_UNSUPPORTED", message="Point cloud requires convex_hull mode."))
        try:
            return loaded.convex_hull
        except Exception:
            vertices = np.asarray(loaded.vertices, dtype=float)
            if vertices.shape[0] < 4 or not np.isfinite(vertices).all():
                raise AssetLoadError(AlgorithmError(code="POINT_CLOUD_UNSUPPORTED", message="Point cloud cannot be converted to a proxy mesh."))
            extents = vertices.max(axis=0) - vertices.min(axis=0)
            center = (vertices.max(axis=0) + vertices.min(axis=0)) / 2.0
            mesh = trimesh.creation.box(extents=np.maximum(extents, 1e-6))
            mesh.apply_translation(center)
            return mesh
    if isinstance(loaded, trimesh.Trimesh):
        return loaded
    raise AssetLoadError(AlgorithmError(code="UNSUPPORTED_FORMAT", message=f"Unsupported asset type: {type(loaded).__name__}"))


def validate_mesh(mesh: trimesh.Trimesh, frame_id: str) -> None:
    if not frame_id:
        raise AssetLoadError(AlgorithmError(code="INVALID_MESH", message="frame_id is required."))
    if mesh.vertices is None or len(mesh.vertices) == 0 or mesh.faces is None or len(mesh.faces) == 0:
        raise AssetLoadError(AlgorithmError(code="EMPTY_MESH", message="Mesh has no vertices or faces."))
    if not np.isfinite(mesh.vertices).all():
        raise AssetLoadError(AlgorithmError(code="NON_FINITE_VERTICES", message="Mesh contains NaN or infinite vertices."))
    extents = np.asarray(mesh.extents, dtype=float)
    if not np.isfinite(extents).all() or float(extents.max(initial=0.0)) <= 0:
        raise AssetLoadError(AlgorithmError(code="INVALID_MESH", message="Mesh has invalid extents."))


def simplify_mesh(mesh: trimesh.Trimesh, target_faces: int) -> trimesh.Trimesh:
    if target_faces <= 0 or len(mesh.faces) <= target_faces:
        return mesh
    try:
        return mesh.simplify_quadratic_decimation(target_faces)
    except Exception:
        face_ids = np.linspace(0, len(mesh.faces) - 1, target_faces, dtype=int)
        return trimesh.Trimesh(vertices=mesh.vertices.copy(), faces=mesh.faces[face_ids].copy(), process=True)
