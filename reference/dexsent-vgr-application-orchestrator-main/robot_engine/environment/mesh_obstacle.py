from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh

from robot_engine.assets.mesh_converter import (
    matrix_to_coal_transform,
    trimesh_to_solid_coal_geometry,
)
from robot_engine.environment.base_obstacle import CollisionObstacle


class MeshObstacle(CollisionObstacle):
    """
    Static mesh obstacle loaded from an STL / OBJ / DAE / GLB file.

    Parameters
    ----------
    name : str
    mesh_path : str or Path
        Path to the mesh file (trimesh-loadable).
    T_world_mesh : (4, 4) array-like
        Homogeneous transform placing the mesh in world coordinates.
    scale : float, optional
        Uniform scale factor applied to the mesh (default 1.0).
    simplify_faces : int or None
        If given, decimate the mesh to at most this many faces.
    """

    def __init__(
        self,
        name: str,
        mesh_path: str | Path,
        T_world_mesh: np.ndarray,
        scale: float = 1.0,
        simplify_faces: int | None = None,
    ) -> None:
        super().__init__(name)
        self._T_world_mesh = np.asarray(T_world_mesh, dtype=np.float64)
        self.mesh_path = str(mesh_path)
        self.mesh, self._coal_geom = _load_mesh(mesh_path, scale, simplify_faces)

    # ------------------------------------------------------------------
    # CollisionObstacle interface
    # ------------------------------------------------------------------

    def coal_geometry(self):
        return self._coal_geom

    def coal_transform(self):
        return matrix_to_coal_transform(self._T_world_mesh)

    def world_aabb(self) -> np.ndarray:
        verts = np.asarray(self.mesh.vertices, dtype=float)
        ones = np.ones((len(verts), 1))
        world = (self._T_world_mesh @ np.hstack([verts, ones]).T).T[:, :3]
        return np.vstack([world.min(axis=0), world.max(axis=0)])

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def set_transform(self, T_world_mesh: np.ndarray) -> None:
        self._T_world_mesh = np.asarray(T_world_mesh, dtype=np.float64)


def _load_mesh(
    path: str | Path,
    scale: float,
    simplify_faces: int | None,
):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Mesh file not found: {path}")

    loaded = trimesh.load(str(path), force=None)

    if isinstance(loaded, trimesh.Scene):
        meshes = [g for g in loaded.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if not meshes:
            raise ValueError(f"No triangle meshes in scene: {path}")
        mesh = trimesh.util.concatenate(meshes)
    elif isinstance(loaded, trimesh.Trimesh):
        mesh = loaded
    else:
        raise ValueError(f"Unsupported mesh type {type(loaded)} from {path}")

    if scale != 1.0:
        mesh = mesh.copy()
        mesh.apply_scale(scale)

    if simplify_faces is not None and len(mesh.faces) > simplify_faces:
        try:
            mesh = mesh.simplify_quadratic_decimation(simplify_faces)
        except Exception:
            pass

    coal_geom = trimesh_to_solid_coal_geometry(mesh)
    return mesh, coal_geom
