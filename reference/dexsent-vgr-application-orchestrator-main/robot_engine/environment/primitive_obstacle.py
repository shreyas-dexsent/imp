from __future__ import annotations

from enum import Enum

import numpy as np

from robot_engine.assets.mesh_converter import (
    box_to_coal_geometry,
    capsule_to_coal_geometry,
    cylinder_to_coal_geometry,
    matrix_to_coal_transform,
    sphere_to_coal_geometry,
)
from robot_engine.environment.base_obstacle import CollisionObstacle


class PrimitiveShape(str, Enum):
    BOX = "box"
    SPHERE = "sphere"
    CYLINDER = "cylinder"
    CAPSULE = "capsule"


class PrimitiveObstacle(CollisionObstacle):
    """
    Analytic primitive obstacle: box, sphere, cylinder, or capsule.

    Parameters
    ----------
    name : str
    shape : PrimitiveShape or str ("box", "sphere", "cylinder", "capsule")
    T_world_primitive : (4, 4) array-like
    size : dict with keys depending on shape:
        box       → {"x": float, "y": float, "z": float}
        sphere    → {"radius": float}
        cylinder  → {"radius": float, "length": float}
        capsule   → {"radius": float, "length": float}
    """

    def __init__(
        self,
        name: str,
        shape: PrimitiveShape | str,
        T_world_primitive: np.ndarray,
        size: dict,
    ) -> None:
        super().__init__(name)
        self.shape = PrimitiveShape(shape)
        self._T = np.asarray(T_world_primitive, dtype=np.float64)
        self.size = size
        self._coal_geom = _build_geometry(self.shape, size)
        self._aabb_half = _compute_half_extents(self.shape, size)

    # ------------------------------------------------------------------
    # CollisionObstacle interface
    # ------------------------------------------------------------------

    def coal_geometry(self):
        return self._coal_geom

    def coal_transform(self):
        return matrix_to_coal_transform(self._T)

    def world_aabb(self) -> np.ndarray:
        center = self._T[:3, 3]
        return np.vstack([center - self._aabb_half, center + self._aabb_half])

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def set_transform(self, T: np.ndarray) -> None:
        self._T = np.asarray(T, dtype=np.float64)


def _build_geometry(shape: PrimitiveShape, size: dict):
    if shape == PrimitiveShape.BOX:
        return box_to_coal_geometry([size["x"], size["y"], size["z"]])
    if shape == PrimitiveShape.SPHERE:
        return sphere_to_coal_geometry(size["radius"])
    if shape == PrimitiveShape.CYLINDER:
        return cylinder_to_coal_geometry(size["radius"], size["length"])
    if shape == PrimitiveShape.CAPSULE:
        return capsule_to_coal_geometry(size["radius"], size["length"])
    raise ValueError(f"Unknown primitive shape: {shape}")


def _compute_half_extents(shape: PrimitiveShape, size: dict) -> np.ndarray:
    if shape == PrimitiveShape.BOX:
        return np.array([size["x"] / 2, size["y"] / 2, size["z"] / 2])
    if shape == PrimitiveShape.SPHERE:
        r = size["radius"]
        return np.array([r, r, r])
    if shape in (PrimitiveShape.CYLINDER, PrimitiveShape.CAPSULE):
        r = size["radius"]
        half_l = size["length"] / 2
        return np.array([r, r, half_l])
    return np.zeros(3)
