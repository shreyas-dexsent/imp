from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class CollisionObstacle(ABC):
    """
    Uniform interface for every obstacle type in the planning scene.

    All subclasses expose:
      - coal_geometry()  → a coal collision geometry (BVH, Box, Sphere, OcTree, …)
      - coal_transform() → a coal.Transform3s placing the geometry in world coordinates
      - world_aabb()     → (2, 3) numpy array [[min_x, min_y, min_z], [max_x, max_y, max_z]]
    """

    def __init__(self, name: str) -> None:
        self.name = name

    @abstractmethod
    def coal_geometry(self):
        """Return the coal collision geometry object."""

    @abstractmethod
    def coal_transform(self):
        """Return a coal.Transform3s for the world pose of this obstacle."""

    @abstractmethod
    def world_aabb(self) -> np.ndarray:
        """Return (2, 3) AABB in world frame: [[xmin, ymin, zmin], [xmax, ymax, zmax]]."""
