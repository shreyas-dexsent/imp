from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict

import numpy as np

from robot_engine.assets.collision_geometry import CollisionGeometry
from robot_engine.assets.mesh_converter import matrix_to_coal_transform
from robot_engine.interfaces.schemas import Transform3D
from robot_engine.math_utils import as_matrix


@dataclass
class CollisionObject:
    object_id: str
    geometry: CollisionGeometry
    pose: Transform3D
    group: str = "world"
    enabled: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def matrix(self) -> np.ndarray:
        return as_matrix(self.pose)

    def set_pose(self, pose: Transform3D) -> None:
        self.pose = pose

    def coal_object(self):
        """Return a coal.CollisionObject or None if Coal / geometry unavailable."""
        try:
            import coal
        except Exception:
            return None
        if self.geometry.coal_geometry is None:
            return None
        T = matrix_to_coal_transform(self.matrix)
        return coal.CollisionObject(self.geometry.coal_geometry, T)

    def world_aabb(self):
        bounds = self.geometry.aabb_bounds
        corners = np.array(
            [
                [bounds[x, 0], bounds[y, 1], bounds[z, 2], 1.0]
                for x in (0, 1)
                for y in (0, 1)
                for z in (0, 1)
            ],
            dtype=float,
        )
        world = (self.matrix @ corners.T).T[:, :3]
        return np.vstack([world.min(axis=0), world.max(axis=0)])
