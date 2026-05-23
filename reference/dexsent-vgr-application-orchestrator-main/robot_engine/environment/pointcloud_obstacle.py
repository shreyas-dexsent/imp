from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from robot_engine.assets.mesh_converter import matrix_to_coal_transform
from robot_engine.environment.base_obstacle import CollisionObstacle
from robot_engine.environment.pointcloud_processing import (
    PointCloudCollisionConfig,
    PointCloudProcessor,
)


class PointCloudObstacle(CollisionObstacle):
    """
    Point-cloud obstacle: raw points → crop/filter/downsample → Coal OcTree.

    Parameters
    ----------
    name : str
    pcd : open3d.geometry.PointCloud
        Already preprocessed point cloud (use PointCloudProcessor.preprocess).
    T_world_cloud : (4, 4) array-like
        Transform from cloud frame to world frame.
    config : PointCloudCollisionConfig
        Processing / resolution config (used to build the octree).
    """

    def __init__(
        self,
        name: str,
        pcd,
        T_world_cloud: np.ndarray,
        config: Optional[PointCloudCollisionConfig] = None,
    ) -> None:
        super().__init__(name)
        self._config = config or PointCloudCollisionConfig()
        self._T = np.asarray(T_world_cloud, dtype=np.float64)

        # Store raw (preprocessed) cloud
        self._pcd = pcd
        self._points = np.asarray(pcd.points, dtype=np.float64)

        # Build Coal octree
        self._processor = PointCloudProcessor(self._config)
        self._coal_geom = self._processor.to_coal_octree(pcd)

        # Cache AABB for quick rejection
        if len(self._points) > 0:
            self._local_aabb = np.vstack([self._points.min(axis=0), self._points.max(axis=0)])
        else:
            self._local_aabb = np.zeros((2, 3))

    # ------------------------------------------------------------------
    # CollisionObstacle interface
    # ------------------------------------------------------------------

    def coal_geometry(self):
        return self._coal_geom

    def coal_transform(self):
        return matrix_to_coal_transform(self._T)

    def world_aabb(self) -> np.ndarray:
        if len(self._points) == 0:
            center = self._T[:3, 3]
            return np.vstack([center, center])
        ones = np.ones((len(self._points), 1))
        world = (self._T @ np.hstack([self._points, ones]).T).T[:, :3]
        return np.vstack([world.min(axis=0), world.max(axis=0)])

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def update_points(self, pcd) -> None:
        """Replace the point cloud with a new preprocessed open3d.PointCloud."""
        self._pcd = pcd
        self._points = np.asarray(pcd.points, dtype=np.float64)
        self._coal_geom = self._processor.to_coal_octree(pcd)
        if len(self._points) > 0:
            self._local_aabb = np.vstack([self._points.min(axis=0), self._points.max(axis=0)])
        else:
            self._local_aabb = np.zeros((2, 3))

    def set_transform(self, T: np.ndarray) -> None:
        self._T = np.asarray(T, dtype=np.float64)

    @property
    def num_points(self) -> int:
        return len(self._points)

    @property
    def config(self) -> PointCloudCollisionConfig:
        return self._config


# ------------------------------------------------------------------
# Convenience factory functions
# ------------------------------------------------------------------

def pointcloud_obstacle_from_file(
    name: str,
    path: str | Path,
    T_world_cloud: np.ndarray,
    config: Optional[PointCloudCollisionConfig] = None,
) -> PointCloudObstacle:
    """Load a PLY/PCD/XYZ file and build a PointCloudObstacle."""
    cfg = config or PointCloudCollisionConfig()
    processor = PointCloudProcessor(cfg)
    raw = processor.load_from_file(path)
    processed = processor.preprocess(raw)
    return PointCloudObstacle(name, processed, T_world_cloud, cfg)


def pointcloud_obstacle_from_numpy(
    name: str,
    points_xyz: np.ndarray,
    T_world_cloud: np.ndarray,
    config: Optional[PointCloudCollisionConfig] = None,
) -> PointCloudObstacle:
    """Build a PointCloudObstacle from a (N, 3) NumPy array."""
    cfg = config or PointCloudCollisionConfig()
    processor = PointCloudProcessor(cfg)
    raw = processor.from_numpy(points_xyz)
    processed = processor.preprocess(raw)
    return PointCloudObstacle(name, processed, T_world_cloud, cfg)
