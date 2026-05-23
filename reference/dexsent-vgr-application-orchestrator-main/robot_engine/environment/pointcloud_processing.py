from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class PointCloudCollisionConfig:
    """
    Configuration for point-cloud collision preprocessing.

    All distances are in metres.
    """

    voxel_size: float = 0.03
    """Side length of each occupied voxel (= octree resolution)."""

    inflation_radius: float = 0.02
    """Inflate each occupied voxel by this amount (effective_voxel = voxel_size + 2 * inflation_radius)."""

    remove_statistical_outliers: bool = True
    statistical_nb_neighbors: int = 20
    statistical_std_ratio: float = 2.0

    crop_enabled: bool = False
    crop_min: np.ndarray = field(default_factory=lambda: np.array([-1.0, -1.0, 0.0]))
    crop_max: np.ndarray = field(default_factory=lambda: np.array([1.5, 1.0, 1.5]))

    @classmethod
    def from_dict(cls, d: dict) -> "PointCloudCollisionConfig":
        cfg = cls()
        cfg.voxel_size = float(d.get("voxel_size", cfg.voxel_size))
        cfg.inflation_radius = float(d.get("inflation_radius", cfg.inflation_radius))
        cfg.remove_statistical_outliers = bool(d.get("remove_statistical_outliers", cfg.remove_statistical_outliers))
        cfg.statistical_nb_neighbors = int(d.get("statistical_nb_neighbors", cfg.statistical_nb_neighbors))
        cfg.statistical_std_ratio = float(d.get("statistical_std_ratio", cfg.statistical_std_ratio))
        crop = d.get("crop", {})
        cfg.crop_enabled = bool(crop.get("enabled", cfg.crop_enabled))
        if "min" in crop:
            cfg.crop_min = np.asarray(crop["min"], dtype=float)
        if "max" in crop:
            cfg.crop_max = np.asarray(crop["max"], dtype=float)
        return cfg


def _import_o3d():
    try:
        import open3d as o3d
        return o3d
    except ImportError as exc:
        raise RuntimeError(
            "open3d is required for point-cloud preprocessing. "
            "Install the frozen stack with: pip install pin coal trimesh open3d scipy ompl ruckig pyyaml meshcat numpy"
        ) from exc


# ---------------------------------------------------------------------------
# PointCloudProcessor
# ---------------------------------------------------------------------------

class PointCloudProcessor:
    """
    Preprocessing pipeline: load → crop → outlier removal → voxel downsample.

    Uses Open3D as the single point-cloud processing backend.
    """

    def __init__(self, config: Optional[PointCloudCollisionConfig] = None) -> None:
        self.config = config or PointCloudCollisionConfig()
        self._o3d = _import_o3d()

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load_from_file(self, path: str | Path):
        """Load a PLY / PCD / XYZ / … file and return a point cloud object."""
        o3d = self._o3d
        pcd = o3d.io.read_point_cloud(str(path))
        if pcd.is_empty():
            raise ValueError(f"Point cloud is empty: {path}")
        return pcd

    def from_numpy(self, points_xyz: np.ndarray):
        """Wrap a (N, 3) NumPy array as a point cloud object."""
        pts = np.asarray(points_xyz, dtype=float)
        if pts.ndim != 2 or pts.shape[1] != 3:
            raise ValueError("points_xyz must have shape (N, 3)")

        o3d = self._o3d
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        return pcd

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    def preprocess(self, pcd):
        """Apply crop → outlier removal → voxel downsample per config."""
        o3d = self._o3d
        cfg = self.config
        return self._preprocess_o3d(pcd, o3d, cfg)

    def _preprocess_o3d(self, pcd, o3d, cfg):
        if cfg.crop_enabled:
            aabb = o3d.geometry.AxisAlignedBoundingBox(
                min_bound=cfg.crop_min.astype(np.float64),
                max_bound=cfg.crop_max.astype(np.float64),
            )
            pcd = pcd.crop(aabb)

        if cfg.remove_statistical_outliers and len(pcd.points) > cfg.statistical_nb_neighbors:
            pcd, _ = pcd.remove_statistical_outlier(
                nb_neighbors=cfg.statistical_nb_neighbors,
                std_ratio=cfg.statistical_std_ratio,
            )

        pcd = pcd.voxel_down_sample(cfg.voxel_size)
        return pcd

    # ------------------------------------------------------------------
    # Coal octree construction
    # ------------------------------------------------------------------

    def to_coal_octree(self, pcd):
        """
        Convert a preprocessed point cloud to a coal.OcTree.

        The octree resolution is `voxel_size + 2 * inflation_radius`.
        """
        import coal

        pts = np.asarray(pcd.points, dtype=np.float64)
        if len(pts) == 0:
            raise ValueError("Cannot build octree from empty point cloud.")

        effective_resolution = self.config.voxel_size + 2.0 * self.config.inflation_radius
        return coal.makeOctree(pts, effective_resolution)

    # ------------------------------------------------------------------
    # Voxel utilities
    # ------------------------------------------------------------------

    def to_voxel_centers(self, pcd) -> np.ndarray:
        """Return (M, 3) array of unique voxel centre positions from the processed cloud."""
        pts = np.asarray(pcd.points, dtype=float)
        if len(pts) == 0:
            return np.zeros((0, 3))
        indices = np.floor(pts / self.config.voxel_size).astype(int)
        unique_idx = np.unique(indices, axis=0)
        return (unique_idx.astype(float) + 0.5) * self.config.voxel_size

    def points(self, pcd) -> np.ndarray:
        """Return (N, 3) NumPy array of points."""
        return np.asarray(pcd.points, dtype=float)
