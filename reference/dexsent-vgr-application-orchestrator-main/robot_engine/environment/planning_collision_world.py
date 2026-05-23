from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from robot_engine.environment.base_obstacle import CollisionObstacle
from robot_engine.environment.mesh_obstacle import MeshObstacle
from robot_engine.environment.pointcloud_obstacle import (
    PointCloudObstacle,
    pointcloud_obstacle_from_file,
    pointcloud_obstacle_from_numpy,
)
from robot_engine.environment.pointcloud_processing import PointCloudCollisionConfig
from robot_engine.environment.primitive_obstacle import PrimitiveObstacle, PrimitiveShape


def _coal_collides(geom_a, T_a, geom_b, T_b) -> bool:
    """Low-level Coal pairwise collision test."""
    try:
        import coal
        req = coal.CollisionRequest()
        res = coal.CollisionResult()
        return bool(coal.collide(geom_a, T_a, geom_b, T_b, req, res) > 0)
    except Exception:
        return False


def _coal_distance(geom_a, T_a, geom_b, T_b) -> float:
    """Signed distance between two Coal geometries (negative = penetrating)."""
    try:
        import coal
        req = coal.DistanceRequest()
        res = coal.DistanceResult()
        return float(coal.distance(geom_a, T_a, geom_b, T_b, req, res))
    except Exception:
        return float("inf")


class PlanningCollisionWorld:
    """
    Planning-time collision scene.

    Holds:
      - mesh obstacles       (MeshObstacle)
      - point-cloud obstacles (PointCloudObstacle)
      - primitive obstacles   (PrimitiveObstacle)

    All three are stored in a unified ``obstacles`` dict keyed by name so that
    collision checking only needs to iterate once.

    Designed to be used by PinocchioRobot.is_state_valid().
    """

    def __init__(self) -> None:
        self._obstacles: Dict[str, CollisionObstacle] = {}

    # ------------------------------------------------------------------
    # Obstacle management
    # ------------------------------------------------------------------

    def add_obstacle(self, obstacle: CollisionObstacle) -> None:
        self._obstacles[obstacle.name] = obstacle

    def remove_obstacle(self, name: str) -> None:
        self._obstacles.pop(name, None)

    def clear_obstacles(self) -> None:
        self._obstacles.clear()

    def list_obstacles(self) -> List[str]:
        return sorted(self._obstacles)

    def get_obstacle(self, name: str) -> CollisionObstacle:
        return self._obstacles[name]

    def has_obstacle(self, name: str) -> bool:
        return name in self._obstacles

    # ------------------------------------------------------------------
    # Typed add helpers
    # ------------------------------------------------------------------

    def add_mesh_obstacle(
        self,
        name: str,
        mesh_path: str,
        T_world_mesh: np.ndarray,
        scale: float = 1.0,
        simplify_faces: Optional[int] = None,
    ) -> MeshObstacle:
        obs = MeshObstacle(name, mesh_path, T_world_mesh, scale=scale, simplify_faces=simplify_faces)
        self.add_obstacle(obs)
        return obs

    def add_primitive_obstacle(
        self,
        name: str,
        shape: PrimitiveShape | str,
        T_world_primitive: np.ndarray,
        size: dict,
    ) -> PrimitiveObstacle:
        obs = PrimitiveObstacle(name, shape, T_world_primitive, size)
        self.add_obstacle(obs)
        return obs

    def add_pointcloud_obstacle_from_file(
        self,
        name: str,
        path: str,
        T_world_cloud: np.ndarray,
        config: Optional[PointCloudCollisionConfig] = None,
    ) -> PointCloudObstacle:
        obs = pointcloud_obstacle_from_file(name, path, T_world_cloud, config)
        self.add_obstacle(obs)
        return obs

    def add_pointcloud_obstacle_from_numpy(
        self,
        name: str,
        points_xyz: np.ndarray,
        T_world_cloud: np.ndarray,
        config: Optional[PointCloudCollisionConfig] = None,
    ) -> PointCloudObstacle:
        obs = pointcloud_obstacle_from_numpy(name, points_xyz, T_world_cloud, config)
        self.add_obstacle(obs)
        return obs

    def update_pointcloud_obstacle_from_numpy(
        self,
        name: str,
        points_xyz: np.ndarray,
        T_world_cloud: np.ndarray,
        config: Optional[PointCloudCollisionConfig] = None,
    ) -> PointCloudObstacle:
        """Replace (or create) a point-cloud obstacle from a live sensor array."""
        self.remove_obstacle(name)
        return self.add_pointcloud_obstacle_from_numpy(name, points_xyz, T_world_cloud, config)

    # ------------------------------------------------------------------
    # Collision checking
    # ------------------------------------------------------------------

    def robot_in_world_collision(
        self,
        robot_geometries: list,
        robot_transforms: list,
    ) -> bool:
        """
        Check all robot link geometries against all world obstacles.

        Parameters
        ----------
        robot_geometries : list of coal geometry objects
        robot_transforms : list of coal.Transform3s (one per geometry, matching order)

        Returns True immediately on first collision.
        """
        for obs in self._obstacles.values():
            obs_geom = obs.coal_geometry()
            obs_T = obs.coal_transform()
            if obs_geom is None:
                continue
            for r_geom, r_T in zip(robot_geometries, robot_transforms):
                if r_geom is None:
                    continue
                if _coal_collides(r_geom, r_T, obs_geom, obs_T):
                    return True
        return False

    def robot_min_clearance(
        self,
        robot_geometries: list,
        robot_transforms: list,
    ) -> float:
        """Return the minimum signed distance between any robot link and any obstacle."""
        min_dist = float("inf")
        for obs in self._obstacles.values():
            obs_geom = obs.coal_geometry()
            obs_T = obs.coal_transform()
            if obs_geom is None:
                continue
            for r_geom, r_T in zip(robot_geometries, robot_transforms):
                if r_geom is None:
                    continue
                d = _coal_distance(r_geom, r_T, obs_geom, obs_T)
                if d < min_dist:
                    min_dist = d
        return min_dist

    def __len__(self) -> int:
        return len(self._obstacles)
