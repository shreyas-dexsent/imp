from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

from robot_engine.assets.collision_geometry import box_geometry, geometry_from_asset
from robot_engine.collision.collision_matrix import CollisionMatrix
from robot_engine.collision.collision_object import CollisionObject
from robot_engine.interfaces.schemas import CollisionMatrix as CollisionMatrixSchema, CollisionObjectConfig, ObjectAssetConfig, Transform3D


class CollisionWorld:
    def __init__(self, matrix: CollisionMatrixSchema | None = None):
        self.objects: Dict[str, CollisionObject] = {}
        self.matrix = CollisionMatrix(matrix)

    def add_object(self, obj: CollisionObject) -> None:
        self.objects[obj.object_id] = obj

    def remove_object(self, object_id: str) -> None:
        self.objects.pop(object_id, None)

    def add_from_config(self, config: CollisionObjectConfig) -> CollisionObject:
        if config.asset_path:
            geom = geometry_from_asset(ObjectAssetConfig(
                object_id=config.object_id,
                mesh_path=config.asset_path,
                frame_id=config.frame_id,
                scale=float(getattr(config, "scale", 1.0) or 1.0),
            ))
        elif config.size_xyz:
            geom = box_geometry(config.object_id, config.frame_id, config.size_xyz)
        else:
            raise ValueError("CollisionObjectConfig requires asset_path or size_xyz")
        obj = CollisionObject(config.object_id, geom, config.pose, config.group)
        self.add_object(obj)
        return obj

    def update_pose(self, object_id: str, pose: Transform3D) -> None:
        self.objects[object_id].set_pose(pose)

    def update_object_pose(self, object_id: str, pose: Transform3D) -> None:
        self.update_pose(object_id, pose)

    def update_robot_state(self, q) -> None:
        self.robot_state = dict(q) if isinstance(q, dict) else q

    def attach_object(self, object_id: str, link_or_tcp_frame: str, grasp_transform) -> None:
        obj = self.objects[object_id]
        obj.group = "attached_object"
        obj.metadata["attached_to"] = link_or_tcp_frame
        obj.metadata["grasp_transform"] = grasp_transform

    def detach_object(self, object_id: str) -> None:
        obj = self.objects[object_id]
        obj.group = "object"
        obj.metadata.pop("attached_to", None)
        obj.metadata.pop("grasp_transform", None)

    def list_objects(self) -> List[str]:
        return sorted(self.objects)

    def set_matrix(self, matrix: CollisionMatrixSchema) -> None:
        self.matrix = CollisionMatrix(matrix)

    def active_pairs(self) -> List[Tuple[str, str]]:
        return self.matrix.active_pairs(self.objects.keys())

    def get(self, object_id: str) -> CollisionObject:
        return self.objects[object_id]

    def get_object(self, object_id: str) -> CollisionObject:
        return self.get(object_id)

    def check_state(self, q):
        self.update_robot_state(q)
        from robot_engine.collision.collision_checker import check_active_pairs

        return check_active_pairs(self)

    def clone(self):
        other = CollisionWorld()
        other.objects = dict(self.objects)
        other.matrix = self.matrix
        return other

    @classmethod
    def from_configs(cls, configs: Iterable[CollisionObjectConfig], matrix: CollisionMatrixSchema | None = None):
        world = cls(matrix)
        for config in configs:
            world.add_from_config(config)
        return world
