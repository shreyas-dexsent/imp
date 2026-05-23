from __future__ import annotations

from robot_engine.assets.collision_geometry import box_geometry, geometry_from_asset
from robot_engine.interfaces.schemas import BinAssetConfig, CollisionObjectConfig, GripperConfig, ObjectAssetConfig


def load_collision_geometry(config):
    if isinstance(config, CollisionObjectConfig):
        if config.asset_path:
            return geometry_from_asset(ObjectAssetConfig(object_id=config.object_id, mesh_path=config.asset_path, frame_id=config.frame_id))
        if config.size_xyz:
            return box_geometry(config.object_id, config.frame_id, config.size_xyz)
    if isinstance(config, ObjectAssetConfig):
        return geometry_from_asset(config)
    raise ValueError("Unsupported collision geometry config.")


def load_robot_collision_geometry(robot_model):
    return []


def load_gripper_collision_geometry(gripper_config: GripperConfig):
    if not gripper_config.mesh_path:
        return []
    return [geometry_from_asset(ObjectAssetConfig(object_id=gripper_config.gripper_id, mesh_path=gripper_config.mesh_path, frame_id=gripper_config.root_frame))]


def load_object_collision_geometry(object_config: ObjectAssetConfig):
    return geometry_from_asset(object_config)


def load_bin_collision_geometry(bin_config: BinAssetConfig):
    if bin_config.mesh_path:
        return geometry_from_asset(ObjectAssetConfig(object_id=bin_config.bin_id, mesh_path=bin_config.mesh_path, frame_id=bin_config.frame_id))
    if bin_config.size_xyz:
        return box_geometry(bin_config.bin_id, bin_config.frame_id, bin_config.size_xyz)
    raise ValueError("Bin geometry requires mesh_path or size_xyz.")

