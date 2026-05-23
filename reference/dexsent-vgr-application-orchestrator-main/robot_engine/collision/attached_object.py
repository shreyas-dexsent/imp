from __future__ import annotations

from dataclasses import dataclass

from robot_engine.math_utils import as_matrix, to_transform


@dataclass
class AttachedObject:
    object_id: str
    link_or_tcp_frame: str
    grasp_transform: object


def attach_object(world, object_id: str, link_or_tcp_frame: str, grasp_transform):
    obj = world.get(object_id)
    obj.metadata["attached_to"] = link_or_tcp_frame
    obj.metadata["grasp_transform"] = grasp_transform
    obj.group = "attached_object"
    return AttachedObject(object_id, link_or_tcp_frame, grasp_transform)


def update_attached_object_pose(world, object_id: str, parent_pose):
    obj = world.get(object_id)
    grasp = obj.metadata.get("grasp_transform")
    if grasp is None:
        return obj.pose
    pose = to_transform(parent_pose.parent_frame, obj.pose.child_frame, as_matrix(parent_pose) @ as_matrix(grasp))
    world.update_pose(object_id, pose)
    return pose

