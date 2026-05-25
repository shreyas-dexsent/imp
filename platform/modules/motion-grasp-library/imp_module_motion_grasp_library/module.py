"""Compute-Runtime wrapper for synthesize_grasps.

Loads a grasp library at ``configure`` from a JSON file, subscribes the
object's world-frame ``Pose6D``, and publishes a ``Grasps`` message with
every candidate composed into the world frame (sorted by score).
"""

from __future__ import annotations

from typing import Dict

import numpy as np
from scipy.spatial.transform import Rotation

from imp_sdk import Input, Module, Output, QosClass, keyexpr
from imp_sdk.schemas import imp_pb2

from .library import GraspLibrary
from .synthesize import synthesize_grasps


def _pose_to_matrix(position_m, quat_xyzw) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat(np.asarray(quat_xyzw, dtype=float)).as_matrix()
    T[:3, 3] = np.asarray(position_m, dtype=float)
    return T


class SynthesizeGraspsModule(Module):
    name = "motion-grasp-library-synthesize"

    def __init__(
        self,
        station: str,
        object_pose_key: str,
        grasps_path: str,
        plan: str = "grasps",
    ):
        self.station = station
        self.object_pose_key = object_pose_key
        self.grasps_path = grasps_path
        self.out_key = keyexpr.motion(station, plan, "candidates")
        self.library: GraspLibrary | None = None

    def inputs(self):
        return [Input("pose", self.object_pose_key, imp_pb2.Pose6D)]

    def outputs(self):
        return [Output("candidates", self.out_key, imp_pb2.Grasps, QosClass.STATE)]

    def configure(self) -> None:
        self.library = GraspLibrary.from_json(self.grasps_path)

    def compute(self, latest: Dict[str, object]) -> Dict[str, object]:
        pose: imp_pb2.Pose6D = latest["pose"]  # type: ignore[assignment]
        if not pose.valid:
            return {}
        T_world_obj = _pose_to_matrix(pose.position_m, pose.quat_xyzw)
        world_grasps = synthesize_grasps(self.library, T_world_obj)
        return {
            "candidates": imp_pb2.Grasps(
                header=imp_pb2.Header(
                    seq=pose.header.seq,
                    stamp_ns=pose.header.stamp_ns,
                    frame_id=pose.header.frame_id,
                    schema="imp.Grasps/1",
                ),
                candidates=[
                    imp_pb2.Grasp(
                        grasp_id=g.grasp_id,
                        score=g.score,
                        t_obj_gripper=g.t_world_gripper.flatten().tolist(),
                    )
                    for g in world_grasps
                ],
            )
        }
