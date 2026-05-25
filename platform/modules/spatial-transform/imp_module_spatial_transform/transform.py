"""Camera-frame ``Pose6D`` -> base-frame ``PoseTarget`` via a tf chain.

The Compute-Runtime wrapper: subscribes the perception pose, the tf stream,
and (in eye-in-hand mode) the robot state; delegates the actual SE(3) math
to :mod:`imp_module_spatial_transform.lift`.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from imp_sdk import Input, Module, Output, QosClass, keyexpr
from imp_sdk.schemas import imp_pb2

from imp_module_spatial_tf.graph import TfGraph

from .lift import lift_pose


class TransformModule(Module):
    name = "spatial-transform"

    def __init__(
        self,
        station: str,
        pose_key: str,
        out_plan: str = "transform",
        base_frame: str = "base",
        *,
        eye_in_hand: bool = False,
        robot: Optional[str] = None,
        robot_system_path: Optional[str] = None,
        chain: str = "arm",
        tcp_frame: Optional[str] = None,
    ):
        self.station = station
        self.pose_key = pose_key
        self.base_frame = base_frame
        self.out_key = keyexpr.motion(station, out_plan, "target")
        self.tf_key = keyexpr.tf(station)

        self.eye_in_hand = eye_in_hand
        self.robot = robot
        self.robot_system_path = robot_system_path
        self.chain = chain
        self.tcp_frame = tcp_frame
        self.state_key = keyexpr.hal(station, robot, "state") if robot else None

        if eye_in_hand and not robot_system_path:
            raise ValueError("eye_in_hand=True requires robot_system_path")
        if eye_in_hand and not robot:
            raise ValueError("eye_in_hand=True requires robot id (for the state topic)")

        self.graph = TfGraph()
        self.model = None
        self.n_active = 0

    # --- Compute Runtime contract ---------------------------------------

    def inputs(self):
        specs = [
            Input("pose", self.pose_key, imp_pb2.Pose6D),
            Input("edge", self.tf_key, imp_pb2.TfEdge),
        ]
        if self.eye_in_hand:
            specs.append(Input("state", self.state_key, imp_pb2.RobotState))
        return specs

    def outputs(self):
        return [Output("target", self.out_key, imp_pb2.PoseTarget, QosClass.STATE)]

    def configure(self) -> None:
        if not self.eye_in_hand:
            return
        # Lazy import: motion-core deps only needed for eye-in-hand mode.
        from algorithms.descriptions import RobotSystemDescription
        from algorithms.resolved import KinematicModel

        system = RobotSystemDescription.from_yaml(self.robot_system_path)
        self.model = KinematicModel.from_robot_system(system)
        if self.tcp_frame is None:
            spec = self.model.chain(self.chain)
            self.tcp_frame = spec.tcp_frame or spec.tip_frame
        if self.base_frame is None:
            self.base_frame = system.robot.base_frame
        self.n_active = len(self.model.active_joint_names)

    def compute(self, latest: Dict[str, object]) -> Dict[str, object]:
        # 1) ingest the latest tf edge
        edge: imp_pb2.TfEdge = latest["edge"]  # type: ignore[assignment]
        if len(edge.matrix) == 16 and edge.parent_frame and edge.child_frame:
            self.graph.add_edge(
                edge.parent_frame,
                edge.child_frame,
                [edge.matrix[i * 4 : (i + 1) * 4] for i in range(4)],
            )

        # 2) eye-in-hand: inject base->tcp from FK on the live joint state
        if self.eye_in_hand:
            from algorithms.kinematics.fk import fk_local

            state: imp_pb2.RobotState = latest["state"]  # type: ignore[assignment]
            q = np.asarray(state.q, dtype=float)
            if q.shape[0] != self.n_active:
                return {}
            T_base_tcp = fk_local(self.model, q, self.tcp_frame)
            self.graph.add_edge(self.base_frame, self.tcp_frame, T_base_tcp)

        # 3) compose and publish
        pose: imp_pb2.Pose6D = latest["pose"]  # type: ignore[assignment]
        if not pose.valid:
            return {}
        lifted = lift_pose(
            self.graph,
            self.base_frame,
            pose.header.frame_id or "",
            pose.position_m,
            pose.quat_xyzw,
        )
        if lifted is None:
            return {}
        pos, quat = lifted
        return {
            "target": imp_pb2.PoseTarget(
                header=imp_pb2.Header(
                    seq=pose.header.seq,
                    stamp_ns=pose.header.stamp_ns,
                    frame_id=self.base_frame,
                    schema="imp.PoseTarget/1",
                ),
                target_frame=self.base_frame,
                position_m=pos,
                quat_xyzw=quat,
            )
        }
