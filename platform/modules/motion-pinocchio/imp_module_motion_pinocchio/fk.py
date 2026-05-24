"""Forward-kinematics imp module (spec §9).

Thin Compute-Runtime wrapper over motion-core: builds the resolved
``KinematicModel`` from a robot-system YAML at ``configure``, then on each
``RobotState`` computes the TCP pose with `algorithms.kinematics.fk_local` and
publishes a `Pose6D` in the robot base frame.

The heavy lifting (Pinocchio model composition, mimic expansion, chain slicing)
lives in motion-core and is exercised by its own test suite; this module only
adapts topic messages to/from the library.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from scipy.spatial.transform import Rotation

from imp_sdk import Input, Module, Output, QosClass, keyexpr
from imp_sdk.schemas import imp_pb2

from algorithms.descriptions import RobotSystemDescription
from algorithms.kinematics.fk import fk_local
from algorithms.resolved import KinematicModel


class FkModule(Module):
    name = "motion-pinocchio-fk"

    def __init__(
        self,
        station: str,
        robot: str,
        robot_system_path: str,
        chain: str = "arm",
        tcp_frame: Optional[str] = None,
        plan: str = "fk",
    ):
        self.station = station
        self.robot = robot
        self.robot_system_path = robot_system_path
        self.chain = chain
        self.tcp_frame = tcp_frame
        self.state_key = keyexpr.hal(station, robot, "state")
        self.out_key = keyexpr.motion(station, plan, "tcp")
        self.model: Optional[KinematicModel] = None

    def inputs(self):
        return [Input("state", self.state_key, imp_pb2.RobotState)]

    def outputs(self):
        return [Output("tcp", self.out_key, imp_pb2.Pose6D, QosClass.STATE)]

    def configure(self) -> None:
        system = RobotSystemDescription.from_yaml(self.robot_system_path)
        self.model = KinematicModel.from_robot_system(system)
        if self.tcp_frame is None:
            spec = self.model.chain(self.chain)
            self.tcp_frame = spec.tcp_frame or spec.tip_frame
        self.base_frame = system.robot.base_frame
        self.n_active = len(self.model.active_joint_names)

    def compute(self, latest: Dict[str, object]) -> Dict[str, object]:
        state = latest["state"]
        q = np.asarray(state.q, dtype=float)
        if q.shape[0] != self.n_active:
            return {}  # state DOF doesn't match this model; ignore
        t = fk_local(self.model, q, self.tcp_frame)
        pos = t[:3, 3]
        quat_xyzw = Rotation.from_matrix(t[:3, :3]).as_quat()  # x, y, z, w
        pose = imp_pb2.Pose6D(
            header=imp_pb2.Header(
                seq=state.header.seq,
                stamp_ns=state.header.stamp_ns,
                frame_id=self.base_frame,
                schema="imp.Pose6D/1",
            ),
            object_id=self.tcp_frame,
            position_m=pos.tolist(),
            quat_xyzw=quat_xyzw.tolist(),
            confidence=1.0,
            valid=True,
        )
        return {"tcp": pose}
