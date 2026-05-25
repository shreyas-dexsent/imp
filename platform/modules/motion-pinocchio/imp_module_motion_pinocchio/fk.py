"""Forward-kinematics imp module (spec §9).

Thin Compute-Runtime wrapper over motion-core: builds the resolved
``KinematicModel`` and a ``Scene`` from a ``world.yaml`` at ``configure``, then
on each ``RobotState`` computes the TCP pose in **world coordinates** using
``algorithms.kinematics.fk(scene, robot_id, q, frame_id)`` and publishes a
``Pose6D`` whose ``frame_id`` is the world frame. This closes the Scene-fill
seam for FK (debt D3 in PLAN.md): FK now composes through the robot's
``base_pose`` recorded in ``world.yaml`` rather than dropping back into the
robot-local base frame.

The heavy lifting (Pinocchio model composition, mimic expansion, chain slicing,
world<-base composition) lives in motion-core and is exercised by its own test
suite; this module only adapts topic messages to and from the library.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from scipy.spatial.transform import Rotation

from imp_sdk import Input, Module, Output, QosClass, keyexpr
from imp_sdk.schemas import imp_pb2

from algorithms.descriptions import WorldDescription
from algorithms.kinematics.fk import fk
from algorithms.resolved import KinematicModel, Scene


class FkModule(Module):
    name = "motion-pinocchio-fk"

    def __init__(
        self,
        station: str,
        robot: str,
        world_path: str,
        world_robot: str = "arm",
        chain: str = "arm",
        tcp_frame: Optional[str] = None,
        plan: str = "fk",
    ):
        self.station = station
        self.robot = robot
        self.world_path = world_path
        self.world_robot = world_robot
        self.chain = chain
        self.tcp_frame = tcp_frame
        self.state_key = keyexpr.hal(station, robot, "state")
        self.out_key = keyexpr.motion(station, plan, "tcp")
        self.model: Optional[KinematicModel] = None
        self.scene: Optional[Scene] = None
        self.world_frame: str = "world"

    def inputs(self):
        return [Input("state", self.state_key, imp_pb2.RobotState)]

    def outputs(self):
        return [Output("tcp", self.out_key, imp_pb2.Pose6D, QosClass.STATE)]

    def configure(self) -> None:
        world = WorldDescription.from_yaml(self.world_path)
        self.scene = Scene.from_world(world)
        self.world_frame = world.world_frame
        self.model = KinematicModel.from_robot_system(world.robot(self.world_robot).robot_system)
        if self.tcp_frame is None:
            spec = self.model.chain(self.chain)
            self.tcp_frame = spec.tcp_frame or spec.tip_frame
        self.n_active = len(self.model.active_joint_names)

    def compute(self, latest: Dict[str, object]) -> Dict[str, object]:
        state = latest["state"]
        q = np.asarray(state.q, dtype=float)
        if q.shape[0] != self.n_active:
            return {}  # state DOF doesn't match this model; ignore

        # Scene-fill: live joint state in the Scene before any operation reads it.
        self.scene.set_robot_state(self.world_robot, q)

        T_world_tcp = fk(self.scene, self.world_robot, q, self.tcp_frame)
        pos = T_world_tcp[:3, 3]
        quat_xyzw = Rotation.from_matrix(T_world_tcp[:3, :3]).as_quat()
        pose = imp_pb2.Pose6D(
            header=imp_pb2.Header(
                seq=state.header.seq,
                stamp_ns=state.header.stamp_ns,
                frame_id=self.world_frame,
                schema="imp.Pose6D/1",
            ),
            object_id=self.tcp_frame,
            position_m=pos.tolist(),
            quat_xyzw=quat_xyzw.tolist(),
            confidence=1.0,
            valid=True,
        )
        return {"tcp": pose}
