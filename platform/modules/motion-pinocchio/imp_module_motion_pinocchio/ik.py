"""Inverse-kinematics imp module (spec §9).

Compute-Runtime wrapper over motion-core: builds the resolved `KinematicModel`
at `configure`, then given a `PoseTarget` (base frame) and a `RobotState` seed,
solves with `algorithms.kinematics.ik.ik_local` and publishes a `JointSolution`.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from scipy.spatial.transform import Rotation

from imp_sdk import Input, Module, Output, QosClass, keyexpr
from imp_sdk.schemas import imp_pb2

from algorithms.descriptions import RobotSystemDescription
from algorithms.kinematics.ik import IKStatus, ik_local
from algorithms.resolved import KinematicModel


def _pose_to_matrix(position_m, quat_xyzw) -> np.ndarray:
    t = np.eye(4)
    t[:3, :3] = Rotation.from_quat(np.asarray(quat_xyzw, dtype=float)).as_matrix()
    t[:3, 3] = np.asarray(position_m, dtype=float)
    return t


class IkModule(Module):
    name = "motion-pinocchio-ik"

    def __init__(
        self,
        station: str,
        robot: str,
        robot_system_path: str,
        chain: str = "arm",
        tcp_frame: Optional[str] = None,
        plan: str = "ik",
    ):
        self.station = station
        self.robot = robot
        self.robot_system_path = robot_system_path
        self.chain = chain
        self.tcp_frame = tcp_frame
        self.target_key = keyexpr.motion(station, plan, "target")
        self.seed_key = keyexpr.hal(station, robot, "state")
        self.out_key = keyexpr.motion(station, plan, "solution")
        self.model: Optional[KinematicModel] = None

    def inputs(self):
        return [
            Input("target", self.target_key, imp_pb2.PoseTarget),
            Input("seed", self.seed_key, imp_pb2.RobotState),
        ]

    def outputs(self):
        return [Output("solution", self.out_key, imp_pb2.JointSolution, QosClass.STATE)]

    def configure(self) -> None:
        system = RobotSystemDescription.from_yaml(self.robot_system_path)
        self.model = KinematicModel.from_robot_system(system)
        if self.tcp_frame is None:
            spec = self.model.chain(self.chain)
            self.tcp_frame = spec.tcp_frame or spec.tip_frame
        self.n_active = len(self.model.active_joint_names)

    def compute(self, latest: Dict[str, object]) -> Dict[str, object]:
        target = latest["target"]
        seed = latest["seed"]
        q_seed = np.asarray(seed.q, dtype=float)
        if q_seed.shape[0] != self.n_active:
            return {}
        t_target = _pose_to_matrix(target.position_m, target.quat_xyzw)
        result = ik_local(self.model, self.tcp_frame, t_target, q_seed)
        ok = result.status is IKStatus.SUCCESS and result.q is not None
        return {
            "solution": imp_pb2.JointSolution(
                header=imp_pb2.Header(schema="imp.JointSolution/1"),
                q=(result.q.tolist() if ok else []),
                valid=ok,
                reject_reason=("" if ok else result.status.value),
            )
        }
