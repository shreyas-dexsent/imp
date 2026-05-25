"""Cartesian (straight-line) planning imp module (spec §9).

Compute-Runtime wrapper over motion-core's ``plan_cartesian``: subscribes a
world-frame ``PoseTarget`` for the TCP and a seed ``RobotState``, samples the
linear pose path, solves IK at each sample (using the prior q as the next
seed), and publishes the resulting joint-space ``Path``.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from scipy.spatial.transform import Rotation

from imp_sdk import Input, Module, Output, QosClass, keyexpr
from imp_sdk.schemas import imp_pb2

from algorithms.descriptions import WorldDescription
from algorithms.planning import PlanOptions, plan_cartesian
from algorithms.resolved import CollisionModel, KinematicModel, Scene


def _pose_to_matrix(position_m, quat_xyzw) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat(np.asarray(quat_xyzw, dtype=float)).as_matrix()
    T[:3, 3] = np.asarray(position_m, dtype=float)
    return T


class CartesianPlanModule(Module):
    name = "motion-cartesian-plan"

    def __init__(
        self,
        station: str,
        robot: str,
        world_path: str,
        world_robot: str = "arm",
        chain: str = "arm",
        tcp_frame: Optional[str] = None,
        plan: str = "cartesian",
        random_seed: int = 0,
    ):
        self.station = station
        self.robot = robot
        self.world_path = world_path
        self.world_robot = world_robot
        self.chain = chain
        self.tcp_frame = tcp_frame
        self.random_seed = random_seed
        self.start_key = keyexpr.hal(station, robot, "state")
        self.goal_key = keyexpr.motion(station, plan, "goal")
        self.out_key = keyexpr.motion(station, plan, "path")
        self.model: Optional[KinematicModel] = None

    def inputs(self):
        return [
            Input("goal", self.goal_key, imp_pb2.PoseTarget),
            Input("start", self.start_key, imp_pb2.RobotState),
        ]

    def outputs(self):
        return [Output("path", self.out_key, imp_pb2.Path, QosClass.STATE)]

    def configure(self) -> None:
        world = WorldDescription.from_yaml(self.world_path)
        self.scene = Scene.from_world(world, CollisionModel.from_world(world))
        self.model = KinematicModel.from_robot_system(world.robot(self.world_robot).robot_system)
        if self.tcp_frame is None:
            spec = self.model.chain(self.chain)
            self.tcp_frame = spec.tcp_frame or spec.tip_frame
        self.n_active = len(self.model.active_joint_names)

    def compute(self, latest: Dict[str, object]) -> Dict[str, object]:
        start = latest["start"]
        goal = latest["goal"]
        q_seed = np.asarray(start.q, dtype=float)
        if q_seed.shape[0] != self.n_active:
            return {}

        T_goal = _pose_to_matrix(goal.position_m, goal.quat_xyzw)

        # Keep robot_states fresh for any other Scene consumer.
        self.scene.set_robot_state(self.world_robot, q_seed)

        result = plan_cartesian(
            self.scene,
            self.world_robot,
            self.tcp_frame,
            T_start=None,
            T_goal=T_goal,
            q_seed=q_seed,
            options=PlanOptions(random_seed=self.random_seed),
        )

        header = imp_pb2.Header(schema="imp.Path/1")
        if result.success and result.path is not None:
            wp = np.asarray(result.path.waypoints, dtype=float)
            return {"path": imp_pb2.Path(header=header, q_wp=wp.flatten().tolist(), n_dof=wp.shape[1])}
        return {"path": imp_pb2.Path(header=header, q_wp=[], n_dof=self.n_active)}
