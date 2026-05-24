"""Joint-space planning imp module (spec §9).

Compute-Runtime wrapper over motion-core: builds the resolved `CollisionModel` +
`Scene` + `KinematicModel` from a `world.yaml` at `configure`, then given a start
`RobotState` and a goal `JointSolution` plans a collision-free joint path with
`algorithms.planning.plan_joint` (OMPL) and publishes a `Path`.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from imp_sdk import Input, Module, Output, QosClass, keyexpr
from imp_sdk.schemas import imp_pb2

from algorithms.descriptions import WorldDescription
from algorithms.planning import PlanOptions, plan_joint
from algorithms.resolved import CollisionModel, KinematicModel, Scene


class PlanModule(Module):
    name = "motion-ompl-plan"

    def __init__(
        self,
        station: str,
        robot: str,
        world_path: str,
        plan: str = "plan",
        random_seed: int = 0,
    ):
        self.station = station
        self.robot = robot
        self.world_path = world_path
        self.random_seed = random_seed
        self.start_key = keyexpr.hal(station, robot, "state")
        self.goal_key = keyexpr.motion(station, plan, "goal")
        self.out_key = keyexpr.motion(station, plan, "path")
        self.model: Optional[KinematicModel] = None

    def inputs(self):
        return [
            Input("goal", self.goal_key, imp_pb2.JointSolution),
            Input("start", self.start_key, imp_pb2.RobotState),
        ]

    def outputs(self):
        return [Output("path", self.out_key, imp_pb2.Path, QosClass.STATE)]

    def configure(self) -> None:
        world = WorldDescription.from_yaml(self.world_path)
        self.scene = Scene.from_world(world, CollisionModel.from_world(world))
        self.model = KinematicModel.from_robot_system(world.robots[0].robot_system)
        self.n_active = len(self.model.active_joint_names)

    def compute(self, latest: Dict[str, object]) -> Dict[str, object]:
        start, goal = latest["start"], latest["goal"]
        q_start = np.asarray(start.q, dtype=float)
        q_goal = np.asarray(goal.q, dtype=float)
        if not goal.valid or q_start.shape[0] != self.n_active or q_goal.shape[0] != self.n_active:
            return {}
        result = plan_joint(self.model, self.scene, q_start, q_goal,
                            options=PlanOptions(random_seed=self.random_seed))
        header = imp_pb2.Header(schema="imp.Path/1")
        if result.success and result.path is not None:
            wp = np.asarray(result.path.waypoints, dtype=float)
            return {"path": imp_pb2.Path(header=header, q_wp=wp.flatten().tolist(), n_dof=wp.shape[1])}
        return {"path": imp_pb2.Path(header=header, q_wp=[], n_dof=self.n_active)}
