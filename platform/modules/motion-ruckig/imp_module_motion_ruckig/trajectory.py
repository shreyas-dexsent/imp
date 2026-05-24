"""Time-parameterization imp module (spec §9).

Compute-Runtime wrapper over motion-core: builds the `KinematicModel` (for the
velocity/acceleration/jerk limits) at `configure`, then turns each incoming
`Path` into a time-stamped `Trajectory` via `algorithms.trajectory.time_parameterize`
(Ruckig backend).
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from imp_sdk import Input, Module, Output, QosClass, keyexpr
from imp_sdk.schemas import imp_pb2

from algorithms.descriptions import RobotSystemDescription
from algorithms.planning import Path
from algorithms.resolved import KinematicModel
from algorithms.trajectory import TrajectoryStatus, time_parameterize


class TrajectoryModule(Module):
    name = "motion-ruckig-trajectory"

    def __init__(self, station: str, robot: str, robot_system_path: str, plan: str = "plan"):
        self.station = station
        self.robot = robot
        self.robot_system_path = robot_system_path
        self.path_key = keyexpr.motion(station, plan, "path")
        self.out_key = keyexpr.motion(station, plan, "trajectory")
        self.model: Optional[KinematicModel] = None

    def inputs(self):
        return [Input("path", self.path_key, imp_pb2.Path)]

    def outputs(self):
        return [Output("trajectory", self.out_key, imp_pb2.Trajectory, QosClass.STATE)]

    def configure(self) -> None:
        system = RobotSystemDescription.from_yaml(self.robot_system_path)
        self.model = KinematicModel.from_robot_system(system)
        self.n_active = len(self.model.active_joint_names)

    def compute(self, latest: Dict[str, object]) -> Dict[str, object]:
        path = latest["path"]
        n_dof = path.n_dof
        if not n_dof or len(path.q_wp) < 2 * n_dof:  # need at least two waypoints
            return {}
        waypoints = np.asarray(path.q_wp, dtype=float).reshape(-1, n_dof)
        mc_path = Path(waypoints=waypoints, joint_names=tuple(self.model.active_joint_names))
        result = time_parameterize(mc_path, self.model)
        if result.status is not TrajectoryStatus.SUCCESS or result.trajectory is None:
            return {}
        traj = result.trajectory
        positions = np.asarray(traj.positions, dtype=float)
        return {
            "trajectory": imp_pb2.Trajectory(
                header=imp_pb2.Header(schema="imp.Trajectory/1"),
                t_s=np.asarray(traj.times, dtype=float).tolist(),
                q_wp=positions.flatten().tolist(),
                n_dof=positions.shape[1],
            )
        }
