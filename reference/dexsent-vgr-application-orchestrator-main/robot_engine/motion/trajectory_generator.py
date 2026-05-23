from __future__ import annotations

from typing import Dict, List

from robot_engine.kinematics.kinematic_chain import KinematicChain
from robot_engine.motion.motion_primitive import ordered_q
from robot_engine.motion.motion_request import TrajectoryOptions
from robot_engine.motion.motion_result import JointTrajectory
from robot_engine.motion.time_parameterization import time_parameterize_joint_path


def generate_joint_trajectory(chain: KinematicChain, waypoints: List[Dict[str, float]], options: TrajectoryOptions) -> JointTrajectory:
    positions = [ordered_q(chain, waypoint) for waypoint in waypoints]
    return time_parameterize_joint_path(chain.joint_names, positions, options.max_joint_velocity, options.time_step)
