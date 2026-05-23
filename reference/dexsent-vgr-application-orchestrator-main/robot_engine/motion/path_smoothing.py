from __future__ import annotations

from typing import Dict, List

import numpy as np

from robot_engine.kinematics.kinematic_chain import KinematicChain
from robot_engine.motion.motion_primitive import ordered_q


def remove_duplicate_joint_waypoints(chain: KinematicChain, waypoints: List[Dict[str, float]], tolerance: float = 1e-12) -> List[Dict[str, float]]:
    if not waypoints:
        return []
    smoothed = [waypoints[0]]
    last = np.asarray(ordered_q(chain, waypoints[0]), dtype=float)
    for waypoint in waypoints[1:]:
        cur = np.asarray(ordered_q(chain, waypoint), dtype=float)
        if float(np.linalg.norm(cur - last)) > tolerance:
            smoothed.append(waypoint)
            last = cur
    return smoothed
