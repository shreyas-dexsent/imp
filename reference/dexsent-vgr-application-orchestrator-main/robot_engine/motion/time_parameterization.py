from __future__ import annotations

from typing import List

import numpy as np

from robot_engine.motion.motion_result import JointTrajectory


def time_parameterize_joint_path(joint_names: List[str], positions: List[List[float]], max_velocity: float, time_step: float) -> JointTrajectory:
    if not positions:
        return JointTrajectory(joint_names=joint_names, positions=[], times=[])
    times = [0.0]
    max_velocity = max(float(max_velocity), 1e-9)
    min_step = max(float(time_step), 1e-9)
    for prev, cur in zip(positions[:-1], positions[1:]):
        delta = float(np.max(np.abs(np.asarray(cur, dtype=float) - np.asarray(prev, dtype=float))))
        times.append(times[-1] + max(min_step, delta / max_velocity))
    velocities = _finite_difference(positions, times)
    accelerations = _finite_difference(velocities, times)
    return JointTrajectory(joint_names=joint_names, positions=positions, times=times, velocities=velocities, accelerations=accelerations)


def _finite_difference(values: List[List[float]], times: List[float]) -> List[List[float]]:
    if not values:
        return []
    out = [[0.0 for _ in values[0]]]
    for prev, cur, t0, t1 in zip(values[:-1], values[1:], times[:-1], times[1:]):
        dt = max(float(t1 - t0), 1e-9)
        out.append(((np.asarray(cur, dtype=float) - np.asarray(prev, dtype=float)) / dt).tolist())
    return out
