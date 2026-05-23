from __future__ import annotations

import numpy as np


def compute_segment_durations(q_waypoints, velocity_limits, acceleration_limits=None):
    q = np.asarray(q_waypoints, dtype=float)
    if len(q) < 2:
        return []
    v = np.maximum(np.asarray(velocity_limits, dtype=float), 1e-9)
    return [float(np.max(np.abs(b - a) / v)) for a, b in zip(q[:-1], q[1:])]


def synchronize_segment_times(segment_times):
    return list(np.maximum(segment_times, 1e-9))


def time_scale_path_constant_velocity(q_waypoints, velocity_limits):
    return synchronize_segment_times(compute_segment_durations(q_waypoints, velocity_limits))


def time_scale_path_trapezoidal(q_waypoints, velocity_limits, acceleration_limits):
    return synchronize_segment_times(compute_segment_durations(q_waypoints, velocity_limits, acceleration_limits))

