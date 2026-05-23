from __future__ import annotations

import numpy as np


def remove_redundant_waypoints(q_waypoints, tolerance: float = 1e-9):
    out = []
    for q in q_waypoints:
        if not out or np.linalg.norm(np.asarray(q) - np.asarray(out[-1])) > tolerance:
            out.append(q)
    return out


def validate_shortcut(q_i, q_j, collision_checker=None):
    if collision_checker is None:
        return True
    result = collision_checker.check_segment(q_i, q_j)
    return result.success and not result.collision


def shortcut_smooth_path(q_waypoints, collision_checker=None, iterations: int = 100):
    path = remove_redundant_waypoints(q_waypoints)
    if len(path) <= 2:
        return path, {"accepted": 0, "attempted": 0}
    accepted = 0
    rng = np.random.default_rng(11)
    for _ in range(iterations):
        if len(path) <= 2:
            break
        i, j = sorted(rng.choice(len(path), size=2, replace=False))
        if j <= i + 1:
            continue
        if validate_shortcut(path[i], path[j], collision_checker):
            path = path[: i + 1] + path[j:]
            accepted += 1
    return path, {"accepted": accepted, "attempted": iterations}

