from __future__ import annotations

from itertools import combinations

import numpy as np


def aabb_overlap(a_bounds, b_bounds, margin: float = 0.0) -> bool:
    a = np.asarray(a_bounds, dtype=float)
    b = np.asarray(b_bounds, dtype=float)
    return bool(np.all(a[0] <= b[1] + margin) and np.all(b[0] <= a[1] + margin))


def broadphase_candidates(objects, matrix=None, margin: float = 0.0):
    ids = sorted(objects.keys()) if isinstance(objects, dict) else [obj.object_id for obj in objects]
    obj_map = objects if isinstance(objects, dict) else {obj.object_id: obj for obj in objects}
    pairs = matrix.active_pairs(ids) if matrix is not None else combinations(ids, 2)
    out = []
    for a, b in pairs:
        if aabb_overlap(obj_map[a].world_aabb(), obj_map[b].world_aabb(), margin):
            out.append((a, b))
    return out

