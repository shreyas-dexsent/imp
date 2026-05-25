"""``synthesize_grasps``: compose object-frame grasp candidates into the world.

Pure function -- the imp module wrapper marshals protobuf around it. Keeps
testing trivial: no imp_sdk / zenoh / motion-core dependency.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np

from .library import Grasp, GraspLibrary


@dataclass
class WorldGrasp:
    """A grasp candidate already lifted into the world frame."""

    grasp_id: str
    score: float
    t_world_gripper: np.ndarray


def synthesize_grasps(
    library: GraspLibrary,
    T_world_object: np.ndarray,
) -> List[WorldGrasp]:
    """Compose every candidate in ``library`` with the live object pose.

    ``t_world_gripper = T_world_object @ t_obj_gripper`` for each candidate.
    Returned list is ordered by descending score.
    """
    T_world_object = np.asarray(T_world_object, dtype=float)
    if T_world_object.shape != (4, 4):
        raise ValueError(f"T_world_object must be (4,4); got {T_world_object.shape}")
    return [
        WorldGrasp(
            grasp_id=g.grasp_id,
            score=g.score,
            t_world_gripper=T_world_object @ g.t_obj_gripper,
        )
        for g in library.list()
    ]
