"""Pure-math helpers for spatial-transform.

Kept dependency-free (numpy + scipy + TfGraph only) so unit tests can run
without ``imp_sdk`` / ``zenoh`` installed. The wire-marshalling shim lives
in ``transform.py``.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np
from scipy.spatial.transform import Rotation

from imp_module_spatial_tf.graph import TfGraph, TfLookupError


def pose_to_matrix(position_m: Sequence[float], quat_xyzw: Sequence[float]) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat(np.asarray(quat_xyzw, dtype=float)).as_matrix()
    T[:3, 3] = np.asarray(position_m, dtype=float)
    return T


def matrix_to_pose(T: np.ndarray) -> Tuple[list, list]:
    pos = T[:3, 3].tolist()
    quat = Rotation.from_matrix(T[:3, :3]).as_quat().tolist()  # xyzw
    return pos, quat


def lift_pose(
    graph: TfGraph,
    base_frame: str,
    source_frame: str,
    position_m: Sequence[float],
    quat_xyzw: Sequence[float],
) -> Optional[Tuple[list, list]]:
    """Compose ``T_base_source @ pose`` and return (position, quat_xyzw) in
    ``base_frame``. Returns ``None`` if the tf chain isn't connected yet."""
    if not source_frame:
        return None
    try:
        T_base_src = graph.lookup(base_frame, source_frame)
    except TfLookupError:
        return None
    T_src_obj = pose_to_matrix(position_m, quat_xyzw)
    return matrix_to_pose(T_base_src @ T_src_obj)
