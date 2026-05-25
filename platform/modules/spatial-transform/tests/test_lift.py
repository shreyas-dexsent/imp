"""Pure-math tests for spatial-transform: TfGraph + SE(3) composition.

Runs without ``imp_sdk`` / ``zenoh`` / motion-core installed -- only numpy
and scipy. The wire-level integration test lives in ``test_module.py`` and
is skipped automatically when ``imp_sdk`` isn't importable.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from imp_module_spatial_tf.graph import TfGraph
from imp_module_spatial_transform.lift import lift_pose, pose_to_matrix


def test_identity_chain_is_passthrough():
    g = TfGraph()
    T = np.eye(4)
    g.add_edge("base", "camera", T)
    out = lift_pose(g, "base", "camera", (0.1, 0.2, 0.3), (0, 0, 0, 1))
    assert out is not None
    pos, _ = out
    assert np.allclose(pos, (0.1, 0.2, 0.3))


def test_static_eye_to_hand_translation():
    g = TfGraph()
    T_base_cam = np.eye(4)
    T_base_cam[:3, 3] = (0.5, 0.0, 0.2)
    g.add_edge("base", "camera", T_base_cam)
    out = lift_pose(g, "base", "camera", (0.1, 0.0, 0.0), (0, 0, 0, 1))
    pos, _ = out
    assert np.allclose(pos, (0.6, 0.0, 0.2))


def test_static_eye_to_hand_rotation():
    g = TfGraph()
    T_base_cam = np.eye(4)
    T_base_cam[:3, :3] = Rotation.from_euler("z", np.pi / 2).as_matrix()
    T_base_cam[:3, 3] = (0.2, 0.0, 0.5)
    g.add_edge("base", "camera", T_base_cam)
    out = lift_pose(g, "base", "camera", (0.4, 0.0, 0.0), (0, 0, 0, 1))
    pos, _ = out
    # (0.4, 0, 0) in camera, rotated 90deg about z + translation (0.2, 0, 0.5):
    assert np.allclose(pos, (0.2, 0.4, 0.5))


def test_two_hop_chain():
    g = TfGraph()
    T_base_tcp = np.eye(4)
    T_base_tcp[:3, 3] = (0.1, 0.0, 0.0)
    T_tcp_cam = np.eye(4)
    T_tcp_cam[:3, 3] = (0.0, 0.0, 0.05)
    g.add_edge("base", "tcp", T_base_tcp)
    g.add_edge("tcp", "camera", T_tcp_cam)
    out = lift_pose(g, "base", "camera", (0.0, 0.0, 0.0), (0, 0, 0, 1))
    pos, _ = out
    assert np.allclose(pos, (0.1, 0.0, 0.05))


def test_orientation_round_trips_through_chain():
    g = TfGraph()
    R_base_cam = Rotation.from_euler("xyz", (0.1, -0.2, 0.3)).as_matrix()
    T_base_cam = np.eye(4)
    T_base_cam[:3, :3] = R_base_cam
    g.add_edge("base", "camera", T_base_cam)

    R_cam_obj = Rotation.from_euler("xyz", (0.4, 0.5, -0.1)).as_matrix()
    quat_cam_obj = Rotation.from_matrix(R_cam_obj).as_quat()
    out = lift_pose(g, "base", "camera", (0.0, 0.0, 0.0), quat_cam_obj)
    _, quat = out
    R_got = Rotation.from_quat(quat).as_matrix()
    assert np.allclose(R_got, R_base_cam @ R_cam_obj)


def test_missing_chain_returns_none():
    g = TfGraph()
    g.add_edge("base", "marker", np.eye(4))
    assert lift_pose(g, "base", "camera", (0, 0, 0), (0, 0, 0, 1)) is None


def test_missing_source_frame_returns_none():
    g = TfGraph()
    g.add_edge("base", "camera", np.eye(4))
    assert lift_pose(g, "base", "", (0, 0, 0), (0, 0, 0, 1)) is None


def test_pose_to_matrix_round_trip():
    R = Rotation.from_euler("xyz", (0.1, 0.2, 0.3)).as_matrix()
    pos = (0.1, -0.2, 0.3)
    quat = Rotation.from_matrix(R).as_quat()
    T = pose_to_matrix(pos, quat)
    assert np.allclose(T[:3, :3], R)
    assert np.allclose(T[:3, 3], pos)
