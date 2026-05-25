"""Unit tests for the pure TfGraph library (no Zenoh)."""

from __future__ import annotations

import numpy as np
import pytest

# Import from the submodule directly so this test does not depend on imp_sdk
# (which the package __init__ pulls in for TfModule). P4 packaging will make
# this moot once editable installs are in place.
from imp_module_spatial_tf.graph import TfGraph, TfLookupError


def _translate(x: float = 0.0, y: float = 0.0, z: float = 0.0) -> np.ndarray:
    T = np.eye(4)
    T[:3, 3] = (x, y, z)
    return T


def _rot_z(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    T = np.eye(4)
    T[:2, :2] = ((c, -s), (s, c))
    return T


def test_identity_self_lookup():
    g = TfGraph()
    g.add_edge("a", "b", _translate(1.0, 0.0, 0.0))
    assert np.allclose(g.lookup("a", "a"), np.eye(4))
    assert np.allclose(g.lookup("b", "b"), np.eye(4))


def test_single_edge_forward_and_inverse():
    g = TfGraph()
    T_ab = _translate(1.0, 2.0, 3.0)
    g.add_edge("a", "b", T_ab)
    assert np.allclose(g.lookup("a", "b"), T_ab)
    assert np.allclose(g.lookup("b", "a"), np.linalg.inv(T_ab))


def test_chain_compose():
    g = TfGraph()
    T_ab = _translate(1.0, 0.0, 0.0)
    T_bc = _rot_z(np.pi / 2) @ _translate(0.0, 1.0, 0.0)
    g.add_edge("a", "b", T_ab)
    g.add_edge("b", "c", T_bc)
    assert np.allclose(g.lookup("a", "c"), T_ab @ T_bc)
    assert np.allclose(g.lookup("c", "a"), np.linalg.inv(T_ab @ T_bc))


def test_rotation_then_translation_composition():
    """T_world_tcp = T_world_base @ T_base_tcp -- the canonical hand-eye style chain."""
    g = TfGraph()
    T_wb = _translate(0.5, 0.0, 1.0)
    T_bt = _rot_z(np.pi / 4) @ _translate(0.1, 0.0, 0.3)
    g.add_edge("world", "base", T_wb)
    g.add_edge("base", "tcp", T_bt)

    p_tcp = np.array([0.0, 0.0, 0.0, 1.0])
    p_world_via_lookup = g.lookup("world", "tcp") @ p_tcp
    p_world_direct = T_wb @ T_bt @ p_tcp
    assert np.allclose(p_world_via_lookup, p_world_direct)


def test_diamond_takes_some_valid_path():
    """Two paths a->d -- BFS will pick one; either must be numerically consistent."""
    g = TfGraph()
    T_ab = _translate(1.0)
    T_ac = _translate(0.0, 1.0)
    T_bd = _translate(0.0, 1.0)
    T_cd = _translate(1.0)
    g.add_edge("a", "b", T_ab)
    g.add_edge("a", "c", T_ac)
    g.add_edge("b", "d", T_bd)
    g.add_edge("c", "d", T_cd)
    got = g.lookup("a", "d")
    # Both legal compositions yield the same world position; check via a point.
    p = np.array([0.0, 0.0, 0.0, 1.0])
    assert np.allclose(got @ p, (T_ab @ T_bd) @ p)
    assert np.allclose(got @ p, (T_ac @ T_cd) @ p)


def test_unknown_frame_raises():
    g = TfGraph()
    g.add_edge("a", "b", _translate(1.0))
    with pytest.raises(TfLookupError):
        g.lookup("a", "z")
    with pytest.raises(TfLookupError):
        g.lookup("z", "a")


def test_disconnected_components_raise():
    g = TfGraph()
    g.add_edge("a", "b", _translate(1.0))
    g.add_edge("c", "d", _translate(1.0))
    with pytest.raises(TfLookupError):
        g.lookup("a", "d")


def test_remove_edge_drops_both_directions():
    g = TfGraph()
    g.add_edge("a", "b", _translate(1.0))
    assert g.has_edge("a", "b") and g.has_edge("b", "a")
    g.remove_edge("a", "b")
    assert not g.has_edge("a", "b")
    assert not g.has_edge("b", "a")


def test_replace_edge_updates_in_place():
    g = TfGraph()
    g.add_edge("a", "b", _translate(1.0))
    g.add_edge("a", "b", _translate(5.0))
    assert np.allclose(g.lookup("a", "b"), _translate(5.0))


def test_frames_includes_both_endpoints():
    g = TfGraph()
    g.add_edge("a", "b", _translate(1.0))
    g.add_edge("b", "c", _translate(1.0))
    assert g.frames() == frozenset({"a", "b", "c"})


def test_validation_rejects_bad_matrix():
    g = TfGraph()
    with pytest.raises(ValueError):
        g.add_edge("a", "b", np.zeros((3, 3)))
    bad = np.eye(4)
    bad[3, 3] = 0.5
    with pytest.raises(ValueError):
        g.add_edge("a", "b", bad)
    bad = np.eye(4)
    bad[0, 0] = np.inf
    with pytest.raises(ValueError):
        g.add_edge("a", "b", bad)


def test_rejects_self_loop_and_empty_names():
    g = TfGraph()
    with pytest.raises(ValueError):
        g.add_edge("a", "a", _translate(1.0))
    with pytest.raises(ValueError):
        g.add_edge("", "a", _translate(1.0))
