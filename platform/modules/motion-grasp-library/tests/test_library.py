"""Pure-library tests for motion-grasp-library (no imp_sdk / zenoh required)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from imp_module_motion_grasp_library import Grasp, GraspLibrary, synthesize_grasps


def _T(x=0.0, y=0.0, z=0.0):
    T = np.eye(4)
    T[:3, 3] = (x, y, z)
    return T


def test_grasp_accepts_nested_4x4_and_flat_16():
    g1 = Grasp(grasp_id="g", score=0.5, t_obj_gripper=np.eye(4))
    g2 = Grasp(grasp_id="g", score=0.5, t_obj_gripper=np.eye(4).flatten().tolist())
    assert np.allclose(g1.t_obj_gripper, g2.t_obj_gripper)


def test_grasp_rejects_bad_score():
    with pytest.raises(ValueError):
        Grasp(grasp_id="g", score=1.5, t_obj_gripper=np.eye(4))


def test_grasp_rejects_bad_matrix():
    bad = np.eye(4)
    bad[3, 3] = 0.7
    with pytest.raises(ValueError):
        Grasp(grasp_id="g", score=0.5, t_obj_gripper=bad)


def test_library_list_sorted_descending():
    lib = GraspLibrary.from_iterable(
        "matka",
        [
            Grasp("a", 0.2, np.eye(4)),
            Grasp("b", 0.9, np.eye(4)),
            Grasp("c", 0.5, np.eye(4)),
        ],
    )
    assert [g.grasp_id for g in lib.list()] == ["b", "c", "a"]
    assert len(lib) == 3


def test_library_upserts_by_id():
    lib = GraspLibrary.from_iterable("matka", [Grasp("a", 0.2, np.eye(4))])
    lib.add(Grasp("a", 0.9, _T(0.1)))  # replace
    assert len(lib) == 1
    assert lib.list()[0].score == 0.9


def test_library_from_json(tmp_path):
    grasps_path = tmp_path / "grasps.json"
    grasps_path.write_text(
        json.dumps(
            {
                "object_id": "matka",
                "grasps": [
                    {"grasp_id": "g1", "score": 0.9, "t_obj_gripper": np.eye(4).tolist()},
                    {"grasp_id": "g2", "score": 0.5, "t_obj_gripper": _T(0.1).tolist()},
                ],
            }
        )
    )
    lib = GraspLibrary.from_json(grasps_path)
    assert lib.object_id == "matka"
    assert [g.grasp_id for g in lib.list()] == ["g1", "g2"]


def test_synthesize_grasps_composes_with_object_pose():
    lib = GraspLibrary.from_iterable(
        "matka",
        [Grasp("g1", 0.9, _T(0.05, 0.0, 0.1)), Grasp("g2", 0.4, _T(-0.05, 0.0, 0.1))],
    )
    T_world_obj = _T(0.5, 0.0, 0.2)
    out = synthesize_grasps(lib, T_world_obj)
    assert [g.grasp_id for g in out] == ["g1", "g2"]  # score-sorted
    assert np.allclose(out[0].t_world_gripper[:3, 3], (0.55, 0.0, 0.3))
    assert np.allclose(out[1].t_world_gripper[:3, 3], (0.45, 0.0, 0.3))


def test_synthesize_grasps_validates_pose_shape():
    lib = GraspLibrary.from_iterable("matka", [])
    with pytest.raises(ValueError):
        synthesize_grasps(lib, np.eye(3))
