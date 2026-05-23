import json

from orchestrator.tasks.bin_picking import (
    _load_pick_contacts_model,
    _select_parallel_jaw_grasp,
)


def _grasp(group_index, center_x, approach_axis):
    return {
        "id": f"grasp_g{group_index}",
        "contact_a_local_m": [center_x - 0.01, 0.0, 0.0],
        "contact_b_local_m": [center_x + 0.01, 0.0, 0.0],
        "center_local_m": [center_x, 0.0, 0.0],
        "jaw_axis_local": [1.0, 0.0, 0.0],
        "approach_axis_local": approach_axis,
        "opening_width_m": 0.02,
        "generator_group_index": group_index,
        "grasp_family_label": "internal" if group_index in (1, 2) else "external",
    }


def _custom_grasp(group_index, center_x, jaw_axis, approach_axis, grasp_id=None):
    return {
        "id": grasp_id or f"grasp_g{group_index}",
        "contact_a_local_m": [center_x - 0.01, 0.0, 0.0],
        "contact_b_local_m": [center_x + 0.01, 0.0, 0.0],
        "center_local_m": [center_x, 0.0, 0.0],
        "jaw_axis_local": jaw_axis,
        "approach_axis_local": approach_axis,
        "opening_width_m": 0.02,
        "generator_group_index": group_index,
        "grasp_family_label": "internal" if group_index in (1, 2) else "external",
    }


def _model_with_policy(grasps):
    return {
        "default_gripper_type": "parallel_jaw",
        "selection_mode": "camera_normal",
        "tool_axis": "z",
        "jaw_axis": "x",
        "grasps": grasps,
        "grasp_family_selection_policy": {
            "enabled": True,
            "mode": "priority_groups",
            "fallback": "next_segmentation",
            "groups": [
                {"group_index": 1, "priority": 1, "order": 0, "enabled": True},
                {"group_index": 2, "priority": 2, "order": 1, "enabled": True},
            ],
        },
    }


def test_parallel_jaw_selection_prefers_first_enabled_priority_group():
    model = _model_with_policy(
        [
            _grasp(2, 0.0, [0.0, 0.0, 1.0]),
            _grasp(1, 0.1, [0.0, 0.0, 1.0]),
        ]
    )

    selected = _select_parallel_jaw_grasp(
        model,
        object_center_base_m=[0.0, 0.0, 0.0],
        object_quat_base_xyzw=[0.0, 0.0, 0.0, 1.0],
        reference_position_m=[0.0, 0.0, 0.0],
        camera_position_m=[0.0, 0.0, 1.0],
        camera_z_axis_base=[0.0, 0.0, 1.0],
        reference_quat_base_xyzw=[0.0, 0.0, 0.0, 1.0],
        jaw_tool_axis="x",
        approach_tool_axis="z",
        max_approach_angle_deg=45.0,
        pick_cfg={},
    )

    assert selected is not None
    assert selected["generator_group_index"] == 1
    assert selected["grasp_family_priority"] == 1


def test_parallel_jaw_selection_requires_horizontal_jaw_axis_over_priority():
    model = _model_with_policy(
        [
            _custom_grasp(
                1,
                0.0,
                jaw_axis=[0.0, 0.0, 1.0],
                approach_axis=[0.0, 0.0, 1.0],
                grasp_id="vertical_priority_grasp",
            ),
            _custom_grasp(
                2,
                0.1,
                jaw_axis=[1.0, 0.0, 0.0],
                approach_axis=[0.0, 0.0, 1.0],
                grasp_id="horizontal_lower_priority_grasp",
            ),
        ]
    )

    selected = _select_parallel_jaw_grasp(
        model,
        object_center_base_m=[0.0, 0.0, 0.0],
        object_quat_base_xyzw=[0.0, 0.0, 0.0, 1.0],
        reference_position_m=[0.0, 0.0, 0.0],
        camera_position_m=[0.0, 0.0, 1.0],
        camera_z_axis_base=[0.0, 0.0, 1.0],
        reference_quat_base_xyzw=[0.0, 0.0, 0.0, 1.0],
        jaw_tool_axis="x",
        approach_tool_axis="z",
        max_approach_angle_deg=120.0,
        pick_cfg={"parallel_jaw_max_jaw_axis_tilt_from_xy_deg": 5.0},
    )

    assert selected is not None
    assert selected["id"] == "horizontal_lower_priority_grasp"
    assert selected["generator_group_index"] == 2
    assert selected["jaw_axis_horizontal_ok"] is True
    assert selected["jaw_axis_abs_z"] == 0.0


def test_parallel_jaw_selection_prefers_approach_vector_toward_camera():
    model = _model_with_policy(
        [
            _custom_grasp(
                1,
                0.0,
                jaw_axis=[1.0, 0.0, 0.0],
                approach_axis=[0.0, 0.0, -1.0],
                grasp_id="away_from_camera",
            ),
            _custom_grasp(
                1,
                0.1,
                jaw_axis=[1.0, 0.0, 0.0],
                approach_axis=[0.0, 0.0, 1.0],
                grasp_id="toward_camera",
            ),
        ]
    )

    selected = _select_parallel_jaw_grasp(
        model,
        object_center_base_m=[0.0, 0.0, 0.0],
        object_quat_base_xyzw=[0.0, 0.0, 0.0, 1.0],
        reference_position_m=[0.0, 0.0, 0.0],
        camera_position_m=[0.0, 0.0, 1.0],
        camera_z_axis_base=[0.0, 0.0, 1.0],
        reference_quat_base_xyzw=[0.0, 0.0, 0.0, 1.0],
        jaw_tool_axis="x",
        approach_tool_axis="z",
        max_approach_angle_deg=120.0,
        pick_cfg={"parallel_jaw_max_jaw_axis_tilt_from_xy_deg": 5.0},
    )

    assert selected is not None
    assert selected["id"] == "toward_camera"
    assert selected["approach_to_camera_score"] > 0.9


def test_parallel_jaw_selection_skips_disabled_priority_groups():
    model = _model_with_policy([_grasp(3, 0.0, [0.0, 0.0, 1.0])])

    selected = _select_parallel_jaw_grasp(
        model,
        object_center_base_m=[0.0, 0.0, 0.0],
        object_quat_base_xyzw=[0.0, 0.0, 0.0, 1.0],
        reference_position_m=[0.0, 0.0, 0.0],
        camera_position_m=[0.0, 0.0, 1.0],
        camera_z_axis_base=[0.0, 0.0, 1.0],
        reference_quat_base_xyzw=[0.0, 0.0, 0.0, 1.0],
        jaw_tool_axis="x",
        approach_tool_axis="z",
        max_approach_angle_deg=45.0,
        pick_cfg={},
    )

    assert selected is None


def test_grasp_family_labels_json_loads_priority_policy(tmp_path):
    contacts_path = tmp_path / "pick_contacts.json"
    contacts_path.write_text(
        json.dumps(
            {
                "format": "vgr_grasp_candidates/v2",
                "default_gripper_type": "parallel_jaw",
                "tool_axis": "z",
                "jaw_axis": "x",
                "grasps": [_grasp(1, 0.0, [0.0, 0.0, 1.0])],
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "grasp_family_labels.json").write_text(
        json.dumps(
            {
                "format": "vgr_grasp_family_labels/v2",
                "labels": {"1": "internal", "2": "internal", "3": "external"},
                "selection_policy": {
                    "mode": "priority_groups",
                    "fallback": "next_segmentation",
                    "groups": [
                        {"priority": 1, "group_index": 1, "label": "internal"},
                        {"priority": 2, "group_index": 2, "label": "internal"},
                        {
                            "priority": 3,
                            "group_index": 3,
                            "label": "external",
                            "enabled": False,
                        },
                    ],
                },
            }
        ),
        encoding="utf-8",
    )

    model = _load_pick_contacts_model(
        module_params={},
        pick_cfg={"pick_contacts_file": str(contacts_path)},
        search_roots=[tmp_path],
    )

    assert model is not None
    assert model["grasp_family_labels"] == {
        "1": "internal",
        "2": "internal",
        "3": "external",
    }
    policy = model["grasp_family_selection_policy"]
    assert policy["fallback"] == "next_segmentation"
    assert [group["group_index"] for group in policy["groups"]] == [1, 2]
    assert [group["group_index"] for group in policy["all_groups"]] == [1, 2, 3]
