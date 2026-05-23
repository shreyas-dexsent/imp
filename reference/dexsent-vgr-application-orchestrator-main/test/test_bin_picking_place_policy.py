import json

from orchestrator.tasks.bin_picking import (
    _load_place_policy_model,
    _resolve_bin_picking_place_plan,
    _select_place_strategy,
)


def test_load_place_policy_model_reads_default_place_json(tmp_path):
    object_folder = tmp_path / "object_a"
    object_folder.mkdir()
    (object_folder / "place.json").write_text(
        json.dumps(
            {
                "format": "vgr_place_policy/v1",
                "enabled": True,
                "placement_goal": {
                    "symmetry": {"type": "axis", "axis": "y"},
                    "primary_axis_alignment": {
                        "object_axis": "y",
                        "target_axis": "+z",
                    },
                },
                "grasp_strategies": [
                    {
                        "name": "direct_group_1",
                        "mode": "direct",
                        "match": {"generator_group_indices": [1]},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    model = _load_place_policy_model(
        module_params={"object_folder": str(object_folder)},
        place_cfg={},
        search_roots=[object_folder],
    )

    assert model is not None
    assert model["enabled"] is True
    assert model["placement_goal"]["primary_axis_alignment"]["object_axis"] == "y"
    assert model["strategies"][0]["match"]["generator_group_indices"] == [1]


def test_select_place_strategy_prefers_specific_group_match():
    place_policy_model = {
        "enabled": True,
        "strategies": [
            {
                "name": "default_direct",
                "mode": "direct",
                "priority": 0,
                "order": 0,
                "match": {},
            },
            {
                "name": "group_2_regrasp",
                "mode": "intermediate_regrasp",
                "priority": 0,
                "order": 1,
                "match": {"generator_group_indices": [2]},
            },
        ],
    }

    selected = _select_place_strategy(
        place_policy_model,
        {
            "id": "grasp_042",
            "label": "grasp_042",
            "generator_group_index": 2,
        },
    )

    assert selected is not None
    assert selected["name"] == "group_2_regrasp"
    assert selected["mode"] == "intermediate_regrasp"


def test_resolve_bin_picking_place_plan_resolves_named_intermediate_poses():
    runtime_state = {
        "selected_pick_contact": {
            "id": "grasp_002",
            "label": "grasp_002",
            "generator_group_index": 2,
        },
        "selected_place_strategy": {
            "name": "group_2_regrasp",
            "mode": "intermediate_regrasp",
            "match": {"generator_group_indices": [2]},
            "intermediate_regrasp": {
                "place_pose_name": "int_place_2",
                "pick_pose_name": "int_pick_2",
                "settle_time_s": 0.2,
            },
            "final_place": {
                "pose_name": "final_place_pose",
            },
        },
        "place_policy_model": {
            "enabled": True,
            "strategies": [],
        },
    }
    pose_index = {
        "int_place_2": {
            "name": "int_place_2",
            "tcp_pose": {
                "position_m": [0.1, 0.2, 0.3],
                "quat_xyzw": [0.0, 0.0, 0.0, 1.0],
                "frame": "base",
            },
        },
        "int_pick_2": {
            "name": "int_pick_2",
            "tcp_pose": {
                "position_m": [0.4, 0.5, 0.6],
                "quat_xyzw": [0.0, 0.0, 0.0, 1.0],
                "frame": "base",
            },
        },
        "final_place_pose": {
            "name": "final_place_pose",
            "tcp_pose": {
                "position_m": [0.7, 0.8, 0.9],
                "quat_xyzw": [0.0, 0.0, 0.0, 1.0],
                "frame": "base",
            },
        },
    }

    plan = _resolve_bin_picking_place_plan(
        ctx=None,
        state=None,
        recipe={},
        vision_cfg={},
        robot_cfg={},
        pick_cfg={},
        module_params={},
        runtime_state=runtime_state,
        execution={
            "cycle": 0,
            "attempt": 1,
            "pose_index": pose_index,
            "default_profile": "slow",
        },
        log_debug=lambda *args, **kwargs: None,
    )

    assert plan is not None
    assert plan["strategy_name"] == "group_2_regrasp"
    assert plan["place_pose"]["name"] == "final_place_pose"
    assert plan["intermediate_regrasp"]["place_pose_name"] == "int_place_2"
    assert plan["intermediate_regrasp"]["pick_pose_name"] == "int_pick_2"
    assert plan["intermediate_regrasp"]["settle_time_s"] == 0.2
