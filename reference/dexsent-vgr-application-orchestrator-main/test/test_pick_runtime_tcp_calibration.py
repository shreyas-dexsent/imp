import json
from types import SimpleNamespace

import pytest

from orchestrator.tasks._pick_runtime import (
    _apply_tcp_calibration_to_base,
    _command_pose_for_desired_tcp,
    _compose_transform,
    _normalize_tcp_calibration,
    _resolve_runtime_tcp_calibration,
    _rpy_deg_to_quat_xyzw,
)


def _assert_vec_close(actual, expected, tol=1e-9):
    assert len(actual) == len(expected)
    for got, want in zip(actual, expected):
        assert got == pytest.approx(want, abs=tol)


def _assert_quat_close(actual, expected, tol=1e-9):
    dot = sum(float(a) * float(b) for a, b in zip(actual, expected))
    if dot < 0.0:
        actual = [-float(v) for v in actual]
    _assert_vec_close(actual, expected, tol=tol)


def _fake_ctx(tmp_path, station_id="station-1"):
    class _Processes:
        def get(self, process_id):
            return {"station_id": station_id, "asset_id": process_id}

    class _DataPaths:
        def station_calibration_dir(self, sid):
            return tmp_path / "stations" / sid / "calibration"

    return SimpleNamespace(processes=_Processes(), data_paths=_DataPaths())


def test_identity_tcp_command_pose_is_unchanged():
    calibration = _normalize_tcp_calibration({})
    desired = {
        "position_m": [0.1, 0.2, 0.3],
        "quat_xyzw": [0.0, 0.0, 0.0, 1.0],
        "frame": "base",
    }

    command = _command_pose_for_desired_tcp(desired, calibration)

    assert command == desired


def test_translation_tcp_command_pose_subtracts_rotated_offset():
    calibration = _normalize_tcp_calibration({"tcp_offset_m": [0.1, 0.0, 0.0]})
    desired = {
        "position_m": [1.0, 2.0, 3.0],
        "quat_xyzw": [0.0, 0.0, 0.0, 1.0],
        "frame": "base",
    }

    command = _command_pose_for_desired_tcp(desired, calibration)

    _assert_vec_close(command["position_m"], [0.9, 2.0, 3.0])
    _assert_quat_close(command["quat_xyzw"], desired["quat_xyzw"])


def test_rotated_tcp_command_pose_preserves_desired_orientation():
    calibration = _normalize_tcp_calibration(
        {
            "tcp_offset_m": [-0.01, 0.0, 0.065],
            "tcp_offset_rpy_deg": [0.0, -45.0, 0.0],
        }
    )
    desired = {
        "position_m": [0.4, -0.2, 0.5],
        "quat_xyzw": _rpy_deg_to_quat_xyzw(10.0, 20.0, 30.0),
        "frame": "base",
    }

    command = _command_pose_for_desired_tcp(desired, calibration)
    round_trip = _apply_tcp_calibration_to_base(
        {
            "translation_m": command["position_m"],
            "rotation_quat_xyzw": command["quat_xyzw"],
        },
        calibration,
    )

    _assert_vec_close(round_trip["translation_m"], desired["position_m"])
    _assert_quat_close(command["quat_xyzw"], desired["quat_xyzw"])


def test_command_pose_preserves_desired_yaw_with_franka_ee_rotation():
    calibration = _normalize_tcp_calibration(
        {
            "tcp_offset_m": [0.0, 0.0, 0.0],
            "tcp_offset_rpy_deg": [0.0, 0.0, 0.0],
            "franka_f_t_ee": {
                "translation_m": [0.0, 0.0, 0.1034],
                "rotation_rpy_deg": [0.0, 0.0, 45.0],
            },
        }
    )
    desired = {
        "position_m": [0.1, 0.2, 0.3],
        "quat_xyzw": _rpy_deg_to_quat_xyzw(0.0, 0.0, 0.0),
        "frame": "base",
    }

    command = _command_pose_for_desired_tcp(desired, calibration)

    _assert_vec_close(command["position_m"], [0.1, 0.2, 0.4034])
    _assert_quat_close(command["quat_xyzw"], desired["quat_xyzw"])


def test_station_tcp_alias_fields_load_like_preferred_fields(tmp_path):
    ctx = _fake_ctx(tmp_path)
    calib_dir = ctx.data_paths.station_calibration_dir("station-1")
    calib_dir.mkdir(parents=True)
    (calib_dir / "tcp.json").write_text(
        json.dumps(
            {
                "translation_m": [0.01, 0.02, 0.03],
                "rotation_rpy_deg": [1.0, 2.0, 3.0],
            }
        ),
        encoding="utf-8",
    )

    calibration, source = _resolve_runtime_tcp_calibration(ctx, "asset-1")

    assert source == "station_tcp"
    assert calibration["tcp_offset_m"] == [0.01, 0.02, 0.03]
    assert calibration["tcp_offset_rpy_deg"] == [1.0, 2.0, 3.0]


def test_handeye_translation_is_not_mistaken_for_tcp_alias(tmp_path):
    ctx = _fake_ctx(tmp_path)
    handeye_only = {
        "translation_m": [9.0, 9.0, 9.0],
        "rotation_rpy_deg": [10.0, 20.0, 30.0],
        "hand_eye_frame": "camera_in_gripper",
    }

    calibration, source = _resolve_runtime_tcp_calibration(
        ctx,
        "asset-1",
        handeye_only,
    )

    assert source == "identity"
    assert calibration["tcp_offset_m"] == [0.0, 0.0, 0.0]
    assert calibration["tcp_offset_rpy_deg"] == [0.0, 0.0, 0.0]


def test_camera_in_gripper_extrinsic_stays_on_robot_reported_ee_frame():
    base_to_flange = {
        "translation_m": [1.0, 0.0, 0.0],
        "rotation_quat_xyzw": [0.0, 0.0, 0.0, 1.0],
    }
    calibration = _normalize_tcp_calibration({"tcp_offset_m": [0.0, 0.0, 0.1]})
    handeye = {
        "translation_m": [0.0, 0.0, 0.2],
        "rotation_quat_xyzw": [0.0, 0.0, 0.0, 1.0],
    }

    _ = _apply_tcp_calibration_to_base(base_to_flange, calibration)
    base_to_cam = _compose_transform(base_to_flange, handeye)

    _assert_vec_close(base_to_cam["translation_m"], [1.0, 0.0, 0.2])
