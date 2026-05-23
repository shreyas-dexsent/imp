from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from orchestrator.tasks._pick_runtime import _vision_timeout_s
from orchestrator.tasks.bin_picking import _prepare_bin_picking_task


class _FakeObjects:
    def __init__(self, metadata):
        self._metadata = metadata

    def get_metadata(self, *_args):
        return self._metadata


class _FakeProcesses:
    def get(self, process_id):
        return {"station_id": "station-1", "asset_id": process_id}


class _FakeDataPaths:
    def __init__(self, root: Path):
        self.root = root
        self.runs = root / "runs"

    def process_objects_dir(self, station_id, process_id):
        return self.root / "stations" / station_id / "assets" / process_id / "objects"

    def process_bin_path(self, station_id, process_id):
        return self.root / "stations" / station_id / "assets" / process_id / "bin_roi.json"


def _ctx(tmp_path, metadata=None):
    return SimpleNamespace(
        config={"vision_engine": {"transport": "zmq"}},
        data_root=tmp_path,
        data_paths=_FakeDataPaths(tmp_path),
        objects=_FakeObjects(metadata or {}),
        processes=_FakeProcesses(),
    )


def _state(task=None):
    return SimpleNamespace(process_id="asset-1", run_id="", task=task or {})


def test_prepare_bin_picking_task_preserves_saved_ppf_module(tmp_path):
    object_folder = tmp_path / "object_a"
    object_folder.mkdir()
    task = {
        "vision": {
            "module": "ppf_icp_bin_picking",
            "params": {
                "object_id": "object_a",
                "object_folder": str(object_folder),
                "ppf_num_angles": 24,
                "icp_enabled": False,
                "model": "rgbd",
            },
        },
        "robot": {"max_pick_attempts": 0},
    }

    prepared = _prepare_bin_picking_task(_ctx(tmp_path), _state(task), task)

    vision = prepared["vision"]
    params = vision["params"]
    assert vision["module"] == "ppf_icp_bin_picking"
    assert vision["process_mode"] == "trigger_only"
    assert params["ppf_num_angles"] == 24
    assert params["icp_enabled"] is False
    assert params["first_frame_timeout_s"] == 5.0
    assert "model" not in params
    assert "refiner_iterations" not in params
    assert "ppf_icp_no_candidate" in prepared["robot"]["retry_errors"]
    assert "megapose_no_candidate" in prepared["robot"]["retry_errors"]


def test_prepare_bin_picking_task_defaults_missing_module_to_megapose(tmp_path):
    object_folder = tmp_path / "object_a"
    object_folder.mkdir()
    task = {
        "vision": {
            "params": {
                "object_id": "object_a",
                "object_folder": str(object_folder),
            },
        },
        "robot": {},
    }

    prepared = _prepare_bin_picking_task(_ctx(tmp_path), _state(task), task)

    params = prepared["vision"]["params"]
    assert prepared["vision"]["module"] == "megapose_bin_picking"
    assert params["model"] == "rgbd"
    assert "ppf_num_angles" not in params


def test_prepare_bin_picking_task_preserves_remote_posix_object_folder_in_websocket_mode(tmp_path):
    local_object_folder = (
        tmp_path / "stations" / "station-1" / "assets" / "asset-1" / "objects" / "black"
    )
    local_object_folder.mkdir(parents=True)
    remote_object_folder = "/home/imp/imp/data/stations/station-1/assets/asset-1/objects/black"
    task = {
        "vision": {
            "module": "megapose_bin_picking",
            "params": {
                "object_id": "black",
                "object_folder": remote_object_folder,
            },
        },
        "robot": {},
    }
    ctx = _ctx(tmp_path)
    ctx.config["vision_engine"]["transport"] = "websocket"

    prepared = _prepare_bin_picking_task(ctx, _state(task), task)

    assert prepared["vision"]["params"]["object_folder"] == remote_object_folder


def test_prepare_bin_picking_task_removes_retry_errors_when_single_attempt(tmp_path):
    object_folder = tmp_path / "object_a"
    object_folder.mkdir()
    task = {
        "vision": {
            "module": "ppf_icp_bin_picking",
            "params": {
                "object_id": "object_a",
                "object_folder": str(object_folder),
            },
        },
        "robot": {
            "max_pick_attempts": 1,
            "retry_errors": ["megapose_no_candidate", "ppf_icp_no_candidate", "vision_timeout"],
        },
    }

    prepared = _prepare_bin_picking_task(_ctx(tmp_path), _state(task), task)

    assert prepared["robot"]["retry_errors"] == ["vision_timeout"]


def test_ppf_icp_uses_bin_picking_vision_timeout_default():
    assert _vision_timeout_s({}, "ppf_icp_bin_picking") == 15.0
