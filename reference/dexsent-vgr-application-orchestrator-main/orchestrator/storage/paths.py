"""Implementation for `orchestrator.storage.paths`."""

from pathlib import Path
from typing import Optional


class DataPaths:
    def __init__(self, root: Path):
        self.root = root
        self.stations = root / "stations"
        self.runs = root / "runs"
        self.legacy_station = root / "station"
        self.legacy_objects = root / "object_library"
        self.legacy_recipes = root / "recipes"

    def ensure(self, create_legacy: bool = False) -> None:
        paths = [self.root, self.stations, self.runs]
        if create_legacy:
            paths.extend(
                [self.legacy_station, self.legacy_objects, self.legacy_recipes]
            )
        for p in paths:
            p.mkdir(parents=True, exist_ok=True)

    def station_dir(self, station_id: str) -> Path:
        return self.stations / station_id

    def station_calibration_dir(self, station_id: str) -> Path:
        return self.station_dir(station_id) / "calibration"

    def station_processes_dir(self, station_id: str) -> Path:
        return self.station_dir(station_id) / "assets"

    def station_legacy_processes_dir(self, station_id: str) -> Path:
        return self.station_dir(station_id) / "processes"

    def process_dir(self, station_id: str, process_id: str) -> Path:
        return self.station_processes_dir(station_id) / process_id

    def process_objects_dir(self, station_id: str, process_id: str) -> Path:
        return self.process_dir(station_id, process_id) / "objects"

    def process_poses_dir(self, station_id: str, process_id: str) -> Path:
        return self.process_dir(station_id, process_id) / "poses"

    def process_tasks_dir(self, station_id: str, process_id: str) -> Path:
        return self.process_dir(station_id, process_id) / "tasks"

    def process_task_scene_dir(
        self, station_id: str, process_id: str, task_id: str
    ) -> Path:
        return self.process_tasks_dir(station_id, process_id) / task_id

    def process_bin_dir(self, station_id: str, process_id: str) -> Path:
        return self.process_dir(station_id, process_id) / "bin"

    def process_bin_path(self, station_id: str, process_id: str) -> Path:
        return self.process_bin_dir(station_id, process_id) / "bin.json"

    def process_robot_dir(self, station_id: str, process_id: str) -> Path:
        return self.process_dir(station_id, process_id) / "robot"

    def process_gripper_dir(self, station_id: str, process_id: str) -> Path:
        return self.process_dir(station_id, process_id) / "gripper"

    def process_dummy_testing_dir(self, station_id: str, process_id: str) -> Path:
        return self.process_dir(station_id, process_id) / "dummy_testing"

    def process_trash_dir(self, station_id: str, process_id: str) -> Path:
        return self.process_dir(station_id, process_id) / "trash"

    def legacy_station_poses_dir(self) -> Path:
        return self.legacy_station / "poses"

    def legacy_station_calibration_dir(self) -> Path:
        return self.legacy_station / "calibration"

    def legacy_object_dir(self, object_id: Optional[str] = None) -> Path:
        return self.legacy_objects / object_id if object_id else self.legacy_objects
