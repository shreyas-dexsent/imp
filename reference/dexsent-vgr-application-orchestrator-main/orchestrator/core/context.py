"""Implementation for `orchestrator.core.context`."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

from orchestrator.camera.cache import CameraFrameCache
from orchestrator.core.runs import RunManager
from orchestrator.robot.base import RobotAdapter
from orchestrator.runs.store import RunStore
from orchestrator.storage.object_store import ObjectLibraryStore
from orchestrator.storage.paths import DataPaths
from orchestrator.storage.pose_store import PoseStore
from orchestrator.storage.process_store import ProcessStore
from orchestrator.storage.station_store import StationStore
from orchestrator.storage.task_store import TaskStore
from orchestrator.vision.cache import VisionResultCache
from orchestrator.vision.client import VisionEngineClient
from orchestrator.vision.results import VisionResultSubscriber


@dataclass
class StationContext:
    config: Dict[str, Any]
    runtime_state: Dict[str, Any]
    data_root: Path
    data_paths: DataPaths
    stations: StationStore
    processes: ProcessStore
    tasks: TaskStore
    objects: ObjectLibraryStore
    poses: PoseStore
    runs: RunStore
    run_manager: RunManager
    vision: VisionEngineClient
    vision_results: VisionResultSubscriber
    vision_cache: VisionResultCache
    camera_cache: CameraFrameCache
    robot: RobotAdapter
    executor: Any
