"""Implementation for `orchestrator.core.runtime`."""

from pathlib import Path
from typing import Any, Dict

from orchestrator.camera.cache import CameraFrameCache
from orchestrator.core.context import StationContext
from orchestrator.core.executor import RunExecutor
from orchestrator.core.runs import RunManager
from orchestrator.robot.null import NullRobotAdapter
from orchestrator.robot.standard import StandardRobotAdapter
from orchestrator.robot.switchable import SwitchableRobotAdapter
from orchestrator.robot.zmq import ZmqRobotAdapter
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


def build_context(cfg: Dict[str, Any]) -> StationContext:
    data_root = Path(cfg.get("data", {}).get("root", "../data")).resolve()
    data_paths = DataPaths(data_root)
    create_legacy = bool(cfg.get("data", {}).get("create_legacy_dirs", False))
    data_paths.ensure(create_legacy=create_legacy)
    stations = StationStore(data_paths.stations)
    default_station = stations.ensure_default()
    persisted_robot_enabled = default_station.get("robot_execution_enabled")
    runtime_state = {
        "robot_enabled": (
            bool(persisted_robot_enabled)
            if isinstance(persisted_robot_enabled, bool)
            else bool(cfg.get("runtime", {}).get("robot_enabled", True))
        ),
        "robot_mode_station_id": default_station.get("station_id"),
    }

    vision_cfg = cfg.get("vision_engine", {})
    camera_cfg = cfg.get("camera_core", {})
    camera_cache = CameraFrameCache(
        camera_cfg.get("pub_endpoint", "tcp://127.0.0.1:5555"),
        camera_cfg.get("topic", "camera"),
    )
    transport = str(vision_cfg.get("transport") or "zmq").strip().lower()
    vision = VisionEngineClient(
        push_endpoint=vision_cfg.get("control_push", "tcp://127.0.0.1:5556"),
        topic=vision_cfg.get("control_topic", "camera"),
        transport=transport,
        websocket_url=vision_cfg.get("websocket_url", "ws://127.0.0.1:8000/ws"),
        results_push_endpoint=vision_cfg.get("results_push", "tcp://127.0.0.1:5556"),
        results_topic=vision_cfg.get("results_topic", "vision"),
        camera_cache=camera_cache,
        frame_poll_s=float(vision_cfg.get("websocket_frame_poll_s", 0.02)),
        max_frame_fps=float(vision_cfg.get("websocket_max_frame_fps", 0.0)),
    )

    robot_cfg = cfg.get("robot", {})
    adapter = robot_cfg.get("adapter", "standard")
    if adapter == "standard":
        primary_robot = StandardRobotAdapter()
    elif adapter == "zmq":
        primary_robot = ZmqRobotAdapter(
            command_endpoint=robot_cfg.get("command_endpoint", "tcp://127.0.0.1:5571"),
            state_endpoint=robot_cfg.get("state_endpoint", "tcp://127.0.0.1:5572"),
            timeout_ms=int(robot_cfg.get("timeout_ms", 2000)),
            connect_on_init=False,
        )
    else:
        raise ValueError(
            f"Unsupported robot adapter '{adapter}'. "
            "Orchestrator supports only 'standard' or 'zmq'."
        )
    robot = SwitchableRobotAdapter(primary_robot, NullRobotAdapter(), runtime_state)

    run_store = RunStore(data_paths.runs)
    processes = ProcessStore(data_paths)
    tasks = TaskStore(data_paths, processes)
    objects = ObjectLibraryStore(data_paths, processes)
    poses = PoseStore(data_paths, processes)
    run_manager = RunManager(run_store)

    processes.ensure_default(default_station["station_id"])
    results = VisionResultSubscriber(
        vision_cfg.get("results_sub", "tcp://127.0.0.1:5555"),
        vision_cfg.get("results_topic", "vision"),
    )
    cache = VisionResultCache(
        vision_cfg.get("results_sub", "tcp://127.0.0.1:5555"),
        vision_cfg.get("results_topic", "vision"),
    )
    ctx = StationContext(
        config=cfg,
        runtime_state=runtime_state,
        data_root=data_root,
        data_paths=data_paths,
        stations=stations,
        processes=processes,
        tasks=tasks,
        objects=objects,
        poses=poses,
        runs=run_store,
        run_manager=run_manager,
        vision=vision,
        vision_results=results,
        vision_cache=cache,
        camera_cache=camera_cache,
        robot=robot,
        executor=None,
    )
    ctx.executor = RunExecutor(ctx)
    return ctx
