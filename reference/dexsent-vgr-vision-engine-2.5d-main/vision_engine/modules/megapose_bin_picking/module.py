from __future__ import annotations

import json
import queue
import threading
import time
from typing import Any, Dict

from vision_engine.core.module_base import VisionModule
from vision_engine.io.data_plane.frame_bundle import FrameBundle
from vision_engine.modules.megapose_bin_picking.runtime import (
    load_persisted_megapose_startup_params,
    persist_megapose_startup_params,
    prewarm_megapose_bin_picking,
    run_megapose_bin_picking,
)


class _MegaPoseWorker:
    def __init__(self) -> None:
        self._tasks: "queue.Queue[dict[str, Any] | None]" = queue.Queue()
        self._thread = threading.Thread(
            target=self._run,
            daemon=True,
            name="megapose-bin-picking-worker",
        )
        self._thread.start()

    @property
    def is_alive(self) -> bool:
        return self._thread.is_alive()

    def submit(
        self,
        *,
        bgr,
        depth,
        params: Dict[str, Any],
        request_id: str,
    ) -> Dict[str, Any]:
        task: Dict[str, Any] = {
            "bgr": bgr,
            "depth": depth,
            "params": params,
            "request_id": request_id,
            "queued_at_perf": time.perf_counter(),
            "done": threading.Event(),
        }
        self._tasks.put(task)
        task["done"].wait()
        exc = task.get("exception")
        if exc is not None:
            raise exc
        return task.get("result") or {}

    def prewarm(
        self,
        *,
        params: Dict[str, Any],
        request_id: str = "",
        persist_params: bool = True,
    ) -> Dict[str, Any]:
        task: Dict[str, Any] = {
            "kind": "prewarm",
            "params": dict(params or {}),
            "request_id": request_id,
            "persist_params": bool(persist_params),
            "done": threading.Event(),
        }
        self._tasks.put(task)
        task["done"].wait()
        exc = task.get("exception")
        if exc is not None:
            raise exc
        return task.get("result") or {}

    def _run(self) -> None:
        while True:
            task = self._tasks.get()
            if task is None:
                return
            try:
                kind = str(task.get("kind") or "run").strip().lower()
                if kind == "prewarm":
                    result = prewarm_megapose_bin_picking(
                        task.get("params") or {},
                        force_bundle_warm=True,
                    )
                    if bool(task.get("persist_params", False)):
                        try:
                            persist_megapose_startup_params(task.get("params") or {})
                        except Exception:
                            pass
                    task["result"] = result
                else:
                    run_params = dict(task.get("params") or {})
                    queued_at_perf = task.get("queued_at_perf")
                    if isinstance(queued_at_perf, (int, float)):
                        run_params["_worker_queue_wait_seconds"] = max(
                            0.0,
                            time.perf_counter() - float(queued_at_perf),
                        )
                    task["result"] = run_megapose_bin_picking(
                        bgr=task["bgr"],
                        depth=task["depth"],
                        params=run_params,
                        request_id=str(task.get("request_id") or ""),
                    )
            except Exception as exc:
                task["exception"] = exc
            finally:
                done_evt = task.get("done")
                if isinstance(done_evt, threading.Event):
                    done_evt.set()


class MegaPoseBinPickingModule(VisionModule):
    _worker_lock = threading.Lock()
    _prewarm_cache_lock = threading.Lock()
    _worker: _MegaPoseWorker | None = None
    _last_prewarm_signature: str | None = None
    _last_prewarm_result: Dict[str, Any] | None = None

    def __init__(self, name: str, params: Dict[str, Any]):
        super().__init__(name, params)
        self._no_candidate_streak = 0

    @classmethod
    def _get_worker(cls) -> _MegaPoseWorker:
        with cls._worker_lock:
            if cls._worker is None or not cls._worker.is_alive:
                cls._worker = _MegaPoseWorker()
                with cls._prewarm_cache_lock:
                    cls._last_prewarm_signature = None
                    cls._last_prewarm_result = None
            return cls._worker

    @staticmethod
    def _prewarm_signature(params: Dict[str, Any]) -> str:
        return json.dumps(params or {}, sort_keys=True, separators=(",", ":"), default=str)

    @classmethod
    def prewarm(
        cls,
        params: Dict[str, Any],
        *,
        request_id: str = "",
        persist_params: bool = True,
    ) -> Dict[str, Any]:
        signature = cls._prewarm_signature(params)
        worker = cls._get_worker()
        with cls._prewarm_cache_lock:
            if (
                cls._last_prewarm_signature == signature
                and worker.is_alive
                and isinstance(cls._last_prewarm_result, dict)
            ):
                cached = dict(cls._last_prewarm_result)
                cached["prewarm_reused"] = True
                return cached
        result = worker.prewarm(
            params=params,
            request_id=request_id,
            persist_params=persist_params,
        )
        with cls._prewarm_cache_lock:
            cls._last_prewarm_signature = signature
            cls._last_prewarm_result = dict(result or {})
        return result

    @classmethod
    def startup_prewarm(
        cls,
        default_params: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        params = load_persisted_megapose_startup_params(default_params or {})
        if not isinstance(params, dict):
            return {"status": "skipped", "reason": "no_persisted_startup_params"}
        return cls.prewarm(
            params,
            request_id="startup-prewarm",
            persist_params=False,
        )

    def run(self, frame_bundle: FrameBundle) -> Dict[str, Any]:
        try:
            if frame_bundle.rgb is None:
                return {
                    "valid": False,
                    "matches": [],
                    "terminal": True,
                    "error": "megapose_rgb_missing",
                }
            runtime_params = dict(self.params)
            meta = frame_bundle.meta if isinstance(frame_bundle.meta, dict) else {}
            session = meta.get("session") if isinstance(meta, dict) else {}
            calibration = session.get("calibration") if isinstance(session, dict) else {}
            if isinstance(calibration, dict):
                if calibration.get("camera_data") and "camera_data" not in runtime_params:
                    runtime_params["camera_data"] = calibration.get("camera_data")
                if calibration.get("K") and "K" not in runtime_params:
                    runtime_params["K"] = calibration.get("K")
                if calibration.get("intrinsics") and "intrinsics" not in runtime_params:
                    runtime_params["intrinsics"] = calibration.get("intrinsics")
                if (
                    calibration.get("intrinsics_resolution")
                    and "intrinsics_resolution" not in runtime_params
                ):
                    runtime_params["intrinsics_resolution"] = calibration.get(
                        "intrinsics_resolution"
                    )
                if calibration.get("depth_scale_m_per_unit") and "depth_scale" not in runtime_params:
                    runtime_params["depth_scale"] = calibration.get("depth_scale_m_per_unit")
            if isinstance(calibration, dict) and calibration:
                calibration_camera_data = calibration.get("camera_data")
                calibration_source = (
                    calibration.get("source")
                    or (
                        calibration_camera_data.get("source")
                        if isinstance(calibration_camera_data, dict)
                        else None
                    )
                    or "session_calibration"
                )
                runtime_params.setdefault("camera_calibration_source", str(calibration_source))
            else:
                runtime_params.setdefault("camera_calibration_source", "module_params")
            frame_camera_data = meta.get("camera_data") if isinstance(meta, dict) else {}
            if isinstance(frame_camera_data, dict) and frame_camera_data:
                merged_camera_data = runtime_params.get("camera_data")
                if not isinstance(merged_camera_data, dict):
                    merged_camera_data = {}
                merged_camera_data = dict(merged_camera_data)
                for key, value in frame_camera_data.items():
                    if value is None:
                        continue
                    if (
                        isinstance(calibration, dict)
                        and calibration
                        and key in {"K", "intrinsics", "resolution", "image_size", "source"}
                    ):
                        continue
                    if (
                        key in {"K", "intrinsics", "resolution", "image_size"}
                        and merged_camera_data.get(key) is not None
                    ):
                        continue
                    merged_camera_data[key] = value
                if merged_camera_data:
                    runtime_params["camera_data"] = merged_camera_data
                if (
                    frame_camera_data.get("depth_scale_m_per_unit") is not None
                    and "depth_scale" not in runtime_params
                ):
                    runtime_params["depth_scale"] = frame_camera_data.get(
                        "depth_scale_m_per_unit"
                    )
                if isinstance(calibration, dict) and calibration:
                    runtime_params.setdefault(
                        "frame_camera_data_source",
                        str(frame_camera_data.get("source") or "pyrealsense2_frame"),
                    )
                else:
                    runtime_params["camera_calibration_source"] = str(
                        frame_camera_data.get("source") or "pyrealsense2_frame"
                    )
            request_id = str(
                meta.get("session", {}).get("request_id")
                or frame_bundle.frame_id
            )
            result = self._get_worker().submit(
                bgr=frame_bundle.rgb,
                depth=frame_bundle.depth,
                params=runtime_params,
                request_id=request_id,
            )
            if (
                isinstance(result, dict)
                and result.get("error") == "megapose_no_candidate"
            ):
                self._no_candidate_streak += 1
                try:
                    max_no_candidate_frames = max(
                        1,
                        int(runtime_params.get("max_no_candidate_frames", 1) or 1),
                    )
                except Exception:
                    max_no_candidate_frames = 1
                if self._no_candidate_streak < max_no_candidate_frames:
                    details = result.get("details")
                    return {
                        "valid": False,
                        "matches": [],
                        "terminal": False,
                        "status": "no_candidate_retrying",
                        "retry_frame": self._no_candidate_streak,
                        "retry_limit": max_no_candidate_frames,
                        "details": details,
                    }
            else:
                self._no_candidate_streak = 0
            return result
        except Exception as exc:
            self._no_candidate_streak = 0
            message = str(exc).strip()
            if message.startswith("megapose_"):
                error_code = message
            elif isinstance(exc, ModuleNotFoundError):
                error_code = "megapose_dependency_missing"
            elif isinstance(exc, ImportError):
                error_code = "megapose_import_failed"
            else:
                error_code = "megapose_inference_failed"
            result = {
                "valid": False,
                "matches": [],
                "terminal": True,
                "error": error_code,
            }
            if message and error_code != message:
                result["error_detail"] = f"{exc.__class__.__name__}: {message}"
            return result
