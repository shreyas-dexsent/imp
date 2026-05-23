"""Implementation for `vision_engine.core.engine`."""

import inspect
import time
from collections import defaultdict
from queue import Empty, Queue
from threading import Event, Lock, Thread
from typing import Any, Dict, Optional

from vision_engine.core.registry import get_module_class
from vision_engine.io.control_plane.publisher import ZmqPublisher
from vision_engine.io.control_plane.subscriber import ZmqSubscriber
from vision_engine.io.data_plane.frame_bundle import FrameBundle
from vision_engine.io.data_plane.shm_reader import ShmReader
from vision_engine.io.data_plane.shm_result_writer import ShmResultWriter


class RequestThread:
    """
    Represents a single continuous processing request.
    Each request runs in its own thread, processes frames at the requested FPS,
    and publishes results continuously until stopped.
    """

    def __init__(
        self,
        request_id: str,
        camera_id: str,
        module_name: str,
        fps_limit: float,
        module_instance: Any,
        publisher: ZmqPublisher,
        shm_reader: Optional[ShmReader] = None,
        session_meta: Optional[Dict[str, Any]] = None,
        enable_shm_output: bool = True,
        shm_max_result_size: Optional[int] = None,
        one_shot: bool = False,
    ):
        self.request_id = request_id
        self.camera_id = camera_id
        self.module_name = module_name
        self.fps_limit = fps_limit
        self.module = module_instance
        self.publisher = publisher
        self.shm_reader = shm_reader
        self.session_meta = session_meta or {}
        self.enable_shm_output = enable_shm_output
        self.one_shot = one_shot
        params = self.session_meta.get("params") if isinstance(self.session_meta, dict) else {}
        if not isinstance(params, dict):
            params = {}
        try:
            self.startup_warmup_frames = max(
                0, int(params.get("startup_warmup_frames", 0) or 0)
            )
        except Exception:
            self.startup_warmup_frames = 0
        try:
            self.min_frame_timestamp_ns = max(
                0,
                int(
                    self.session_meta.get("min_frame_timestamp_ns", 0)
                    if isinstance(self.session_meta, dict)
                    else 0
                ),
            )
        except Exception:
            self.min_frame_timestamp_ns = 0
        try:
            self.first_frame_timeout_s = max(
                0.0, float(params.get("first_frame_timeout_s", 0.0) or 0.0)
            )
        except Exception:
            self.first_frame_timeout_s = 0.0
        self._startup_warmup_skipped = 0
        self._startup_warmup_logged = False

        # Frame queue for this request (latest-frame only to minimize tracking lag).
        self.frame_queue = Queue(maxsize=1)

        # Control
        self.running = Event()
        self.thread = None

        # FPS tracking
        self.last_process_time = 0.0
        self.last_process_ts_ns = 0
        self.min_interval = 1.0 / fps_limit if fps_limit > 0 else 0
        self.last_source_seq = 0
        self.last_source_ts_ns = 0

        # Statistics
        self.frames_processed = 0
        self.frames_received = 0
        self.start_time = None
        self.first_processed_at = None
        self.last_processed_at = None
        self.process_time_ms_sum = 0.0
        self.process_time_ms_count = 0
        self.produced_latency_ms_sum = 0.0
        self.produced_latency_ms_count = 0
        self.drop_queue_full = 0
        self.skip_throttle = 0
        self.skip_invalid_meta = 0
        self.skip_no_shm_reader = 0
        self.skip_rgb_missing = 0
        self.skip_rgb_seq_mismatch = 0
        self.depth_seq_mismatch = 0
        self.skip_errors = 0
        self.skip_duplicate_source = 0
        self.skip_pre_start_frame = 0

        # Shared memory result writer
        self.shm_writer: Optional[ShmResultWriter] = None
        if self.enable_shm_output:
            try:
                if shm_max_result_size:
                    self.shm_writer = ShmResultWriter(
                        request_id, max_result_size=shm_max_result_size
                    )
                else:
                    self.shm_writer = ShmResultWriter(request_id)
            except Exception as e:
                print(f"[{self.request_id}] Failed to create SHM writer: {e}")
                self.shm_writer = None

    def _publish_terminal_error(self, error: str, details: Optional[Dict[str, Any]] = None) -> None:
        result = {
            "valid": False,
            "matches": [],
            "terminal": True,
            "error": str(error),
        }
        if details:
            result["details"] = details
        produced_timestamp_ns = time.time_ns()
        produced_monotonic_ns = time.perf_counter_ns()
        event = {
            "event": "VISION_RESULT",
            "request_id": self.request_id,
            "camera_id": self.camera_id,
            "module": self.module_name,
            "frame_id": "",
            "sequence_id": 0,
            "timestamp_ns": produced_timestamp_ns,
            "produced_timestamp_ns": produced_timestamp_ns,
            "produced_monotonic_ns": produced_monotonic_ns,
            "produced_latency_ms": 0.0,
            "result": result,
            "process_time_ms": 0.0,
            "frames_processed": self.frames_processed,
        }
        self.publisher.publish(event)
        if self.shm_writer:
            self.shm_writer.write_result(
                timestamp_ns=produced_timestamp_ns,
                sequence_id=0,
                result=result,
                request_id=self.request_id,
                camera_id=self.camera_id,
                module=self.module_name,
                process_time_ms=0.0,
            )

    def _session_calibration(self) -> Dict[str, Any]:
        calibration = self.session_meta.get("calibration") if isinstance(self.session_meta, dict) else {}
        return calibration if isinstance(calibration, dict) else {}

    def _resolve_frame_camera_data(self, frame_camera_data: Any) -> Any:
        calibration = self._session_calibration()
        calibration_camera_data = calibration.get("camera_data")
        if not isinstance(calibration_camera_data, dict) or not calibration_camera_data:
            return frame_camera_data

        merged = dict(calibration_camera_data)
        if isinstance(frame_camera_data, dict):
            for key, value in frame_camera_data.items():
                if value is None or key in {"K", "intrinsics", "resolution", "image_size", "source"}:
                    continue
                if key not in merged:
                    merged[key] = value
        return merged

    def start(self):
        """Start the request processing thread."""
        self.running.set()
        self.start_time = time.time()
        self.thread = Thread(
            target=self._run, daemon=True, name=f"req-{self.request_id}"
        )
        self.thread.start()

    def stop(
        self,
        *,
        wait: bool = True,
        join_timeout: float = 2.0,
        close_shm_writer: bool = True,
    ):
        """Stop the request processing thread."""
        self.running.clear()
        try:
            self.module.stop()
        except Exception as exc:
            print(f"[{self.request_id}] Module stop hook failed: {exc}")
        if wait and self.thread and self.thread.is_alive():
            self.thread.join(timeout=max(0.0, float(join_timeout)))
            if self.thread.is_alive():
                print(
                    f"[{self.request_id}] Stop timeout after {join_timeout:.2f}s; "
                    "request thread will finish in background"
                )

        # Cleanup shared memory writer
        if close_shm_writer and self.shm_writer:
            self.shm_writer.close()
            self.shm_writer = None

    def push_frame(self, frame_data: Dict[str, Any]):
        """
        Push a new frame to this request's queue.
        Non-blocking - drops old frames if queue is full.
        """
        self.frames_received += 1

        # Drop oldest frame if queue is full
        if self.frame_queue.full():
            try:
                self.frame_queue.get_nowait()
                self.drop_queue_full += 1
            except Empty:
                pass

        try:
            self.frame_queue.put_nowait(frame_data)
        except Exception:
            pass  # Queue full, frame dropped

    def _should_process(self, frame_timestamp_ns: Optional[int] = None):
        """Check if enough time has passed based on FPS limit."""
        if self.fps_limit <= 0:
            return True

        if frame_timestamp_ns is not None and frame_timestamp_ns > 0:
            if self.last_process_ts_ns <= 0:
                self.last_process_ts_ns = int(frame_timestamp_ns)
                return True
            min_interval_ns = int(self.min_interval * 1e9)
            jitter_tolerance_ns = min(
                1_000_000,  # 1ms max tolerance
                max(100_000, min_interval_ns // 40),  # ~2.5%, >=0.1ms
            )
            if (
                int(frame_timestamp_ns) - self.last_process_ts_ns + jitter_tolerance_ns
                >= min_interval_ns
            ):
                self.last_process_ts_ns = int(frame_timestamp_ns)
                return True
            return False

        current_time = time.perf_counter()

        if self.last_process_time <= 0:
            self.last_process_time = current_time
            return True

        elapsed = current_time - self.last_process_time
        if elapsed >= self.min_interval:
            self.last_process_time = current_time
            return True

        return False

    def _is_new_source_frame(self, sequence_id: int, timestamp_ns: int) -> bool:
        """Guard against duplicate/out-of-order source frames per request."""
        if sequence_id > 0:
            if self.last_source_seq <= 0:
                self.last_source_seq = int(sequence_id)
                if timestamp_ns > 0:
                    self.last_source_ts_ns = int(timestamp_ns)
                return True
            if int(sequence_id) > self.last_source_seq:
                self.last_source_seq = int(sequence_id)
                if timestamp_ns > 0:
                    self.last_source_ts_ns = int(timestamp_ns)
                return True
            # Allow sequence reset only after a long gap (camera restart/reconnect).
            if (
                timestamp_ns > self.last_source_ts_ns
                and (timestamp_ns - self.last_source_ts_ns) >= int(5e9)
            ):
                self.last_source_seq = int(sequence_id)
                self.last_source_ts_ns = int(timestamp_ns)
                return True
            return False

        if timestamp_ns > 0:
            if timestamp_ns > self.last_source_ts_ns:
                self.last_source_ts_ns = int(timestamp_ns)
                return True
            return False
        return True

    def _run(self):
        """Main processing loop for this request."""
        fps_label = (
            f"{self.fps_limit:.1f}"
            if self.fps_limit > 0
            else "unlimited(camera-bound)"
        )
        print(
            f"[{self.request_id}] Started: {self.module_name} @ {fps_label} FPS on {self.camera_id}"
        )

        self.publisher.publish(
            {
                "event": "VISION_REQUEST_STARTED",
                "request_id": self.request_id,
                "camera_id": self.camera_id,
                "module": self.module_name,
                "fps_limit": self.fps_limit,
            }
        )

        while self.running.is_set():
            frame_data = None
            try:
                # Wait for next frame with timeout
                frame_data = self.frame_queue.get(timeout=0.5)

                frame_id = str(frame_data.get("frame_id") or "")
                timestamp_ns = int(frame_data.get("timestamp_ns") or 0)
                sequence_id = int(frame_data.get("sequence_id") or 0)
                if not frame_id or timestamp_ns <= 0:
                    self.skip_invalid_meta += 1
                    continue

                if (
                    self.min_frame_timestamp_ns > 0
                    and timestamp_ns < self.min_frame_timestamp_ns
                ):
                    self.skip_pre_start_frame += 1
                    continue

                if not self._is_new_source_frame(sequence_id, timestamp_ns):
                    self.skip_duplicate_source += 1
                    continue

                if self._startup_warmup_skipped < self.startup_warmup_frames:
                    if not self._startup_warmup_logged:
                        print(
                            f"[{self.request_id}] Warmup skipping "
                            f"{self.startup_warmup_frames} fresh frame(s) before processing"
                        )
                        self._startup_warmup_logged = True
                    self._startup_warmup_skipped += 1
                    if self._startup_warmup_skipped >= self.startup_warmup_frames:
                        print(
                            f"[{self.request_id}] Warmup complete; processing next fresh frame"
                        )
                    continue

                # Throttle by source timestamps to avoid wall-time jitter artifacts.
                if not self._should_process(timestamp_ns):
                    self.skip_throttle += 1
                    continue

                # Decode frame lazily in request thread (avoid unnecessary copy in engine main loop).
                if "rgb" in frame_data:
                    rgb = frame_data["rgb"]
                    depth = frame_data.get("depth")
                    calib_version = frame_data.get("calib_version")
                else:
                    if self.shm_reader is None:
                        self.skip_no_shm_reader += 1
                        continue
                    rgb_packet = self.shm_reader.read_rgb(
                        frame_data["rgb_shm"],
                        tuple(frame_data["rgb_shape"]),
                        frame_data["rgb_dtype"],
                    )
                    if rgb_packet is None:
                        self.skip_rgb_missing += 1
                        continue
                    packet_seq = int(rgb_packet.get("sequence_id", -1))
                    # If SHM slot has rotated, use the freshest packet instead of dropping.
                    # Latest-frame processing prefers freshness over strict metadata match.
                    if packet_seq != sequence_id and packet_seq >= 0:
                        self.skip_rgb_seq_mismatch += 1
                        sequence_id = packet_seq
                        timestamp_ns = int(rgb_packet.get("timestamp_ns") or timestamp_ns)
                        frame_id = f"{self.camera_id}:{sequence_id}"
                    rgb = rgb_packet["rgb"]
                    calib_version = rgb_packet.get("calib_version")
                    if not calib_version:
                        calib_version = frame_data.get("calib_version")
                    depth = None
                    if frame_data.get("depth_shm"):
                        depth_packet = self.shm_reader.read_image(
                            frame_data["depth_shm"],
                            tuple(frame_data["depth_shape"]),
                            frame_data["depth_dtype"],
                        )
                        if depth_packet is not None:
                            depth_seq = int(depth_packet.get("sequence_id", -1))
                            # Depth is published in a separate triple-buffer from RGB,
                            # so its slot can rotate independently. Prefer freshness
                            # over a strict sequence match (same policy as RGB above):
                            # accept the depth packet if it is the matching sequence,
                            # or close to it (within a small window). Rejecting it here
                            # makes the runtime fall back to a flat synthetic depth
                            # plane, which collapses the safety/scene point cloud.
                            if depth_seq < 0 or depth_seq == sequence_id:
                                depth = depth_packet["image"]
                            elif abs(depth_seq - sequence_id) <= 2:
                                depth = depth_packet["image"]
                                self.depth_seq_mismatch += 1
                            else:
                                self.depth_seq_mismatch += 1

                # Run the vision module
                frame_bundle = FrameBundle(
                    frame_id=frame_id,
                    camera_id=self.camera_id,
                    sequence_id=sequence_id,
                    timestamp_ns=timestamp_ns,
                    rgb=rgb,
                    depth=depth,
                    meta={
                        "calib_version": calib_version,
                        "camera_data": self._resolve_frame_camera_data(frame_data.get("camera_data")),
                        "session": self.session_meta,
                    },
                )

                start_time = time.time()
                result = self.module.run(frame_bundle)
                process_time_ms = (time.time() - start_time) * 1000

                self.frames_processed += 1
                now_wall = time.time()
                if self.first_processed_at is None:
                    self.first_processed_at = now_wall
                self.last_processed_at = now_wall
                self.process_time_ms_sum += process_time_ms
                self.process_time_ms_count += 1

                if not isinstance(result, dict):
                    result = {"valid": True, "data": result}

                produced_timestamp_ns = time.time_ns()
                produced_monotonic_ns = time.perf_counter_ns()
                produced_latency_ms = (produced_timestamp_ns - int(timestamp_ns)) / 1e6
                self.produced_latency_ms_sum += produced_latency_ms
                self.produced_latency_ms_count += 1

                # Publish result via ZMQ
                self.publisher.publish(
                    {
                        "event": "VISION_RESULT",
                        "request_id": self.request_id,
                        "camera_id": self.camera_id,
                        "module": self.module_name,
                        "frame_id": frame_id,
                        "sequence_id": sequence_id,
                        "timestamp_ns": timestamp_ns,
                        "produced_timestamp_ns": produced_timestamp_ns,
                        "produced_monotonic_ns": produced_monotonic_ns,
                        "produced_latency_ms": round(produced_latency_ms, 2),
                        "result": result,
                        "process_time_ms": round(process_time_ms, 2),
                        "frames_processed": self.frames_processed,
                    }
                )

                # Write result to shared memory (if enabled)
                if self.shm_writer:
                    self.shm_writer.write_result(
                        timestamp_ns=timestamp_ns,
                        sequence_id=sequence_id,
                        result=result,
                        request_id=self.request_id,
                        camera_id=self.camera_id,
                        module=self.module_name,
                        process_time_ms=round(process_time_ms, 2),
                    )

                if self.one_shot:
                    keep_running = (
                        isinstance(result, dict)
                        and result.get("terminal") is False
                        and not bool(result.get("valid"))
                    )
                    if not keep_running:
                        self.running.clear()

            except Empty:
                # No frame available, continue waiting
                if (
                    self.one_shot
                    and self.first_frame_timeout_s > 0.0
                    and self.frames_processed <= 0
                    and self.start_time is not None
                    and (time.time() - self.start_time) >= self.first_frame_timeout_s
                ):
                    details = {
                        "camera_id": self.camera_id,
                        "module": self.module_name,
                        "first_frame_timeout_s": float(self.first_frame_timeout_s),
                        "frames_received": int(self.frames_received),
                        "frames_processed": int(self.frames_processed),
                        "skip_pre_start_frame": int(self.skip_pre_start_frame),
                        "skip_duplicate_source": int(self.skip_duplicate_source),
                        "skip_invalid_meta": int(self.skip_invalid_meta),
                        "skip_no_shm_reader": int(self.skip_no_shm_reader),
                        "skip_rgb_missing": int(self.skip_rgb_missing),
                        "skip_rgb_seq_mismatch": int(self.skip_rgb_seq_mismatch),
                        "depth_seq_mismatch": int(self.depth_seq_mismatch),
                    }
                    print(
                        f"[{self.request_id}] First frame timeout after "
                        f"{self.first_frame_timeout_s:.2f}s on camera {self.camera_id}"
                    )
                    self._publish_terminal_error("vision_no_frame", details)
                    self.running.clear()
                continue
            except Exception as e:
                self.skip_errors += 1
                timestamp_ns = time.time_ns()
                sequence_id = 0
                if isinstance(frame_data, dict):
                    timestamp_ns = frame_data.get("timestamp_ns", timestamp_ns)
                    sequence_id = frame_data.get("sequence_id", sequence_id)

                print(f"[{self.request_id}] Error processing frame: {e}")
                self.publisher.publish(
                    {
                        "event": "VISION_PROCESSING_ERROR",
                        "request_id": self.request_id,
                        "camera_id": self.camera_id,
                        "module": self.module_name,
                        "error": str(e),
                    }
                )
                if self.shm_writer:
                    self.shm_writer.write_result(
                        timestamp_ns=timestamp_ns,
                        sequence_id=sequence_id,
                        result={},
                        request_id=self.request_id,
                        camera_id=self.camera_id,
                        module=self.module_name,
                        process_time_ms=0.0,
                        error=str(e),
                    )

        # Calculate statistics
        elapsed_time = time.time() - self.start_time if self.start_time else 0
        input_fps = self.frames_received / elapsed_time if elapsed_time > 0 else 0
        actual_fps = self.frames_processed / elapsed_time if elapsed_time > 0 else 0
        active_span = 0.0
        active_fps = 0.0
        if (
            self.first_processed_at is not None
            and self.last_processed_at is not None
            and self.last_processed_at >= self.first_processed_at
        ):
            active_span = self.last_processed_at - self.first_processed_at
            active_fps = (
                self.frames_processed / active_span if active_span > 1e-6 else 0.0
            )
        dropped = max(0, self.frames_received - self.frames_processed)
        drop_pct = (
            (100.0 * dropped / self.frames_received)
            if self.frames_received > 0
            else 0.0
        )
        avg_process_ms = (
            self.process_time_ms_sum / self.process_time_ms_count
            if self.process_time_ms_count > 0
            else 0.0
        )
        avg_latency_ms = (
            self.produced_latency_ms_sum / self.produced_latency_ms_count
            if self.produced_latency_ms_count > 0
            else 0.0
        )

        print(
            f"[{self.request_id}] Stopped: processed {self.frames_processed}/"
            f"{self.frames_received} frames, input FPS: {input_fps:.2f}, "
            f"output FPS: {actual_fps:.2f}, active FPS: {active_fps:.2f}, "
            f"drop: {dropped} ({drop_pct:.1f}%), "
            f"proc_avg: {avg_process_ms:.1f}ms, latency_avg: {avg_latency_ms:.1f}ms"
        )
        print(
            f"[{self.request_id}] Drop reasons: queue_full={self.drop_queue_full}, "
            f"throttle_skip={self.skip_throttle}, dup_source={self.skip_duplicate_source}, "
            f"invalid_meta={self.skip_invalid_meta}, "
            f"no_shm_reader={self.skip_no_shm_reader}, rgb_missing={self.skip_rgb_missing}, "
            f"rgb_seq_mismatch={self.skip_rgb_seq_mismatch}, depth_seq_mismatch={self.depth_seq_mismatch}, "
            f"pre_start={self.skip_pre_start_frame}, errors={self.skip_errors}"
        )

        self.publisher.publish(
            {
                "event": "VISION_REQUEST_STOPPED",
                "request_id": self.request_id,
                "camera_id": self.camera_id,
                "module": self.module_name,
                "frames_processed": self.frames_processed,
                "frames_received": self.frames_received,
                "elapsed_time_sec": round(elapsed_time, 2),
                "input_fps": round(input_fps, 2),
                "actual_fps": round(actual_fps, 2),
                "active_fps": round(active_fps, 2),
                "dropped_frames": dropped,
                "drop_percent": round(drop_pct, 1),
                "avg_process_time_ms": round(avg_process_ms, 2),
                "avg_latency_ms": round(avg_latency_ms, 2),
                "drop_queue_full": int(self.drop_queue_full),
                "skip_throttle": int(self.skip_throttle),
                "skip_duplicate_source": int(self.skip_duplicate_source),
                "skip_invalid_meta": int(self.skip_invalid_meta),
                "skip_no_shm_reader": int(self.skip_no_shm_reader),
                "skip_rgb_missing": int(self.skip_rgb_missing),
                "skip_rgb_seq_mismatch": int(self.skip_rgb_seq_mismatch),
                "depth_seq_mismatch": int(self.depth_seq_mismatch),
                "skip_pre_start_frame": int(self.skip_pre_start_frame),
                "skip_errors": int(self.skip_errors),
            }
        )


class VisionEngine:
    """
    Vision Engine with per-request thread management.
    Supports START/STOP commands to create/destroy continuous processing threads.
    Each request runs independently at its own FPS.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.running = False
        self.default_fps_limit = cfg.get("vision", {}).get("fps_limit", 0)
        self.default_process_mode = cfg.get("vision", {}).get(
            "process_mode", "continuous"
        )
        self.frame_cache_max_age_s = float(
            cfg.get("vision", {}).get("frame_cache_max_age_s", 2.0)
        )

        # Camera state tracking
        self.camera_states = {}  # camera_id -> {last_ts, last_seq, fps_estimate}
        self.camera_available = set()
        self.last_frames: Dict[str, Dict[str, Any]] = {}
        self.last_frame_ts: Dict[str, float] = {}

        # Request management (thread-safe)
        self.req_lock = Lock()
        self.active_requests: Dict[str, RequestThread] = (
            {}
        )  # request_id -> RequestThread
        self.requests_by_camera: Dict[str, set] = defaultdict(
            set
        )  # camera_id -> set of request_ids

        # ZMQ communication
        camera_cfg = cfg.get("camera_events") or cfg.get("control_plane") or {}
        results_cfg = cfg.get("results") or cfg.get("control_plane") or {}

        self.sub = ZmqSubscriber(
            camera_cfg["subscribe_endpoint"],
            camera_cfg.get("subscribe_topic", "camera"),
        )

        self.pub = ZmqPublisher(
            results_cfg["publish_endpoint"], results_cfg.get("publish_topic", "vision")
        )

        # Shared memory reader
        shm_cfg = cfg.get("shm", {})
        data_cfg = cfg.get("data_plane", {})
        header_bytes = shm_cfg.get("frame_header_bytes") or data_cfg.get(
            "header_bytes", 64
        )
        self.reader = ShmReader(header_bytes=header_bytes)
        self.result_max_json_bytes = shm_cfg.get(
            "result_max_json_bytes"
        ) or data_cfg.get("result_max_json_bytes")

        # Load vision modules
        self.module_defs: Dict[str, Dict[str, Any]] = {}
        for mod_cfg in cfg.get("vision", {}).get("modules", []):
            name = mod_cfg["name"]
            cls = get_module_class(name)
            self.module_defs[name] = {
                "cls": cls,
                "params": mod_cfg.get("params", {}),
            }

        print(f"[engine] Initialized with modules: {list(self.module_defs.keys())}")

    def _create_module_instance(self, module_name: str, params: Dict[str, Any]) -> Any:
        mod_def = self.module_defs.get(module_name)
        if not mod_def:
            raise ValueError(f"Unknown module: {module_name}")

        merged_params = dict(mod_def["params"])
        merged_params.update(params or {})

        cls = mod_def["cls"]
        try:
            sig = inspect.signature(cls.__init__)
        except (TypeError, ValueError):
            sig = None

        if sig:
            params_spec = list(sig.parameters.values())
            accepts_kwargs = any(
                p.kind == inspect.Parameter.VAR_KEYWORD for p in params_spec
            )
            accepts_params = any(p.name in ("name", "params") for p in params_spec)
            if accepts_kwargs or accepts_params:
                return cls(name=module_name, params=merged_params)

        return cls(**merged_params)

    def _register_request(self, req_thread: RequestThread):
        """Register a new request thread (thread-safe)."""
        with self.req_lock:
            self.active_requests[req_thread.request_id] = req_thread
            self.requests_by_camera[req_thread.camera_id].add(req_thread.request_id)

    def _unregister_request(self, request_id: str):
        """Unregister a request thread (thread-safe)."""
        with self.req_lock:
            req_thread = self.active_requests.pop(request_id, None)
            if req_thread:
                cam_reqs = self.requests_by_camera.get(req_thread.camera_id)
                if cam_reqs:
                    cam_reqs.discard(request_id)
                    if not cam_reqs:
                        self.requests_by_camera.pop(req_thread.camera_id, None)
            return req_thread

    def _get_requests_for_camera(self, camera_id: str):
        """Get all active request threads for a camera (thread-safe)."""
        with self.req_lock:
            request_ids = self.requests_by_camera.get(camera_id, set()).copy()
            return [
                self.active_requests[rid]
                for rid in request_ids
                if rid in self.active_requests
            ]

    def _handle_start_trigger(self, evt: Dict[str, Any], publisher: Optional[Any] = None):
        """
        Handle VISION_START trigger to create a new continuous processing request.

        Expected event format:
        {
            "event": "VISION_START",
            "request_id": "req-001",
            "camera_id": "cam_webcam",
            "module": "template_matching",
            "fps_limit": 5.0
        }
        """
        pub = publisher or self.pub
        req_id = evt.get("request_id")
        cam_id = evt.get("camera_id")
        module_name = evt.get("module")
        fps_limit = evt.get("fps_limit", self.default_fps_limit)  # 0 = unlimited
        module_params = evt.get("params") or evt.get("module_params") or {}
        calibration = evt.get("calibration", {})
        process_mode = evt.get("process_mode", self.default_process_mode)
        one_shot = bool(evt.get("one_shot", False)) or process_mode == "trigger_only"
        try:
            startup_warmup_frames = max(
                0, int((module_params or {}).get("startup_warmup_frames", 0) or 0)
            )
        except Exception:
            startup_warmup_frames = 0
        try:
            min_frame_timestamp_ns = max(
                0,
                int(
                    evt.get("min_frame_timestamp_ns")
                    or (module_params or {}).get("min_frame_timestamp_ns")
                    or 0
                ),
            )
        except Exception:
            min_frame_timestamp_ns = 0
        enable_shm_output = bool(evt.get("enable_shm_output", True))
        shm_max_result_size = evt.get("shm_max_result_size")

        print(
            f"[engine] START request received: request_id={req_id}, "
            f"camera={cam_id}, module={module_name}, fps={fps_limit}"
        )

        # Validate required fields
        if not req_id:
            print("[engine] START rejected: missing_request_id")
            pub.publish(
                {
                    "event": "VISION_START_REJECTED",
                    "reason": "missing_request_id",
                }
            )
            return

        if not cam_id:
            print(f"[engine] START rejected (missing_camera_id): request_id={req_id}")
            pub.publish(
                {
                    "event": "VISION_START_REJECTED",
                    "reason": "missing_camera_id",
                    "request_id": req_id,
                }
            )
            return

        if not module_name:
            print(
                f"[engine] START rejected (missing_module): request_id={req_id} camera={cam_id}"
            )
            pub.publish(
                {
                    "event": "VISION_START_REJECTED",
                    "reason": "missing_module",
                    "request_id": req_id,
                }
            )
            return

        # Check if request_id already exists
        with self.req_lock:
            if req_id in self.active_requests:
                print(
                    f"[engine] START rejected (duplicate_request_id): request_id={req_id}"
                )
                pub.publish(
                    {
                        "event": "VISION_START_REJECTED",
                        "reason": "duplicate_request_id",
                        "request_id": req_id,
                        "camera_id": cam_id,
                    }
                )
                return

        # Resolve module from runtime registry if it is missing from static config.
        if module_name not in self.module_defs:
            try:
                cls = get_module_class(module_name)
                self.module_defs[module_name] = {
                    "cls": cls,
                    "params": {},
                }
                print(
                    f"[engine] Module '{module_name}' was not preconfigured; "
                    "loaded dynamically from registry"
                )
            except Exception:
                print(
                    f"[engine] START rejected (unknown_module): request_id={req_id} "
                    f"camera={cam_id} module={module_name}"
                )
                pub.publish(
                    {
                        "event": "VISION_START_REJECTED",
                        "reason": "unknown_module",
                        "request_id": req_id,
                        "camera_id": cam_id,
                        "module": module_name,
                        "available_modules": list(self.module_defs.keys()),
                    }
                )
                return

        # Check if camera is available (allow start; wait for frames if needed)
        if cam_id not in self.camera_available:
            print(
                f"[engine] Camera '{cam_id}' not yet available; "
                "starting request and waiting for frames..."
            )

        # Validate FPS limit
        try:
            fps_limit = float(fps_limit)
            if fps_limit < 0:
                fps_limit = 0
        except (ValueError, TypeError):
            fps_limit = 0

        # Get camera's actual FPS for adaptive throttling
        cam_state = self.camera_states.get(cam_id, {})
        try:
            cam_fps = float(cam_state.get("fps_estimate", 30.0))
        except Exception:
            cam_fps = 30.0
        if cam_fps <= 0:
            cam_fps = 30.0

        # If requested FPS is effectively camera-rate, disable software throttling and
        # let camera cadence bound processing. This avoids jitter-induced under-run.
        throttle_mode = "camera_bound"
        effective_fps = 0.0
        if fps_limit > 0:
            near_camera_rate = fps_limit >= (0.95 * cam_fps)
            common_30fps_case = cam_fps <= 35.0 and fps_limit >= 29.0
            if near_camera_rate or common_30fps_case:
                effective_fps = 0.0
                throttle_mode = "camera_bound"
            else:
                effective_fps = fps_limit
                throttle_mode = "software"

        # Create module instance for this request
        try:
            module_instance = self._create_module_instance(module_name, module_params)
        except Exception as e:
            print(
                f"[engine] START rejected (module_init_failed): "
                f"request_id={req_id} camera={cam_id} module={module_name} error={e}"
            )
            pub.publish(
                {
                    "event": "VISION_START_REJECTED",
                    "reason": "module_init_failed",
                    "request_id": req_id,
                    "camera_id": cam_id,
                    "module": module_name,
                    "error": str(e),
                }
            )
            return

        # Create request thread
        req_thread = RequestThread(
            request_id=req_id,
            camera_id=cam_id,
            module_name=module_name,
            fps_limit=effective_fps,
            module_instance=module_instance,
            publisher=pub,
            shm_reader=self.reader,
            session_meta={
                "params": module_params,
                "calibration": calibration,
                "min_frame_timestamp_ns": min_frame_timestamp_ns,
            },
            enable_shm_output=enable_shm_output,
            shm_max_result_size=shm_max_result_size or self.result_max_json_bytes,
            one_shot=one_shot,
        )

        # Register and start
        self._register_request(req_thread)
        req_thread.start()

        cached = self.last_frames.get(cam_id)
        if cached:
            age_s = time.time() - self.last_frame_ts.get(cam_id, 0.0)
            if one_shot and startup_warmup_frames > 0:
                print(
                    f"[engine] One-shot warmup enabled for {req_id}; "
                    f"waiting for {startup_warmup_frames} fresh frame(s)"
                )
            elif one_shot and age_s <= self.frame_cache_max_age_s:
                req_thread.push_frame(cached)
            elif age_s > self.frame_cache_max_age_s:
                print(
                    f"[engine] No recent frame for {cam_id} (last {age_s:.2f}s ago), waiting..."
                )
            else:
                # For continuous requests, avoid immediate cached-frame injection.
                # It can create startup burst artifacts in observed FPS.
                print(
                    f"[engine] Live stream active for {cam_id}; waiting for next FRAME_READY"
                )
        else:
            print(f"[engine] No cached frame for {cam_id}, waiting for frames...")

        effective_label = (
            f"{effective_fps:.1f}" if effective_fps > 0 else "unlimited(camera-bound)"
        )
        print(
            f"[engine] Request {req_id} is now RUNNING: {module_name} @ {effective_label} FPS "
            f"(requested: {fps_limit}, camera: {cam_fps:.1f}, throttle={throttle_mode})"
        )
        print(
            f"[engine] Waiting for frames from camera '{cam_id}' to begin analysis..."
        )

        pub.publish(
            {
                "event": "VISION_START_ACCEPTED",
                "request_id": req_id,
                "camera_id": cam_id,
                "module": module_name,
                "fps_limit_requested": fps_limit,
                "fps_limit_effective": round(effective_fps, 2),
                "camera_fps": round(cam_fps, 2),
                "throttle_mode": throttle_mode,
            }
        )

    def _handle_stop_trigger(self, evt: Dict[str, Any], publisher: Optional[Any] = None):
        """
        Handle VISION_STOP trigger to stop a running request.

        Expected event format:
        {
            "event": "VISION_STOP",
            "request_id": "req-001"
        }
        """
        pub = publisher or self.pub
        req_id = evt.get("request_id")

        if not req_id:
            pub.publish(
                {
                    "event": "VISION_STOP_REJECTED",
                    "reason": "missing_request_id",
                }
            )
            return

        # Get and unregister the request thread
        req_thread = self._unregister_request(req_id)

        if not req_thread:
            pub.publish(
                {
                    "event": "VISION_STOP_REJECTED",
                    "reason": "request_not_found",
                    "request_id": req_id,
                }
            )
            return

        # Stop the thread
        req_thread.stop()

        print(f"[engine] Stopped request {req_id}")

        pub.publish(
            {
                "event": "VISION_STOP_ACCEPTED",
                "request_id": req_id,
                "camera_id": req_thread.camera_id,
                "module": req_thread.module_name,
            }
        )

    def _update_camera_fps(self, camera_id: str, timestamp_ns: int, sequence_id: int):
        """Estimate camera FPS based on frame timing."""
        state = self.camera_states.get(camera_id)

        if not state:
            return

        try:
            timestamp_ns = int(timestamp_ns)
            sequence_id = int(sequence_id)
        except Exception:
            return

        last_ts = int(state.get("last_ts", 0))
        last_seq = int(state.get("last_seq", 0))

        # Ignore out-of-order/duplicate samples; only allow reset after long silence.
        if last_seq > 0 and sequence_id <= last_seq:
            if timestamp_ns <= last_ts:
                return
            if (timestamp_ns - last_ts) < int(5e9):
                return

        if last_ts > 0 and last_seq > 0:
            time_diff_sec = (timestamp_ns - last_ts) / 1e9
            seq_diff = sequence_id - last_seq

            if time_diff_sec > 0 and seq_diff > 0:
                instant_fps = seq_diff / time_diff_sec

                # Smooth FPS estimate with exponential moving average
                current_estimate = state.get("fps_estimate", instant_fps)
                alpha = 0.2  # Smoothing factor
                state["fps_estimate"] = (
                    alpha * instant_fps + (1 - alpha) * current_estimate
                )

        state["last_ts"] = timestamp_ns
        state["last_seq"] = sequence_id

    def _distribute_frame_to_requests(self, camera_id: str, frame_data: Dict[str, Any]):
        """Distribute frame to all active requests for this camera."""
        request_threads = self._get_requests_for_camera(camera_id)

        for req_thread in request_threads:
            req_thread.push_frame(frame_data)

    def handle_inline_frame(self, evt: Dict[str, Any]) -> None:
        """Accept a decoded RGB/depth frame from a network transport.

        Local camera-core events only carry shared-memory names. Remote websocket
        clients send image arrays inline because shared memory is host-local.
        """
        cam_id = evt.get("camera_id")
        seq_id = evt.get("sequence_id")
        ts_ns = evt.get("timestamp_ns")
        rgb = evt.get("rgb")
        if not cam_id or seq_id is None or ts_ns is None or rgb is None:
            return

        if cam_id not in self.camera_states:
            print(f"[engine] Discovered remote camera: {cam_id}")
            self.camera_states[cam_id] = {
                "last_ts": 0,
                "last_seq": 0,
                "fps_estimate": 30.0,
                "last_dispatched_seq": 0,
            }
            self.camera_available.add(cam_id)

        self._update_camera_fps(cam_id, ts_ns, seq_id)
        cam_state = self.camera_states.get(cam_id) or {}
        last_dispatched_seq = int(cam_state.get("last_dispatched_seq", 0))
        try:
            seq_int = int(seq_id)
        except Exception:
            seq_int = 0
        if seq_int > 0 and seq_int <= last_dispatched_seq:
            return
        if seq_int > 0:
            cam_state["last_dispatched_seq"] = seq_int

        frame_data = {
            "rgb": rgb,
            "depth": evt.get("depth"),
            "calib_version": evt.get("calib_version"),
            "camera_data": evt.get("camera_data"),
            "frame_id": str(evt.get("frame_id") or f"{cam_id}:{seq_id}"),
            "timestamp_ns": ts_ns,
            "sequence_id": seq_id,
        }
        self.last_frames[str(cam_id)] = frame_data
        self.last_frame_ts[str(cam_id)] = time.time()
        self._distribute_frame_to_requests(str(cam_id), frame_data)

    def run(self):
        """Main event loop."""
        print("[engine] Vision Engine started")
        print("[engine] Listening for VISION_START and VISION_STOP triggers...")
        self.running = True

        while self.running:
            # Receive event from ZMQ
            try:
                _, evt = self.sub.recv(timeout_ms=100)
            except Exception as exc:
                if not self.running:
                    break
                print(f"[engine] Subscriber recv failed: {exc}")
                continue

            if evt is None:
                continue

            event_type = evt.get("event")

            # Handle START trigger
            if event_type == "VISION_START":
                print("[engine] Received VISION_START trigger")
                self._handle_start_trigger(evt)
                continue

            # Handle STOP trigger
            if event_type == "VISION_STOP":
                self._handle_stop_trigger(evt)
                continue

            # Handle FRAME_READY events
            if event_type == "FRAME_READY":
                cam_id = evt.get("camera_id")
                seq_id = evt.get("sequence_id")
                ts_ns = evt.get("timestamp_ns")

                if not cam_id or seq_id is None or ts_ns is None:
                    continue

                # Mark camera as available on first frame
                if cam_id not in self.camera_states:
                    print(f"[engine] Discovered camera: {cam_id}")
                    self.camera_states[cam_id] = {
                        "last_ts": 0,
                        "last_seq": 0,
                        "fps_estimate": 30.0,  # Default assumption
                        "last_dispatched_seq": 0,
                    }
                    self.camera_available.add(cam_id)

                # Update camera FPS estimate
                self._update_camera_fps(cam_id, ts_ns, seq_id)
                cam_state = self.camera_states.get(cam_id) or {}
                last_dispatched_seq = int(cam_state.get("last_dispatched_seq", 0))
                if int(seq_id) <= last_dispatched_seq:
                    continue
                cam_state["last_dispatched_seq"] = int(seq_id)

                try:
                    rgb_shm = evt.get("rgb_shm")
                    rgb_shape = evt.get("rgb_shape")
                    rgb_dtype = evt.get("rgb_dtype")
                    if not rgb_shm or not rgb_shape or not rgb_dtype:
                        continue

                    # Push SHM metadata only; request thread reads/copies if it actually processes.
                    frame_data = {
                        "rgb_shm": rgb_shm,
                        "rgb_shape": tuple(rgb_shape),
                        "rgb_dtype": rgb_dtype,
                        "depth_shm": evt.get("depth_shm"),
                        "depth_shape": tuple(evt.get("depth_shape") or ()),
                        "depth_dtype": evt.get("depth_dtype"),
                        "calib_version": evt.get("calib_version"),
                        "camera_data": evt.get("camera_data"),
                        "frame_id": f"{cam_id}:{seq_id}",
                        "timestamp_ns": ts_ns,
                        "sequence_id": seq_id,
                    }
                    self.last_frames[cam_id] = frame_data
                    self.last_frame_ts[cam_id] = time.time()

                    # Update the latest frame cache even when no requests are active.
                    # This keeps trigger-only / one-shot sessions responsive on the
                    # next start instead of forcing them to begin from a stale cache.
                    if not self._get_requests_for_camera(cam_id):
                        continue

                    # Distribute to all active requests
                    self._distribute_frame_to_requests(cam_id, frame_data)

                except Exception as e:
                    print(f"[engine] Error reading frame: {e}")
                    continue

        print("[engine] Vision Engine stopped")

    def stop(
        self,
        *,
        wait_for_requests: bool = True,
        request_join_timeout: float = 2.0,
        close_request_shm: bool = True,
    ):
        """Stop the engine and all active requests."""
        print("[engine] Stopping Vision Engine...")
        self.running = False

        # Stop all active request threads
        with self.req_lock:
            request_threads = list(self.active_requests.values())
            self.active_requests.clear()
            self.requests_by_camera.clear()

        for req_thread in request_threads:
            print(f"[engine] Stopping request {req_thread.request_id}...")
            req_thread.stop(
                wait=wait_for_requests,
                join_timeout=request_join_timeout,
                close_shm_writer=close_request_shm,
            )

        # Close ZMQ sockets
        self.sub.close()
        self.pub.close()
        try:
            self.reader.close()
        except Exception:
            pass

        print("[engine] All requests stopped")
