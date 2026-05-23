"""Implementation for `orchestrator.vision.cache`."""

import logging
import threading
import time
from collections import deque
from typing import Any, Deque, Dict, Optional

from orchestrator.vision.results import VisionResultSubscriber

log = logging.getLogger("vision_cache")


class VisionResultCache:
    def __init__(self, endpoint: str, topic: str = "vision"):
        self.sub = VisionResultSubscriber(endpoint, topic)
        self._latest: Dict[str, Dict[str, Any]] = {}
        self._latest_any: Optional[Dict[str, Any]] = None
        self._latest_image: Dict[str, Dict[str, Any]] = {}
        self._latest_image_any: Optional[Dict[str, Any]] = None
        self._history: Dict[str, Deque[Dict[str, Any]]] = {}
        self._history_any: Deque[Dict[str, Any]] = deque(maxlen=60)
        self._fps_estimate: Dict[str, float] = {}
        self._fps_instant: Dict[str, float] = {}
        self._last_ts_ns: Dict[str, int] = {}
        self._result_fps_estimate: Dict[str, float] = {}
        self._result_fps_instant: Dict[str, float] = {}
        self._last_recv_ns: Dict[str, int] = {}
        self._publish_fps_estimate: Dict[str, float] = {}
        self._publish_fps_instant: Dict[str, float] = {}
        self._last_produced_ns: Dict[str, int] = {}
        self._last_frame_id: Dict[str, str] = {}
        self._last_sequence_id: Dict[str, int] = {}
        self._telemetry_window_start = time.perf_counter()
        self._telemetry_result_count = 0
        self._telemetry_result_by_request: Dict[str, int] = {}
        self._telemetry_latest: Optional[Dict[str, Any]] = None
        self._telemetry_process_ms_sum = 0.0
        self._telemetry_process_ms_count = 0
        self._telemetry_latency_ms_sum = 0.0
        self._telemetry_latency_ms_count = 0
        self._lock = threading.Lock()
        self._running = True
        # Daemon thread avoids process hang if shutdown races with subscriber poll.
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _emit_telemetry_if_due(self) -> None:
        now = time.perf_counter()
        elapsed = now - self._telemetry_window_start
        if elapsed < 1.0:
            return
        self._telemetry_window_start = now

        total = self._telemetry_result_count
        by_request = self._telemetry_result_by_request
        latest = self._telemetry_latest

        self._telemetry_result_count = 0
        self._telemetry_result_by_request = {}
        self._telemetry_latest = None
        proc_ms_sum = self._telemetry_process_ms_sum
        proc_ms_count = self._telemetry_process_ms_count
        latency_ms_sum = self._telemetry_latency_ms_sum
        latency_ms_count = self._telemetry_latency_ms_count
        self._telemetry_process_ms_sum = 0.0
        self._telemetry_process_ms_count = 0
        self._telemetry_latency_ms_sum = 0.0
        self._telemetry_latency_ms_count = 0

        if total == 0:
            log.info("[telemetry] vision_results incoming=no rps=0.00")
            return

        rps = total / elapsed
        top = sorted(by_request.items(), key=lambda kv: kv[1], reverse=True)[:3]
        top_str = ",".join([f"{rid}:{cnt}" for rid, cnt in top])
        latest_req = latest.get("request_id") if latest else "unknown"
        latest_frame = latest.get("frame_id") if latest else "unknown"
        latest_latency = latest.get("produced_latency_ms") if latest else None
        latest_latency_s = (
            f"{float(latest_latency):.1f}ms"
            if isinstance(latest_latency, (int, float))
            else "n/a"
        )
        avg_proc_ms = (
            (proc_ms_sum / proc_ms_count) if proc_ms_count > 0 else None
        )
        avg_latency_ms = (
            (latency_ms_sum / latency_ms_count) if latency_ms_count > 0 else None
        )
        if avg_proc_ms is not None and avg_latency_ms is not None:
            avg_overhead_ms = max(0.0, avg_latency_ms - avg_proc_ms)
            perf_part = (
                f" proc_avg={avg_proc_ms:.1f}ms latency_avg={avg_latency_ms:.1f}ms"
                f" overhead_avg={avg_overhead_ms:.1f}ms"
            )
        elif avg_proc_ms is not None:
            perf_part = f" proc_avg={avg_proc_ms:.1f}ms"
        elif avg_latency_ms is not None:
            perf_part = f" latency_avg={avg_latency_ms:.1f}ms"
        else:
            perf_part = ""

        log.info(
            "[telemetry] vision_results incoming=yes rps=%.2f total=%d reqs=%d top=%s latest=%s frame=%s latency=%s%s",
            rps,
            total,
            len(by_request),
            top_str or "none",
            latest_req,
            latest_frame,
            latest_latency_s,
            perf_part,
        )

    def _run(self) -> None:
        recv_count = 0
        skip_count = 0
        store_count = 0

        while self._running:
            try:
                evt = self.sub.recv(timeout_ms=200)
            except Exception:
                if not self._running:
                    break
                continue
            self._emit_telemetry_if_due()
            if not evt:
                continue

            recv_count += 1
            event_type = evt.get("event")
            request_id = evt.get("request_id")

            # Keep high-frequency pipeline traces at DEBUG to avoid terminal spam.
            if recv_count <= 5 or recv_count % 10 == 0:
                log.debug(
                    "[vision_cache._run] Received event #%s: event=%s, request_id=%s",
                    recv_count,
                    event_type,
                    request_id,
                )

            if evt.get("event") != "VISION_RESULT":
                skip_count += 1
                if skip_count <= 3:
                    log.debug(
                        f"[vision_cache._run] Skipping non-VISION_RESULT: {event_type}"
                    )
                continue

            if not request_id:
                log.warning("[vision_cache._run] VISION_RESULT missing request_id")
                continue

            frame_id = str(evt.get("frame_id") or "")
            seq_raw = evt.get("sequence_id")
            ts_raw = evt.get("timestamp_ns")
            try:
                sequence_id = int(seq_raw) if seq_raw is not None else 0
            except Exception:
                sequence_id = 0
            try:
                ts_ns = int(ts_raw) if ts_raw is not None else None
            except Exception:
                ts_ns = None

            with self._lock:
                prev_frame_id = self._last_frame_id.get(request_id)
                prev_seq = int(self._last_sequence_id.get(request_id, 0))
                prev_ts = self._last_ts_ns.get(request_id)

                if frame_id and prev_frame_id == frame_id:
                    continue

                if sequence_id > 0 and prev_seq > 0 and sequence_id <= prev_seq:
                    # Allow sequence reset only after long silence (camera reconnect).
                    if (
                        not isinstance(ts_ns, int)
                        or not isinstance(prev_ts, int)
                        or ts_ns <= prev_ts
                        or (ts_ns - prev_ts) < int(5e9)
                    ):
                        continue

                if not frame_id and sequence_id <= 0:
                    if isinstance(ts_ns, int) and isinstance(prev_ts, int) and ts_ns <= prev_ts:
                        continue

                if frame_id:
                    self._last_frame_id[request_id] = frame_id
                if sequence_id > 0:
                    self._last_sequence_id[request_id] = sequence_id

                if isinstance(ts_ns, int):
                    if isinstance(prev_ts, int) and ts_ns > prev_ts:
                        dt_s = (ts_ns - prev_ts) / 1e9
                        if dt_s > 1e-6:
                            fps_inst = 1.0 / dt_s
                            if 0.0 < fps_inst < 200.0:
                                self._fps_instant[request_id] = fps_inst
                                prev_ema = self._fps_estimate.get(request_id, fps_inst)
                                self._fps_estimate[request_id] = prev_ema + 0.2 * (
                                    fps_inst - prev_ema
                                )
                                log.debug(
                                    "[vision_cache] fps request_id=%s frame_id=%s fps=%.1f",
                                    request_id,
                                    evt.get("frame_id") or "unknown",
                                    fps_inst,
                                )
                    if prev_ts is None or ts_ns > prev_ts:
                        self._last_ts_ns[request_id] = ts_ns

                source_fps_cap = self._fps_estimate.get(request_id)
                if source_fps_cap is None:
                    source_fps_cap = self._fps_instant.get(request_id)

                recv_ns = time.time_ns()
                produced_ns = evt.get("produced_monotonic_ns")
                if not isinstance(produced_ns, int):
                    produced_ns = evt.get("produced_timestamp_ns")
                if isinstance(produced_ns, int):
                    prev_produced_ns = self._last_produced_ns.get(request_id)
                    if prev_produced_ns is not None and produced_ns > prev_produced_ns:
                        produced_dt_s = (produced_ns - prev_produced_ns) / 1e9
                        if produced_dt_s > 1e-6:
                            publish_fps_inst = 1.0 / produced_dt_s
                            if (
                                isinstance(source_fps_cap, (int, float))
                                and source_fps_cap > 0
                            ):
                                publish_fps_inst = min(
                                    publish_fps_inst, max(5.0, float(source_fps_cap) * 1.15)
                                )
                            if 0.0 < publish_fps_inst < 200.0:
                                self._publish_fps_instant[request_id] = publish_fps_inst
                                prev_publish_ema = self._publish_fps_estimate.get(
                                    request_id, publish_fps_inst
                                )
                                self._publish_fps_estimate[request_id] = (
                                    prev_publish_ema
                                    + 0.2 * (publish_fps_inst - prev_publish_ema)
                                )
                    self._last_produced_ns[request_id] = produced_ns

                prev_recv_ns = self._last_recv_ns.get(request_id)
                if prev_recv_ns is not None and recv_ns > prev_recv_ns:
                    recv_dt_s = (recv_ns - prev_recv_ns) / 1e9
                    if recv_dt_s > 1e-6:
                        result_fps_inst = 1.0 / recv_dt_s
                        if (
                            isinstance(source_fps_cap, (int, float))
                            and source_fps_cap > 0
                        ):
                            result_fps_inst = min(
                                result_fps_inst, max(5.0, float(source_fps_cap) * 1.15)
                            )
                        if 0.0 < result_fps_inst < 200.0:
                            self._result_fps_instant[request_id] = result_fps_inst
                            prev_recv_ema = self._result_fps_estimate.get(
                                request_id, result_fps_inst
                            )
                            self._result_fps_estimate[request_id] = (
                                prev_recv_ema + 0.2 * (result_fps_inst - prev_recv_ema)
                            )
                self._last_recv_ns[request_id] = recv_ns

                self._latest[request_id] = evt
                self._latest_any = evt
                history = self._history.setdefault(request_id, deque(maxlen=60))
                history.append(evt)
                self._history_any.append(evt)
                result = evt.get("result") or {}
                if result.get("image_b64"):
                    self._latest_image[request_id] = evt
                    self._latest_image_any = evt

            store_count += 1
            self._telemetry_result_count += 1
            self._telemetry_result_by_request[request_id] = (
                self._telemetry_result_by_request.get(request_id, 0) + 1
            )
            self._telemetry_latest = evt
            process_ms = evt.get("process_time_ms")
            if isinstance(process_ms, (int, float)):
                self._telemetry_process_ms_sum += float(process_ms)
                self._telemetry_process_ms_count += 1
            latency_ms = evt.get("produced_latency_ms")
            if isinstance(latency_ms, (int, float)):
                self._telemetry_latency_ms_sum += float(latency_ms)
                self._telemetry_latency_ms_count += 1
            result = evt.get("result") or {}
            matches_count = len(result.get("matches") or [])
            log.info(
                "vision_result_cached request_id=%s frame_id=%s sequence_id=%s matches=%s",
                request_id,
                frame_id,
                sequence_id,
                matches_count,
            )
            print(
                "[orchestrator] vision_result_cached "
                f"request_id={request_id} frame_id={frame_id} "
                f"sequence_id={sequence_id} matches={matches_count}",
                flush=True,
            )

            if store_count <= 5 or store_count % 10 == 0:
                log.debug(
                    "[vision_cache._run] Storing VISION_RESULT #%s: request_id=%s, matches=%s",
                    store_count,
                    request_id,
                    matches_count,
                )

        log.debug(
            "[vision_cache._run] Exiting: received=%s, skipped=%s, stored=%s",
            recv_count,
            skip_count,
            store_count,
        )

    def get_latest(self, request_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            return self._latest.get(request_id)

    def get_latest_any(self) -> Optional[Dict[str, Any]]:
        with self._lock:
            return self._latest_any

    def get_latest_image(self, request_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            return self._latest_image.get(request_id)

    def get_latest_image_any(self) -> Optional[Dict[str, Any]]:
        with self._lock:
            return self._latest_image_any

    def get_by_frame(self, request_id: str, frame_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            history = self._history.get(request_id)
            if not history:
                return None
            for evt in reversed(history):
                if evt.get("frame_id") == frame_id:
                    return evt
        return None

    def get_by_frame_any(self, frame_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            for evt in reversed(self._history_any):
                if evt.get("frame_id") == frame_id:
                    return evt
        return None

    def wait_latest(
        self,
        request_id: str,
        timeout_s: float = 1.0,
    ) -> Optional[Dict[str, Any]]:
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            evt = self.get_latest(request_id)
            if evt:
                return evt
            time.sleep(0.05)
        return None

    def get_fps_estimate(self, request_id: str) -> Optional[float]:
        with self._lock:
            return self._fps_estimate.get(request_id)

    def get_fps_instant(self, request_id: str) -> Optional[float]:
        with self._lock:
            return self._fps_instant.get(request_id)

    def get_result_fps_estimate(self, request_id: str) -> Optional[float]:
        with self._lock:
            return self._result_fps_estimate.get(request_id)

    def get_result_fps_instant(self, request_id: str) -> Optional[float]:
        with self._lock:
            return self._result_fps_instant.get(request_id)

    def get_publish_fps_estimate(self, request_id: str) -> Optional[float]:
        with self._lock:
            return self._publish_fps_estimate.get(request_id)

    def get_publish_fps_instant(self, request_id: str) -> Optional[float]:
        with self._lock:
            return self._publish_fps_instant.get(request_id)

    def close(self) -> None:
        self._running = False
        self.sub.close()
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)
