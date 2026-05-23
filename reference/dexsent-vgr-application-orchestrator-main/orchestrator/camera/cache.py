"""Implementation for `orchestrator.camera.cache`."""

import json
import logging
import threading
import time
from collections import deque
from typing import Any, Dict, List, Optional

import zmq

log = logging.getLogger("orchestrator.telemetry")


class CameraEventSubscriber:
    def __init__(self, endpoint: str, topic: str = "camera") -> None:
        self.ctx = zmq.Context.instance()
        self.sock = self.ctx.socket(zmq.SUB)
        self.sock.setsockopt(zmq.LINGER, 0)
        self.sock.connect(endpoint)
        self.sock.setsockopt_string(zmq.SUBSCRIBE, topic)

        self.poller = zmq.Poller()
        self.poller.register(self.sock, zmq.POLLIN)

    def recv(self, timeout_ms: int = 100) -> Optional[Dict[str, Any]]:
        try:
            events = dict(self.poller.poll(timeout_ms))
        except zmq.ZMQError:
            return None
        if self.sock not in events:
            return None
        try:
            raw = self.sock.recv_string()
        except zmq.ZMQError:
            return None
        if " " not in raw:
            return None
        _, payload = raw.split(" ", 1)
        try:
            return json.loads(payload)
        except json.JSONDecodeError:
            return None

    def close(self) -> None:
        try:
            self.poller.unregister(self.sock)
        except Exception:
            pass
        self.sock.close(0)


class CameraFrameCache:
    def __init__(self, endpoint: str, topic: str = "camera") -> None:
        self.sub = CameraEventSubscriber(endpoint, topic)
        self._latest: Dict[str, Dict[str, Any]] = {}
        self._latest_any: Optional[Dict[str, Any]] = None
        self._history: Dict[str, deque[Dict[str, Any]]] = {}
        self._cameras: List[str] = []
        self._fps_history: Dict[str, deque[int]] = {}
        self._fps_history_len = 12
        self._frame_history_len = 180
        self._telemetry_window_start = time.perf_counter()
        self._telemetry_frames: Dict[str, int] = {}
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

        with self._lock:
            cameras = list(self._cameras)
            fps_histories = {
                cam_id: self._fps_from_history(self._fps_history.get(cam_id))
                for cam_id in cameras
            }

        if not cameras:
            log.info("[telemetry] camera_fps no_camera_events")
            return

        parts = []
        for cam_id in cameras:
            count = self._telemetry_frames.get(cam_id, 0)
            window_fps = count / elapsed
            hist_fps = fps_histories.get(cam_id)
            if hist_fps is None:
                parts.append(f"{cam_id}=window:{window_fps:.2f}")
            else:
                parts.append(f"{cam_id}=window:{window_fps:.2f},est:{hist_fps:.2f}")
            self._telemetry_frames[cam_id] = 0

        log.info("[telemetry] camera_fps %s", " | ".join(parts))

    def _run(self) -> None:
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
            event_type = evt.get("event")
            if event_type not in ("FRAME_READY", "CAMERA_STARTED"):
                continue
            camera_id = evt.get("camera_id")
            if not camera_id:
                continue
            with self._lock:
                if camera_id not in self._cameras:
                    self._cameras.append(camera_id)
                # Only update frame cache for FRAME_READY events
                if event_type == "FRAME_READY":
                    self._latest[camera_id] = evt
                    self._latest_any = evt
                    self._telemetry_frames[camera_id] = (
                        self._telemetry_frames.get(camera_id, 0) + 1
                    )
                    history = self._history.setdefault(
                        camera_id, deque(maxlen=self._frame_history_len)
                    )
                    history.append(evt)
                    ts_ns = evt.get("timestamp_ns")
                    if isinstance(ts_ns, (int, float)):
                        self._record_timestamp(camera_id, int(ts_ns))

    def list_cameras(self) -> List[str]:
        with self._lock:
            return list(self._cameras)

    def get_latest(self, camera_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            return self._latest.get(camera_id)

    def get_latest_any(self) -> Optional[Dict[str, Any]]:
        with self._lock:
            return self._latest_any

    def get_by_frame_id(
        self, frame_id: str, camera_id: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        if not frame_id:
            return None
        with self._lock:
            if camera_id:
                return self._find_in_camera_history(camera_id, frame_id)
            for cam_id in self._cameras:
                evt = self._find_in_camera_history(cam_id, frame_id)
                if evt is not None:
                    return evt
        return None

    def _find_in_camera_history(
        self, camera_id: str, frame_id: str
    ) -> Optional[Dict[str, Any]]:
        history = self._history.get(camera_id)
        if not history:
            return None
        for evt in reversed(history):
            seq = evt.get("sequence_id")
            evt_cam = evt.get("camera_id") or camera_id
            evt_frame_id = f"{evt_cam}:{seq}" if evt_cam and seq is not None else ""
            if evt_frame_id == frame_id:
                return evt
        return None

    def get_camera_fps(self, camera_id: str) -> Optional[float]:
        with self._lock:
            history = self._fps_history.get(camera_id)
            return self._fps_from_history(history)

    def get_all_camera_fps(self) -> Dict[str, Optional[float]]:
        with self._lock:
            return {
                cam_id: self._fps_from_history(history)
                for cam_id, history in self._fps_history.items()
            }

    def _record_timestamp(self, camera_id: str, timestamp_ns: int) -> None:
        history = self._fps_history.setdefault(
            camera_id, deque(maxlen=self._fps_history_len)
        )
        history.append(timestamp_ns)

    def _fps_from_history(self, history: Optional[deque[int]]) -> Optional[float]:
        if not history or len(history) < 2:
            return None
        duration_ns = history[-1] - history[0]
        if duration_ns <= 0:
            return None
        frame_count = len(history) - 1
        return frame_count * 1e9 / duration_ns

    def close(self) -> None:
        self._running = False
        self.sub.close()
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)
