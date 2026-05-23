"""Implementation for `orchestrator.vision.client`."""

import asyncio
import base64
import json
import logging
import struct
import threading
import time
from pathlib import Path
from multiprocessing import resource_tracker
from multiprocessing import shared_memory
from typing import Any, Dict, Optional, Set

import numpy as np
import zmq

log = logging.getLogger("orchestrator.vision.client")

HEADER_FMT = "QQII40s"
HEADER_SIZE = 64
FLAG_VALID = 0x01


def _read_shm_image(name: str, shape: Any, dtype: Any) -> Optional[np.ndarray]:
    if not name or not shape:
        return None
    dtype_np = np.dtype(dtype or "uint8")
    shm = None
    try:
        try:
            shm = shared_memory.SharedMemory(name=str(name), create=False, track=False)
        except TypeError:
            shm = shared_memory.SharedMemory(name=str(name), create=False)
            try:
                resource_tracker.unregister(shm._name, "shared_memory")
            except Exception:
                pass
        expected = int(np.prod(shape)) * dtype_np.itemsize
        if len(shm.buf) < HEADER_SIZE + expected:
            return None
        header = bytes(shm.buf[:HEADER_SIZE])
        _, _, _, flags, _ = struct.unpack(HEADER_FMT, header)
        if not (flags & FLAG_VALID):
            return None
        arr = np.ndarray(
            shape=tuple(shape),
            dtype=dtype_np,
            buffer=shm.buf,
            offset=HEADER_SIZE,
        )
        return arr.copy()
    finally:
        if shm is not None:
            try:
                shm.close()
            except Exception:
                pass


def _array_payload(arr: Optional[np.ndarray]) -> Dict[str, Any]:
    if arr is None:
        return {}
    contiguous = np.ascontiguousarray(arr)
    return {
        "shape": list(contiguous.shape),
        "dtype": str(contiguous.dtype),
        "b64": base64.b64encode(contiguous.tobytes()).decode("ascii"),
    }


class VisionEngineClient:
    def __init__(
        self,
        push_endpoint: str,
        topic: str = "camera",
        *,
        transport: str = "zmq",
        websocket_url: str = "ws://127.0.0.1:8000/ws",
        results_push_endpoint: str = "tcp://127.0.0.1:5556",
        results_topic: str = "vision",
        camera_cache: Any = None,
        frame_poll_s: float = 0.02,
        max_frame_fps: float = 0.0,
    ):
        self.ctx = zmq.Context.instance()
        self.push_endpoint = push_endpoint
        self.topic = topic
        self.transport = str(transport or "zmq").strip().lower()
        self.websocket_url = websocket_url
        self.results_push_endpoint = results_push_endpoint
        self.results_topic = results_topic
        self.camera_cache = camera_cache
        self.frame_poll_s = max(0.005, float(frame_poll_s or 0.02))
        self.max_frame_fps = max(0.0, float(max_frame_fps or 0.0))
        self._lock = threading.Lock()
        self._active_sessions: Set[str] = set()
        self._session_cameras: Dict[str, str] = {}
        self._last_sent_seq: Dict[str, int] = {}
        self._last_sent_at: Dict[str, float] = {}
        # Maps request_id -> local output_root Path for artifact relay
        self._remote_artifact_roots: Dict[str, Path] = {}
        self._closed = False

        self.sock = None
        self._results_sock = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._loop_thread: Optional[threading.Thread] = None
        self._ws = None
        self._send_queue = None

        if self.transport == "websocket":
            self._start_ws_loop()
        else:
            self.transport = "zmq"
            self.sock = self.ctx.socket(zmq.PUSH)
            # Keep a short linger so stop messages can flush on graceful shutdown.
            self.sock.setsockopt(zmq.LINGER, 200)
            self.sock.connect(push_endpoint)

    def _send(self, msg: Dict[str, Any]) -> None:
        if self.transport == "websocket":
            self._send_ws(msg)
            return
        if self.sock is None:
            return
        self.sock.send_string(json.dumps(msg, separators=(",", ":"), ensure_ascii=False))

    def _start_ws_loop(self) -> None:
        self._loop = asyncio.new_event_loop()
        self._loop_thread = threading.Thread(
            target=self._run_ws_loop,
            daemon=True,
            name="vision-ws-client",
        )
        self._loop_thread.start()

    def _run_ws_loop(self) -> None:
        if self._loop is None:
            return
        asyncio.set_event_loop(self._loop)
        self._loop.create_task(self._ws_main())
        self._loop.run_forever()

    def _send_ws(self, msg: Dict[str, Any]) -> None:
        if self._loop is None or self._closed:
            return
        asyncio.run_coroutine_threadsafe(self._send_queue_put(msg), self._loop)

    async def _send_queue_put(self, msg: Dict[str, Any]) -> None:
        if self._send_queue is None:
            self._send_queue = asyncio.Queue()
        await self._send_queue.put(dict(msg))

    async def _ws_main(self) -> None:
        try:
            import websockets
        except Exception as exc:
            log.error("websocket transport requires the 'websockets' package: %s", exc)
            return

        if self._send_queue is None:
            self._send_queue = asyncio.Queue()
        while not self._closed:
            try:
                async with websockets.connect(
                    self.websocket_url,
                    max_size=None,
                    ping_interval=None,  # we use our own PING/PONG; auto-ping causes concurrent write crashes during artifact relay
                ) as ws:
                    self._ws = ws
                    recv_task = asyncio.create_task(self._ws_recv_loop(ws))
                    send_task = asyncio.create_task(self._ws_send_loop(ws))
                    frame_task = asyncio.create_task(self._ws_frame_loop())
                    done, pending = await asyncio.wait(
                        {recv_task, send_task, frame_task},
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    for task in pending:
                        task.cancel()
                    for task in done:
                        exc = task.exception()
                        if exc:
                            raise exc
            except asyncio.CancelledError:
                break
            except Exception as exc:
                if not self._closed:
                    log.warning("vision websocket disconnected: %s", exc)
                    await asyncio.sleep(1.0)
            finally:
                self._ws = None

    async def _ws_send_loop(self, ws: Any) -> None:
        while not self._closed:
            msg = await self._send_queue.get()
            await ws.send(json.dumps(msg, separators=(",", ":"), ensure_ascii=False))

    async def _ws_recv_loop(self, ws: Any) -> None:
        async for raw in ws:
            try:
                msg = json.loads(raw)
            except Exception:
                continue
            event_type = msg.get("event")
            if event_type == "VISION_RESULT":
                result = msg.get("result") if isinstance(msg.get("result"), dict) else {}
                matches = result.get("matches") if isinstance(result, dict) else None
                log.info(
                    "remote_vision_result_received request_id=%s frame_id=%s matches=%s",
                    msg.get("request_id"),
                    msg.get("frame_id"),
                    len(matches) if isinstance(matches, list) else None,
                )
                print(
                    "[orchestrator] remote_vision_result_received "
                    f"request_id={msg.get('request_id')} frame_id={msg.get('frame_id')} "
                    f"matches={len(matches) if isinstance(matches, list) else None}",
                    flush=True,
                )
            if event_type in {
                "VISION_RESULT",
                "VISION_REQUEST_STARTED",
                "VISION_REQUEST_STOPPED",
                "VISION_START_ACCEPTED",
                "VISION_START_REJECTED",
                "VISION_STOP_ACCEPTED",
                "VISION_STOP_REJECTED",
                "VISION_PROCESSING_ERROR",
            }:
                self._publish_remote_result(msg)
            elif event_type == "ARTIFACT_FILE":
                await asyncio.to_thread(self._write_artifact, msg)
            elif event_type == "ARTIFACTS_DONE":
                rid = str(msg.get("request_id") or "").strip()
                with self._lock:
                    self._remote_artifact_roots.pop(rid, None)

    def _write_artifact(self, msg: Dict[str, Any]) -> None:
        """Write a single artifact file relayed from the remote vision engine."""
        request_id = str(msg.get("request_id") or "").strip()
        rel_path = str(msg.get("rel_path") or "").strip()
        content_b64 = msg.get("content_b64") or ""
        if not request_id or not rel_path or not content_b64:
            return
        with self._lock:
            output_root = self._remote_artifact_roots.get(request_id)
        if output_root is None:
            log.warning(
                "remote artifact root missing, skipping request_id=%s rel_path=%s",
                request_id,
                rel_path,
            )
            return
        try:
            dest = (output_root / rel_path).resolve()
            # Guard against path traversal
            if not str(dest).startswith(str(output_root.resolve())):
                log.warning("artifact path escapes output_root, skipping: %s", rel_path)
                return
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(base64.b64decode(content_b64))
            log.info(
                "remote_artifact_written request_id=%s rel_path=%s path=%s bytes=%s",
                request_id,
                rel_path,
                dest,
                dest.stat().st_size,
            )
        except Exception as exc:
            log.warning("failed to write remote artifact %s: %s", rel_path, exc)

    def _publish_remote_result(self, msg: Dict[str, Any]) -> None:
        try:
            if self._results_sock is None:
                self._results_sock = self.ctx.socket(zmq.PUSH)
                self._results_sock.setsockopt(zmq.LINGER, 200)
                self._results_sock.connect(self.results_push_endpoint)
            payload = dict(msg)
            payload.setdefault("__topic__", self.results_topic)
            self._results_sock.send_string(
                json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
            )
            if payload.get("event") == "VISION_RESULT":
                log.info(
                    "remote_vision_result_republished request_id=%s frame_id=%s endpoint=%s topic=%s",
                    payload.get("request_id"),
                    payload.get("frame_id"),
                    self.results_push_endpoint,
                    self.results_topic,
                )
                print(
                    "[orchestrator] remote_vision_result_republished "
                    f"request_id={payload.get('request_id')} frame_id={payload.get('frame_id')} "
                    f"endpoint={self.results_push_endpoint} topic={self.results_topic}",
                    flush=True,
                )
        except Exception as exc:
            log.warning("failed to republish remote vision result: %s", exc)

    async def _ws_frame_loop(self) -> None:
        while not self._closed:
            await asyncio.sleep(self.frame_poll_s)
            if self.camera_cache is None:
                continue
            with self._lock:
                cameras = set(self._session_cameras.values())
            for camera_id in cameras:
                await self._send_latest_camera_frame(camera_id)

    async def _send_latest_camera_frame(self, camera_id: str) -> None:
        evt = self.camera_cache.get_latest(camera_id)
        if not evt:
            return
        try:
            sequence_id = int(evt.get("sequence_id") or 0)
        except Exception:
            sequence_id = 0
        if sequence_id <= 0 or sequence_id <= int(self._last_sent_seq.get(camera_id, 0)):
            return
        now = time.monotonic()
        if self.max_frame_fps > 0:
            min_interval = 1.0 / self.max_frame_fps
            if now - float(self._last_sent_at.get(camera_id, 0.0)) < min_interval:
                return
        rgb = _read_shm_image(evt.get("rgb_shm"), evt.get("rgb_shape"), evt.get("rgb_dtype"))
        if rgb is None:
            return
        depth = _read_shm_image(
            evt.get("depth_shm"),
            evt.get("depth_shape"),
            evt.get("depth_dtype"),
        )
        rgb_payload = _array_payload(rgb)
        depth_payload = _array_payload(depth)
        msg = {
            "event": "FRAME_READY_INLINE",
            "camera_id": camera_id,
            "sequence_id": sequence_id,
            "timestamp_ns": evt.get("timestamp_ns"),
            "frame_id": f"{camera_id}:{sequence_id}",
            "calib_version": evt.get("calib_version"),
            "camera_data": evt.get("camera_data"),
            "rgb_shape": rgb_payload["shape"],
            "rgb_dtype": rgb_payload["dtype"],
            "rgb_b64": rgb_payload["b64"],
        }
        if depth_payload:
            msg.update(
                {
                    "depth_shape": depth_payload["shape"],
                    "depth_dtype": depth_payload["dtype"],
                    "depth_b64": depth_payload["b64"],
                }
            )
        await self._send_queue_put(msg)
        self._last_sent_seq[camera_id] = sequence_id
        self._last_sent_at[camera_id] = now

    def start_session(self, payload: Dict[str, Any]) -> None:
        msg = dict(payload)
        msg.setdefault("event", "VISION_START")
        msg.setdefault("__topic__", self.topic)
        if self.transport == "websocket":
            msg["enable_shm_output"] = False
        request_id = str(msg.get("request_id") or "").strip()
        camera_id = str(msg.get("camera_id") or "").strip()
        with self._lock:
            self._send(msg)
            if request_id:
                self._active_sessions.add(request_id)
                if camera_id:
                    self._session_cameras[request_id] = camera_id
                # Track local output_root so artifact files can be written here
                if self.transport == "websocket":
                    output_root = (msg.get("params") or {}).get("output_root")
                    if output_root:
                        self._remote_artifact_roots[request_id] = Path(output_root)

    def stop_session(self, session_id: str) -> None:
        sid = str(session_id or "").strip()
        if not sid:
            return
        msg = {
            "event": "VISION_STOP",
            "request_id": sid,
            "__topic__": self.topic,
        }
        with self._lock:
            self._send(msg)
            self._active_sessions.discard(sid)
            self._session_cameras.pop(sid, None)
            if self.transport != "websocket":
                self._remote_artifact_roots.pop(sid, None)

    def stop_all_sessions(self) -> None:
        with self._lock:
            sessions = list(self._active_sessions)
        for sid in sessions:
            try:
                self.stop_session(sid)
            except Exception:
                pass

    def set_transport(self, transport: str, websocket_url: Optional[str] = None) -> None:
        next_transport = str(transport or "zmq").strip().lower()
        if next_transport not in {"zmq", "websocket"}:
            raise ValueError(f"unsupported_vision_transport:{transport}")
        if websocket_url:
            self.websocket_url = str(websocket_url)
        self.stop_all_sessions()
        self._shutdown_transport()
        self.transport = next_transport
        self._closed = False
        self._last_sent_seq = {}
        self._last_sent_at = {}
        if self.transport == "websocket":
            self._start_ws_loop()
        else:
            self.sock = self.ctx.socket(zmq.PUSH)
            self.sock.setsockopt(zmq.LINGER, 200)
            self.sock.connect(self.push_endpoint)

    def _shutdown_transport(self) -> None:
        self._closed = True
        try:
            if self.sock is not None:
                self.sock.close()
        except Exception:
            pass
        self.sock = None
        try:
            if self._results_sock is not None:
                self._results_sock.close()
        except Exception:
            pass
        self._results_sock = None
        if self._loop is not None:
            try:
                self._loop.call_soon_threadsafe(self._loop.stop)
            except Exception:
                pass
        self._loop = None
        self._loop_thread = None
        self._send_queue = None
        self._ws = None

    def close(self) -> None:
        try:
            self.stop_all_sessions()
        except Exception:
            pass
        self._shutdown_transport()
