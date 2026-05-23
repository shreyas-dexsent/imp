"""Application entrypoint for `vgr-vision-engine-2.5d`."""

import argparse
import asyncio
import base64
import contextlib
import json
import logging
import os
import shutil
import tempfile
import time
from pathlib import Path
from queue import Empty, Queue
from threading import Thread
from typing import Any, Dict

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field
from vision_engine.core.engine import VisionEngine
from vision_engine.modules.megapose_bin_picking.module import MegaPoseBinPickingModule
from vision_engine.modules.ppf_icp_bin_picking.module import PpfIcpBinPickingModule

app = FastAPI(title="DexSent VGR Vision Engine")
log = logging.getLogger("vision_engine.app")

engine: VisionEngine | None = None
engine_thread: Thread | None = None
CONFIG_ENV = "VISION_ENGINE_CONFIG"
DEFAULT_CONFIG = "configs/engine.local.json"


class MegaPosePrewarmRequest(BaseModel):
    params: Dict[str, Any] = Field(default_factory=dict)


class WebSocketPublisher:
    def __init__(self) -> None:
        self.queue: Queue[Dict[str, Any]] = Queue()

    def publish(self, msg: Dict[str, Any]) -> None:
        self.queue.put(dict(msg))


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, bytes):
        return base64.b64encode(value).decode("ascii")
    raise TypeError(f"{value.__class__.__name__} is not JSON serializable")


async def _send_json_safe(ws: WebSocket, msg: Dict[str, Any]) -> None:
    raw = json.dumps(
        msg,
        default=_json_default,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    await ws.send_text(raw)


def _decode_array(data_b64: Any, shape: Any, dtype: Any) -> np.ndarray | None:
    if not data_b64 or not shape:
        return None
    dtype_np = np.dtype(dtype or "uint8")
    raw = base64.b64decode(str(data_b64))
    arr = np.frombuffer(raw, dtype=dtype_np)
    return arr.reshape(tuple(shape)).copy()


def _resolve_config_path() -> Path:
    cfg_path = os.getenv(CONFIG_ENV, DEFAULT_CONFIG)
    return Path(cfg_path).resolve()


@app.on_event("startup")
def startup():
    global engine, engine_thread

    config_path = _resolve_config_path()
    with config_path.open() as f:
        cfg = json.load(f)

    engine = VisionEngine(cfg)
    engine_thread = Thread(target=engine.run, daemon=True)
    engine_thread.start()

    megapose_def = (
        engine.module_defs.get("megapose_bin_picking", {}) if engine else {}
    )
    megapose_defaults = (
        megapose_def.get("params", {}) if isinstance(megapose_def, dict) else {}
    )
    try:
        startup_prewarm = MegaPoseBinPickingModule.startup_prewarm(megapose_defaults)
        status = str(startup_prewarm.get("status") or "").strip().lower()
        if status == "ok":
            print(
                "[fastapi] megapose startup prewarm ready "
                f"label={startup_prewarm.get('label')} device={startup_prewarm.get('device')}"
            )
        elif status == "skipped":
            print(
                "[fastapi] megapose startup prewarm skipped "
                f"reason={startup_prewarm.get('reason')}"
            )
    except Exception as exc:
        print(f"[fastapi] megapose startup prewarm failed: {exc}")

    print("[fastapi] vision engine thread started")


@app.on_event("shutdown")
def shutdown():
    global engine, engine_thread
    if engine:
        engine.stop(
            wait_for_requests=False,
            request_join_timeout=0.0,
            close_request_shm=False,
        )
        if engine_thread and engine_thread.is_alive():
            engine_thread.join(timeout=1.0)
        engine_thread = None
        engine = None
        print("[fastapi] vision engine stopped")


@app.get("/health")
def health():
    return {"status": "alive"}


@app.get("/ready")
def ready():
    camera_details: Dict[str, Any] = {}
    if engine:
        now = time.time()
        camera_ids = set(engine.camera_states.keys()) | set(engine.last_frames.keys())
        for camera_id in sorted(camera_ids):
            state = engine.camera_states.get(camera_id, {}) or {}
            last_seen = engine.last_frame_ts.get(camera_id)
            camera_details[camera_id] = {
                "fps_estimate": state.get("fps_estimate"),
                "last_sequence_id": state.get("last_seq"),
                "last_frame_age_s": None
                if last_seen is None
                else round(max(0.0, now - float(last_seen)), 3),
                "has_cached_frame": camera_id in engine.last_frames,
                "active_requests": [
                    req.request_id for req in engine._get_requests_for_camera(camera_id)
                ],
            }
    return {
        "engine_running": engine.running if engine else False,
        "known_cameras": list(engine.camera_states.keys()) if engine else [],
        "cameras": camera_details,
    }


@app.websocket("/ws")
async def websocket_transport(ws: WebSocket):
    await ws.accept()
    if engine is None:
        await ws.send_json({"event": "ERROR", "error": "engine_not_ready"})
        await ws.close()
        return

    publisher = WebSocketPublisher()
    closed = asyncio.Event()
    session_ids: set[str] = set()
    # Maps request_id -> (tmp_dir, original_output_root) for artifact relay + path rewriting
    artifact_tmpdirs: dict[str, tuple[str, str]] = {}

    async def _send_artifacts(request_id: str, tmp_dir: Path) -> None:
        """Send all files in tmp_dir to the client then clean up."""
        try:
            for fpath in sorted(tmp_dir.rglob("*")):
                if not fpath.is_file():
                    continue
                rel = fpath.relative_to(tmp_dir).as_posix()
                content = await asyncio.to_thread(fpath.read_bytes)
                await _send_json_safe(
                    ws,
                    {
                        "event": "ARTIFACT_FILE",
                        "request_id": request_id,
                        "rel_path": rel,
                        "content_b64": base64.b64encode(content).decode(),
                    }
                )
        finally:
            await _send_json_safe(ws, {"event": "ARTIFACTS_DONE", "request_id": request_id})
            await asyncio.to_thread(shutil.rmtree, str(tmp_dir), True)

    async def pump_results() -> None:
        while not closed.is_set():
            try:
                msg = await asyncio.to_thread(publisher.queue.get, True, 0.2)
            except Empty:
                continue
            if msg.get("event") == "VISION_RESULT":
                result = msg.get("result") if isinstance(msg.get("result"), dict) else {}
                matches = result.get("matches") if isinstance(result, dict) else None
                log.info(
                    "websocket_queue_vision_result request_id=%s frame_id=%s matches=%s",
                    msg.get("request_id"),
                    msg.get("frame_id"),
                    len(matches) if isinstance(matches, list) else None,
                )
                print(
                    "[fastapi] websocket_queue_vision_result "
                    f"request_id={msg.get('request_id')} frame_id={msg.get('frame_id')} "
                    f"matches={len(matches) if isinstance(matches, list) else None}",
                    flush=True,
                )
            # Before forwarding VISION_RESULT, rewrite server tmp paths → client's original output_root
            if msg.get("event") == "VISION_RESULT":
                rid = str(msg.get("request_id") or "").strip()
                if rid and rid in artifact_tmpdirs:
                    tmp, original_root = artifact_tmpdirs[rid]
                    print(
                        "[fastapi] websocket_artifact_paths_pending "
                        f"request_id={rid} tmp_dir={tmp} original_root={original_root}",
                        flush=True,
                    )
            try:
                if msg.get("event") == "VISION_RESULT":
                    result = msg.get("result") if isinstance(msg.get("result"), dict) else {}
                    matches = result.get("matches") if isinstance(result, dict) else None
                    t0 = time.perf_counter()
                    print(
                        "[fastapi] websocket_serializing_vision_result "
                        f"request_id={msg.get('request_id')} frame_id={msg.get('frame_id')} "
                        f"matches={len(matches) if isinstance(matches, list) else None}",
                        flush=True,
                    )
                    raw = json.dumps(
                        msg,
                        default=_json_default,
                        separators=(",", ":"),
                        ensure_ascii=False,
                        allow_nan=False,
                    )
                    rid = str(msg.get("request_id") or "").strip()
                    if rid and rid in artifact_tmpdirs:
                        tmp, original_root = artifact_tmpdirs[rid]
                        if tmp and original_root:
                            escaped_original_root = json.dumps(
                                original_root,
                                ensure_ascii=False,
                            )[1:-1]
                            raw = raw.replace(tmp, escaped_original_root)
                            print(
                                "[fastapi] websocket_result_paths_rewritten "
                                f"request_id={rid} tmp_dir={tmp} original_root={original_root}",
                                flush=True,
                            )
                    serialize_dt = time.perf_counter() - t0
                    print(
                        "[fastapi] websocket_sending_vision_result "
                        f"request_id={msg.get('request_id')} frame_id={msg.get('frame_id')} "
                        f"matches={len(matches) if isinstance(matches, list) else None} "
                        f"bytes={len(raw)} serialize_dt={serialize_dt:.3f}s",
                        flush=True,
                    )
                    await asyncio.wait_for(ws.send_text(raw), timeout=5.0)
                    log.info(
                        "websocket_sent_vision_result request_id=%s frame_id=%s matches=%s bytes=%s",
                        msg.get("request_id"),
                        msg.get("frame_id"),
                        len(matches) if isinstance(matches, list) else None,
                        len(raw),
                    )
                    print(
                        "[fastapi] websocket_sent_vision_result "
                        f"request_id={msg.get('request_id')} frame_id={msg.get('frame_id')} "
                        f"matches={len(matches) if isinstance(matches, list) else None} bytes={len(raw)}",
                        flush=True,
                    )
                else:
                    await _send_json_safe(ws, msg)
            except Exception as exc:
                rid = str(msg.get("request_id") or "").strip()
                event = str(msg.get("event") or "")
                log.exception("websocket_send_failed event=%s request_id=%s", event, rid)
                print(
                    "[fastapi] websocket_send_failed "
                    f"event={event} request_id={rid} error={exc.__class__.__name__}: {exc}",
                    flush=True,
                )
                if event == "VISION_RESULT" and rid:
                    await _send_json_safe(
                        ws,
                        {
                            "event": "VISION_PROCESSING_ERROR",
                            "request_id": rid,
                            "error": f"websocket_send_failed:{exc.__class__.__name__}: {exc}",
                        },
                    )
                continue
            # After forwarding the final result, relay any saved artifacts back
            if msg.get("event") == "VISION_RESULT":
                rid = str(msg.get("request_id") or "").strip()
                if rid and rid in artifact_tmpdirs:
                    tmp, _ = artifact_tmpdirs.pop(rid)
                    print(
                        "[fastapi] websocket_artifact_relay_start "
                        f"request_id={rid} tmp_dir={tmp}",
                        flush=True,
                    )
                    await _send_artifacts(rid, Path(tmp))
                    print(
                        "[fastapi] websocket_artifact_relay_done "
                        f"request_id={rid} tmp_dir={tmp}",
                        flush=True,
                    )

    pump_task = asyncio.create_task(pump_results())
    try:
        await _send_json_safe(ws, {"event": "WS_READY", "status": "ok"})
        while True:
            msg = await ws.receive_json()
            event_type = msg.get("event")
            if event_type == "VISION_START":
                payload = dict(msg)
                payload.setdefault("enable_shm_output", False)
                request_id = str(payload.get("request_id") or "").strip()
                if request_id:
                    session_ids.add(request_id)
                # If the client asked the engine to save artifacts, redirect the
                # output_root to a temp dir so we can relay files back over WS.
                params = dict(payload.get("params") or {})
                if request_id and params.get("save_outputs"):
                    original_root = str(params.get("output_root") or "")
                    tmp = tempfile.mkdtemp(prefix=f"ve_artifacts_{request_id}_")
                    artifact_tmpdirs[request_id] = (tmp, original_root)
                    params["output_root"] = tmp
                    payload["params"] = params
                engine._handle_start_trigger(payload, publisher=publisher)
            elif event_type == "VISION_STOP":
                request_id = str(msg.get("request_id") or "").strip()
                if request_id:
                    session_ids.discard(request_id)
                    entry = artifact_tmpdirs.pop(request_id, None)
                    if entry:
                        await asyncio.to_thread(shutil.rmtree, entry[0], True)
                engine._handle_stop_trigger(msg, publisher=publisher)
            elif event_type == "FRAME_READY_INLINE":
                try:
                    rgb = _decode_array(
                        msg.get("rgb_b64"),
                        msg.get("rgb_shape"),
                        msg.get("rgb_dtype"),
                    )
                    depth = _decode_array(
                        msg.get("depth_b64"),
                        msg.get("depth_shape"),
                        msg.get("depth_dtype"),
                    )
                    if rgb is None:
                        continue
                    engine.handle_inline_frame(
                        {
                            "camera_id": msg.get("camera_id"),
                            "sequence_id": msg.get("sequence_id"),
                            "timestamp_ns": msg.get("timestamp_ns"),
                            "frame_id": msg.get("frame_id"),
                            "rgb": rgb,
                            "depth": depth,
                            "calib_version": msg.get("calib_version"),
                            "camera_data": msg.get("camera_data"),
                        }
                    )
                except Exception as exc:
                    publisher.publish(
                        {
                            "event": "FRAME_REJECTED",
                            "reason": "decode_failed",
                            "error": str(exc),
                        }
                    )
            elif event_type == "PING":
                await ws.send_json({"event": "PONG", "timestamp_ns": time.time_ns()})
    except WebSocketDisconnect:
        pass
    finally:
        for request_id in list(session_ids):
            with contextlib.suppress(Exception):
                engine._handle_stop_trigger({"request_id": request_id}, publisher=publisher)
        for tmp, _ in artifact_tmpdirs.values():
            with contextlib.suppress(Exception):
                shutil.rmtree(tmp, True)
        artifact_tmpdirs.clear()
        closed.set()
        pump_task.cancel()
        with contextlib.suppress(BaseException):
            await pump_task


@app.post("/modules/megapose_bin_picking/prewarm")
def megapose_prewarm(req: MegaPosePrewarmRequest):
    try:
        return MegaPoseBinPickingModule.prewarm(req.params or {}, request_id="http-prewarm")
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"{exc.__class__.__name__}: {exc}",
        ) from exc


@app.post("/modules/ppf_icp_bin_picking/prewarm")
def ppf_icp_prewarm(req: MegaPosePrewarmRequest):
    try:
        return PpfIcpBinPickingModule.prewarm(
            req.params or {},
            request_id="http-prewarm",
        )
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"{exc.__class__.__name__}: {exc}",
        ) from exc


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config", default=DEFAULT_CONFIG, help="Path to engine config JSON"
    )
    ap.add_argument("--host", default="127.0.0.1", help="Bind host")
    ap.add_argument("--port", type=int, default=8000, help="Bind port")
    args = ap.parse_args()

    os.environ[CONFIG_ENV] = args.config
    uvicorn.run("app:app", host=args.host, port=args.port)
