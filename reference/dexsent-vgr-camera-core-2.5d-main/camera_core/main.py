"""Implementation for `camera_core.main`."""

import argparse
import logging
import queue
import threading
import time

import numpy as np
import uvicorn
import yaml
import zmq
from camera_core.calibration.control import run_control_server
from camera_core.calibration.manager import CalibrationManager
from camera_core.config import load_config
from camera_core.drivers.factory import create_camera_driver
from camera_core.health.server import create_health_app
from camera_core.ipc.event_bus import start_event_bus
from camera_core.ipc.zmq_pub import ZmqPublisher
from camera_core.logger import get_logger
from camera_core.shm.header import FLAG_VALID, pack_header
from camera_core.shm.triple_buffer import TripleBuffer

log = logging.getLogger("camera-core")


def run_health_server(app, host, port, log_level="warning"):
    uvicorn.run(app, host=host, port=port, log_level=log_level)


class CameraPipeline(threading.Thread):
    """
    One pipeline = one camera config
    Runs independently in its own thread:
      - its own driver
      - its own shared memory buffers
      - its own ZMQ publisher socket (connect)
      - its own health server
      - its own calibration manager
    """

    def __init__(self, cfg, push_addr: str, log, stop_event: threading.Event):
        super().__init__(daemon=True)
        self.cfg = cfg
        self.push_addr = push_addr
        self.log = log
        self.stop_event = stop_event

        self.seq = 0

        # health state (per camera)
        self.state = {
            "camera_id": cfg.camera.camera_id,
            "alive": True,
            "fps_target": cfg.camera.fps,
            "fps_recent": 0.0,
            "capture_ms_recent": 0.0,
            "shm_publish_ms_recent": 0.0,
            "frames_ok": 0,
            "frames_err": 0,
            "last_sequence_id": 0,
            "last_timestamp_ns": 0,
            "last_shm": "",
        }

    def run(self):
        cam = self.cfg.camera
        pub = None
        tb = None
        depth_tb = None
        driver = None

        # ---------- Calibration (per camera) ----------
        calib = CalibrationManager()
        calib.load()

        try:
            # ---------- ZMQ publisher (per camera thread) ----------
            pub = ZmqPublisher(self.push_addr)
            self.log.info(
                f"[{cam.camera_id}] ZMQ PUSH connected to event bus: {self.push_addr}"
            )

            time.sleep(0.5)  # let connect settle

            pub.publish(
                {
                    "event": "CAMERA_STARTED",
                    "camera_id": cam.camera_id,
                    "timestamp_ns": time.time_ns(),
                }
            )
            self.log.info(f"[{cam.camera_id}] CAMERA_STARTED event published")

            # ---------- Shared Memory (RGB) ----------
            rgb_cfg = self.cfg.shared_memory.rgb
            shape = (rgb_cfg.height, rgb_cfg.width, rgb_cfg.channels)
            dtype = np.uint8

            tb = TripleBuffer(
                name_prefix=rgb_cfg.name_prefix,
                suffixes=rgb_cfg.triple_buffers,
                shape=shape,
                dtype=dtype,
                create=True,
            )
            self.log.info(f"[{cam.camera_id}] Shared memory created: {tb.names}")

            # ---------- Shared Memory (Depth, optional) ----------
            depth_cfg = getattr(self.cfg.shared_memory, "depth", None)
            if depth_cfg:
                depth_shape = (depth_cfg.height, depth_cfg.width)
                depth_dtype = np.dtype(depth_cfg.dtype)
                depth_tb = TripleBuffer(
                    name_prefix=depth_cfg.name_prefix,
                    suffixes=depth_cfg.triple_buffers,
                    shape=depth_shape,
                    dtype=depth_dtype,
                    create=True,
                )
                self.log.info(
                    f"[{cam.camera_id}] Depth shared memory created: {depth_tb.names}"
                )

            # ---------- Driver ----------
            driver = create_camera_driver(cam.type, self.log)
            driver.initialize(cam.sdk.model_dump())
            driver.start()
        except Exception as exc:
            self.state["alive"] = False
            self.log.exception(f"[{cam.camera_id}] Pipeline startup failed: {exc}")
            if driver is not None:
                try:
                    driver.stop()
                except Exception:
                    pass
            if tb is not None:
                try:
                    tb.close(unlink=True)
                except Exception:
                    pass
            if depth_tb is not None:
                try:
                    depth_tb.close(unlink=True)
                except Exception:
                    pass
            if pub is not None:
                try:
                    pub.close()
                except Exception:
                    pass
            return

        # ---------- Health server (per camera) ----------
        def get_status():
            return dict(self.state)

        health_app = create_health_app(get_status)
        threading.Thread(
            target=run_health_server,
            args=(health_app, self.cfg.health.host, self.cfg.health.port),
            daemon=True,
        ).start()
        self.log.info(
            f"[{cam.camera_id}] Health server started: {self.cfg.health.host}:{self.cfg.health.port}"
        )

        # ---------- Capture loop ----------
        period = 1.0 / max(1, cam.fps)
        use_driver_timing = str(cam.type).startswith("realsense_")
        self.log.info(f"[{cam.camera_id}] Starting capture loop fps={cam.fps}")
        fps_report_interval_s = 5.0
        fps_report_t0 = time.perf_counter()
        fps_report_frames_ok = 0
        fps_report_frames_err = 0
        timing_sums_ms = {
            "capture_total": 0.0,
            "wait_for_frames": 0.0,
            "align": 0.0,
            "post_processing": 0.0,
            "frame_to_numpy": 0.0,
            "depth_to_meters": 0.0,
            "calib_apply": 0.0,
            "shm_publish": 0.0,
        }
        timing_sample_count = 0

        try:
            while not self.stop_event.is_set():
                t0 = time.perf_counter()

                try:
                    frame = driver.capture()
                    frame_timings_ms = frame.get("timings_ms") or {}
                    calib_t0 = time.perf_counter()
                    frame = calib.apply(frame)
                    calib_apply_ms = (time.perf_counter() - calib_t0) * 1000.0

                    rgb = frame["rgb"]
                    depth = frame.get("depth")
                    ts_ns = int(frame["timestamp_ns"])
                    self.seq += 1
                    shm_publish_t0 = time.perf_counter()

                    wb = tb.get_write_buffer()

                    if rgb.shape != wb.img.shape:
                        raise RuntimeError(
                            f"RGB shape mismatch: got {rgb.shape}, expected {wb.img.shape}"
                        )

                    # Write pixels first
                    wb.img[:] = rgb

                    # Commit header last
                    hdr = pack_header(ts_ns, self.seq, calib.version(), FLAG_VALID)
                    wb.write_header(hdr)

                    # Rotate and publish which buffer is safe
                    tb.rotate()
                    ready_name = tb.get_last_complete_name()

                    event = {
                        "event": "FRAME_READY",
                        "camera_id": cam.camera_id,
                        "sequence_id": self.seq,
                        "timestamp_ns": ts_ns,
                        "calib_version": calib.version(),
                        "status_flags": int(FLAG_VALID),
                        "rgb_shm": ready_name,
                        "rgb_shape": list(wb.img.shape),
                        "rgb_dtype": str(wb.img.dtype),
                    }
                    camera_data = frame.get("camera_data")
                    if isinstance(camera_data, dict) and camera_data:
                        event["camera_data"] = camera_data

                    if depth_tb is not None and depth is not None:
                        depth_wb = depth_tb.get_write_buffer()
                        if depth.shape != depth_wb.img.shape:
                            raise RuntimeError(
                                f"Depth shape mismatch: got {depth.shape}, expected {depth_wb.img.shape}"
                            )
                        depth_wb.img[:] = depth
                        depth_hdr = pack_header(
                            ts_ns, self.seq, calib.version(), FLAG_VALID
                        )
                        depth_wb.write_header(depth_hdr)
                        depth_tb.rotate()
                        depth_ready = depth_tb.get_last_complete_name()
                        event.update(
                            {
                                "depth_shm": depth_ready,
                                "depth_shape": list(depth_wb.img.shape),
                                "depth_dtype": str(depth_wb.img.dtype),
                            }
                        )
                    pub.publish(event)
                    shm_publish_ms = (time.perf_counter() - shm_publish_t0) * 1000.0

                    # Update health state
                    self.state["frames_ok"] += 1
                    self.state["last_sequence_id"] = self.seq
                    self.state["last_timestamp_ns"] = ts_ns
                    self.state["last_shm"] = ready_name
                    timing_sums_ms["capture_total"] += float(
                        frame_timings_ms.get("capture_total", 0.0)
                    )
                    timing_sums_ms["wait_for_frames"] += float(
                        frame_timings_ms.get("wait_for_frames", 0.0)
                    )
                    timing_sums_ms["align"] += float(frame_timings_ms.get("align", 0.0))
                    timing_sums_ms["post_processing"] += float(
                        frame_timings_ms.get("post_processing", 0.0)
                    )
                    timing_sums_ms["frame_to_numpy"] += float(
                        frame_timings_ms.get("frame_to_numpy", 0.0)
                    )
                    timing_sums_ms["depth_to_meters"] += float(
                        frame_timings_ms.get("depth_to_meters", 0.0)
                    )
                    timing_sums_ms["calib_apply"] += calib_apply_ms
                    timing_sums_ms["shm_publish"] += shm_publish_ms
                    timing_sample_count += 1

                except Exception as e:
                    self.state["frames_err"] += 1
                    self.log.exception(f"[{cam.camera_id}] Capture/write failed: {e}")

                    pub.publish(
                        {
                            "event": "CAMERA_ERROR",
                            "camera_id": cam.camera_id,
                            "sequence_id": self.seq,
                            "timestamp_ns": time.time_ns(),
                            "error": str(e),
                        }
                    )

                report_now = time.perf_counter()
                report_dt = report_now - fps_report_t0
                if report_dt >= fps_report_interval_s:
                    frames_ok_now = int(self.state["frames_ok"])
                    frames_err_now = int(self.state["frames_err"])
                    delta_ok = frames_ok_now - fps_report_frames_ok
                    delta_err = frames_err_now - fps_report_frames_err
                    fps_recent = delta_ok / report_dt if report_dt > 0 else 0.0
                    self.state["fps_recent"] = fps_recent
                    self.log.info(
                        f"[{cam.camera_id}] SHM publish FPS={fps_recent:.2f} "
                        f"over {report_dt:.1f}s ok={delta_ok} err={delta_err} "
                        f"target={cam.fps}"
                    )
                    if timing_sample_count > 0:
                        capture_ms_recent = (
                            timing_sums_ms["capture_total"] / timing_sample_count
                        )
                        shm_publish_ms_recent = (
                            timing_sums_ms["shm_publish"] / timing_sample_count
                        )
                        self.state["capture_ms_recent"] = capture_ms_recent
                        self.state["shm_publish_ms_recent"] = shm_publish_ms_recent
                        self.log.info(
                            f"[{cam.camera_id}] Avg timing ms "
                            f"wait={timing_sums_ms['wait_for_frames'] / timing_sample_count:.1f} "
                            f"align={timing_sums_ms['align'] / timing_sample_count:.1f} "
                            f"post={timing_sums_ms['post_processing'] / timing_sample_count:.1f} "
                            f"numpy={timing_sums_ms['frame_to_numpy'] / timing_sample_count:.1f} "
                            f"depth_m={timing_sums_ms['depth_to_meters'] / timing_sample_count:.1f} "
                            f"calib={timing_sums_ms['calib_apply'] / timing_sample_count:.1f} "
                            f"shm_pub={shm_publish_ms_recent:.1f} "
                            f"capture_total={capture_ms_recent:.1f}"
                        )
                    else:
                        self.state["capture_ms_recent"] = 0.0
                        self.state["shm_publish_ms_recent"] = 0.0
                    fps_report_t0 = report_now
                    fps_report_frames_ok = frames_ok_now
                    fps_report_frames_err = frames_err_now
                    for key in timing_sums_ms:
                        timing_sums_ms[key] = 0.0
                    timing_sample_count = 0

                # RealSense already blocks on sensor frame cadence. Adding an
                # extra sleep on Windows drifts the effective publish rate
                # below the configured FPS, so only pace non-RealSense drivers.
                if not use_driver_timing:
                    dt = time.perf_counter() - t0
                    sleep_s = period - dt
                    if sleep_s > 0:
                        time.sleep(sleep_s)

        finally:
            self.state["alive"] = False
            try:
                driver.stop()
            except Exception:
                pass
            try:
                tb.close(unlink=True)
            except Exception:
                pass
            if depth_tb is not None:
                try:
                    depth_tb.close(unlink=True)
                except Exception:
                    pass
            try:
                pub.close()
            except Exception:
                pass
            self.log.info(f"[{cam.camera_id}] Pipeline stopped")


# --------------------------------------------------
# ZMQ EVENT PUBLISHER THREAD (SINGLE PUB SOCKET)
# --------------------------------------------------
class EventPublisher(threading.Thread):
    def __init__(self, bind_addr: str, event_queue: queue.Queue):
        super().__init__(daemon=True)
        self.bind_addr = bind_addr
        self.event_queue = event_queue

        self.ctx = zmq.Context.instance()
        self.sock = self.ctx.socket(zmq.PUB)
        self.sock.bind(self.bind_addr)

        log.info(f"[EVENT BUS] ZMQ PUB bound at {self.bind_addr}")

    def run(self):
        # Allow subscribers to connect (slow joiner protection)
        time.sleep(1.0)

        log.info("[EVENT BUS] Publisher thread running")

        while True:
            event = self.event_queue.get()  # blocks
            topic = event.pop("__topic__", "camera")
            payload = yaml.safe_dump(event, default_flow_style=True)

            self.sock.send_string(f"{topic} {payload}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config",
        action="append",
        required=True,
        help="Camera config YAML (repeat for multiple cameras)",
    )
    args = ap.parse_args()

    configs = [load_config(p) for p in args.config]
    base_cfg = configs[0]
    log = get_logger("camera-core", base_cfg.runtime.log_level)

    log.info("Starting Camera Core (multi-camera) with PUSH-PULL-PUB architecture")
    log.info(f"Configs: {args.config}")

    # --------------------------------------------------
    # Bind ZMQ event bus ONCE (in main)
    # --------------------------------------------------
    pub_addr = base_cfg.ipc.zmq_pub  # e.g., tcp://127.0.0.1:5555 (for subscribers)
    pull_addr = "tcp://127.0.0.1:5556"  # Internal address for cameras to push to
    topic = base_cfg.ipc.topic

    t_bus = threading.Thread(
        target=start_event_bus, args=(pub_addr, pull_addr, topic, log), daemon=True
    )
    t_bus.start()

    # Give bind time to settle
    time.sleep(0.5)

    # --------------------------------------------------
    # Calibration control server (starts calibration server on demand)
    # --------------------------------------------------
    calib_cfg = getattr(base_cfg, "calibration", None)
    if calib_cfg and getattr(calib_cfg, "enable", True):
        threading.Thread(
            target=run_control_server,
            kwargs=dict(
                host=calib_cfg.control_host,
                port=calib_cfg.control_port,
                server_host=calib_cfg.server_host,
                server_port=calib_cfg.server_port,
                pub_endpoint=base_cfg.ipc.zmq_pub,
                topic=base_cfg.ipc.topic,
                log_level=base_cfg.runtime.log_level,
                robot_command_endpoint=base_cfg.robot.command_endpoint,
                robot_timeout_ms=base_cfg.robot.timeout_ms,
                cors_origins=list(calib_cfg.cors_origins),
                # Fusion pushes FRAME_READY events into the broker's PULL
                # socket (same endpoint camera pipelines use), NOT the PUB
                # endpoint that subscribers read from.
                push_endpoint=pull_addr,
            ),
            daemon=True,
        ).start()
        log.info(
            "[CALIBRATION] Control server started at %s:%s",
            calib_cfg.control_host,
            calib_cfg.control_port,
        )

    # --------------------------------------------------
    # Start all camera pipelines
    # --------------------------------------------------
    stop_event = threading.Event()
    pipelines = []

    for cfg in configs:
        pipe = CameraPipeline(cfg, push_addr=pull_addr, log=log, stop_event=stop_event)
        pipe.start()
        pipelines.append(pipe)

    # --------------------------------------------------
    # Supervisor loop
    # --------------------------------------------------
    try:
        while True:
            # optional: detect dead pipelines
            for p in pipelines:
                if not p.is_alive():
                    log.error(
                        f"Pipeline thread died for camera_id={p.cfg.camera.camera_id}"
                    )
            time.sleep(1)
    except KeyboardInterrupt:
        log.info("Stopping all camera pipelines (KeyboardInterrupt)")
        stop_event.set()
        time.sleep(0.5)


if __name__ == "__main__":
    main()
