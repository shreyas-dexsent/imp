#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Optional
from urllib.error import URLError
from urllib.parse import urlencode
from urllib.request import urlopen

import cv2
import numpy as np
from ultralytics import YOLO


# ---------------------------------------------------------------------------
# Matplotlib fallback viewer
# ---------------------------------------------------------------------------

class MatplotlibViewer:
    def __init__(self, window_name: str) -> None:
        import matplotlib.pyplot as plt

        self.plt = plt
        self.window_name = window_name
        self.figure, self.axis = plt.subplots(num=window_name)
        self.axis.axis("off")
        self.image_artist = None
        self.closed = False
        self.figure.canvas.mpl_connect("close_event", self._handle_close)
        plt.show(block=False)

    def _handle_close(self, _event: Any) -> None:
        self.closed = True

    def show(self, bgr_frame: np.ndarray) -> None:
        rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        if self.image_artist is None:
            self.image_artist = self.axis.imshow(rgb_frame)
        else:
            self.image_artist.set_data(rgb_frame)
        self.figure.canvas.draw_idle()
        self.plt.pause(0.001)

    def is_closed(self) -> bool:
        return self.closed or not self.plt.fignum_exists(self.figure.number)

    def close(self) -> None:
        if not self.is_closed():
            self.plt.close(self.figure)


# ---------------------------------------------------------------------------
# Direct pyrealsense2 D405 capture
# ---------------------------------------------------------------------------

class RealsenseCapture:
    """Minimal RGB-only capture from a RealSense camera via pyrealsense2."""

    def __init__(
        self,
        serial: str = "",
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
        initial_gain: Optional[float] = None,
    ) -> None:
        import pyrealsense2 as rs  # type: ignore

        self._rs = rs
        self._pipeline = rs.pipeline()
        cfg = rs.config()
        if serial:
            cfg.enable_device(serial)
        cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        self._profile = self._pipeline.start(cfg)
        self._color_sensor = self._get_color_sensor()
        if initial_gain is not None:
            self.set_gain(initial_gain)

    def _get_color_sensor(self) -> Any:
        rs = self._rs
        device = self._profile.get_device()
        for sensor in device.query_sensors():
            try:
                sensor.get_stream_profiles()
                name = sensor.get_info(rs.camera_info.name).lower()
                if "rgb" in name or "color" in name or "stereo" in name:
                    return sensor
            except Exception:
                pass
        # fallback: first sensor
        sensors = device.query_sensors()
        return sensors[0] if sensors else None

    def set_gain(self, value: float) -> None:
        if self._color_sensor is None:
            return
        rs = self._rs
        try:
            self._color_sensor.set_option(rs.option.enable_auto_exposure, 0)
        except Exception:
            pass
        try:
            self._color_sensor.set_option(rs.option.gain, float(value))
        except Exception as exc:
            print(f"[yolo-live] gain set failed: {exc}", flush=True)

    def get_gain(self) -> float:
        if self._color_sensor is None:
            return 0.0
        rs = self._rs
        try:
            return float(self._color_sensor.get_option(rs.option.gain))
        except Exception:
            return 0.0

    def read_bgr(self) -> np.ndarray:
        frames = self._pipeline.wait_for_frames(timeout_ms=5000)
        color_frame = frames.get_color_frame()
        if not color_frame:
            raise RuntimeError("no_color_frame")
        return np.asarray(color_frame.get_data())

    def stop(self) -> None:
        try:
            self._pipeline.stop()
        except Exception:
            pass


def load_d405_config(config_path: str) -> dict:
    """Load a camera YAML config and return the sdk section."""
    import yaml  # type: ignore

    with open(config_path) as fh:
        raw = yaml.safe_load(fh)
    camera = raw.get("camera", {})
    sdk = camera.get("sdk", {})
    return {
        "camera_id": camera.get("camera_id", "cam_rs_d405"),
        "serial_number": sdk.get("serial_number", ""),
        "color_width": sdk.get("color_width") or sdk.get("width", 1280),
        "color_height": sdk.get("color_height") or sdk.get("height", 720),
        "color_fps": sdk.get("color_fps") or sdk.get("fps", 30),
        "depth_gain": (sdk.get("options") or {}).get("depth", {}).get("gain", 60),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Run a YOLO segmentation model on live camera frames."
    )
    ap.add_argument(
        "--source",
        choices=("realsense", "http"),
        default="realsense",
        help="Camera source: 'realsense' for direct pyrealsense2, 'http' for camera-core HTTP.",
    )
    ap.add_argument(
        "--camera-config",
        default="/home/imp/imp/dexsent-vgr-camera-core-2.5d/config/cam_realsense_d405.yaml",
        help="Path to camera YAML config (used for realsense source).",
    )
    ap.add_argument(
        "--serial",
        default="",
        help="RealSense serial number override (overrides --camera-config serial).",
    )
    ap.add_argument(
        "--camera-url",
        default="http://127.0.0.1:8100/camera/frame",
        help="HTTP endpoint for frames (only used with --source http).",
    )
    ap.add_argument(
        "--camera-id",
        default="cam_rs_d405",
        help="Camera ID for HTTP endpoint (only used with --source http).",
    )
    ap.add_argument(
        "--object-folder",
        default="/home/imp/imp/data/stations/station-1/assets/asset-1/objects/barel2",
        help="Object folder containing the YOLO segmentation model.",
    )
    ap.add_argument(
        "--model-path",
        default="",
        help="Optional direct path to a YOLO .pt model. Overrides --object-folder.",
    )
    ap.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Initial YOLO confidence threshold (live slider in OpenCV mode).",
    )
    ap.add_argument(
        "--gain",
        type=float,
        default=-1.0,
        help="Initial camera gain (0-128). -1 = use value from config.",
    )
    ap.add_argument(
        "--imgsz",
        type=int,
        default=1024,
        help="YOLO inference image size (matches core pipeline default of 1024).",
    )
    ap.add_argument(
        "--device",
        default="auto",
        help="YOLO device: auto, cpu, cuda, 0.",
    )
    ap.add_argument(
        "--quality",
        type=int,
        default=80,
        help="Requested JPEG quality from HTTP camera endpoint.",
    )
    ap.add_argument(
        "--poll-interval",
        type=float,
        default=0.05,
        help="Delay between frame polls in seconds (http source only).",
    )
    ap.add_argument(
        "--viewer",
        choices=("auto", "opencv", "matplotlib"),
        default="auto",
        help="Viewer backend for live output.",
    )
    return ap.parse_args()


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def resolve_model_path(args: argparse.Namespace) -> Path:
    if str(args.model_path or "").strip():
        return Path(args.model_path).expanduser().resolve()
    folder = Path(args.object_folder).expanduser().resolve()
    for name in ("object.pt", f"{folder.name}.pt"):
        candidate = folder / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No YOLO model found in {folder}")


# ---------------------------------------------------------------------------
# HTTP source helpers
# ---------------------------------------------------------------------------

def fetch_frame_jpeg(base_url: str, camera_id: str, quality: int) -> bytes:
    query = urlencode(
        {
            "camera_id": camera_id,
            "quality": int(quality),
            "t": int(time.time() * 1000),
        }
    )
    url = f"{base_url}?{query}"
    with urlopen(url, timeout=5.0) as resp:
        return resp.read()


def decode_bgr(jpeg_bytes: bytes) -> np.ndarray:
    data = np.frombuffer(jpeg_bytes, dtype=np.uint8)
    frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if frame is None:
        raise RuntimeError("jpeg_decode_failed")
    return frame


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

def mask_color(index: int) -> np.ndarray:
    palette = (
        (255, 99, 71),
        (60, 179, 113),
        (65, 105, 225),
        (255, 215, 0),
        (186, 85, 211),
        (0, 206, 209),
        (255, 140, 0),
        (220, 20, 60),
    )
    color = palette[index % len(palette)]
    return np.asarray(color, dtype=np.uint8)


def draw_results(frame: np.ndarray, result: Any) -> tuple[np.ndarray, int]:
    overlay = frame.copy()
    detections = 0
    result_h, result_w = getattr(result, "orig_shape", overlay.shape[:2])
    scale_x = overlay.shape[1] / max(1, int(result_w))
    scale_y = overlay.shape[0] / max(1, int(result_h))

    if result.masks is not None and result.masks.data is not None:
        for idx, mask_tensor in enumerate(result.masks.data):
            mask = mask_tensor.detach().cpu().numpy() > 0.5
            if mask.shape[:2] != overlay.shape[:2]:
                mask = cv2.resize(
                    mask.astype(np.uint8),
                    (overlay.shape[1], overlay.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(bool)
            color = mask_color(idx)
            overlay[mask] = ((0.62 * overlay[mask]) + (0.38 * color)).astype(np.uint8)
            contours, _ = cv2.findContours(
                mask.astype(np.uint8),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )
            cv2.drawContours(
                overlay,
                contours,
                -1,
                tuple(int(v) for v in color.tolist()),
                2,
                cv2.LINE_AA,
            )

    if result.boxes is not None and len(result.boxes) > 0:
        detections = len(result.boxes)
    return overlay, detections


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()
    model_path = resolve_model_path(args)

    # --- resolve realsense config ---
    rs_capture: Optional[RealsenseCapture] = None
    if args.source == "realsense":
        cam_cfg: dict = {}
        if args.camera_config and Path(args.camera_config).exists():
            cam_cfg = load_d405_config(args.camera_config)
        serial = str(args.serial or cam_cfg.get("serial_number", "")).strip()
        width = int(cam_cfg.get("color_width", 1280))
        height = int(cam_cfg.get("color_height", 720))
        fps = int(cam_cfg.get("color_fps", 30))
        initial_gain: Optional[float] = None
        if args.gain >= 0:
            initial_gain = float(args.gain)
        elif cam_cfg.get("depth_gain") is not None:
            initial_gain = float(cam_cfg["depth_gain"])
        print(
            json.dumps(
                {
                    "source": "realsense",
                    "serial": serial or "(any)",
                    "resolution": [width, height],
                    "fps": fps,
                    "initial_gain": initial_gain,
                    "model_path": str(model_path),
                    "conf": float(args.conf),
                    "imgsz": int(args.imgsz),
                    "device": str(args.device),
                },
                indent=2,
            )
        )
        rs_capture = RealsenseCapture(
            serial=serial,
            width=width,
            height=height,
            fps=fps,
            initial_gain=initial_gain,
        )
        # read back actual gain after init
        actual_gain = rs_capture.get_gain()
        if initial_gain is None:
            initial_gain = actual_gain
        print(f"[yolo-live] realsense started, gain={actual_gain:.1f}", flush=True)
    else:
        initial_gain = max(0.0, float(args.gain)) if args.gain >= 0 else 60.0
        print(
            json.dumps(
                {
                    "source": "http",
                    "camera_url": args.camera_url,
                    "camera_id": args.camera_id,
                    "model_path": str(model_path),
                    "conf": float(args.conf),
                    "imgsz": int(args.imgsz),
                    "device": str(args.device),
                    "viewer": str(args.viewer),
                },
                indent=2,
            )
        )

    # resolve device (auto = cuda if available, else cpu)
    device_str = str(args.device)
    if device_str == "auto":
        try:
            import torch
            device_str = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            device_str = "cpu"
    print(f"[yolo-live] device={device_str}", flush=True)

    model = YOLO(str(model_path))
    window_name = f"YOLO Live: {Path(model_path).name}"
    viewer_backend = str(args.viewer)
    mpl_viewer = None
    use_opencv_window = False

    if viewer_backend in {"auto", "opencv"}:
        try:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            use_opencv_window = True
            viewer_backend = "opencv"
        except cv2.error as exc:
            if str(args.viewer) == "opencv":
                raise
            print(
                f"[yolo-live] OpenCV window unavailable, falling back to matplotlib: {exc}",
                flush=True,
            )

    if not use_opencv_window:
        mpl_viewer = MatplotlibViewer(window_name)
        viewer_backend = "matplotlib"

    # --- OpenCV trackbars ---
    # conf trackbar: integer 1..99 maps to conf 0.01..0.99
    # gain trackbar: integer 0..128
    live_conf = [float(args.conf)]
    live_gain = [float(initial_gain if initial_gain is not None else 60.0)]

    if use_opencv_window:
        conf_init = max(1, min(99, int(round(live_conf[0] * 100))))

        def _on_conf(val: int) -> None:
            live_conf[0] = max(0.01, val / 100.0)

        cv2.createTrackbar("Conf x100", window_name, conf_init, 99, _on_conf)

        if rs_capture is not None:
            gain_init = max(0, min(128, int(round(live_gain[0]))))

            def _on_gain(val: int) -> None:
                live_gain[0] = float(val)
                assert rs_capture is not None
                rs_capture.set_gain(float(val))

            cv2.createTrackbar("Gain", window_name, gain_init, 128, _on_gain)

    frame_count = 0
    last_report_t = time.perf_counter()

    try:
        while True:
            started = time.perf_counter()
            try:
                if rs_capture is not None:
                    frame = rs_capture.read_bgr()
                else:
                    jpeg_bytes = fetch_frame_jpeg(args.camera_url, args.camera_id, args.quality)
                    frame = decode_bgr(jpeg_bytes)

                # pipeline passes full-res RGB to YOLO (no separate seg resize)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                conf_val = live_conf[0]
                result = model.predict(
                    source=rgb,
                    conf=conf_val,
                    imgsz=int(args.imgsz),
                    device=device_str,
                    retina_masks=True,
                    verbose=False,
                )[0]
                overlay, detections = draw_results(frame, result)
                elapsed_ms = (time.perf_counter() - started) * 1000.0
                frame_count += 1
                fps_val = frame_count / max(1e-6, time.perf_counter() - last_report_t)
                gain_label = f" gain={live_gain[0]:.0f}" if rs_capture is not None else ""
                status = f"dets={detections} conf={conf_val:.2f}{gain_label} [{device_str}] dt={elapsed_ms:.1f}ms"
                cv2.putText(
                    overlay,
                    status,
                    (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 0),
                    3,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    overlay,
                    status,
                    (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
                if use_opencv_window:
                    cv2.imshow(window_name, overlay)
                else:
                    assert mpl_viewer is not None
                    mpl_viewer.show(overlay)
                if time.perf_counter() - last_report_t >= 5.0:
                    print(
                        f"[yolo-live] fps={fps_val:.2f} dets={detections} conf={conf_val:.2f}{gain_label} [{device_str}] dt={elapsed_ms:.1f}ms",
                        flush=True,
                    )
                    frame_count = 0
                    last_report_t = time.perf_counter()
            except URLError as exc:
                blank = np.zeros((720, 1280, 3), dtype=np.uint8)
                cv2.putText(
                    blank,
                    f"Camera fetch failed: {exc}",
                    (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )
                if use_opencv_window:
                    cv2.imshow(window_name, blank)
                else:
                    assert mpl_viewer is not None
                    mpl_viewer.show(blank)
            except Exception as exc:
                print(f"[yolo-live] error: {exc}", flush=True)

            if use_opencv_window:
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break
            else:
                assert mpl_viewer is not None
                if mpl_viewer.is_closed():
                    break

            if rs_capture is None:
                time.sleep(max(0.0, float(args.poll_interval)))
    finally:
        if rs_capture is not None:
            rs_capture.stop()

    if use_opencv_window:
        cv2.destroyAllWindows()
    elif mpl_viewer is not None:
        mpl_viewer.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
