"""Implementation for `camera_core.drivers.realsense_d435i`."""

import time
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import pyrealsense2 as rs
from camera_core.drivers.base import CameraDriver


class RealSenseD435iDriver(CameraDriver):
    """
    Intel RealSense D435i RGB-D driver
    """

    device_label = "D435i"
    _OPTION_ENUM_TYPES = {
        "visual_preset": ("rs400_visual_preset", "l500_visual_preset"),
        "power_line_frequency": ("power_line_frequency",),
        "host_performance": ("host_performance",),
    }

    def __init__(self, logger):
        self.log = logger
        self.pipeline = None
        self.align = None
        self.depth_scale = None
        self.profile = None
        self.post_processors: List[Tuple[str, Any]] = []
        self.color_stream_format_name = "bgr8"
        self.output_color_format_name = "bgr8"
        self.target_depth_shape: Tuple[int, int] | None = None
        self._warned_depth_resize = False

    @staticmethod
    def _normalize_name(value: Any) -> str:
        return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")

    def _resolve_format(self, format_name: Any, default: Any) -> Any:
        clean = self._normalize_name(format_name)
        if not clean:
            return default
        resolved = getattr(rs.format, clean, None)
        if resolved is None:
            self.log.warning("Unknown RealSense format '%s'; using default", format_name)
            return default
        return resolved

    def _resolve_align_target(self, align_to: Any) -> Any:
        clean = self._normalize_name(align_to)
        if clean in {"", "color", "rgb"}:
            return rs.stream.color
        if clean == "depth":
            return rs.stream.depth
        if clean in {"none", "off", "disabled"}:
            return None
        self.log.warning("Unknown align_to '%s'; using color", align_to)
        return rs.stream.color

    @staticmethod
    def _describe_stream_profile(profile: Any, stream_type: Any) -> str:
        try:
            stream = profile.get_stream(stream_type)
            video = stream.as_video_stream_profile()
            fmt = str(video.format()).split(".")[-1]
            return f"{video.width()}x{video.height()}@{video.fps()}:{fmt}"
        except Exception:
            return "unknown"

    @staticmethod
    def _device_info(device: Any, info_key: Any) -> str:
        try:
            if device.supports(info_key):
                return str(device.get_info(info_key))
        except Exception:
            pass
        return ""

    @staticmethod
    def _get_frame_intrinsics(frame: Any) -> Any:
        try:
            return frame.profile.as_video_stream_profile().intrinsics
        except Exception:
            return None

    def _build_camera_data(self, color_frame: Any) -> Dict[str, Any]:
        intr = self._get_frame_intrinsics(color_frame)
        if intr is None:
            return {}
        try:
            dist_coeffs = [float(v) for v in list(getattr(intr, "coeffs", []) or [])]
        except Exception:
            dist_coeffs = []
        try:
            distortion_model = str(getattr(intr, "model", "") or "")
        except Exception:
            distortion_model = ""
        return {
            "K": [
                [float(intr.fx), 0.0, float(intr.ppx)],
                [0.0, float(intr.fy), float(intr.ppy)],
                [0.0, 0.0, 1.0],
            ],
            "resolution": [int(intr.height), int(intr.width)],
            "dist_coeffs": dist_coeffs,
            "distortion_model": distortion_model,
            "intrinsics": {
                "fx": float(intr.fx),
                "fy": float(intr.fy),
                "cx": float(intr.ppx),
                "cy": float(intr.ppy),
                "dist_coeffs": dist_coeffs,
                "distortion_model": distortion_model,
                "resolution": {
                    "width": int(intr.width),
                    "height": int(intr.height),
                },
            },
            "depth_scale_m_per_unit": float(self.depth_scale or 0.001),
            "color_format": self.output_color_format_name,
            "stream_color_format": self.color_stream_format_name,
            "device_model": self.device_label,
            "source": "pyrealsense2",
        }

    def _sensor_name(self, sensor: Any) -> str:
        try:
            if sensor.supports(rs.camera_info.name):
                return str(sensor.get_info(rs.camera_info.name) or "")
        except Exception:
            pass
        return sensor.__class__.__name__

    def _discover_sensors(self, device: Any) -> Dict[str, Any]:
        sensors: Dict[str, Any] = {}
        try:
            sensor_list = list(device.query_sensors())
        except Exception:
            sensor_list = []
        for sensor in sensor_list:
            raw_name = self._sensor_name(sensor)
            clean = self._normalize_name(raw_name)
            if "rgb" in clean or "color" in clean:
                sensors["color"] = sensor
            elif "stereo" in clean or "depth" in clean:
                sensors["depth"] = sensor
            elif "motion" in clean:
                sensors["motion"] = sensor
            sensors.setdefault(clean or f"sensor_{len(sensors)}", sensor)
        return sensors

    def _enum_to_float(self, value: Any) -> float:
        for candidate in (
            lambda v: float(v),
            lambda v: float(int(v)),
            lambda v: float(v.value),
        ):
            try:
                return candidate(value)
            except Exception:
                continue
        raise TypeError(f"Unsupported RealSense enum value type: {type(value)!r}")

    def _resolve_named_option_value(self, option_name: str, value: str) -> Any:
        clean_value = self._normalize_name(value)
        if not clean_value:
            return None
        for enum_type_name in self._OPTION_ENUM_TYPES.get(option_name, ()):
            enum_type = getattr(rs, enum_type_name, None)
            if enum_type is None:
                continue
            for attr_name in dir(enum_type):
                if attr_name.startswith("_"):
                    continue
                if self._normalize_name(attr_name) != clean_value:
                    continue
                try:
                    return getattr(enum_type, attr_name)
                except Exception:
                    continue
        return None

    def _coerce_option_value(self, option_name: str, value: Any) -> float:
        if isinstance(value, bool):
            return 1.0 if value else 0.0
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            enum_value = self._resolve_named_option_value(option_name, value)
            if enum_value is not None:
                return self._enum_to_float(enum_value)
            clean = value.strip().lower()
            if clean in {"true", "yes", "on"}:
                return 1.0
            if clean in {"false", "no", "off"}:
                return 0.0
            return float(value)
        raise TypeError(f"Unsupported RealSense option value type: {type(value)!r}")

    @staticmethod
    def _option_supported(target: Any, option_enum: Any) -> bool:
        try:
            supported_options = target.get_supported_options()
        except Exception:
            supported_options = None
        if supported_options is not None:
            option_text = str(option_enum)
            for supported in supported_options:
                try:
                    if supported == option_enum or str(supported) == option_text:
                        return True
                except Exception:
                    continue
            return False
        try:
            return bool(target.supports(option_enum))
        except Exception:
            return False

    def _apply_option_map(self, target: Any, options: Dict[str, Any], target_name: str) -> None:
        if target is None or not isinstance(options, dict):
            return
        for raw_option_name, raw_value in options.items():
            option_name = self._normalize_name(raw_option_name)
            option_enum = getattr(rs.option, option_name, None)
            if option_enum is None:
                self.log.warning(
                    "Ignoring unknown RealSense option '%s' for %s sensor",
                    raw_option_name,
                    target_name,
                )
                continue
            try:
                supported = self._option_supported(target, option_enum)
            except Exception:
                supported = False
            if not supported:
                self.log.warning(
                    "RealSense %s sensor does not support option '%s'",
                    target_name,
                    option_name,
                )
                continue
            try:
                option_value = self._coerce_option_value(option_name, raw_value)
            except Exception as exc:
                self.log.warning(
                    "Invalid value for RealSense option '%s' on %s sensor: %s",
                    option_name,
                    target_name,
                    exc,
                )
                continue
            try:
                target.set_option(option_enum, option_value)
                self.log.info(
                    "RealSense %s option %s=%s",
                    target_name,
                    option_name,
                    option_value,
                )
            except Exception as exc:
                self.log.warning(
                    "Failed setting RealSense %s option '%s': %s",
                    target_name,
                    option_name,
                    exc,
                )

    def _apply_roi(self, sensors: Dict[str, Any], config: Dict[str, Any]) -> None:
        roi_cfg = config.get("roi") or {}
        if not isinstance(roi_cfg, dict) or not roi_cfg.get("enabled", False):
            return

        target_name = self._normalize_name(roi_cfg.get("target", "depth")) or "depth"
        if target_name == "both":
            target_keys = ("color", "depth")
        elif target_name in {"color", "depth"}:
            target_keys = (target_name,)
        else:
            self.log.warning("Unknown RealSense ROI target '%s'; using depth", target_name)
            target_keys = ("depth",)

        try:
            region = rs.region_of_interest()
            region.min_x = int(roi_cfg.get("min_x", 0))
            region.min_y = int(roi_cfg.get("min_y", 0))
            region.max_x = int(roi_cfg.get("max_x", 0))
            region.max_y = int(roi_cfg.get("max_y", 0))
        except Exception as exc:
            self.log.warning("Invalid RealSense ROI configuration: %s", exc)
            return

        for sensor_key in target_keys:
            sensor = sensors.get(sensor_key)
            if sensor is None:
                continue
            try:
                roi_sensor = sensor.as_roi_sensor()
            except Exception:
                roi_sensor = None
            if roi_sensor is None:
                self.log.warning(
                    "RealSense %s sensor does not support ROI configuration",
                    sensor_key,
                )
                continue
            try:
                roi_sensor.set_region_of_interest(region)
                self.log.info(
                    "RealSense %s ROI set to (%d,%d)-(%d,%d)",
                    sensor_key,
                    region.min_x,
                    region.min_y,
                    region.max_x,
                    region.max_y,
                )
            except Exception as exc:
                self.log.warning("Failed setting RealSense %s ROI: %s", sensor_key, exc)

    def _normalize_color_output(self, color: np.ndarray) -> np.ndarray:
        if self.color_stream_format_name == "rgb8":
            self.output_color_format_name = "bgr8"
            return np.ascontiguousarray(color[..., ::-1])
        self.output_color_format_name = self.color_stream_format_name or "bgr8"
        return color

    def _build_post_processors(self, config: Dict[str, Any]) -> None:
        self.post_processors = []
        pp_cfg = config.get("post_processing") or {}
        if not isinstance(pp_cfg, dict) or not pp_cfg.get("enabled", False):
            return

        default_order = [
            "decimation",
            "rotation",
            "hdr_merge",
            "sequence_id_filter",
            "threshold",
            "depth_to_disparity",
            "spatial",
            "temporal",
            "hole_filling",
            "disparity_to_depth",
        ]
        raw_order = pp_cfg.get("filter_order")
        filter_order = raw_order if isinstance(raw_order, list) and raw_order else default_order

        for raw_name in filter_order:
            name = self._normalize_name(raw_name)
            spec = pp_cfg.get(name, {})
            if isinstance(spec, bool):
                enabled = spec
                options = {}
            elif isinstance(spec, dict):
                enabled = bool(spec.get("enabled", False))
                raw_options = spec.get("options", {})
                options = raw_options if isinstance(raw_options, dict) else {}
            else:
                enabled = False
                options = {}
            if not enabled:
                continue

            if name == "decimation":
                filter_obj = rs.decimation_filter()
            elif name == "rotation":
                filter_obj = rs.rotation_filter()
            elif name == "hdr_merge":
                filter_obj = rs.hdr_merge()
            elif name == "sequence_id_filter":
                filter_obj = rs.sequence_id_filter()
            elif name == "threshold":
                filter_obj = rs.threshold_filter()
            elif name == "spatial":
                filter_obj = rs.spatial_filter()
            elif name == "temporal":
                filter_obj = rs.temporal_filter()
            elif name == "hole_filling":
                filter_obj = rs.hole_filling_filter()
            elif name == "depth_to_disparity":
                filter_obj = rs.disparity_transform(True)
            elif name == "disparity_to_depth":
                filter_obj = rs.disparity_transform(False)
            else:
                self.log.warning("Ignoring unknown RealSense post-processing filter '%s'", raw_name)
                continue

            self._apply_option_map(filter_obj, options, f"post_filter:{name}")
            self.post_processors.append((name, filter_obj))

        if self.post_processors:
            self.log.info(
                "RealSense post-processing chain enabled: %s",
                ", ".join(name for name, _ in self.post_processors),
            )

    def initialize(self, config: dict) -> None:
        self.pipeline = rs.pipeline()
        cfg = rs.config()

        serial_number = str(config.get("serial_number") or "").strip()
        if serial_number:
            cfg.enable_device(serial_number)

        width = int(config.get("width", 640))
        height = int(config.get("height", 480))
        fps = int(config.get("fps", 30))
        color_width = int(config.get("color_width") or width)
        color_height = int(config.get("color_height") or height)
        color_fps = int(config.get("color_fps") or fps)
        depth_width = int(config.get("depth_width") or width)
        depth_height = int(config.get("depth_height") or height)
        depth_fps = int(config.get("depth_fps") or fps)
        self.target_depth_shape = (depth_height, depth_width)
        self._warned_depth_resize = False
        self.color_stream_format_name = (
            self._normalize_name(config.get("color_format", "bgr8")) or "bgr8"
        )
        color_format = self._resolve_format(config.get("color_format", "bgr8"), rs.format.bgr8)
        depth_format = self._resolve_format(config.get("depth_format", "z16"), rs.format.z16)

        cfg.enable_stream(
            rs.stream.color, color_width, color_height, color_format, color_fps
        )
        cfg.enable_stream(
            rs.stream.depth, depth_width, depth_height, depth_format, depth_fps
        )

        self.profile = self.pipeline.start(cfg)
        device = self.profile.get_device()
        sensors = self._discover_sensors(device)

        align_target = self._resolve_align_target(config.get("align_to", "color"))
        self.align = rs.align(align_target) if align_target is not None else None

        color_sensor = sensors.get("color")
        depth_sensor = sensors.get("depth")
        depth_scale_sensor = None
        try:
            depth_scale_sensor = device.first_depth_sensor()
        except Exception:
            depth_scale_sensor = None
        if depth_sensor is None:
            depth_sensor = depth_scale_sensor
        if color_sensor is None and depth_sensor is not None:
            color_sensor = depth_sensor
            self.log.info(
                "RealSense %s has no dedicated color sensor; applying color options to shared %s sensor",
                self.device_label,
                self._sensor_name(depth_sensor),
            )

        options_cfg = config.get("options") or {}
        if not isinstance(options_cfg, dict):
            options_cfg = {}

        shared_rgbd_options = (
            options_cfg.get("shared_rgbd") or options_cfg.get("common_rgbd") or {}
        )
        if isinstance(shared_rgbd_options, dict):
            for sensor_key in ("color", "depth"):
                self._apply_option_map(
                    sensors.get(sensor_key), shared_rgbd_options, sensor_key
                )

        all_options = options_cfg.get("all")
        if isinstance(all_options, dict):
            for sensor_key in ("color", "depth", "motion"):
                self._apply_option_map(sensors.get(sensor_key), all_options, sensor_key)

        self._apply_option_map(color_sensor, options_cfg.get("color", {}), "color")
        self._apply_option_map(depth_sensor, options_cfg.get("depth", {}), "depth")
        self._apply_option_map(sensors.get("motion"), options_cfg.get("motion", {}), "motion")
        self._apply_roi(sensors, config)

        legacy_color_options: Dict[str, Any] = {}
        if "exposure_time_us" in config and "color" not in options_cfg:
            legacy_color_options["exposure"] = config.get("exposure_time_us")
        if "gain" in config and "color" not in options_cfg:
            legacy_color_options["gain"] = config.get("gain")
        if legacy_color_options:
            self._apply_option_map(color_sensor, legacy_color_options, "color")

        self._build_post_processors(config)
        try:
            scale_source = depth_scale_sensor or depth_sensor
            if scale_source is None:
                raise RuntimeError("no_depth_scale_sensor")
            self.depth_scale = float(scale_source.get_depth_scale())
        except Exception:
            depth_options = options_cfg.get("depth", {})
            fallback_scale = 0.001
            if isinstance(depth_options, dict):
                fallback_scale = float(depth_options.get("depth_units", fallback_scale))
            self.depth_scale = fallback_scale
            self.log.warning(
                "Falling back to configured depth scale %.6f because get_depth_scale() was unavailable",
                self.depth_scale,
            )

        self.log.info(
            f"RealSense {self.device_label} initialized "
            f"color={color_width}x{color_height}@{color_fps}, "
            f"depth={depth_width}x{depth_height}@{depth_fps}, "
            f"align_to={self._normalize_name(config.get('align_to', 'color')) or 'color'}, "
            f"depth_scale={self.depth_scale}"
        )
        active_color = self._describe_stream_profile(self.profile, rs.stream.color)
        active_depth = self._describe_stream_profile(self.profile, rs.stream.depth)
        usb_type = self._device_info(device, rs.camera_info.usb_type_descriptor)
        self.log.info(
            "RealSense active profile color=%s depth=%s usb=%s",
            active_color,
            active_depth,
            usb_type or "unknown",
        )

    def start(self) -> None:
        pass  # pipeline already started

    def stop(self) -> None:
        if self.pipeline:
            self.pipeline.stop()

    def capture(self) -> dict:
        capture_t0 = time.perf_counter()
        frames = self.pipeline.wait_for_frames()
        wait_ms = (time.perf_counter() - capture_t0) * 1000.0

        align_ms = 0.0
        if self.align is not None:
            align_t0 = time.perf_counter()
            frames = self.align.process(frames)
            align_ms = (time.perf_counter() - align_t0) * 1000.0

        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

        if not color_frame or not depth_frame:
            raise RuntimeError("Incomplete RealSense frame")

        post_ms = 0.0
        for filter_name, filter_obj in self.post_processors:
            try:
                filter_t0 = time.perf_counter()
                depth_frame = filter_obj.process(depth_frame)
                post_ms += (time.perf_counter() - filter_t0) * 1000.0
            except Exception as exc:
                self.log.warning(
                    "RealSense post-processing filter '%s' failed: %s",
                    filter_name,
                    exc,
                )
                break

        array_t0 = time.perf_counter()
        color = self._normalize_color_output(np.asanyarray(color_frame.get_data()))
        depth_raw = np.asanyarray(depth_frame.get_data())
        array_ms = (time.perf_counter() - array_t0) * 1000.0

        # Convert depth to meters (float32)
        depth_convert_t0 = time.perf_counter()
        depth_m = depth_raw.astype(np.float32) * self.depth_scale
        if self.target_depth_shape and depth_m.shape != self.target_depth_shape:
            depth_m = cv2.resize(
                depth_m,
                (self.target_depth_shape[1], self.target_depth_shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
            if not self._warned_depth_resize:
                self.log.info(
                    "RealSense depth post-processing output %s resized to configured output %s",
                    depth_raw.shape,
                    self.target_depth_shape,
                )
                self._warned_depth_resize = True
        depth_convert_ms = (time.perf_counter() - depth_convert_t0) * 1000.0

        ts_ns = time.time_ns()
        capture_total_ms = (time.perf_counter() - capture_t0) * 1000.0

        return {
            "rgb": color,  # uint8 HxWx3
            "depth": depth_m,  # float32 HxW (meters)
            "camera_data": self._build_camera_data(color_frame),
            "timestamp_ns": ts_ns,
            "timings_ms": {
                "wait_for_frames": wait_ms,
                "align": align_ms,
                "post_processing": post_ms,
                "frame_to_numpy": array_ms,
                "depth_to_meters": depth_convert_ms,
                "capture_total": capture_total_ms,
            },
        }


class RealSenseD405Driver(RealSenseD435iDriver):
    """Intel RealSense D405 RGB-D driver."""

    device_label = "D405"
