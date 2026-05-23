#!/usr/bin/env python3
"""
Capture RGB-D frames for segmentation dataset creation with a RealSense camera.

This script is self-contained and is meant to replace the older Windows-only
copy that depended on missing helper modules.

Typical usage:
    python seg_obj/capture_from_realsense.py \
        --output /home/imp/imp/data/object_library/barel2/dataset \
        --camera-config /home/imp/imp/dexsent-vgr-camera-core-2.5d/config/cam_realsense_d435i.yaml

Preview controls:
    s -> save current RGB/depth frame pair
    q -> quit
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pyrealsense2 as rs
import yaml


OPTION_ENUM_TYPES = {
    "visual_preset": ("rs400_visual_preset", "l500_visual_preset"),
    "power_line_frequency": ("power_line_frequency",),
    "host_performance": ("host_performance",),
}


def _normalize_name(value: Any) -> str:
    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


def _default_camera_config_path() -> Path | None:
    candidate = Path(__file__).resolve().parents[1] / "dexsent-vgr-camera-core-2.5d" / "config" / "cam_realsense_d435i.yaml"
    return candidate if candidate.exists() else None


def _resolve_format(format_name: str, default: Any) -> Any:
    clean = _normalize_name(format_name)
    if not clean:
        return default
    resolved = getattr(rs.format, clean, None)
    return default if resolved is None else resolved


def _resolve_align_target(align_to: str) -> Any:
    clean = _normalize_name(align_to)
    if clean in {"", "color", "rgb"}:
        return rs.stream.color
    if clean == "depth":
        return rs.stream.depth
    if clean in {"none", "off", "disabled"}:
        return None
    return rs.stream.color


def _sensor_name(sensor: Any) -> str:
    try:
        if sensor.supports(rs.camera_info.name):
            return str(sensor.get_info(rs.camera_info.name) or "")
    except Exception:
        pass
    return sensor.__class__.__name__


def _discover_sensors(device: Any) -> dict[str, Any]:
    sensors: dict[str, Any] = {}
    for sensor in device.query_sensors():
        clean = _normalize_name(_sensor_name(sensor))
        if "rgb" in clean or "color" in clean:
            sensors["color"] = sensor
        elif "stereo" in clean or "depth" in clean:
            sensors["depth"] = sensor
        sensors.setdefault(clean or f"sensor_{len(sensors)}", sensor)
    return sensors


def _enum_to_float(value: Any) -> float:
    for candidate in (
        lambda current: float(current),
        lambda current: float(int(current)),
        lambda current: float(current.value),
    ):
        try:
            return candidate(value)
        except Exception:
            continue
    raise TypeError(f"Unsupported RealSense enum value type: {type(value)!r}")


def _resolve_named_option_value(option_name: str, value: str) -> Any:
    clean_value = _normalize_name(value)
    if not clean_value:
        return None
    for enum_type_name in OPTION_ENUM_TYPES.get(option_name, ()):
        enum_type = getattr(rs, enum_type_name, None)
        if enum_type is None:
            continue
        for attr_name in dir(enum_type):
            if attr_name.startswith("_"):
                continue
            if _normalize_name(attr_name) != clean_value:
                continue
            try:
                return getattr(enum_type, attr_name)
            except Exception:
                continue
    return None


def _coerce_option_value(option_name: str, value: Any) -> float:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        enum_value = _resolve_named_option_value(option_name, value)
        if enum_value is not None:
            return _enum_to_float(enum_value)
        clean = value.strip().lower()
        if clean in {"true", "yes", "on"}:
            return 1.0
        if clean in {"false", "no", "off"}:
            return 0.0
        return float(value)
    raise TypeError(f"Unsupported RealSense option value type: {type(value)!r}")


def _bool_or_none(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    clean = str(value).strip().lower()
    if clean in {"true", "1", "yes", "on"}:
        return True
    if clean in {"false", "0", "no", "off"}:
        return False
    return None


def _option_supported(sensor: Any, option_enum: Any) -> bool:
    try:
        supported_options = sensor.get_supported_options()
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
        return bool(sensor.supports(option_enum))
    except Exception:
        return False


def _apply_single_option(sensor: Any, option_name: str, value: Any, target_name: str) -> None:
    option_enum = getattr(rs.option, _normalize_name(option_name), None)
    if option_enum is None:
        print(f"[WARN] Unknown RealSense option '{option_name}' for {target_name}")
        return
    if not _option_supported(sensor, option_enum):
        print(f"[WARN] {target_name} sensor does not support option '{option_name}'")
        return
    try:
        coerced = _coerce_option_value(_normalize_name(option_name), value)
        sensor.set_option(option_enum, coerced)
        print(f"[INFO] {target_name} option {option_name}={value}")
    except Exception as exc:
        print(f"[WARN] Failed to set {target_name} option '{option_name}': {exc}")


def _apply_option_map(sensor: Any, options: dict[str, Any], target_name: str) -> None:
    if sensor is None or not options:
        return

    for key in ("enable_auto_exposure", "enable_auto_white_balance"):
        if key in options:
            _apply_single_option(sensor, key, options[key], target_name)

    skip_keys = {"enable_auto_exposure", "enable_auto_white_balance", "exposure", "gain", "white_balance"}
    for key, value in options.items():
        if key in skip_keys:
            continue
        _apply_single_option(sensor, key, value, target_name)

    auto_exposure = _bool_or_none(options.get("enable_auto_exposure"))
    if auto_exposure is not True:
        for key in ("exposure", "gain"):
            if key in options:
                _apply_single_option(sensor, key, options[key], target_name)

    auto_white_balance = _bool_or_none(options.get("enable_auto_white_balance"))
    if auto_white_balance is not True and "white_balance" in options:
        _apply_single_option(sensor, "white_balance", options["white_balance"], target_name)


def _read_camera_config(config_path: Path | None) -> dict[str, Any]:
    defaults = {
        "config_path": str(config_path) if config_path else "",
        "serial_number": "",
        "align_to": "color",
        "color_width": 1280,
        "color_height": 720,
        "color_fps": 30,
        "color_format": "bgr8",
        "depth_width": 1280,
        "depth_height": 720,
        "depth_fps": 30,
        "depth_format": "z16",
        "all_options": {},
        "color_options": {},
        "depth_options": {},
    }
    if config_path is None:
        return defaults
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    sdk = (((raw.get("camera") or {}).get("sdk")) or {})
    options = (sdk.get("options") or {})

    defaults.update(
        {
            "serial_number": str(sdk.get("serial_number") or ""),
            "align_to": str(sdk.get("align_to") or "color"),
            "color_width": int(sdk.get("color_width", sdk.get("width", 1280))),
            "color_height": int(sdk.get("color_height", sdk.get("height", 720))),
            "color_fps": int(sdk.get("color_fps", sdk.get("fps", 30))),
            "color_format": str(sdk.get("color_format") or "bgr8"),
            "depth_width": int(sdk.get("depth_width", sdk.get("width", 1280))),
            "depth_height": int(sdk.get("depth_height", sdk.get("height", 720))),
            "depth_fps": int(sdk.get("depth_fps", sdk.get("fps", 30))),
            "depth_format": str(sdk.get("depth_format") or "z16"),
            "all_options": dict(options.get("all") or {}),
            "color_options": dict(options.get("color") or {}),
            "depth_options": dict(options.get("depth") or {}),
        }
    )
    return defaults


def _merge_capture_settings(config_data: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    merged = dict(config_data)
    if args.serial_number is not None:
        merged["serial_number"] = args.serial_number
    if args.align_to is not None:
        merged["align_to"] = args.align_to

    for key in (
        "color_width",
        "color_height",
        "color_fps",
        "depth_width",
        "depth_height",
        "depth_fps",
    ):
        value = getattr(args, key)
        if value is not None:
            merged[key] = value

    color_options = dict(merged.get("color_options") or {})
    depth_options = dict(merged.get("depth_options") or {})

    cli_overrides = {
        "color_auto_exposure": ("enable_auto_exposure", color_options),
        "color_exposure": ("exposure", color_options),
        "color_gain": ("gain", color_options),
        "color_auto_white_balance": ("enable_auto_white_balance", color_options),
        "color_white_balance": ("white_balance", color_options),
        "depth_auto_exposure": ("enable_auto_exposure", depth_options),
        "depth_exposure": ("exposure", depth_options),
        "depth_gain": ("gain", depth_options),
        "laser_power": ("laser_power", depth_options),
    }
    for arg_name, (option_name, target_options) in cli_overrides.items():
        value = getattr(args, arg_name)
        if value is not None:
            target_options[option_name] = value

    merged["color_options"] = color_options
    merged["depth_options"] = depth_options
    return merged


def _build_parser() -> argparse.ArgumentParser:
    default_config = _default_camera_config_path()
    parser = argparse.ArgumentParser(description="Capture RGB-D samples for segmentation training.")
    parser.add_argument("--output", required=True, help="Folder where rgb_*.png and depth_*.png files will be saved")
    parser.add_argument(
        "--camera-config",
        default=str(default_config) if default_config else None,
        help="Path to the camera YAML used by dexsent-vgr-camera-core-2.5d",
    )
    parser.add_argument("--serial-number", default=None, help="Optional RealSense serial number override")
    parser.add_argument("--align-to", choices=("color", "depth", "none"), default=None, help="Frame alignment target")
    parser.add_argument("--color-width", type=int, default=None)
    parser.add_argument("--color-height", type=int, default=None)
    parser.add_argument("--color-fps", type=int, default=None)
    parser.add_argument("--depth-width", type=int, default=None)
    parser.add_argument("--depth-height", type=int, default=None)
    parser.add_argument("--depth-fps", type=int, default=None)

    parser.add_argument("--color-auto-exposure", action="store_true", dest="color_auto_exposure")
    parser.add_argument("--no-color-auto-exposure", action="store_false", dest="color_auto_exposure")
    parser.set_defaults(color_auto_exposure=None)
    parser.add_argument("--color-exposure", type=float, default=None)
    parser.add_argument("--color-gain", type=float, default=None)
    parser.add_argument("--color-auto-white-balance", action="store_true", dest="color_auto_white_balance")
    parser.add_argument("--no-color-auto-white-balance", action="store_false", dest="color_auto_white_balance")
    parser.set_defaults(color_auto_white_balance=None)
    parser.add_argument("--color-white-balance", type=float, default=None)

    parser.add_argument("--depth-auto-exposure", action="store_true", dest="depth_auto_exposure")
    parser.add_argument("--no-depth-auto-exposure", action="store_false", dest="depth_auto_exposure")
    parser.set_defaults(depth_auto_exposure=None)
    parser.add_argument("--depth-exposure", type=float, default=None)
    parser.add_argument("--depth-gain", type=float, default=None)
    parser.add_argument("--laser-power", type=float, default=None)

    parser.add_argument("--warmup-frames", type=int, default=15, help="Frames to discard after startup")
    return parser


def _start_pipeline(settings: dict[str, Any]) -> tuple[Any, Any, Any]:
    pipeline = rs.pipeline()
    config = rs.config()

    serial_number = str(settings.get("serial_number") or "").strip()
    if serial_number:
        config.enable_device(serial_number)

    config.enable_stream(
        rs.stream.color,
        int(settings["color_width"]),
        int(settings["color_height"]),
        _resolve_format(settings["color_format"], rs.format.bgr8),
        int(settings["color_fps"]),
    )
    config.enable_stream(
        rs.stream.depth,
        int(settings["depth_width"]),
        int(settings["depth_height"]),
        _resolve_format(settings["depth_format"], rs.format.z16),
        int(settings["depth_fps"]),
    )

    profile = pipeline.start(config)
    align_target = _resolve_align_target(str(settings.get("align_to", "color")))
    align = None if align_target is None else rs.align(align_target)
    return pipeline, align, profile


def _warmup(pipeline: Any, align: Any, frame_count: int) -> None:
    for _ in range(max(0, frame_count)):
        frames = pipeline.wait_for_frames()
        if align is not None:
            align.process(frames)


def _fit_preview(image: np.ndarray, max_width: int = 1600, max_height: int = 900) -> np.ndarray:
    height, width = image.shape[:2]
    scale = min(max_width / width, max_height / height, 1.0)
    if scale >= 1.0:
        return image
    new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
    return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)


def _match_image_height(image: np.ndarray, target_height: int) -> np.ndarray:
    if image.shape[0] == target_height:
        return image
    width = max(1, int(image.shape[1] * (target_height / image.shape[0])))
    return cv2.resize(image, (width, target_height), interpolation=cv2.INTER_NEAREST)


def _intrinsics_to_dict(intrinsics: Any, depth_scale: float) -> dict[str, Any]:
    return {
        "width": int(intrinsics.width),
        "height": int(intrinsics.height),
        "fx": float(intrinsics.fx),
        "fy": float(intrinsics.fy),
        "cx": float(intrinsics.ppx),
        "cy": float(intrinsics.ppy),
        "coeffs": [float(value) for value in list(intrinsics.coeffs)],
        "depth_scale_m_per_unit": float(depth_scale),
    }


def _next_capture_index(output_dir: Path) -> int:
    max_index = -1
    for image_path in output_dir.glob("rgb_*.png"):
        try:
            max_index = max(max_index, int(image_path.stem.split("_")[-1]))
        except ValueError:
            continue
    return max_index + 1


def _save_capture(
    output_dir: Path,
    capture_index: int,
    color_image: np.ndarray,
    depth_image: np.ndarray,
    depth_vis: np.ndarray,
    intrinsics: dict[str, Any],
    settings: dict[str, Any],
) -> None:
    stem = f"{capture_index:03d}"
    cv2.imwrite(str(output_dir / f"rgb_{stem}.png"), color_image)
    cv2.imwrite(str(output_dir / f"depth_{stem}.png"), depth_image)
    cv2.imwrite(str(output_dir / f"depth_vis_{stem}.png"), depth_vis)
    (output_dir / f"intrinsics_{stem}.json").write_text(json.dumps(intrinsics, indent=2), encoding="utf-8")
    (output_dir / f"settings_{stem}.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")
    print(f"[DONE] Saved capture #{stem} to {output_dir}")


def _read_sensor_option(sensor: Any, option_name: str) -> Any:
    option_enum = getattr(rs.option, _normalize_name(option_name), None)
    if option_enum is None or sensor is None:
        return None
    try:
        if sensor.supports(option_enum):
            return sensor.get_option(option_enum)
    except Exception:
        return None
    return None


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    output_dir = Path(args.output).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    camera_config = None if not args.camera_config else Path(args.camera_config).expanduser().resolve()
    if camera_config is not None and not camera_config.exists():
        parser.error(f"camera config not found: {camera_config}")

    settings = _merge_capture_settings(_read_camera_config(camera_config), args)
    print("[INFO] Capture output:", output_dir)
    if camera_config is not None:
        print("[INFO] Camera config:", camera_config)
    print(
        "[INFO] Streams:",
        f"color={settings['color_width']}x{settings['color_height']}@{settings['color_fps']}",
        f"depth={settings['depth_width']}x{settings['depth_height']}@{settings['depth_fps']}",
        f"align_to={settings['align_to']}",
    )

    pipeline = None
    try:
        pipeline, align, profile = _start_pipeline(settings)
        device = profile.get_device()
        sensors = _discover_sensors(device)
        color_sensor = sensors.get("color")
        depth_sensor = sensors.get("depth") or device.first_depth_sensor()
        depth_scale_sensor = device.first_depth_sensor()

        for sensor in sensors.values():
            _apply_option_map(sensor, settings.get("all_options") or {}, _sensor_name(sensor))
        _apply_option_map(color_sensor, settings.get("color_options") or {}, "color")
        _apply_option_map(depth_sensor, settings.get("depth_options") or {}, "depth")

        depth_scale = float(depth_scale_sensor.get_depth_scale())
        colorizer = rs.colorizer()
        _warmup(pipeline, align, args.warmup_frames)
        print("[INFO] Depth scale:", depth_scale)
        print("[INFO] Preview ready. Press 's' to save, 'q' to quit.")

        capture_index = _next_capture_index(output_dir)
        window_name = "RealSense Segmentation Capture"

        while True:
            frames = pipeline.wait_for_frames()
            if align is not None:
                frames = align.process(frames)

            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            if not color_frame or not depth_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())
            depth_vis = np.asanyarray(colorizer.colorize(depth_frame).get_data())
            preview = np.hstack((color_image, _match_image_height(depth_vis, color_image.shape[0])))

            rgb_ae = _read_sensor_option(color_sensor, "enable_auto_exposure")
            rgb_exp = _read_sensor_option(color_sensor, "exposure")
            rgb_gain = _read_sensor_option(color_sensor, "gain")
            depth_ae = _read_sensor_option(depth_sensor, "enable_auto_exposure")
            depth_exp = _read_sensor_option(depth_sensor, "exposure")
            depth_gain = _read_sensor_option(depth_sensor, "gain")

            overlay_lines = [
                f"Output: {output_dir}",
                (
                    "RGB AE: "
                    f"{'ON' if rgb_ae else 'OFF'} | "
                    f"Exp: {0 if rgb_exp is None else rgb_exp:.0f} | "
                    f"Gain: {0 if rgb_gain is None else rgb_gain:.0f}"
                ),
                (
                    "Depth AE: "
                    f"{'ON' if depth_ae else 'OFF'} | "
                    f"Exp: {0 if depth_exp is None else depth_exp:.0f} | "
                    f"Gain: {0 if depth_gain is None else depth_gain:.0f}"
                ),
                "Keys: s save, q quit",
            ]

            y = 28
            for line in overlay_lines:
                cv2.putText(preview, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 3, cv2.LINE_AA)
                cv2.putText(preview, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (20, 20, 20), 1, cv2.LINE_AA)
                y += 28

            cv2.imshow(window_name, _fit_preview(preview))
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            if key == ord("s"):
                intrinsics = color_frame.profile.as_video_stream_profile().intrinsics
                applied_settings = {
                    "camera_config_path": str(camera_config) if camera_config else "",
                    "color_stream": {
                        "width": int(settings["color_width"]),
                        "height": int(settings["color_height"]),
                        "fps": int(settings["color_fps"]),
                        "format": str(settings["color_format"]),
                    },
                    "depth_stream": {
                        "width": int(settings["depth_width"]),
                        "height": int(settings["depth_height"]),
                        "fps": int(settings["depth_fps"]),
                        "format": str(settings["depth_format"]),
                    },
                    "align_to": str(settings["align_to"]),
                    "color_options": dict(settings.get("color_options") or {}),
                    "depth_options": dict(settings.get("depth_options") or {}),
                    "all_options": dict(settings.get("all_options") or {}),
                }
                _save_capture(
                    output_dir=output_dir,
                    capture_index=capture_index,
                    color_image=color_image,
                    depth_image=depth_image,
                    depth_vis=depth_vis,
                    intrinsics=_intrinsics_to_dict(intrinsics, depth_scale),
                    settings=applied_settings,
                )
                capture_index += 1

        return 0
    except RuntimeError as exc:
        print(f"[ERROR] RealSense capture failed: {exc}")
        return 1
    finally:
        if pipeline is not None:
            try:
                pipeline.stop()
            except Exception:
                pass
        cv2.destroyAllWindows()


if __name__ == "__main__":
    raise SystemExit(main())
