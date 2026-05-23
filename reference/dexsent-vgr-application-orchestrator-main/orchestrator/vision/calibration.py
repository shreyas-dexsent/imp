"""Helpers for attaching station-owned camera calibration to vision sessions."""

from __future__ import annotations

import json
import math
from typing import Any, Dict, Optional

from orchestrator.storage.paths import DataPaths
from orchestrator.storage.station_store import StationStore


def _finite_float(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except Exception:
        return None
    return number if math.isfinite(number) else None


def _normalize_resolution(value: Any) -> Optional[Dict[str, int]]:
    if not isinstance(value, dict):
        return None
    width = _finite_float(value.get("width"))
    height = _finite_float(value.get("height"))
    if width is None or height is None or width <= 0 or height <= 0:
        return None
    return {
        "width": int(round(width)),
        "height": int(round(height)),
    }


def load_station_intrinsics(
    data_paths: DataPaths,
    station_id: Optional[str],
) -> Optional[Dict[str, Any]]:
    sid = str(station_id or "").strip()
    if not sid:
        return None
    path = data_paths.station_calibration_dir(sid) / "intrinsics.json"
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(raw, dict):
        return None

    fx = _finite_float(raw.get("fx"))
    fy = _finite_float(raw.get("fy"))
    cx = _finite_float(raw.get("cx"))
    cy = _finite_float(raw.get("cy"))
    if fx is None or fy is None or cx is None or cy is None:
        return None

    intrinsics: Dict[str, Any] = {
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy,
    }
    resolution = _normalize_resolution(raw.get("resolution"))
    if resolution is not None:
        intrinsics["resolution"] = resolution

    dist_coeffs = raw.get("dist_coeffs")
    if isinstance(dist_coeffs, list):
        parsed_dist_coeffs = []
        for value in dist_coeffs:
            coeff = _finite_float(value)
            if coeff is None:
                parsed_dist_coeffs = []
                break
            parsed_dist_coeffs.append(coeff)
        if parsed_dist_coeffs:
            intrinsics["dist_coeffs"] = parsed_dist_coeffs

    distortion_model = raw.get("distortion_model")
    if distortion_model is not None:
        intrinsics["distortion_model"] = str(distortion_model)

    return intrinsics


def build_station_camera_calibration(
    data_paths: DataPaths,
    station_id: Optional[str],
) -> Optional[Dict[str, Any]]:
    intrinsics = load_station_intrinsics(data_paths, station_id)
    if not isinstance(intrinsics, dict):
        return None

    fx = float(intrinsics["fx"])
    fy = float(intrinsics["fy"])
    cx = float(intrinsics["cx"])
    cy = float(intrinsics["cy"])
    resolution = intrinsics.get("resolution")
    if not isinstance(resolution, dict):
        resolution = {}
    width = int(resolution.get("width") or 0)
    height = int(resolution.get("height") or 0)

    K = [
        [fx, 0.0, cx],
        [0.0, fy, cy],
        [0.0, 0.0, 1.0],
    ]
    camera_intrinsics: Dict[str, Any] = dict(intrinsics)
    if width > 0 and height > 0:
        camera_intrinsics["resolution"] = {
            "width": width,
            "height": height,
        }

    camera_data: Dict[str, Any] = {
        "K": K,
        "intrinsics": camera_intrinsics,
        "source": "station_calibration",
    }
    if width > 0 and height > 0:
        camera_data["resolution"] = [height, width]
    if "dist_coeffs" in camera_intrinsics:
        camera_data["dist_coeffs"] = list(camera_intrinsics["dist_coeffs"])
    if "distortion_model" in camera_intrinsics:
        camera_data["distortion_model"] = camera_intrinsics["distortion_model"]

    calibration: Dict[str, Any] = {
        "source": "station_calibration",
        "K": K,
        "intrinsics": camera_intrinsics,
        "camera_data": camera_data,
    }
    if width > 0 and height > 0:
        calibration["intrinsics_resolution"] = {
            "width": width,
            "height": height,
        }
    return calibration


def infer_station_id_for_camera(
    stations: StationStore,
    camera_id: Optional[str],
    preferred_station_id: Optional[str] = None,
) -> Optional[str]:
    preferred = str(preferred_station_id or "").strip()
    if preferred:
        return preferred

    camera = str(camera_id or "").strip()
    station_rows = stations.list()
    if camera:
        matches = []
        for station in station_rows:
            camera_ids = station.get("camera_ids")
            if isinstance(camera_ids, list) and camera in camera_ids:
                matches.append(str(station.get("station_id") or "").strip())
        matches = [station_id for station_id in matches if station_id]
        if len(matches) == 1:
            return matches[0]

    if len(station_rows) == 1:
        station_id = str(station_rows[0].get("station_id") or "").strip()
        if station_id:
            return station_id
    return None
