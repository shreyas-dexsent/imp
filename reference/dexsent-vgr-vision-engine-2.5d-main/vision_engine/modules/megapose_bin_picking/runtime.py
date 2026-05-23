from __future__ import annotations

import json
import math
import os
import platform
import shutil
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import cv2
import numpy as np

# NumPy 2 removed a few legacy aliases that older third-party geometry stacks
# still reference during import/runtime (for example trimesh and vendored
# MegaPose helpers). Restore the aliases locally so the vision runtime stays
# compatible without patching site-packages.
if not hasattr(np, "float_"):
    np.float_ = np.float64  # type: ignore[attr-defined]
if not hasattr(np, "complex_"):
    np.complex_ = np.complex128  # type: ignore[attr-defined]

MODEL_CONFIGS: dict[str, dict[str, Any]] = {
    "rgb": {
        "coarse_run_id": "coarse-rgb-906902141",
        "refiner_run_id": "refiner-rgb-653307694",
        "requires_depth": False,
        "n_refiner_iterations": 5,
        "n_pose_hypotheses": 1,
    },
    "rgb-multi-hypothesis": {
        "coarse_run_id": "coarse-rgb-906902141",
        "refiner_run_id": "refiner-rgb-653307694",
        "requires_depth": False,
        "n_refiner_iterations": 5,
        "n_pose_hypotheses": 5,
    },
    "rgbd": {
        "coarse_run_id": "coarse-rgb-906902141",
        "refiner_run_id": "refiner-rgbd-288182519",
        "requires_depth": True,
        "n_refiner_iterations": 5,
        "n_pose_hypotheses": 5,
    },
}

VISION_ENGINE_ROOT = Path(__file__).resolve().parents[3]
WORKSPACE_ROOT = VISION_ENGINE_ROOT.parent
# Keep the editable MegaPose runtime copy separate from the read-only
# reference tree under repo-level third_party/.
DEFAULT_MEGAPOSE_SOURCE_ROOT = (
    VISION_ENGINE_ROOT / "vision_engine" / "modules" / "megapose_bin_picking" / "vendor_runtime"
)
DEFAULT_MEGAPOSE_WEIGHTS_ROOT = VISION_ENGINE_ROOT / "weights"
DEFAULT_RUNTIME_DATA_ROOT = VISION_ENGINE_ROOT / "runtime_data"
DEFAULT_OUTPUT_ROOT = WORKSPACE_ROOT / "data" / "vision" / "megapose"
DEFAULT_STARTUP_PREWARM_PARAMS_PATH = (
    DEFAULT_RUNTIME_DATA_ROOT / "megapose_startup_prewarm_params.json"
)

_MEGAPOSE_IMPORT_LOCK = threading.Lock()
_IMPORTED_MODULES: dict[str, dict[str, Any]] = {}
_YOLO_CACHE_LOCK = threading.Lock()
_YOLO_MODELS: dict[str, Any] = {}
_YOLO_INFER_LOCK = threading.Lock()
_SAM_CACHE_LOCK = threading.Lock()
_SAM_MODELS: dict[str, Any] = {}
_SAM_INFER_LOCK = threading.Lock()
_ESTIMATOR_CACHE_LOCK = threading.Lock()
_ESTIMATOR_CACHE: dict[tuple[Any, ...], "EstimatorBundle"] = {}
_PREWARM_CACHE_LOCK = threading.Lock()
_YOLO_WARMED: set[str] = set()
_SAM_WARMED: set[str] = set()
_ESTIMATOR_WARMED: set[tuple[Any, ...]] = set()
_MESH_ANNOTATION_CACHE_LOCK = threading.Lock()
_MESH_ANNOTATION_CACHE: dict[tuple[str, str, float], dict[str, Any]] = {}


def _runtime_log(request_id: str | None, message: str) -> None:
    prefix = f"[megapose_bin_picking:{request_id}]" if request_id else "[megapose_bin_picking]"
    print(f"{prefix} {message}", flush=True)


class _BinPickingTimingLog:
    def __init__(
        self,
        *,
        enabled: bool,
        path: Path | None,
        run_id: str,
        request_id: str,
    ) -> None:
        self.enabled = bool(enabled and path is not None)
        self.path = path
        self.run_id = run_id
        self.request_id = request_id
        self.started_perf = time.perf_counter()
        self.started_ns = time.time_ns()
        self.rows: list[dict[str, Any]] = []

    @staticmethod
    def _enabled_from_params(params: Dict[str, Any]) -> bool:
        for key in ("timing_log_enabled", "enable_timing_log", "bin_picking_timing"):
            if key in params:
                return bool(params.get(key))
        timing_cfg = params.get("timing")
        if isinstance(timing_cfg, dict):
            return bool(timing_cfg.get("enabled", False))
        return False

    @staticmethod
    def _run_id_from_request(request_id: str) -> str:
        if "-vision-" in request_id:
            return request_id.split("-vision-", 1)[0]
        return request_id or "unknown-run"

    @classmethod
    def from_params(
        cls,
        params: Dict[str, Any],
        request_id: str | None,
    ) -> "_BinPickingTimingLog":
        clean_request_id = str(request_id or "").strip()
        run_id = str(params.get("timing_run_id") or "").strip()
        if not run_id:
            run_id = cls._run_id_from_request(clean_request_id)
        raw_path = params.get("timing_log_path")
        if raw_path:
            path = resolve_workspace_path(raw_path, workspace_root=WORKSPACE_ROOT)
        else:
            path = WORKSPACE_ROOT / "data" / "vision" / "runs" / run_id / "bin_picking_timing.jsonl"
        return cls(
            enabled=cls._enabled_from_params(params),
            path=path,
            run_id=run_id,
            request_id=clean_request_id,
        )

    def span(self, stage: str, started_perf: float, **fields: Any) -> None:
        if not self.enabled:
            return
        ended_perf = time.perf_counter()
        row = {
            "type": "stage",
            "run_id": self.run_id,
            "request_id": self.request_id,
            "stage": str(stage),
            "duration_s": float(max(0.0, ended_perf - started_perf)),
            "elapsed_s": float(max(0.0, ended_perf - self.started_perf)),
            "timestamp_ns": time.time_ns(),
        }
        row.update(fields)
        self.rows.append(row)
        self._append(row)

    def summary(self, **fields: Any) -> None:
        if not self.enabled:
            return
        total_s = float(max(0.0, time.perf_counter() - self.started_perf))
        stage_totals = {
            str(row.get("stage")): float(row.get("duration_s", 0.0))
            for row in self.rows
            if row.get("type") == "stage"
        }
        row = {
            "type": "summary",
            "run_id": self.run_id,
            "request_id": self.request_id,
            "total_s": total_s,
            "started_ns": self.started_ns,
            "timestamp_ns": time.time_ns(),
            "stage_totals_s": stage_totals,
        }
        row.update(fields)
        self._append(row)

    def _append(self, row: Dict[str, Any]) -> None:
        if not self.enabled or self.path is None:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(_json_safe(row), separators=(",", ":")) + "\n")
        except Exception as exc:
            _runtime_log(self.request_id, f"timing_log_write_failed path={self.path} error={exc}")
            self.enabled = False


def _scaled_rgb_mask_dilation_px(
    image_shape: tuple[int, int] | tuple[int, int, int],
    base_pixels: float = 5.0,
    reference_width: float = 1280.0,
    reference_height: float = 720.0,
) -> int:
    if not image_shape or len(image_shape) < 2:
        return max(0, int(round(base_pixels)))
    height = max(1.0, float(image_shape[0]))
    width = max(1.0, float(image_shape[1]))
    scale_x = width / max(1.0, float(reference_width))
    scale_y = height / max(1.0, float(reference_height))
    scale = math.sqrt(max(1e-8, scale_x * scale_y))
    return max(0, int(round(float(base_pixels) * scale)))


def _coerce_float(value: Any, default: float | None = None) -> float | None:
    try:
        if value is None or value == "":
            return default
        num = float(value)
        if not math.isfinite(num):
            return default
        return num
    except (TypeError, ValueError):
        return default


def _make_feature_revealing_light_rig(modules: dict[str, Any]) -> list[Any]:
    Panda3dLightData = modules["Panda3dLightData"]

    def _scene_center_radius_camera(root_node: Any) -> tuple[np.ndarray, float, Any | None]:
        bounds = root_node.getBounds()
        try:
            radius = float(bounds.radius)
        except Exception:
            radius = 0.0
        if not math.isfinite(radius) or radius <= 1e-4:
            radius = 0.05
        center_pt = bounds.getApproxCenter()
        center = np.array(
            [float(center_pt[0]), float(center_pt[1]), float(center_pt[2])],
            dtype=np.float32,
        )
        try:
            camera_np = root_node.find("**/+Camera")
        except Exception:
            camera_np = None
        if camera_np is None:
            return center, radius, None
        try:
            if camera_np.isEmpty():
                return center, radius, None
        except Exception:
            pass
        return center, radius, camera_np

    def _camera_relative_position(
        root_node: Any,
        offset: tuple[float, float, float],
        radius_scale: float = 4.0,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        center, radius, camera_np = _scene_center_radius_camera(root_node)
        if camera_np is None:
            toward_camera = np.array([0.0, -1.0, 0.0], dtype=np.float32)
        else:
            cam_pos_pt = camera_np.getPos(root_node)
            cam_pos = np.array(
                [float(cam_pos_pt[0]), float(cam_pos_pt[1]), float(cam_pos_pt[2])],
                dtype=np.float32,
            )
            toward_camera = cam_pos - center
            toward_camera_norm = float(np.linalg.norm(toward_camera))
            if not math.isfinite(toward_camera_norm) or toward_camera_norm <= 1e-6:
                toward_camera = np.array([0.0, -1.0, 0.0], dtype=np.float32)
            else:
                toward_camera = toward_camera / toward_camera_norm

        helper_up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        if abs(float(np.dot(toward_camera, helper_up))) > 0.95:
            helper_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)

        right = np.cross(helper_up, toward_camera)
        right_norm = float(np.linalg.norm(right))
        if not math.isfinite(right_norm) or right_norm <= 1e-6:
            right = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        else:
            right = right / right_norm

        up = np.cross(toward_camera, right)
        up_norm = float(np.linalg.norm(up))
        if not math.isfinite(up_norm) or up_norm <= 1e-6:
            up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        else:
            up = up / up_norm

        scale = radius * float(radius_scale)
        position = center + (
            scale
            * (
                (float(offset[0]) * right)
                + (float(offset[1]) * toward_camera)
                + (float(offset[2]) * up)
            )
        )
        return position.astype(np.float32), center.astype(np.float32), radius

    def _set_light_pose(
        root_node: Any,
        light_node: Any,
        offset: tuple[float, float, float],
        radius_scale: float = 4.0,
        aim_at_center: bool = True,
    ) -> None:
        position, center, radius = _camera_relative_position(
            root_node=root_node,
            offset=offset,
            radius_scale=radius_scale,
        )
        light_node.setPos(tuple(position.tolist()))
        if aim_at_center:
            light_node.lookAt(float(center[0]), float(center[1]), float(center[2]))
        light = light_node.node()
        lens = light.get_lens() if hasattr(light, "get_lens") else None
        if lens is not None:
            distance = float(np.linalg.norm(position - center))
            lens.set_near_far(
                max(0.01, radius * 0.30),
                max(radius * 10.0, distance + (radius * 3.5)),
            )

    def _camera_aligned_light_position(
        root_node: Any,
        distance_scale: float = 4.8,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        center, radius, _ = _scene_center_radius_camera(root_node)
        position, _, _ = _camera_relative_position(
            root_node=root_node,
            offset=(0.0, 1.0, 0.0),
            radius_scale=distance_scale,
        )
        return position.astype(np.float32), center.astype(np.float32), radius

    def _fill_left_light_position(root_node: Any, light_node: Any) -> None:
        _set_light_pose(
            root_node=root_node,
            light_node=light_node,
            offset=(-0.95, 1.05, 0.45),
            radius_scale=4.4,
            aim_at_center=True,
        )

    def _fill_right_light_position(root_node: Any, light_node: Any) -> None:
        _set_light_pose(
            root_node=root_node,
            light_node=light_node,
            offset=(0.95, 1.05, 0.45),
            radius_scale=4.4,
            aim_at_center=True,
        )

    def _frontal_light_position(root_node: Any, light_node: Any) -> None:
        position, center, radius = _camera_aligned_light_position(
            root_node=root_node,
            distance_scale=4.6,
        )
        light_node.setPos(tuple(position.tolist()))
        light_node.lookAt(float(center[0]), float(center[1]), float(center[2]))
        light = light_node.node()
        lens = light.get_lens() if hasattr(light, "get_lens") else None
        if lens is not None:
            distance = float(np.linalg.norm(position - center))
            lens.set_near_far(
                max(0.01, radius * 0.30),
                max(radius * 10.0, distance + (radius * 3.5)),
            )

    def _position_axis_point_light(
        root_node: Any,
        light_node: Any,
        offset: tuple[float, float, float],
        radius_scale: float = 4.0,
    ) -> None:
        _set_light_pose(
            root_node=root_node,
            light_node=light_node,
            offset=offset,
            radius_scale=4.0,
            aim_at_center=False,
        )

    def _pos_x_light_position(root_node: Any, light_node: Any) -> None:
        _position_axis_point_light(root_node, light_node, offset=(1.0, 0.0, 0.0))

    def _neg_x_light_position(root_node: Any, light_node: Any) -> None:
        _position_axis_point_light(root_node, light_node, offset=(-1.0, 0.0, 0.0))

    def _pos_y_light_position(root_node: Any, light_node: Any) -> None:
        _position_axis_point_light(root_node, light_node, offset=(0.0, 1.0, 0.0))

    def _neg_y_light_position(root_node: Any, light_node: Any) -> None:
        _position_axis_point_light(root_node, light_node, offset=(0.0, -1.0, 0.0))

    def _pos_z_light_position(root_node: Any, light_node: Any) -> None:
        _position_axis_point_light(root_node, light_node, offset=(0.0, 0.0, 1.0))

    def _neg_z_light_position(root_node: Any, light_node: Any) -> None:
        _position_axis_point_light(root_node, light_node, offset=(0.0, 0.0, -1.0))

    def _back_rim_light_position(root_node: Any, light_node: Any) -> None:
        # Placed behind and above the object to reveal silhouette and edges.
        _set_light_pose(
            root_node=root_node,
            light_node=light_node,
            offset=(0.0, -1.2, 0.8),
            radius_scale=4.2,
            aim_at_center=True,
        )

    return [
        Panda3dLightData(
            light_type="point",
            color=(0.32, 0.32, 0.32, 1.0),
            positioning_function=_pos_x_light_position,
            attenuation=(1.0, 0.0, 0.04),
        ),
        Panda3dLightData(
            light_type="point",
            color=(0.32, 0.32, 0.32, 1.0),
            positioning_function=_neg_x_light_position,
            attenuation=(1.0, 0.0, 0.04),
        ),
        Panda3dLightData(
            light_type="point",
            color=(0.32, 0.32, 0.32, 1.0),
            positioning_function=_pos_y_light_position,
            attenuation=(1.0, 0.0, 0.04),
        ),
        Panda3dLightData(
            light_type="point",
            color=(0.32, 0.32, 0.32, 1.0),
            positioning_function=_neg_y_light_position,
            attenuation=(1.0, 0.0, 0.04),
        ),
        Panda3dLightData(
            light_type="point",
            color=(0.32, 0.32, 0.32, 1.0),
            positioning_function=_pos_z_light_position,
            attenuation=(1.0, 0.0, 0.04),
        ),
        Panda3dLightData(
            light_type="point",
            color=(0.32, 0.32, 0.32, 1.0),
            positioning_function=_neg_z_light_position,
            attenuation=(1.0, 0.0, 0.04),
        ),
    ]


def _resolve_megapose_src_dir(source_root: Path) -> Path:
    candidates = [
        source_root / "third_party" / "megapose6d" / "src",
        source_root / "third_party" / "src",
        source_root / "src",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("megapose_source_missing")


def _estimator_cache_scope(device: Any, renderer_workers: Any = 0) -> str:
    device_type = str(getattr(device, "type", device)).strip().lower()
    try:
        renderer_workers_count = int(renderer_workers or 0)
    except (TypeError, ValueError):
        renderer_workers_count = 0
    # Panda3D renderers created in-process are thread-affine on this stack.
    # Prewarm runs on FastAPI request threads while inference runs on the module's
    # dedicated worker thread, so sharing a direct renderer across threads can
    # produce empty GPU readback buffers during MegaPose refinement.
    if renderer_workers_count == 0:
        return f"thread:{threading.get_ident()}"
    if platform.system().lower().startswith("win") and device_type == "cpu":
        return f"thread:{threading.get_ident()}"
    return "shared"


@dataclass(frozen=True)
class ResolvedObjectAssets:
    object_folder: Path
    label: str
    mesh_path: Path
    segmentation_model_path: Path


@dataclass
class EstimatorBundle:
    modules: dict[str, Any]
    object_dataset: Any
    pose_estimator: Any
    mesh_obj_path: Path
    device: Any
    selected_model: str
    pose_lock: threading.Lock
    cache_key: tuple[Any, ...]
    object_runtime_config: dict[str, Any]


def _safe_slug(raw: str) -> str:
    value = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(raw))
    return value or "object"


def _clean_path_string(raw_path: str | Path | None) -> str:
    value = str(raw_path or "").strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        value = value[1:-1].strip()
    return value


def _iter_unique_paths(paths: Iterable[Path]) -> Iterable[Path]:
    seen: set[str] = set()
    for path in paths:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        yield path


def resolve_workspace_path(
    raw_path: str | Path,
    workspace_root: Path = WORKSPACE_ROOT,
    extra_roots: Optional[Iterable[Path]] = None,
) -> Path:
    candidate = Path(_clean_path_string(raw_path))
    if candidate.is_absolute():
        return candidate.resolve()

    search_roots = [workspace_root, workspace_root / "data", Path.cwd()]
    if extra_roots:
        search_roots.extend(extra_roots)
    for root in _iter_unique_paths(Path(p) for p in search_roots):
        resolved = (root / candidate).resolve()
        if resolved.exists():
            return resolved
    return (workspace_root / candidate).resolve()


def _startup_prewarm_params_path(params: Optional[Dict[str, Any]] = None) -> Path:
    raw_value = None
    if isinstance(params, dict):
        raw_value = params.get("startup_prewarm_params_path")
    if raw_value:
        return resolve_workspace_path(raw_value, workspace_root=WORKSPACE_ROOT)
    return DEFAULT_STARTUP_PREWARM_PARAMS_PATH


def load_persisted_megapose_startup_params(
    default_params: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    params_path = _startup_prewarm_params_path(default_params)
    if not params_path.exists() or not params_path.is_file():
        return None
    try:
        loaded = json.loads(params_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(loaded, dict):
        return None
    merged = dict(loaded)
    merged.update(default_params or {})
    if not _clean_path_string(merged.get("object_folder")):
        return None
    return merged


def persist_megapose_startup_params(params: Dict[str, Any]) -> Path:
    params_path = _startup_prewarm_params_path(params)
    params_path.parent.mkdir(parents=True, exist_ok=True)
    serializable = dict(params or {})
    params_path.write_text(
        json.dumps(serializable, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return params_path


def _resolve_named_file(
    folder: Path,
    base_names: list[str],
    extensions: list[str],
) -> Optional[Path]:
    for base_name in base_names:
        for ext in extensions:
            candidate = folder / f"{base_name}{ext}"
            if candidate.exists():
                return candidate
    for ext in extensions:
        matches = sorted(folder.glob(f"*{ext}"))
        if len(matches) == 1:
            return matches[0]
    return None


def _load_object_runtime_config(object_folder: Path) -> dict[str, Any]:
    config_path = object_folder / "megapose.json"
    if not config_path.exists():
        return {}
    try:
        loaded = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"megapose_object_config_invalid: {exc}") from exc
    return loaded if isinstance(loaded, dict) else {}


def _axis_vector_from_value(value: Any) -> Optional[np.ndarray]:
    if isinstance(value, str):
        clean = value.strip().lower()
        mapping = {
            "x": np.array([1.0, 0.0, 0.0], dtype=np.float32),
            "+x": np.array([1.0, 0.0, 0.0], dtype=np.float32),
            "-x": np.array([-1.0, 0.0, 0.0], dtype=np.float32),
            "y": np.array([0.0, 1.0, 0.0], dtype=np.float32),
            "+y": np.array([0.0, 1.0, 0.0], dtype=np.float32),
            "-y": np.array([0.0, -1.0, 0.0], dtype=np.float32),
            "z": np.array([0.0, 0.0, 1.0], dtype=np.float32),
            "+z": np.array([0.0, 0.0, 1.0], dtype=np.float32),
            "-z": np.array([0.0, 0.0, -1.0], dtype=np.float32),
        }
        return mapping.get(clean)
    if isinstance(value, (list, tuple, np.ndarray)) and len(value) >= 3:
        axis = np.asarray(value[:3], dtype=np.float32)
        norm = float(np.linalg.norm(axis))
        if norm > 1e-8:
            return axis / norm
    return None


def _build_symmetry_config(
    params: Dict[str, Any],
    object_config: dict[str, Any],
) -> dict[str, Any]:
    symmetry_cfg = object_config.get("symmetry") if isinstance(object_config.get("symmetry"), dict) else {}
    symmetry_type = str(
        params.get("symmetry_type")
        or symmetry_cfg.get("type")
        or ""
    ).strip().lower()
    axis_value = params.get("symmetry_axis", symmetry_cfg.get("axis"))
    axis = _axis_vector_from_value(axis_value)
    samples_raw = params.get("continuous_symmetry_samples", symmetry_cfg.get("samples", 64))
    try:
        samples = max(1, int(samples_raw))
    except Exception:
        samples = 64
    if symmetry_type != "continuous" or axis is None:
        return {"enabled": False, "type": "", "axis": None, "samples": samples}
    return {
        "enabled": True,
        "type": "continuous",
        "axis": axis,
        "samples": samples,
    }


def _build_mesh_export_config(
    params: Dict[str, Any],
    object_config: dict[str, Any],
) -> dict[str, Any]:
    cfg = object_config.get("mesh_export") if isinstance(object_config.get("mesh_export"), dict) else {}

    def _bool_param(name: str, default: bool) -> bool:
        value = params.get(name, cfg.get(name, default))
        if value is None:
            return default
        return bool(value)

    return {
        "recompute_vertex_normals": _bool_param("mesh_recompute_vertex_normals", True),
        "merge_vertices": _bool_param("mesh_merge_vertices", True),
        "process": _bool_param("mesh_process_before_export", True),
    }


def _resolve_pose_render_size(
    params: Dict[str, Any],
    object_config: dict[str, Any],
) -> tuple[int, int]:
    render_cfg = object_config.get("render_size") if isinstance(object_config.get("render_size"), (dict, list, tuple)) else {}

    def _read_dimension(param_key: str, cfg_key: str, default: int) -> int:
        value = params.get(param_key)
        if value is None:
            if isinstance(render_cfg, dict):
                value = render_cfg.get(cfg_key)
            elif isinstance(render_cfg, (list, tuple)) and len(render_cfg) >= 2:
                value = render_cfg[0 if cfg_key == "height" else 1]
        try:
            dim = int(value)
        except Exception:
            dim = default
        return max(64, dim)

    return (
        _read_dimension("crop_render_height", "height", 320),
        _read_dimension("crop_render_width", "width", 448),
    )


def _rotation_angle_between(R_a: np.ndarray, R_b: np.ndarray) -> float:
    relative = np.asarray(R_a, dtype=np.float64).T @ np.asarray(R_b, dtype=np.float64)
    cos_theta = max(-1.0, min(1.0, float((np.trace(relative) - 1.0) * 0.5)))
    return math.degrees(math.acos(cos_theta))


def _make_refiner_guard_info(
    params: Dict[str, Any],
    object_config: dict[str, Any],
    object_extents_m: Optional[np.ndarray] = None,
) -> dict[str, Any]:
    guard_cfg = object_config.get("refiner_guard") if isinstance(object_config.get("refiner_guard"), dict) else {}
    enabled_value = params.get("refiner_guard_enabled", guard_cfg.get("enabled"))
    if enabled_value is None:
        enabled = False
    else:
        enabled = bool(enabled_value)
    try:
        max_translation = float(
            params.get("refiner_guard_max_translation_delta_m", guard_cfg.get("max_translation_delta_m", 0.015))
        )
    except Exception:
        max_translation = 0.015
    try:
        max_rotation = float(
            params.get("refiner_guard_max_rotation_delta_deg", guard_cfg.get("max_rotation_delta_deg", 12.0))
        )
    except Exception:
        max_rotation = 12.0
    fallback = str(
        params.get("refiner_guard_fallback", guard_cfg.get("fallback", "coarse_input"))
    ).strip().lower() or "coarse_input"
    return {
        "enabled": enabled,
        "max_translation_delta_m": max_translation,
        "max_rotation_delta_deg": max_rotation,
        "fallback": fallback,
        "object_extents_m": None if object_extents_m is None else [float(v) for v in object_extents_m.tolist()],
    }


def resolve_object_assets(
    raw_object_folder: str | Path,
    *,
    workspace_root: Path = WORKSPACE_ROOT,
    extra_roots: Optional[Iterable[Path]] = None,
    label_override: str | None = None,
    segmentation_model_override: str | Path | None = None,
    mesh_extension_priority: Optional[list[str]] = None,
) -> ResolvedObjectAssets:
    object_folder = resolve_workspace_path(
        raw_object_folder,
        workspace_root=workspace_root,
        extra_roots=extra_roots,
    )
    if not object_folder.exists():
        raise FileNotFoundError("megapose_object_folder_not_found")
    if not object_folder.is_dir():
        raise NotADirectoryError("megapose_object_folder_invalid")

    folder_name = object_folder.name
    label = str(label_override or folder_name).strip() or folder_name
    object_config = _load_object_runtime_config(object_folder)
    mesh_priority = mesh_extension_priority or [".glb", ".obj", ".stl"]
    mesh_priority = [ext if ext.startswith(".") else f".{ext}" for ext in mesh_priority]

    mesh_override = _clean_path_string(object_config.get("mesh_path"))
    if mesh_override:
        mesh_path = resolve_workspace_path(
            mesh_override,
            workspace_root=workspace_root,
            extra_roots=[object_folder, *(list(extra_roots) if extra_roots else [])],
        )
        if not mesh_path.exists():
            raise FileNotFoundError("megapose_mesh_missing")
    else:
        mesh_path = _resolve_named_file(
            object_folder,
            base_names=["object", folder_name],
            extensions=mesh_priority,
        )
        if mesh_path is None:
            raise FileNotFoundError("megapose_mesh_missing")

    segmentation_override = segmentation_model_override
    if segmentation_override in (None, ""):
        config_segmentation_path = _clean_path_string(object_config.get("segmentation_model_path"))
        if config_segmentation_path:
            segmentation_override = config_segmentation_path

    if segmentation_override not in (None, ""):
        seg_model_path = resolve_workspace_path(
            segmentation_override,
            workspace_root=workspace_root,
            extra_roots=[object_folder, *(list(extra_roots) if extra_roots else [])],
        )
        if not seg_model_path.exists():
            raise FileNotFoundError("megapose_segmentation_model_missing")
    else:
        seg_model_path = _resolve_named_file(
            object_folder,
            base_names=["object", folder_name],
            extensions=[".pt"],
        )
        if seg_model_path is None:
            raise FileNotFoundError("megapose_segmentation_model_missing")

    return ResolvedObjectAssets(
        object_folder=object_folder,
        label=label,
        mesh_path=mesh_path,
        segmentation_model_path=seg_model_path,
    )


def normalize_depth_map(depth: np.ndarray | None, depth_scale: float) -> np.ndarray | None:
    if depth is None:
        return None
    if np.issubdtype(depth.dtype, np.integer):
        return depth.astype(np.float32) * float(depth_scale)
    return depth.astype(np.float32, copy=False)


def filter_depth_range(
    depth_m: np.ndarray | None,
    *,
    min_depth_m: float | None = None,
    max_depth_m: float | None = None,
) -> np.ndarray | None:
    if depth_m is None:
        return None
    depth = np.asarray(depth_m, dtype=np.float32).copy()
    valid = np.isfinite(depth) & (depth > 0.0)
    if min_depth_m is not None and float(min_depth_m) > 0.0:
        valid &= depth >= float(min_depth_m)
    if max_depth_m is not None and float(max_depth_m) > 0.0:
        valid &= depth <= float(max_depth_m)
    depth[~valid] = 0.0
    return depth


def resize_inputs_for_processing(
    rgb: np.ndarray,
    depth_m: np.ndarray | None,
    *,
    max_width: int,
    max_height: int,
) -> tuple[np.ndarray, np.ndarray | None]:
    src_h, src_w = rgb.shape[:2]
    if max_width <= 0 or max_height <= 0:
        return rgb, depth_m
    scale = min(float(max_width) / float(src_w), float(max_height) / float(src_h), 1.0)
    if scale >= 0.999:
        return rgb, depth_m
    dst_w = max(1, int(round(src_w * scale)))
    dst_h = max(1, int(round(src_h * scale)))
    resized_rgb = cv2.resize(rgb, (dst_w, dst_h), interpolation=cv2.INTER_AREA)
    resized_depth = None
    if depth_m is not None:
        resized_depth = cv2.resize(depth_m, (dst_w, dst_h), interpolation=cv2.INTER_NEAREST)
    return resized_rgb, resized_depth


def _parse_intrinsics_resolution(value: Any) -> tuple[float, float] | None:
    if isinstance(value, dict):
        width = value.get("width")
        height = value.get("height")
        if width and height:
            return float(width), float(height)
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        # MegaPose camera_data.json and the RealSense reference flow store
        # resolution as [height, width].
        return float(value[1]), float(value[0])
    return None


def _parse_camera_matrix(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    matrix = np.asarray(value, dtype=np.float32)
    if matrix.shape != (3, 3):
        return None
    if not np.isfinite(matrix).all():
        return None
    return matrix


def build_camera_data(params: Dict[str, Any], frame_shape: tuple[int, int]) -> dict[str, Any]:
    camera_data = params.get("camera_data") or {}
    if not isinstance(camera_data, dict):
        camera_data = {}
    params_intrinsics = params.get("intrinsics") or {}
    if not isinstance(params_intrinsics, dict):
        params_intrinsics = {}
    camera_intrinsics = camera_data.get("intrinsics") or {}
    if not isinstance(camera_intrinsics, dict):
        camera_intrinsics = {}
    intrinsics = params_intrinsics or camera_intrinsics
    matrix = _parse_camera_matrix(params.get("K") or camera_data.get("K"))

    width = float(frame_shape[1])
    height = float(frame_shape[0])
    src_resolution = (
        _parse_intrinsics_resolution(params_intrinsics.get("resolution"))
        or _parse_intrinsics_resolution(params.get("intrinsics_resolution"))
        or _parse_intrinsics_resolution(params.get("intrinsics_image_size"))
        or _parse_intrinsics_resolution(camera_intrinsics.get("resolution"))
        or _parse_intrinsics_resolution(camera_data.get("resolution"))
        or _parse_intrinsics_resolution(camera_data.get("image_size"))
    )
    if matrix is not None:
        fx_v = float(matrix[0, 0])
        fy_v = float(matrix[1, 1])
        cx_v = float(matrix[0, 2])
        cy_v = float(matrix[1, 2])
    else:
        fx = intrinsics.get("fx", params.get("fx"))
        fy = intrinsics.get("fy", params.get("fy"))
        cx = intrinsics.get("cx", params.get("cx"))
        cy = intrinsics.get("cy", params.get("cy"))
        if fx is None or fy is None or cx is None or cy is None:
            raise ValueError("megapose_intrinsics_missing")
        fx_v = float(fx)
        fy_v = float(fy)
        cx_v = float(cx)
        cy_v = float(cy)
    if src_resolution and src_resolution[0] > 0 and src_resolution[1] > 0:
        scale_x = width / float(src_resolution[0])
        scale_y = height / float(src_resolution[1])
        fx_v *= scale_x
        fy_v *= scale_y
        cx_v *= scale_x
        cy_v *= scale_y

    return {
        "K": np.array(
            [
                [fx_v, 0.0, cx_v],
                [0.0, fy_v, cy_v],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        ),
        "resolution": [int(height), int(width)],
        "intrinsics": {
            "fx": fx_v,
            "fy": fy_v,
            "cx": cx_v,
            "cy": cy_v,
            "resolution": {"width": int(width), "height": int(height)},
        },
    }


def voxel_downsample_points(
    points: np.ndarray,
    voxel_size_m: float,
) -> np.ndarray:
    """Voxel-grid downsample: keep one representative point per voxel cell.

    Used to bound payload size when attaching scene point clouds to match dicts
    that cross the ZMQ/JSON transport to the orchestrator.
    """
    if points is None or len(points) == 0 or voxel_size_m <= 0.0:
        return points if points is not None else np.zeros((0, 3), dtype=np.float32)
    pts = np.asarray(points, dtype=np.float32)
    finite = np.all(np.isfinite(pts), axis=1)
    pts = pts[finite]
    if len(pts) == 0:
        return np.zeros((0, 3), dtype=np.float32)
    keys = np.floor(pts / float(voxel_size_m)).astype(np.int64)
    # Deduplicate voxel keys; take first representative per cell.
    _, unique_idx = np.unique(keys, axis=0, return_index=True)
    return pts[np.sort(unique_idx)]


def depth_to_point_cloud_points(
    depth_m: np.ndarray | None,
    K: np.ndarray,
    mask: np.ndarray | None = None,
    max_depth_m: float | None = None,
) -> np.ndarray:
    if depth_m is None:
        return np.zeros((0, 3), dtype=np.float32)

    valid_mask = np.isfinite(depth_m) & (depth_m > 0)
    if max_depth_m is not None and float(max_depth_m) > 0.0:
        valid_mask &= depth_m <= float(max_depth_m)
    if mask is not None:
        valid_mask &= mask.astype(bool)

    ys, xs = np.where(valid_mask)
    if len(xs) == 0:
        return np.zeros((0, 3), dtype=np.float32)

    z = depth_m[ys, xs].astype(np.float32)
    fx = float(K[0, 0])
    fy = float(K[1, 1])
    cx = float(K[0, 2])
    cy = float(K[1, 2])
    x = (xs.astype(np.float32) - cx) * z / fx
    y = (ys.astype(np.float32) - cy) * z / fy
    return np.column_stack((x, y, z)).astype(np.float32)


def depth_to_point_cloud(
    rgb: np.ndarray,
    depth_m: np.ndarray | None,
    K: np.ndarray,
    mask: np.ndarray | None = None,
    max_depth_m: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if depth_m is None:
        return (
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0, 4), dtype=np.uint8),
        )

    valid_mask = np.isfinite(depth_m) & (depth_m > 0)
    if max_depth_m is not None and float(max_depth_m) > 0.0:
        valid_mask &= depth_m <= float(max_depth_m)
    if mask is not None:
        valid_mask &= mask.astype(bool)

    ys, xs = np.where(valid_mask)
    if len(xs) == 0:
        return (
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0, 4), dtype=np.uint8),
        )

    z = depth_m[ys, xs].astype(np.float32)
    fx = float(K[0, 0])
    fy = float(K[1, 1])
    cx = float(K[0, 2])
    cy = float(K[1, 2])
    x = (xs.astype(np.float32) - cx) * z / fx
    y = (ys.astype(np.float32) - cy) * z / fy
    points = np.column_stack((x, y, z)).astype(np.float32)

    colors_rgb = rgb[ys, xs].astype(np.uint8)
    alpha = np.full((colors_rgb.shape[0], 1), 255, dtype=np.uint8)
    colors_rgba = np.concatenate((colors_rgb, alpha), axis=1)
    return points, colors_rgba


def compute_object_center_world(
    vertices_object: np.ndarray,
    pose_matrix: np.ndarray,
    object_center_object: np.ndarray | None = None,
) -> np.ndarray:
    if object_center_object is None:
        center_object = 0.5 * (vertices_object.min(axis=0) + vertices_object.max(axis=0))
    else:
        center_object = np.asarray(object_center_object, dtype=np.float64).reshape(3)
    center_object_h = np.concatenate((center_object.astype(np.float64), np.array([1.0])))
    center_world_h = pose_matrix.astype(np.float64) @ center_object_h
    return center_world_h[:3]


def subsample_points(points: np.ndarray, max_points: int, rng: Any) -> np.ndarray:
    if len(points) <= max_points:
        return points
    indices = rng.choice(len(points), size=max_points, replace=False)
    return points[indices]


def batched_nearest_neighbors(
    source_points: np.ndarray,
    target_points: np.ndarray,
    batch_size: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    nearest_indices = np.empty(len(source_points), dtype=np.int32)
    nearest_distances = np.empty(len(source_points), dtype=np.float64)
    for start_idx in range(0, len(source_points), batch_size):
        end_idx = min(start_idx + batch_size, len(source_points))
        batch = source_points[start_idx:end_idx]
        deltas = batch[:, None, :] - target_points[None, :, :]
        distances = np.linalg.norm(deltas, axis=2)
        batch_indices = np.argmin(distances, axis=1)
        nearest_indices[start_idx:end_idx] = batch_indices.astype(np.int32)
        nearest_distances[start_idx:end_idx] = distances[
            np.arange(len(batch_indices)),
            batch_indices,
        ]
    return nearest_indices, nearest_distances


def filter_front_surface_layer_points(
    points: np.ndarray,
    layer_thickness_m: float,
    center_keep_ratio: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    if len(points) == 0:
        return points, {"input_count": 0, "layer_count": 0, "center_count": 0}

    layer_thickness_m = max(1e-4, float(layer_thickness_m))
    center_keep_ratio = float(np.clip(center_keep_ratio, 0.05, 1.0))

    z_min = float(np.min(points[:, 2]))
    layer_mask = points[:, 2] <= (z_min + layer_thickness_m)
    layer_points = points[layer_mask]
    if len(layer_points) == 0:
        return points, {
            "input_count": int(len(points)),
            "layer_count": 0,
            "center_count": 0,
            "layer_z_min_m": z_min,
            "layer_thickness_m": layer_thickness_m,
        }

    center_points = layer_points
    if center_keep_ratio < 0.999 and len(layer_points) >= 32:
        center_xy = np.median(layer_points[:, :2], axis=0)
        radii = np.linalg.norm(layer_points[:, :2] - center_xy[None, :], axis=1)
        radius_limit = float(np.quantile(radii, center_keep_ratio))
        keep_mask = radii <= radius_limit
        if int(np.sum(keep_mask)) >= 16:
            center_points = layer_points[keep_mask]

    return center_points, {
        "input_count": int(len(points)),
        "layer_count": int(len(layer_points)),
        "center_count": int(len(center_points)),
        "layer_z_min_m": z_min,
        "layer_thickness_m": layer_thickness_m,
        "center_keep_ratio": center_keep_ratio,
    }


def refine_pose_distance_along_center_ray(
    mesh_path: Path,
    pose_matrix: np.ndarray,
    observed_points: np.ndarray | None,
    mesh_units: str,
    mesh_scale: float,
    iterations: int,
    mesh_samples: int,
    observed_samples: int,
    max_correspondence_m: float,
    max_shift_m: float,
    front_layer_thickness_m: float,
    center_keep_ratio: float,
) -> tuple[np.ndarray, dict[str, Any] | None]:
    import trimesh

    if observed_points is None or len(observed_points) == 0 or iterations <= 0:
        return pose_matrix, {
            "enabled": iterations > 0,
            "applied": False,
            "skip_reason": (
                "no_observed_points"
                if observed_points is None or len(observed_points) == 0
                else "iterations_disabled"
            ),
            "shift_along_center_ray_m": 0.0,
            "iterations": [],
        }

    mesh = trimesh.load_mesh(mesh_path, force="mesh")
    unit_scale = 1.0 if str(mesh_units).strip().lower() == "m" else 0.001
    mesh.apply_scale(unit_scale * float(mesh_scale))

    rng = np.random.default_rng(0)
    observed_points = subsample_points(
        observed_points.astype(np.float64),
        max(128, int(observed_samples)),
        rng,
    )
    observed_points, observed_filter_stats = filter_front_surface_layer_points(
        observed_points,
        layer_thickness_m=front_layer_thickness_m,
        center_keep_ratio=center_keep_ratio,
    )
    if len(observed_points) < 32:
        return pose_matrix, {
            "enabled": True,
            "applied": False,
            "skip_reason": "too_few_observed_front_points",
            "shift_along_center_ray_m": 0.0,
            "observed_front_layer": observed_filter_stats,
            "iterations": [],
        }

    refined_pose = pose_matrix.astype(np.float64).copy()
    total_shift_m = 0.0
    iteration_summaries: list[dict[str, Any]] = []
    skip_reason = None

    for iteration_idx in range(max(0, int(iterations))):
        sampled_points_object, face_indices = trimesh.sample.sample_surface(
            mesh,
            count=max(256, int(mesh_samples)),
        )
        sampled_points_object = sampled_points_object.astype(np.float64)
        face_normals_object = mesh.face_normals[face_indices].astype(np.float64)

        rotation = refined_pose[:3, :3]
        translation = refined_pose[:3, 3]
        sampled_points_camera = (rotation @ sampled_points_object.T).T + translation
        sampled_normals_camera = (rotation @ face_normals_object.T).T

        camera_to_surface = -sampled_points_camera
        visible_mask = (
            (sampled_points_camera[:, 2] > 0.0)
            & (np.einsum("ij,ij->i", sampled_normals_camera, camera_to_surface) > 0.0)
        )
        visible_points = sampled_points_camera[visible_mask]
        visible_points, visible_filter_stats = filter_front_surface_layer_points(
            visible_points,
            layer_thickness_m=front_layer_thickness_m,
            center_keep_ratio=center_keep_ratio,
        )
        if len(visible_points) < 32:
            skip_reason = "too_few_model_front_points"
            break

        center_world = compute_object_center_world(mesh.vertices.copy(), refined_pose)
        center_norm = float(np.linalg.norm(center_world))
        if center_norm < 1e-8:
            skip_reason = "degenerate_center_ray"
            break
        center_ray = center_world / center_norm

        nearest_indices, nearest_distances = batched_nearest_neighbors(
            visible_points,
            observed_points,
        )
        inlier_mask = nearest_distances <= float(max_correspondence_m)
        if int(inlier_mask.sum()) < 32:
            skip_reason = "too_few_inliers"
            break

        visible_inliers = visible_points[inlier_mask]
        observed_inliers = observed_points[nearest_indices[inlier_mask]]
        signed_offsets = np.einsum("ij,j->i", observed_inliers - visible_inliers, center_ray)
        shift_m = float(np.median(signed_offsets))

        remaining_positive = float(max_shift_m) - total_shift_m
        remaining_negative = -float(max_shift_m) - total_shift_m
        shift_m = float(np.clip(shift_m, remaining_negative, remaining_positive))
        if abs(shift_m) < 1e-5:
            iteration_summaries.append(
                {
                    "iteration": iteration_idx + 1,
                    "visible_points": int(len(visible_points)),
                    "inliers": int(inlier_mask.sum()),
                    "median_shift_m": shift_m,
                    "model_front_layer": visible_filter_stats,
                }
            )
            break

        refined_pose[:3, 3] += center_ray * shift_m
        total_shift_m += shift_m
        iteration_summaries.append(
            {
                "iteration": iteration_idx + 1,
                "visible_points": int(len(visible_points)),
                "inliers": int(inlier_mask.sum()),
                "median_shift_m": shift_m,
                "model_front_layer": visible_filter_stats,
            }
        )

        if abs(total_shift_m) >= float(max_shift_m):
            break

    if not iteration_summaries:
        return pose_matrix, {
            "enabled": True,
            "applied": False,
            "skip_reason": skip_reason or "no_valid_refinement_step",
            "shift_along_center_ray_m": 0.0,
            "observed_front_layer": observed_filter_stats,
            "iterations": [],
            "max_correspondence_m": float(max_correspondence_m),
            "max_shift_m": float(max_shift_m),
            "front_layer_thickness_m": float(front_layer_thickness_m),
            "center_keep_ratio": float(center_keep_ratio),
        }

    return refined_pose.astype(np.float32), {
        "enabled": True,
        "applied": bool(abs(total_shift_m) > 0.0),
        "shift_along_center_ray_m": float(total_shift_m),
        "observed_front_layer": observed_filter_stats,
        "iterations": iteration_summaries,
        "max_correspondence_m": float(max_correspondence_m),
        "max_shift_m": float(max_shift_m),
        "front_layer_thickness_m": float(front_layer_thickness_m),
        "center_keep_ratio": float(center_keep_ratio),
    }


def refine_pose_distance_from_rendered_depth_overlap(
    pose_matrix: np.ndarray,
    depth_m: np.ndarray | None,
    selection_mask: np.ndarray | None,
    K: np.ndarray,
    resolution: tuple[int, int],
    label: str,
    modules: dict[str, Any],
    pose_estimator: Any,
    max_shift_m: float,
    front_layer_thickness_m: float,
    center_keep_ratio: float,
) -> tuple[np.ndarray, dict[str, Any] | None]:
    import torch

    if depth_m is None or selection_mask is None:
        return pose_matrix, None

    renderer = pose_estimator.coarse_model.renderer
    TCO = torch.as_tensor(pose_matrix, dtype=torch.float32).unsqueeze(0)
    K_tensor = torch.as_tensor(K, dtype=torch.float32).unsqueeze(0)
    light_datas = [[modules["Panda3dLightData"](light_type="ambient", color=(1.0, 1.0, 1.0, 1.0))]]
    render_output = renderer.render(
        labels=[label],
        TCO=TCO,
        K=K_tensor,
        light_datas=light_datas,
        resolution=resolution,
        render_depth=True,
        render_mask=False,
        render_normals=False,
    )
    rendered_depth = render_output.depths[0].detach().cpu().permute(1, 2, 0).numpy()[..., 0]
    rendered_valid = np.isfinite(rendered_depth) & (rendered_depth > 0)
    observed_valid = np.isfinite(depth_m) & (depth_m > 0) & selection_mask.astype(bool)
    overlap_mask = rendered_valid & observed_valid
    if int(np.sum(overlap_mask)) < 32:
        return pose_matrix, {
            "enabled": True,
            "applied": False,
            "method": "rendered_depth_overlap_camera_z",
            "skip_reason": "too_few_overlap_pixels",
            "shift_along_camera_z_m": 0.0,
            "overlap_pixel_count": int(np.sum(overlap_mask)),
        }

    z_min = float(np.min(rendered_depth[overlap_mask]))
    layer_mask = overlap_mask & (
        rendered_depth <= (z_min + max(1e-4, float(front_layer_thickness_m)))
    )
    ys, xs = np.where(layer_mask)
    if len(xs) < 32:
        return pose_matrix, {
            "enabled": True,
            "applied": False,
            "method": "rendered_depth_overlap_camera_z",
            "skip_reason": "too_few_front_overlap_pixels",
            "shift_along_camera_z_m": 0.0,
            "overlap_pixel_count": int(np.sum(overlap_mask)),
            "front_overlap_pixel_count": int(len(xs)),
        }

    keep_ratio = float(np.clip(center_keep_ratio, 0.05, 1.0))
    if keep_ratio < 0.999 and len(xs) >= 32:
        center_xy = np.array([np.median(xs), np.median(ys)], dtype=np.float64)
        radii = np.sqrt((xs - center_xy[0]) ** 2 + (ys - center_xy[1]) ** 2)
        radius_limit = float(np.quantile(radii, keep_ratio))
        keep = radii <= radius_limit
        xs = xs[keep]
        ys = ys[keep]

    if len(xs) < 32:
        return pose_matrix, {
            "enabled": True,
            "applied": False,
            "method": "rendered_depth_overlap_camera_z",
            "skip_reason": "too_few_center_overlap_pixels",
            "shift_along_camera_z_m": 0.0,
            "overlap_pixel_count": int(np.sum(overlap_mask)),
            "front_overlap_pixel_count": int(np.sum(layer_mask)),
            "center_overlap_pixel_count": int(len(xs)),
        }

    depth_residuals = depth_m[ys, xs] - rendered_depth[ys, xs]
    shift_m = float(np.median(depth_residuals))
    shift_m = float(np.clip(shift_m, -float(max_shift_m), float(max_shift_m)))

    refined_pose = pose_matrix.astype(np.float64).copy()
    refined_pose[2, 3] += shift_m
    return refined_pose.astype(np.float32), {
        "enabled": True,
        "applied": bool(abs(shift_m) > 0.0),
        "method": "rendered_depth_overlap_camera_z",
        "shift_along_camera_z_m": shift_m,
        "overlap_pixel_count": int(np.sum(overlap_mask)),
        "front_overlap_pixel_count": int(np.sum(layer_mask)),
        "center_overlap_pixel_count": int(len(xs)),
        "front_layer_thickness_m": float(front_layer_thickness_m),
        "center_keep_ratio": float(keep_ratio),
    }


def _normalize_metric_map(records: list[dict[str, Any]], key: str) -> dict[int, float]:
    values: list[tuple[int, float]] = []
    for idx, record in enumerate(records):
        try:
            value = float(record.get(key, 0.0))
        except Exception:
            value = 0.0
        if not math.isfinite(value):
            value = 0.0
        values.append((idx, value))
    if not values:
        return {}
    raw = np.asarray([value for _, value in values], dtype=np.float32)
    lo = float(np.min(raw))
    hi = float(np.max(raw))
    if hi - lo < 1e-8:
        return {idx: 1.0 for idx, _ in values}
    return {idx: float((value - lo) / (hi - lo)) for idx, value in values}


def rerank_pose_hypotheses(
    *,
    hypotheses: Any,
    selection_mask: np.ndarray | None,
    depth_selection_mask: np.ndarray | None,
    depth_m: np.ndarray | None,
    K: np.ndarray,
    resolution: tuple[int, int],
    label: str,
    modules: dict[str, Any],
    pose_estimator: Any,
    top_k: int,
    min_depth_overlap_pixels: int = 64,
    depth_error_scale_m: float = 0.004,
    mask_iou_weight: float = 0.30,
    mask_coverage_weight: float = 0.12,
    depth_weight: float = 0.28,
    network_prior_weight: float = 1.0,
) -> dict[str, Any] | None:
    if hypotheses is None:
        return None

    entries = _collect_pose_entries(
        hypotheses,
        score_field="pose_logit",
        limit=max(1, int(top_k)),
    )
    if len(entries) <= 1:
        return None

    import torch

    pose_matrices = np.stack(
        [np.asarray(entry["pose_matrix"], dtype=np.float32) for entry in entries],
        axis=0,
    )
    renderer = pose_estimator.coarse_model.renderer
    TCO = torch.as_tensor(pose_matrices, dtype=torch.float32)
    K_tensor = torch.as_tensor(
        np.repeat(np.asarray(K, dtype=np.float32)[None, :, :], len(entries), axis=0),
        dtype=torch.float32,
    )
    light_datas = [[modules["Panda3dLightData"](light_type="ambient", color=(1.0, 1.0, 1.0, 1.0))]]
    light_datas = light_datas * len(entries)
    render_output = renderer.render(
        labels=[label] * len(entries),
        TCO=TCO,
        K=K_tensor,
        light_datas=light_datas,
        resolution=resolution,
        render_depth=True,
        render_mask=False,
        render_normals=False,
    )

    seg_mask = None if selection_mask is None else selection_mask.astype(bool)
    observed_depth_mask = None
    if depth_m is not None:
        observed_depth_mask = np.isfinite(depth_m) & (depth_m > 0)
        if depth_selection_mask is not None:
            observed_depth_mask &= depth_selection_mask.astype(bool)

    pose_norm = _normalize_metric_map(entries, "pose_logit")
    coarse_norm = _normalize_metric_map(entries, "coarse_logit")
    reranked: list[dict[str, Any]] = []

    for idx, entry in enumerate(entries):
        rendered_depth = render_output.depths[idx].detach().cpu().permute(1, 2, 0).numpy()[..., 0]
        rendered_mask = np.isfinite(rendered_depth) & (rendered_depth > 0)
        render_pixels = int(np.count_nonzero(rendered_mask))

        mask_iou = 0.0
        mask_coverage = 0.0
        if seg_mask is not None:
            intersection = int(np.count_nonzero(rendered_mask & seg_mask))
            union = int(np.count_nonzero(rendered_mask | seg_mask))
            seg_pixels = int(np.count_nonzero(seg_mask))
            mask_iou = float(intersection / union) if union > 0 else 0.0
            mask_coverage = float(intersection / seg_pixels) if seg_pixels > 0 else 0.0

        depth_overlap_pixels = 0
        depth_mae_m = None
        depth_score = 0.0
        if observed_depth_mask is not None:
            overlap_mask = rendered_mask & observed_depth_mask
            depth_overlap_pixels = int(np.count_nonzero(overlap_mask))
            if depth_overlap_pixels >= max(8, int(min_depth_overlap_pixels)):
                residual = np.abs(depth_m[overlap_mask] - rendered_depth[overlap_mask])
                depth_mae_m = float(np.median(residual))
                scale = max(1e-4, float(depth_error_scale_m))
                depth_score = float(math.exp(-depth_mae_m / scale))

        mask_iou_w = max(0.0, float(mask_iou_weight))
        mask_coverage_w = max(0.0, float(mask_coverage_weight))
        depth_w = max(0.0, float(depth_weight))
        network_weight = max(0.0, float(network_prior_weight))
        rerank_score = (
            (mask_iou_w * mask_iou)
            + (mask_coverage_w * mask_coverage)
            + (depth_w * depth_score)
            + network_weight
            * (
                (0.25 * pose_norm.get(idx, 0.0))
                + (0.05 * coarse_norm.get(idx, 0.0))
            )
        )

        ranked_entry = dict(entry)
        ranked_entry["rerank_score"] = float(rerank_score)
        ranked_entry["rerank_metrics"] = {
            "mask_iou": float(mask_iou),
            "mask_coverage": float(mask_coverage),
            "depth_overlap_pixels": int(depth_overlap_pixels),
            "depth_mae_m": None if depth_mae_m is None else float(depth_mae_m),
            "depth_score": float(depth_score),
            "pose_logit_norm": float(pose_norm.get(idx, 0.0)),
            "coarse_logit_norm": float(coarse_norm.get(idx, 0.0)),
            "mask_iou_weight": float(mask_iou_w),
            "mask_coverage_weight": float(mask_coverage_w),
            "depth_weight": float(depth_w),
            "network_prior_weight": float(network_weight),
            "render_pixels": int(render_pixels),
        }
        reranked.append(ranked_entry)

    reranked.sort(key=lambda entry: float(entry.get("rerank_score", float("-inf"))), reverse=True)
    best = reranked[0] if reranked else None
    if best is None:
        return None
    return {
        "applied": True,
        "top_k": int(top_k),
        "selected_hypothesis_id": _entry_hypothesis_id(best),
        "selected_row_index": int(best.get("_row_index", 0)),
        "selected_pose_matrix": np.asarray(best["pose_matrix"], dtype=np.float32).tolist(),
        "selected_pose_logit": float(best.get("pose_logit", 0.0) or 0.0),
        "selected_coarse_logit": float(best.get("coarse_logit", 0.0) or 0.0),
        "selected_rerank_score": float(best.get("rerank_score", 0.0)),
        "candidates": [
            {
                "hypothesis_id": _entry_hypothesis_id(entry),
                "pose_logit": _json_safe(entry.get("pose_logit")),
                "coarse_logit": _json_safe(entry.get("coarse_logit")),
                "rerank_score": float(entry.get("rerank_score", 0.0)),
                "metrics": entry.get("rerank_metrics") or {},
            }
            for entry in reranked
        ],
    }


def _bbox_xyxy_to_xywh(bbox_xyxy: np.ndarray) -> list[float]:
    x1, y1, x2, y2 = bbox_xyxy.tolist()
    return [float(x1), float(y1), float(max(0.0, x2 - x1)), float(max(0.0, y2 - y1))]


def _bbox_from_mask(mask: np.ndarray) -> np.ndarray | None:
    ys, xs = np.where(mask.astype(bool))
    if len(xs) == 0 or len(ys) == 0:
        return None
    return np.array([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], dtype=np.float32)


def resolve_segmentation_backend(backend: str | None, model_path: Path) -> str:
    raw_backend = str(backend or "auto").strip().lower()
    if raw_backend in {"", "auto"}:
        stem = model_path.stem.strip().lower()
        return "sam" if stem.startswith("sam") or "sam2" in stem or "sam3" in stem else "yolo"
    if raw_backend in {"yolo", "yolov8", "yolov11"}:
        return "yolo"
    if raw_backend in {"sam", "sam2", "sam3", "ultralytics_sam"}:
        return "sam"
    raise ValueError(f"unsupported_segmentation_backend:{backend}")


def get_yolo_model(model_path: Path) -> Any:
    key = str(model_path.resolve())
    with _YOLO_CACHE_LOCK:
        model = _YOLO_MODELS.get(key)
        if model is not None:
            return model
        from ultralytics import YOLO

        model = YOLO(str(model_path))
        _YOLO_MODELS[key] = model
        return model


def get_sam_model(model_path: Path) -> Any:
    key = str(model_path.resolve())
    with _SAM_CACHE_LOCK:
        model = _SAM_MODELS.get(key)
        if model is not None:
            return model
        from ultralytics import SAM

        model = SAM(str(model_path))
        _SAM_MODELS[key] = model
        return model


def get_segmentation_model(model_path: Path, backend: str) -> Any:
    resolved_backend = resolve_segmentation_backend(backend, model_path)
    if resolved_backend == "sam":
        return get_sam_model(model_path)
    return get_yolo_model(model_path)


def _warmup_segmentation_predict(
    model: Any,
    backend: str,
    image_shape: tuple[int, int],
    conf: float,
    device: str | None,
    imgsz: int | None = None,
    retina_masks: bool = True,
) -> None:
    height, width = image_shape
    blank = np.zeros((int(height), int(width), 3), dtype=np.uint8)
    infer_lock = _SAM_INFER_LOCK if backend == "sam" else _YOLO_INFER_LOCK
    with infer_lock:
        if backend == "sam":
            model.predict(
                source=blank,
                device=device,
                imgsz=int(imgsz) if imgsz else 1024,
                conf_thres=float(conf),
                verbose=False,
            )
        else:
            model.predict(
                source=blank,
                conf=float(conf),
                device=device,
                imgsz=int(imgsz) if imgsz else None,
                retina_masks=bool(retina_masks),
                verbose=False,
            )


def run_yolo_segmentation(
    rgb: np.ndarray,
    model_path: Path,
    conf: float,
    device: str | None,
    imgsz: int | None = None,
    retina_masks: bool = True,
) -> list[dict[str, Any]]:
    if not model_path.exists():
        raise FileNotFoundError("megapose_segmentation_model_missing")
    model = get_yolo_model(model_path)
    with _YOLO_INFER_LOCK:
        result = model.predict(
            source=rgb,
            conf=float(conf),
            device=device,
            imgsz=int(imgsz) if imgsz else None,
            retina_masks=bool(retina_masks),
            verbose=False,
        )[0]
    if result.boxes is None or len(result.boxes) == 0:
        return []

    scores = result.boxes.conf.detach().cpu().numpy()
    order = np.argsort(-scores)
    detections: list[dict[str, Any]] = []
    for rank, raw_idx in enumerate(order.tolist()):
        idx = int(raw_idx)
        bbox_xyxy = result.boxes.xyxy[idx].detach().cpu().numpy().astype(np.float32)
        mask = None
        if result.masks is not None and result.masks.data is not None:
            mask = result.masks.data[idx].detach().cpu().numpy() > 0.5
            if mask.shape[:2] != rgb.shape[:2]:
                mask = cv2.resize(
                    mask.astype(np.uint8),
                    (int(rgb.shape[1]), int(rgb.shape[0])),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(bool)
            mask_bbox = _bbox_from_mask(mask)
            if mask_bbox is not None:
                bbox_xyxy = mask_bbox
        bbox_xywh = _bbox_xyxy_to_xywh(bbox_xyxy)
        area_px = float(mask.sum()) if mask is not None else float(bbox_xywh[2] * bbox_xywh[3])
        detections.append(
            {
                "bbox_xyxy": bbox_xyxy,
                "bbox_xywh": bbox_xywh,
                "mask": mask,
                "score": float(scores[idx]),
                "area_px": area_px,
                "rank": rank,
            }
        )
    return detections


def run_sam_segmentation(
    rgb: np.ndarray,
    model_path: Path,
    conf: float,
    device: str | None,
    imgsz: int | None = None,
) -> list[dict[str, Any]]:
    if not model_path.exists():
        raise FileNotFoundError("megapose_segmentation_model_missing")
    model = get_sam_model(model_path)
    with _SAM_INFER_LOCK:
        result = model.predict(
            source=rgb,
            device=device,
            imgsz=int(imgsz) if imgsz else 1024,
            conf_thres=float(conf),
            verbose=False,
        )[0]

    masks_obj = getattr(result, "masks", None)
    boxes_obj = getattr(result, "boxes", None)
    if masks_obj is None or getattr(masks_obj, "data", None) is None:
        return []

    mask_tensor = masks_obj.data
    if mask_tensor is None or len(mask_tensor) == 0:
        return []

    masks_np = mask_tensor.detach().cpu().numpy() > 0.5
    scores = np.ones((masks_np.shape[0],), dtype=np.float32)
    if boxes_obj is not None and hasattr(boxes_obj, "conf") and boxes_obj.conf is not None:
        try:
            scores = boxes_obj.conf.detach().cpu().numpy().astype(np.float32)
        except Exception:
            scores = np.ones((masks_np.shape[0],), dtype=np.float32)

    order = np.argsort(-scores)
    detections: list[dict[str, Any]] = []
    for rank, raw_idx in enumerate(order.tolist()):
        idx = int(raw_idx)
        mask = masks_np[idx]
        if mask.shape[:2] != rgb.shape[:2]:
            mask = cv2.resize(
                mask.astype(np.uint8),
                (int(rgb.shape[1]), int(rgb.shape[0])),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
        mask_bbox = _bbox_from_mask(mask)
        if mask_bbox is None:
            continue
        bbox_xywh = _bbox_xyxy_to_xywh(mask_bbox)
        detections.append(
            {
                "bbox_xyxy": mask_bbox,
                "bbox_xywh": bbox_xywh,
                "mask": mask,
                "score": float(scores[idx]),
                "area_px": float(mask.sum()),
                "rank": rank,
            }
        )
    return detections


def run_segmentation(
    rgb: np.ndarray,
    model_path: Path,
    backend: str,
    conf: float,
    device: str | None,
    imgsz: int | None = None,
    retina_masks: bool = True,
) -> list[dict[str, Any]]:
    resolved_backend = resolve_segmentation_backend(backend, model_path)
    if resolved_backend == "sam":
        return run_sam_segmentation(rgb=rgb, model_path=model_path, conf=conf, device=device, imgsz=imgsz)
    return run_yolo_segmentation(
        rgb=rgb,
        model_path=model_path,
        conf=conf,
        device=device,
        imgsz=imgsz,
        retina_masks=retina_masks,
    )


def _point_in_polygon_uv(px: float, py: float, polygon: list[list[float]]) -> bool:
    """Ray-cast point-in-polygon for 2D OBB. Polygon is 4 (u, v) corners."""
    if not polygon or len(polygon) < 3:
        return True
    inside = False
    n = len(polygon)
    x = float(px)
    y = float(py)
    j = n - 1
    for i in range(n):
        xi, yi = float(polygon[i][0]), float(polygon[i][1])
        xj, yj = float(polygon[j][0]), float(polygon[j][1])
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / ((yj - yi) or 1e-12) + xi):
            inside = not inside
        j = i
    return inside


def _detection_passes_bin_roi(
    det: dict[str, Any],
    polygon_uv: list[list[float]] | None,
) -> bool:
    """Reject detections whose centroid OR bbox corners fall outside the bin OBB.

    Centroid alone is not enough — e.g. a tall object partially on a bin edge could
    have centroid inside but corner hanging over the wall. Requiring all 4 bbox
    corners + centroid inside gives a conservative check.
    """
    if not polygon_uv:
        return True
    bbox = det.get("bbox_xywh") or []
    try:
        x = float(bbox[0]); y = float(bbox[1]); w = float(bbox[2]); h = float(bbox[3])
    except (TypeError, ValueError, IndexError):
        return True
    cx = x + w / 2.0
    cy = y + h / 2.0
    if not _point_in_polygon_uv(cx, cy, polygon_uv):
        return False
    corners = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
    for (px, py) in corners:
        if not _point_in_polygon_uv(px, py, polygon_uv):
            return False
    return True


def select_detection_candidate(
    detections: list[dict[str, Any]],
    *,
    image_shape: tuple[int, int],
    depth_m: np.ndarray | None = None,
    K: np.ndarray | None = None,
    top_k: int = 5,
    min_confidence: float = 0.5,
    min_area_px: float = 0.0,
    max_area_px: float | None = None,
    selection_mode: str = "highest_segmentation_score",
    high_confidence_gate: float = 0.8,
    bin_roi_polygon_uv: list[list[float]] | None = None,
    excluded_ranks: list[int] | None = None,
) -> dict[str, Any] | None:
    if not detections:
        return None
    excluded_rank_set = {int(rank) for rank in (excluded_ranks or [])}
    min_area = max(0.0, float(min_area_px or 0.0))
    max_area = (
        float(max_area_px)
        if max_area_px is not None and float(max_area_px) > 0.0
        else None
    )
    eligible = [
        det
        for det in detections
        if float(det.get("score", 0.0)) >= float(min_confidence)
        and int(det.get("rank", -1)) not in excluded_rank_set
        and float(det.get("area_px", 0.0)) >= min_area
        and (max_area is None or float(det.get("area_px", 0.0)) <= max_area)
        and _detection_passes_bin_roi(det, bin_roi_polygon_uv)
    ]
    if not eligible:
        return None
    candidate_pool = eligible[: max(1, int(top_k))]

    def _get_region_depth_metrics(det: dict[str, Any]) -> dict[str, Any]:
        if depth_m is None:
            return {
                "closest_region_distance_m": None,
                "robust_region_distance_m": None,
                "z_min_m": None,
                "z_front_m": None,
                "z_max_m": None,
                "valid_depth_points": 0,
                "support_depth_points": 0,
            }
        selection_mask = make_selection_mask(
            image_shape,
            det.get("mask"),
            np.asarray(det.get("bbox_xyxy"), dtype=np.float32),
        )
        if selection_mask.shape != depth_m.shape:
            return {
                "closest_region_distance_m": None,
                "robust_region_distance_m": None,
                "z_min_m": None,
                "z_front_m": None,
                "z_max_m": None,
                "valid_depth_points": 0,
                "support_depth_points": 0,
            }
        valid_mask = selection_mask & np.isfinite(depth_m) & (depth_m > 0.0)
        if not np.any(valid_mask):
            return {
                "closest_region_distance_m": None,
                "robust_region_distance_m": None,
                "z_min_m": None,
                "z_front_m": None,
                "z_max_m": None,
                "valid_depth_points": 0,
                "support_depth_points": 0,
            }

        ys, xs = np.nonzero(valid_mask)
        z_vals = depth_m[ys, xs].astype(np.float64)
        valid_depth_points = int(z_vals.size)
        support_depth_points = max(3, int(math.ceil(valid_depth_points * 0.02)))
        support_depth_points = min(valid_depth_points, support_depth_points)
        sorted_z = np.sort(z_vals)
        z_min_m = float(sorted_z[0])
        z_front_m = float(sorted_z[support_depth_points - 1])
        z_max_m = float(sorted_z[-1])

        if K is None:
            return {
                "closest_region_distance_m": z_min_m,
                "robust_region_distance_m": z_front_m,
                "z_min_m": z_min_m,
                "z_front_m": z_front_m,
                "z_max_m": z_max_m,
                "valid_depth_points": valid_depth_points,
                "support_depth_points": support_depth_points,
            }

        fx = float(K[0, 0])
        fy = float(K[1, 1])
        cx_intr = float(K[0, 2])
        cy_intr = float(K[1, 2])
        x_m = (xs.astype(np.float64) - cx_intr) * z_vals / fx
        y_m = (ys.astype(np.float64) - cy_intr) * z_vals / fy
        dist_vals = np.sqrt((x_m * x_m) + (y_m * y_m) + (z_vals * z_vals))
        sorted_dist = np.sort(dist_vals)
        return {
            "closest_region_distance_m": float(sorted_dist[0]),
            "robust_region_distance_m": float(sorted_dist[support_depth_points - 1]),
            "z_min_m": z_min_m,
            "z_front_m": z_front_m,
            "z_max_m": z_max_m,
            "valid_depth_points": valid_depth_points,
            "support_depth_points": support_depth_points,
        }

    depth_metrics_cache: dict[int, dict[str, Any]] = {}

    def _get_region_depth_metrics_cached(det: dict[str, Any]) -> dict[str, Any]:
        key = id(det)
        metrics = depth_metrics_cache.get(key)
        if metrics is None:
            metrics = _get_region_depth_metrics(det)
            depth_metrics_cache[key] = metrics
        return metrics

    def _get_region_selection_distance(det: dict[str, Any]) -> float | None:
        metrics = _get_region_depth_metrics_cached(det)
        robust_distance = metrics.get("robust_region_distance_m")
        if robust_distance is not None:
            return float(robust_distance)
        closest_distance = metrics.get("closest_region_distance_m")
        if closest_distance is not None:
            return float(closest_distance)
        return None

    def _get_3d_distance(det: dict[str, Any]) -> float:
        """Calculate fallback distance when region-level closest depth is unavailable."""
        if depth_m is not None and K is not None:
            bbox_xywh = det.get("bbox_xywh") or [0.0, 0.0, 0.0, 0.0]
            x, y, w, h = [float(v) for v in bbox_xywh[:4]]
            cx = int(x + w * 0.5)
            cy = int(y + h * 0.5)
            cx = max(0, min(cx, int(depth_m.shape[1]) - 1))
            cy = max(0, min(cy, int(depth_m.shape[0]) - 1))
            z_m = float(depth_m[cy, cx])
            if z_m > 0 and np.isfinite(z_m):
                fx = float(K[0, 0])
                fy = float(K[1, 1])
                cx_intr = float(K[0, 2])
                cy_intr = float(K[1, 2])
                x_m = (float(cx) - cx_intr) * z_m / fx
                y_m = (float(cy) - cy_intr) * z_m / fy
                distance_3d = math.sqrt(x_m * x_m + y_m * y_m + z_m * z_m)
                return distance_3d
        depth = det.get("mean_depth")
        if depth is not None:
            return float(depth)
        area = float(det.get("area_px", 1.0))
        return 1.0 / max(area, 1.0)

    resolved_selection_mode = str(selection_mode or "highest_segmentation_score").strip().lower()
    if resolved_selection_mode == "high_confidence_highest_z":
        high_conf_pool = [
            det for det in eligible if float(det.get("score", 0.0)) >= float(high_confidence_gate)
        ]
        if high_conf_pool:
            selected = min(
                high_conf_pool,
                key=lambda det: (
                    (
                        _get_region_selection_distance(det)
                        if _get_region_selection_distance(det) is not None
                        else _get_3d_distance(det)
                    ),
                    -float(det.get("score", 0.0)),
                    -float(det.get("area_px", 0.0)),
                ),
            )
            candidate_pool = high_conf_pool
        else:
            selected = min(
                candidate_pool,
                key=lambda det: (
                    (
                        _get_region_selection_distance(det)
                        if _get_region_selection_distance(det) is not None
                        else _get_3d_distance(det)
                    ),
                    -float(det.get("score", 0.0)),
                    -float(det.get("area_px", 0.0)),
                ),
            )
    elif resolved_selection_mode == "lowest_z_front":
        def _z_front_key(det: dict[str, Any]) -> float:
            metrics = _get_region_depth_metrics_cached(det)
            zf = metrics.get("z_front_m")
            if zf is None or not math.isfinite(float(zf)):
                return float("inf")
            return float(zf)

        selected = min(
            candidate_pool,
            key=lambda det: (
                _z_front_key(det),
                -float(det.get("score", 0.0)),
                -float(det.get("area_px", 0.0)),
            ),
        )
    elif resolved_selection_mode == "top_k_highest_z":
        selected = min(
            candidate_pool,
            key=lambda det: (
                (
                    _get_region_selection_distance(det)
                    if _get_region_selection_distance(det) is not None
                    else _get_3d_distance(det)
                ),
                -float(det.get("score", 0.0)),
                -float(det.get("area_px", 0.0)),
            ),
        )
    else:
        selected = max(
            candidate_pool,
            key=lambda det: (
                float(det.get("score", 0.0)),
                float(det.get("area_px", 0.0)),
                -(
                    _get_region_selection_distance(det)
                    if _get_region_selection_distance(det) is not None
                    else _get_3d_distance(det)
                ),
            ),
        )
    chosen = dict(selected)
    chosen["selection_pool"] = [
        ({
            "rank": int(det.get("rank", 0)),
            "score": float(det.get("score", 0.0)),
            "area_px": float(det.get("area_px", 0.0)),
            "bbox_xywh": [float(v) for v in det.get("bbox_xywh", [])],
            "closest_region_distance_m": _get_region_depth_metrics_cached(det).get(
                "closest_region_distance_m"
            ),
            "robust_region_distance_m": _get_region_depth_metrics_cached(det).get(
                "robust_region_distance_m"
            ),
            "z_min_m": _get_region_depth_metrics_cached(det).get("z_min_m"),
            "z_front_m": _get_region_depth_metrics_cached(det).get("z_front_m"),
            "z_max_m": _get_region_depth_metrics_cached(det).get("z_max_m"),
            "valid_depth_points": int(
                _get_region_depth_metrics_cached(det).get("valid_depth_points", 0)
            ),
            "support_depth_points": int(
                _get_region_depth_metrics_cached(det).get("support_depth_points", 0)
            ),
            "distance_3d_m": float(_get_3d_distance(det)),
        })
        for det in candidate_pool[:10]
    ]
    chosen["selection_mode"] = resolved_selection_mode
    chosen["selection_area_filter"] = {
        "min_area_px": min_area,
        "max_area_px": max_area,
        "eligible_count": len(eligible),
        "input_count": len(detections),
    }
    chosen_metrics = _get_region_depth_metrics_cached(chosen)
    chosen["closest_region_distance_m"] = chosen_metrics.get("closest_region_distance_m")
    chosen["robust_region_distance_m"] = chosen_metrics.get("robust_region_distance_m")
    chosen["z_min_m"] = chosen_metrics.get("z_min_m")
    chosen["z_front_m"] = chosen_metrics.get("z_front_m")
    chosen["z_max_m"] = chosen_metrics.get("z_max_m")
    chosen["valid_depth_points"] = int(chosen_metrics.get("valid_depth_points", 0))
    chosen["support_depth_points"] = int(chosen_metrics.get("support_depth_points", 0))
    chosen["distance_3d_m"] = float(_get_3d_distance(chosen))
    return chosen


def suppress_nested_duplicate_detections(
    detections: list[dict[str, Any]],
    *,
    image_shape: tuple[int, int],
    depth_m: np.ndarray | None = None,
    containment_threshold: float = 0.9,
    max_z_front_delta_m: float = 0.008,
    max_center_offset_ratio: float = 0.25,
    max_area_ratio: float = 4.0,
    dilation_px: int = 3,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if len(detections) <= 1:
        return detections, {
            "enabled": True,
            "input_count": len(detections),
            "kept_count": len(detections),
            "suppressed_count": 0,
            "groups": [],
        }

    h, w = image_shape
    kernel = None
    if dilation_px > 0:
        kernel_size = max(1, (int(dilation_px) * 2) + 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    mask_cache: dict[int, np.ndarray] = {}
    depth_stats_cache: dict[int, dict[str, float | None]] = {}

    def _mask(det: dict[str, Any]) -> np.ndarray:
        key = id(det)
        cached = mask_cache.get(key)
        if cached is None:
            cached = make_selection_mask(
                image_shape,
                det.get("mask"),
                np.asarray(det.get("bbox_xyxy"), dtype=np.float32),
            ).astype(bool)
            if kernel is not None and np.any(cached):
                cached = cv2.dilate(cached.astype(np.uint8), kernel, iterations=1).astype(bool)
            mask_cache[key] = cached
        return cached

    def _depth_stats(det: dict[str, Any]) -> dict[str, float | None]:
        key = id(det)
        cached = depth_stats_cache.get(key)
        if cached is None:
            cached = _candidate_depth_stats(det, image_shape=image_shape, depth_m=depth_m)
            depth_stats_cache[key] = cached
        return cached

    def _center(det: dict[str, Any]) -> tuple[float, float]:
        bbox = det.get("bbox_xywh") or [0.0, 0.0, 0.0, 0.0]
        x, y, bw, bh = [float(v) for v in bbox[:4]]
        return (x + (bw * 0.5), y + (bh * 0.5))

    # Prefer larger masks first so smaller nested duplicates get suppressed.
    ordered = sorted(
        detections,
        key=lambda det: (
            -float(det.get("area_px", 0.0)),
            -float(det.get("score", 0.0)),
            int(det.get("rank", 0)),
        ),
    )
    suppressed_ids: set[int] = set()
    groups: list[dict[str, Any]] = []

    for i, larger in enumerate(ordered):
        larger_id = id(larger)
        if larger_id in suppressed_ids:
            continue
        larger_area = max(1.0, float(larger.get("area_px", 0.0)))
        larger_mask = _mask(larger)
        larger_stats = _depth_stats(larger)
        larger_z_front = larger_stats.get("z_front_m")
        larger_center_x, larger_center_y = _center(larger)
        bbox = larger.get("bbox_xywh") or [0.0, 0.0, 1.0, 1.0]
        larger_bw = max(1.0, float(bbox[2]))
        larger_bh = max(1.0, float(bbox[3]))
        suppressed_children: list[dict[str, Any]] = []

        for smaller in ordered[i + 1:]:
            smaller_id = id(smaller)
            if smaller_id in suppressed_ids:
                continue
            smaller_area = max(1.0, float(smaller.get("area_px", 0.0)))
            if smaller_area >= larger_area:
                continue
            area_ratio = larger_area / smaller_area
            if area_ratio > float(max_area_ratio):
                continue

            smaller_mask = _mask(smaller)
            intersection = int(np.count_nonzero(larger_mask & smaller_mask))
            contain_small = float(intersection / max(1.0, float(np.count_nonzero(smaller_mask))))
            if contain_small < float(containment_threshold):
                continue

            smaller_stats = _depth_stats(smaller)
            smaller_z_front = smaller_stats.get("z_front_m")
            if (
                larger_z_front is not None
                and smaller_z_front is not None
                and math.isfinite(float(larger_z_front))
                and math.isfinite(float(smaller_z_front))
                and abs(float(larger_z_front) - float(smaller_z_front)) > float(max_z_front_delta_m)
            ):
                continue

            smaller_center_x, smaller_center_y = _center(smaller)
            if (
                abs(smaller_center_x - larger_center_x) > (larger_bw * float(max_center_offset_ratio))
                or abs(smaller_center_y - larger_center_y) > (larger_bh * float(max_center_offset_ratio))
            ):
                continue

            suppressed_ids.add(smaller_id)
            smaller["duplicate_suppressed_by_rank"] = int(larger.get("rank", 0))
            suppressed_children.append(
                {
                    "rank": int(smaller.get("rank", 0)),
                    "score": float(smaller.get("score", 0.0)),
                    "area_px": float(smaller.get("area_px", 0.0)),
                    "containment_in_parent": float(contain_small),
                    "z_front_m": smaller_z_front,
                }
            )

        if suppressed_children:
            groups.append(
                {
                    "kept_rank": int(larger.get("rank", 0)),
                    "kept_score": float(larger.get("score", 0.0)),
                    "kept_area_px": float(larger.get("area_px", 0.0)),
                    "kept_z_front_m": larger_z_front,
                    "suppressed": suppressed_children,
                }
            )

    filtered = [det for det in detections if id(det) not in suppressed_ids]
    return filtered, {
        "enabled": True,
        "input_count": len(detections),
        "kept_count": len(filtered),
        "suppressed_count": len(detections) - len(filtered),
        "containment_threshold": float(containment_threshold),
        "max_z_front_delta_m": float(max_z_front_delta_m),
        "max_center_offset_ratio": float(max_center_offset_ratio),
        "max_area_ratio": float(max_area_ratio),
        "dilation_px": int(dilation_px),
        "groups": groups,
    }


def make_selection_mask(
    image_shape: tuple[int, int],
    mask: np.ndarray | None,
    bbox_xyxy: np.ndarray,
) -> np.ndarray:
    if mask is not None:
        mask_bool = mask.astype(bool)
        h, w = image_shape
        if mask_bool.shape[:2] != (h, w):
            mask_bool = cv2.resize(
                mask_bool.astype(np.uint8),
                (int(w), int(h)),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
        return mask_bool
    h, w = image_shape
    x1, y1, x2, y2 = bbox_xyxy.astype(int).tolist()
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(x1 + 1, min(x2, w))
    y2 = max(y1 + 1, min(y2, h))
    roi_mask = np.zeros((h, w), dtype=bool)
    roi_mask[y1:y2, x1:x2] = True
    return roi_mask


def extract_segmentation_contours(
    selection_mask: np.ndarray,
    *,
    max_contours: int = 2,
    max_points_per_contour: int = 64,
) -> list[list[list[float]]]:
    mask_u8 = (selection_mask.astype(np.uint8) * 255)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []

    packed: list[list[list[float]]] = []
    ordered = sorted(contours, key=cv2.contourArea, reverse=True)
    for contour in ordered[: max(1, int(max_contours))]:
        arc_len = float(cv2.arcLength(contour, True))
        epsilon = max(1.0, arc_len * 0.003)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        pts = approx.reshape(-1, 2) if approx is not None else contour.reshape(-1, 2)
        if pts.shape[0] < 3:
            continue
        if pts.shape[0] > max_points_per_contour:
            sample_idx = np.linspace(
                0,
                pts.shape[0] - 1,
                num=max_points_per_contour,
                dtype=np.int32,
            )
            pts = pts[sample_idx]
        packed.append([[float(x), float(y)] for x, y in pts.tolist()])
    return packed


def filter_segmented_depth_to_main_cluster(
    selection_mask: np.ndarray,
    depth_m: np.ndarray | None,
    K: np.ndarray | None,
    *,
    cluster_distance_m: float = 0.03,
    min_cluster_points: int = 48,
    min_cluster_ratio: float = 0.01,
) -> tuple[np.ndarray, dict[str, Any] | None]:
    if depth_m is None or K is None:
        return selection_mask.astype(bool), None

    valid_mask = selection_mask.astype(bool) & np.isfinite(depth_m) & (depth_m > 0.0)
    ys, xs = np.where(valid_mask)
    if len(xs) == 0:
        return selection_mask.astype(bool), {
            "enabled": True,
            "applied": False,
            "skip_reason": "no_valid_depth_points",
        }

    fx = float(K[0, 0])
    fy = float(K[1, 1])
    cx = float(K[0, 2])
    cy = float(K[1, 2])
    z = depth_m[ys, xs].astype(np.float32)
    x = (xs.astype(np.float32) - cx) * z / fx
    y = (ys.astype(np.float32) - cy) * z / fy
    points = np.column_stack((x, y, z)).astype(np.float32)

    index_map = np.full(valid_mask.shape, -1, dtype=np.int32)
    point_indices = np.arange(len(xs), dtype=np.int32)
    index_map[ys, xs] = point_indices
    visited = np.zeros(len(xs), dtype=bool)
    neighbor_offsets = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1),
    ]
    cluster_distance_sq = float(cluster_distance_m) * float(cluster_distance_m)
    clusters: list[np.ndarray] = []

    for root_idx in range(len(xs)):
        if visited[root_idx]:
            continue
        stack = [int(root_idx)]
        visited[root_idx] = True
        cluster_members: list[int] = []
        while stack:
            current_idx = stack.pop()
            cluster_members.append(current_idx)
            py = int(ys[current_idx])
            px = int(xs[current_idx])
            current_point = points[current_idx]
            for dy, dx in neighbor_offsets:
                ny = py + dy
                nx = px + dx
                if ny < 0 or nx < 0 or ny >= valid_mask.shape[0] or nx >= valid_mask.shape[1]:
                    continue
                neighbor_idx = int(index_map[ny, nx])
                if neighbor_idx < 0 or visited[neighbor_idx]:
                    continue
                delta = points[neighbor_idx] - current_point
                if float(np.dot(delta, delta)) > cluster_distance_sq:
                    continue
                visited[neighbor_idx] = True
                stack.append(neighbor_idx)
        clusters.append(np.asarray(cluster_members, dtype=np.int32))

    if not clusters:
        return selection_mask.astype(bool), {
            "enabled": True,
            "applied": False,
            "skip_reason": "no_clusters_found",
        }

    largest_cluster_points = max(int(cluster.size) for cluster in clusters)
    dynamic_min_points = max(
        int(max(1, int(min_cluster_points))),
        int(math.ceil(float(min_cluster_ratio) * float(max(1, largest_cluster_points)))),
    )
    retained_clusters = [cluster for cluster in clusters if int(cluster.size) >= dynamic_min_points]
    discarded_small_clusters = len(clusters) - len(retained_clusters)
    if not retained_clusters:
        retained_clusters = [max(clusters, key=lambda cluster: int(cluster.size))]
        discarded_small_clusters = len(clusters) - 1

    best_cluster = max(
        retained_clusters,
        key=lambda cluster: (
            int(cluster.size),
            -float(np.median(points[cluster, 2])),
        ),
    )
    kept_mask = np.zeros_like(selection_mask, dtype=bool)
    kept_mask[ys[best_cluster], xs[best_cluster]] = True
    removed_count = int(np.count_nonzero(valid_mask)) - int(best_cluster.size)
    return kept_mask, {
        "enabled": True,
        "applied": bool(removed_count > 0),
        "cluster_distance_m": float(cluster_distance_m),
        "min_cluster_points": int(dynamic_min_points),
        "min_cluster_ratio": float(min_cluster_ratio),
        "valid_depth_points": int(np.count_nonzero(valid_mask)),
        "cluster_count": int(len(clusters)),
        "retained_cluster_count": int(len(retained_clusters)),
        "discarded_small_clusters": int(discarded_small_clusters),
        "kept_cluster_points": int(best_cluster.size),
        "removed_points": int(max(0, removed_count)),
    }


def mask_observation_to_selected_object(
    rgb: np.ndarray,
    depth_m: np.ndarray | None,
    mask: np.ndarray | None,
    bbox_xyxy: np.ndarray,
    K: np.ndarray | None = None,
    filter_depth_clusters: bool = True,
    cluster_distance_m: float = 0.03,
    min_cluster_points: int = 48,
    min_cluster_ratio: float = 0.01,
    rgb_mask_dilation_base_px: float = 5.0,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray, np.ndarray, dict[str, Any] | None]:
    selection_mask = make_selection_mask(rgb.shape[:2], mask, bbox_xyxy)
    depth_selection_mask = selection_mask.copy()
    filter_info = None
    if filter_depth_clusters and depth_m is not None and K is not None:
        depth_selection_mask, filter_info = filter_segmented_depth_to_main_cluster(
            depth_selection_mask,
            depth_m,
            K,
            cluster_distance_m=float(cluster_distance_m),
            min_cluster_points=int(min_cluster_points),
            min_cluster_ratio=float(min_cluster_ratio),
        )
    rgb_selection_mask = selection_mask.astype(bool)
    dilation_px = 0
    if float(rgb_mask_dilation_base_px) > 0.0:
        dilation_px = _scaled_rgb_mask_dilation_px(
            rgb.shape,
            base_pixels=float(rgb_mask_dilation_base_px),
        )
    if dilation_px > 0 and np.any(rgb_selection_mask):
        kernel_size = max(1, (dilation_px * 2) + 1)
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
        )
        rgb_selection_mask = cv2.dilate(
            rgb_selection_mask.astype(np.uint8), kernel, iterations=1
        ).astype(bool)
    masked_rgb = rgb.copy()
    masked_rgb[~rgb_selection_mask] = 0
    masked_depth = None
    if depth_m is not None:
        masked_depth = depth_m.copy()
        masked_depth[~depth_selection_mask] = 0.0
    return masked_rgb, masked_depth, selection_mask, depth_selection_mask, filter_info


def get_mesh_annotation_meta(
    mesh_path: Path,
    mesh_units: str,
    mesh_scale: float,
) -> dict[str, Any]:
    key = (str(mesh_path.resolve()), str(mesh_units).strip().lower(), float(mesh_scale))
    with _MESH_ANNOTATION_CACHE_LOCK:
        cached = _MESH_ANNOTATION_CACHE.get(key)
        if cached is not None:
            return dict(cached)

    import trimesh

    mesh = trimesh.load_mesh(str(mesh_path), force="mesh")
    if isinstance(mesh, trimesh.Scene):
        if not mesh.geometry:
            raise ValueError("megapose_mesh_missing_geometry")
        mesh = mesh.dump(concatenate=True)
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"Unsupported mesh type: {type(mesh)!r}")
    if mesh.vertices.size == 0:
        raise ValueError("megapose_mesh_empty")

    bounds = np.asarray(mesh.bounds, dtype=np.float64)
    center_mesh_units = 0.5 * (bounds[0] + bounds[1])
    extents_mesh_units = np.maximum(bounds[1] - bounds[0], 0.0)
    scale_to_m = (1.0 if str(mesh_units).strip().lower() == "m" else 0.001) * float(mesh_scale)
    center_m = center_mesh_units * scale_to_m
    extents_m = extents_mesh_units * scale_to_m
    max_extent_m = float(np.max(extents_m)) if extents_m.size else 0.0
    axis_length_m = float(np.clip(max_extent_m * 0.35, 0.015, 0.08))

    meta = {
        "object_center_object_m": [float(v) for v in center_m.tolist()],
        "object_extents_m": [float(v) for v in extents_m.tolist()],
        "annotation_axis_length_m": axis_length_m,
    }
    with _MESH_ANNOTATION_CACHE_LOCK:
        _MESH_ANNOTATION_CACHE[key] = dict(meta)
    return meta


def rotation_matrix_to_quaternion_xyzw(rotation: np.ndarray) -> np.ndarray:
    trace = float(np.trace(rotation))
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (rotation[2, 1] - rotation[1, 2]) / s
        y = (rotation[0, 2] - rotation[2, 0]) / s
        z = (rotation[1, 0] - rotation[0, 1]) / s
    elif rotation[0, 0] > rotation[1, 1] and rotation[0, 0] > rotation[2, 2]:
        s = np.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
        w = (rotation[2, 1] - rotation[1, 2]) / s
        x = 0.25 * s
        y = (rotation[0, 1] + rotation[1, 0]) / s
        z = (rotation[0, 2] + rotation[2, 0]) / s
    elif rotation[1, 1] > rotation[2, 2]:
        s = np.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
        w = (rotation[0, 2] - rotation[2, 0]) / s
        x = (rotation[0, 1] + rotation[1, 0]) / s
        y = 0.25 * s
        z = (rotation[1, 2] + rotation[2, 1]) / s
    else:
        s = np.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
        w = (rotation[1, 0] - rotation[0, 1]) / s
        x = (rotation[0, 2] + rotation[2, 0]) / s
        y = (rotation[1, 2] + rotation[2, 1]) / s
        z = 0.25 * s
    quat = np.array([x, y, z, w], dtype=np.float32)
    return quat / np.linalg.norm(quat)


def _normalize_axis(axis: Any) -> np.ndarray:
    value = np.array(axis if isinstance(axis, (list, tuple, np.ndarray)) else [0.0, 0.0, 1.0], dtype=np.float32)
    if value.shape[0] != 3:
        raise ValueError("megapose_axis_invalid")
    norm = float(np.linalg.norm(value))
    if norm <= 1e-8:
        raise ValueError("megapose_axis_invalid")
    return value / norm


def _yaw_from_axis(axis_camera: np.ndarray) -> float | None:
    planar_norm = math.hypot(float(axis_camera[0]), float(axis_camera[1]))
    if planar_norm <= 1e-6:
        return None
    return math.degrees(math.atan2(float(axis_camera[1]), float(axis_camera[0])))


def _project_point_uv(point_xyz: np.ndarray, K: np.ndarray) -> list[float] | None:
    x = float(point_xyz[0])
    y = float(point_xyz[1])
    z = float(point_xyz[2])
    if not np.isfinite([x, y, z]).all() or z <= 1e-6:
        return None
    fx = float(K[0, 0])
    fy = float(K[1, 1])
    cx = float(K[0, 2])
    cy = float(K[1, 2])
    return [
        float((fx * (x / z)) + cx),
        float((fy * (y / z)) + cy),
    ]


def _save_debug_overlay(
    bgr: np.ndarray,
    detections: list[dict[str, Any]],
    selected_rank: int,
    destination: Path,
) -> Path:
    debug = bgr.copy()
    palette = [
        (0, 255, 255),
        (255, 180, 0),
        (255, 0, 255),
        (0, 200, 0),
        (0, 128, 255),
    ]
    for det in detections:
        rank = int(det.get("rank", 0))
        color = palette[rank % len(palette)]
        mask = det.get("mask")
        if mask is not None:
            overlay = debug.copy()
            overlay[mask.astype(bool)] = color
            alpha = 0.45 if rank == selected_rank else 0.24
            debug = cv2.addWeighted(debug, 1.0 - alpha, overlay, alpha, 0.0)
        x, y, w, h = [int(v) for v in det.get("bbox_xywh", [0, 0, 0, 0])]
        cv2.rectangle(debug, (x, y), (x + w, y + h), color, 2)
        cv2.putText(
            debug,
            f"{rank}: {float(det.get('score', 0.0)):.2f}",
            (x, max(20, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(destination), debug)
    return destination


def _save_segmented_region(
    bgr: np.ndarray,
    selection_mask: np.ndarray,
    bbox_xyxy: np.ndarray,
    destination: Path,
) -> Path:
    x1, y1, x2, y2 = bbox_xyxy.astype(int).tolist()
    x1 = max(0, min(x1, bgr.shape[1] - 1))
    y1 = max(0, min(y1, bgr.shape[0] - 1))
    x2 = max(x1 + 1, min(x2, bgr.shape[1]))
    y2 = max(y1 + 1, min(y2, bgr.shape[0]))
    crop = bgr[y1:y2, x1:x2]
    alpha = (selection_mask[y1:y2, x1:x2].astype(np.uint8)) * 255
    rgba = cv2.cvtColor(crop, cv2.COLOR_BGR2BGRA)
    rgba[:, :, 3] = alpha
    destination.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(destination), rgba)
    return destination


def _save_selected_detection_crop(
    bgr: np.ndarray,
    bbox_xyxy: np.ndarray,
    destination: Path,
) -> Path:
    x1, y1, x2, y2 = bbox_xyxy.astype(int).tolist()
    x1 = max(0, min(x1, bgr.shape[1] - 1))
    y1 = max(0, min(y1, bgr.shape[0] - 1))
    x2 = max(x1 + 1, min(x2, bgr.shape[1]))
    y2 = max(y1 + 1, min(y2, bgr.shape[0]))
    crop = bgr[y1:y2, x1:x2]
    destination.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(destination), crop)
    return destination


def _save_pose_annotated_overlay(
    bgr: np.ndarray,
    selection_mask: np.ndarray,
    bbox_xyxy: np.ndarray,
    *,
    label: str,
    score: float,
    origin_xyz_m: np.ndarray,
    pose_matrix: np.ndarray,
    K: np.ndarray,
    axis_length_m: float,
    destination: Path,
) -> Path:
    debug = bgr.copy()
    overlay = debug.copy()
    overlay[selection_mask.astype(bool)] = (0, 220, 220)
    debug = cv2.addWeighted(debug, 0.78, overlay, 0.22, 0.0)

    mask_u8 = (selection_mask.astype(np.uint8)) * 255
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(debug, contours, -1, (255, 255, 0), 2, cv2.LINE_AA)

    x1, y1, x2, y2 = bbox_xyxy.astype(int).tolist()
    cv2.rectangle(debug, (x1, y1), (x2, y2), (0, 255, 255), 2)

    origin_uv = _project_point_uv(origin_xyz_m, K)
    if origin_uv is not None:
        ox = int(round(origin_uv[0]))
        oy = int(round(origin_uv[1]))
        cv2.circle(debug, (ox, oy), 4, (0, 255, 255), -1, cv2.LINE_AA)

        rotation = pose_matrix[:3, :3]
        axis_length = max(0.005, float(axis_length_m or 0.015))
        axis_defs = [
            ((0, 0, 255), rotation[:, 0], "X"),
            ((0, 255, 0), rotation[:, 1], "Y"),
            ((255, 120, 0), rotation[:, 2], "Z"),
        ]
        for color, axis_vec, axis_name in axis_defs:
            endpoint_xyz = origin_xyz_m + (np.asarray(axis_vec, dtype=np.float32) * axis_length)
            endpoint_uv = _project_point_uv(endpoint_xyz, K)
            if endpoint_uv is None:
                continue
            ex = int(round(endpoint_uv[0]))
            ey = int(round(endpoint_uv[1]))
            cv2.arrowedLine(debug, (ox, oy), (ex, ey), color, 2, cv2.LINE_AA, tipLength=0.18)
            cv2.putText(
                debug,
                axis_name,
                (ex + 4, ey - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                2,
                cv2.LINE_AA,
            )

    label_text = f"{label} | s {float(score):.2f} | z {float(origin_xyz_m[2]):.3f}m"
    label_org = (max(8, x1), max(24, y1 - 10))
    cv2.putText(
        debug,
        label_text,
        label_org,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (15, 15, 15),
        4,
        cv2.LINE_AA,
    )
    cv2.putText(
        debug,
        label_text,
        label_org,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    destination.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(destination), debug)
    return destination


def _json_safe(value: Any) -> Any:
    if isinstance(value, (str, bool)) or value is None:
        return value
    if isinstance(value, (int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe(val) for key, val in value.items()}
    return str(value)


def _serialize_pose_collection(data_tc: Any) -> list[dict[str, Any]]:
    if data_tc is None:
        return []
    infos = getattr(data_tc, "infos", None)
    if infos is None:
        return []
    try:
        df = infos.reset_index(drop=True)
    except Exception:
        return []

    poses_np = None
    bboxes_np = None
    poses = getattr(data_tc, "poses", None)
    if poses is not None:
        try:
            poses_np = poses.detach().cpu().numpy()
        except Exception:
            poses_np = None
    bboxes = getattr(data_tc, "bboxes", None)
    if bboxes is not None:
        try:
            bboxes_np = bboxes.detach().cpu().numpy()
        except Exception:
            bboxes_np = None

    rows: list[dict[str, Any]] = []
    for idx, (_, row) in enumerate(df.iterrows()):
        record = {str(key): _json_safe(val) for key, val in row.to_dict().items()}
        if poses_np is not None and idx < len(poses_np):
            record["pose_matrix"] = _json_safe(np.asarray(poses_np[idx]))
        if bboxes_np is not None and idx < len(bboxes_np):
            record["bbox_xyxy"] = _json_safe(np.asarray(bboxes_np[idx]))
        rows.append(record)
    return rows


def _tensor_image_to_u8(array: np.ndarray) -> np.ndarray:
    image = np.asarray(array)
    if image.ndim == 3:
        # MegaPose coarse debug renders can arrive as channel-first tensors with
        # more than three channels. Keep the leading RGB planes and convert them
        # to HWC before OpenCV writes the montage.
        if image.shape[0] <= 16 and image.shape[1] > 16 and image.shape[2] > 16:
            image = np.transpose(image[: min(3, int(image.shape[0]))], (1, 2, 0))
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=2)
    if image.ndim == 3 and image.shape[2] == 1:
        image = np.repeat(image, 3, axis=2)
    if image.ndim == 3 and image.shape[2] == 2:
        image = np.concatenate([image, image[:, :, :1]], axis=2)
    if image.ndim == 3 and image.shape[2] > 3:
        image = image[:, :, :3]
    if image.dtype.kind == "f":
        image = np.clip(np.round(image * 255.0), 0, 255).astype(np.uint8)
    else:
        image = np.clip(image, 0, 255).astype(np.uint8)
    return image


def _tensor_bank_to_numpy(tensor_bank: Any) -> np.ndarray | None:
    if tensor_bank is None:
        return None
    try:
        if isinstance(tensor_bank, np.ndarray):
            return tensor_bank
        return tensor_bank.detach().cpu().numpy()
    except Exception:
        return None


def _normals_image_to_feature_shading(normals_image: np.ndarray) -> np.ndarray | None:
    image = np.asarray(normals_image)
    if image.ndim == 3 and image.shape[0] <= 16 and image.shape[1] > 16 and image.shape[2] > 16:
        image = np.transpose(image[: min(3, int(image.shape[0]))], (1, 2, 0))
    if image.ndim != 3 or image.shape[2] < 3:
        return None

    image = image[..., :3].astype(np.float32)
    if image.dtype.kind == "f":
        image = np.clip(image, 0.0, 1.0)
    else:
        image = np.clip(image / 255.0, 0.0, 1.0)

    normals = (image * 2.0) - 1.0
    norm = np.linalg.norm(normals, axis=2, keepdims=True)
    valid = norm[..., 0] > 1e-4
    if not np.any(valid):
        return None
    normals[valid] /= norm[valid]

    # Strong raking lights reveal embossed and recessed features much more
    # clearly than the flat default RGB render of an untextured CAD part.
    key = np.array([-0.82, 0.28, 0.50], dtype=np.float32)
    fill = np.array([0.56, -0.48, 0.34], dtype=np.float32)
    rim = np.array([-0.20, 0.95, 0.22], dtype=np.float32)
    key /= np.linalg.norm(key)
    fill /= np.linalg.norm(fill)
    rim /= np.linalg.norm(rim)

    diffuse = np.maximum(0.0, np.einsum("ijk,k->ij", normals, key)) * 0.90
    diffuse += np.maximum(0.0, np.einsum("ijk,k->ij", normals, fill)) * 0.35
    rim_term = 1.0 - np.clip(normals[..., 2], -1.0, 1.0)
    diffuse += np.maximum(0.0, np.einsum("ijk,k->ij", normals, rim)) * rim_term * 0.30

    shade = np.zeros((*image.shape[:2], 3), dtype=np.float32)
    base = np.array([0.30, 0.30, 0.30], dtype=np.float32)
    shade[:] = base
    shade[valid] = np.clip(0.08 + diffuse[valid, None], 0.0, 1.0)

    # Add a subtle contour boost from normal variation so shallow grooves read.
    grad_x = cv2.Sobel(normals[..., 2], cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(normals[..., 2], cv2.CV_32F, 0, 1, ksize=3)
    edge = np.sqrt((grad_x * grad_x) + (grad_y * grad_y))
    if np.any(valid):
        edge_scale = float(np.percentile(edge[valid], 95))
        if edge_scale > 1e-6:
            edge = np.clip(edge / edge_scale, 0.0, 1.0)
            shade[valid] *= (1.0 - (0.28 * edge[valid, None]))

    shade = np.clip(np.round(shade * 255.0), 0, 255).astype(np.uint8)
    shade[~valid] = 0
    return shade


def _save_feature_candidate_montage(
    tensor_bank: Any,
    destination: Path,
    *,
    title_prefix: str,
    candidate_indices: Optional[list[int]] = None,
    title_indices: Optional[list[int]] = None,
    selected_indices: Optional[set[int]] = None,
    limit: int = 24,
) -> Optional[Path]:
    bank = _tensor_bank_to_numpy(tensor_bank)
    if bank is None:
        return None
    if bank.ndim != 5 or bank.shape[0] <= 0 or bank.shape[1] <= 0:
        return None

    if candidate_indices:
        indices = [
            int(idx)
            for idx in candidate_indices
            if 0 <= int(idx) < int(bank.shape[1])
        ][: max(1, int(limit))]
    else:
        candidates = min(int(bank.shape[1]), int(limit))
        indices = list(range(candidates))
    if not indices:
        return None

    tiles: list[np.ndarray] = []
    for idx in indices:
        image = _normals_image_to_feature_shading(bank[0, idx])
        if image is None:
            continue
        canvas = image.copy()
        display_idx = int(title_indices[idx]) if title_indices is not None and 0 <= idx < len(title_indices) else idx
        is_selected = (
            candidate_indices is not None
            or (selected_indices is not None and idx in selected_indices)
        )
        border_color = (255, 215, 0) if is_selected else (90, 90, 90)
        cv2.rectangle(canvas, (0, 0), (canvas.shape[1] - 1, canvas.shape[0] - 1), border_color, 2)
        cv2.putText(
            canvas,
            f"{title_prefix} {display_idx}",
            (8, max(16, int(canvas.shape[0] * 0.12))),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            border_color,
            1,
            cv2.LINE_AA,
        )
        tiles.append(canvas)

    if not tiles:
        return None

    tile_h = max(tile.shape[0] for tile in tiles)
    tile_w = max(tile.shape[1] for tile in tiles)
    cols = min(6, len(tiles))
    rows = int(math.ceil(len(tiles) / cols))
    montage = np.zeros((rows * tile_h, cols * tile_w, 3), dtype=np.uint8)
    for idx, tile in enumerate(tiles):
        row = idx // cols
        col = idx % cols
        y0 = row * tile_h
        x0 = col * tile_w
        montage[y0 : y0 + tile.shape[0], x0 : x0 + tile.shape[1]] = tile

    destination.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(destination), cv2.cvtColor(montage, cv2.COLOR_RGB2BGR))
    return destination


def _save_candidate_montage(
    tensor_bank: Any,
    destination: Path,
    *,
    title_prefix: str,
    candidate_indices: Optional[list[int]] = None,
    title_indices: Optional[list[int]] = None,
    selected_indices: Optional[set[int]] = None,
    limit: int = 24,
) -> Optional[Path]:
    bank = _tensor_bank_to_numpy(tensor_bank)
    if bank is None:
        return None
    if bank.ndim != 5 or bank.shape[0] <= 0 or bank.shape[1] <= 0:
        return None

    if candidate_indices:
        indices = [
            int(idx)
            for idx in candidate_indices
            if 0 <= int(idx) < int(bank.shape[1])
        ][: max(1, int(limit))]
    else:
        candidates = min(int(bank.shape[1]), int(limit))
        indices = list(range(candidates))
    if not indices:
        return None

    tiles: list[np.ndarray] = []
    for idx in indices:
        image = _tensor_image_to_u8(bank[0, idx])
        canvas = image.copy()
        display_idx = int(title_indices[idx]) if title_indices is not None and 0 <= idx < len(title_indices) else idx
        is_selected = (
            candidate_indices is not None
            or (selected_indices is not None and idx in selected_indices)
        )
        border_color = (255, 215, 0) if is_selected else (90, 90, 90)
        cv2.rectangle(canvas, (0, 0), (canvas.shape[1] - 1, canvas.shape[0] - 1), border_color, 2)
        cv2.putText(
            canvas,
            f"{title_prefix} {display_idx}",
            (8, max(16, int(canvas.shape[0] * 0.12))),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            border_color,
            1,
            cv2.LINE_AA,
        )
        tiles.append(canvas)

    if not tiles:
        return None

    tile_h = max(tile.shape[0] for tile in tiles)
    tile_w = max(tile.shape[1] for tile in tiles)
    cols = min(6, len(tiles))
    rows = int(math.ceil(len(tiles) / cols))
    montage = np.zeros((rows * tile_h, cols * tile_w, 3), dtype=np.uint8)
    for idx, tile in enumerate(tiles):
        row = idx // cols
        col = idx % cols
        y0 = row * tile_h
        x0 = col * tile_w
        montage[y0 : y0 + tile.shape[0], x0 : x0 + tile.shape[1]] = tile

    destination.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(destination), cv2.cvtColor(montage, cv2.COLOR_RGB2BGR))
    return destination


def _depth_tile_to_u8(depth_image: np.ndarray) -> np.ndarray | None:
    depth = np.asarray(depth_image)
    if depth.ndim == 3 and depth.shape[2] == 1:
        depth = depth[..., 0]
    if depth.ndim != 2:
        return None
    valid = np.isfinite(depth) & (depth > 0)
    if not np.any(valid):
        return None
    depth_valid = depth[valid]
    depth_min = float(np.min(depth_valid))
    depth_max = float(np.max(depth_valid))
    span = max(depth_max - depth_min, 1e-6)
    normalized = np.zeros_like(depth, dtype=np.float32)
    normalized[valid] = (depth[valid] - depth_min) / span
    normalized = np.clip(np.round(normalized * 255.0), 0, 255).astype(np.uint8)
    color = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
    color[~valid] = 0
    return cv2.cvtColor(color, cv2.COLOR_BGR2RGB)


def _save_depth_candidate_montage(
    tensor_bank: Any,
    destination: Path,
    *,
    title_prefix: str,
    candidate_indices: Optional[list[int]] = None,
    title_indices: Optional[list[int]] = None,
    selected_indices: Optional[set[int]] = None,
    limit: int = 24,
) -> Optional[Path]:
    bank = _tensor_bank_to_numpy(tensor_bank)
    if bank is None:
        return None
    if bank.ndim != 5 or bank.shape[0] <= 0 or bank.shape[1] <= 0:
        return None

    if candidate_indices:
        indices = [
            int(idx)
            for idx in candidate_indices
            if 0 <= int(idx) < int(bank.shape[1])
        ][: max(1, int(limit))]
    else:
        candidates = min(int(bank.shape[1]), int(limit))
        indices = list(range(candidates))
    if not indices:
        return None

    tiles: list[np.ndarray] = []
    for idx in indices:
        image = _depth_tile_to_u8(bank[0, idx])
        if image is None:
            continue
        canvas = image.copy()
        display_idx = int(title_indices[idx]) if title_indices is not None and 0 <= idx < len(title_indices) else idx
        is_selected = (
            candidate_indices is not None
            or (selected_indices is not None and idx in selected_indices)
        )
        border_color = (255, 215, 0) if is_selected else (90, 90, 90)
        cv2.rectangle(canvas, (0, 0), (canvas.shape[1] - 1, canvas.shape[0] - 1), border_color, 2)
        cv2.putText(
            canvas,
            f"{title_prefix} {display_idx}",
            (8, max(16, int(canvas.shape[0] * 0.12))),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            border_color,
            1,
            cv2.LINE_AA,
        )
        tiles.append(canvas)

    if not tiles:
        return None

    tile_h = max(tile.shape[0] for tile in tiles)
    tile_w = max(tile.shape[1] for tile in tiles)
    cols = min(6, len(tiles))
    rows = int(math.ceil(len(tiles) / cols))
    montage = np.zeros((rows * tile_h, cols * tile_w, 3), dtype=np.uint8)
    for idx, tile in enumerate(tiles):
        row = idx // cols
        col = idx % cols
        y0 = row * tile_h
        x0 = col * tile_w
        montage[y0 : y0 + tile.shape[0], x0 : x0 + tile.shape[1]] = tile

    destination.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(destination), cv2.cvtColor(montage, cv2.COLOR_RGB2BGR))
    return destination


def _render_candidate_review_banks(
    *,
    entries: list[dict[str, Any]],
    label: str,
    K: np.ndarray,
    resolution: tuple[int, int],
    frame_size: tuple[int, int],
    modules: dict[str, Any],
    pose_estimator: Any,
    object_center_object_m: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Render top-K pose hypotheses with per-entry K_crop so each object fills the frame.

    K is the full-frame camera matrix (calibrated for frame_size).
    resolution is the desired render target (H, W).

    The key idea: instead of using the full-frame K directly (wrong scale/offset for the
    small render buffer), we compute a per-hypothesis crop box by projecting the mesh
    vertices through the hypothesis pose into the full-frame image plane, then call
    ``get_K_crop_resize`` to derive K_crop whose principal point is centred on the
    projected mesh bounding box.  Panda3D receives K_crop, so the object is centred in
    the render regardless of where the .obj origin sits relative to the mesh geometry.

    Consistent visual scale: flat objects (e.g. washers) seen edge-on have a tiny
    projected 2D bbox even though the object is the same physical size.  We compute the
    3D bounding sphere of the mesh and project its radius at the object depth to get a
    rotation-invariant minimum crop size.  Every render shows the object at the same
    apparent scale regardless of orientation.

    Behind-camera vertex handling: ``project_points_robust`` clips z<0.1 → extreme UV
    → inflated bbox.  We compute camera-space z explicitly and exclude z≤0 vertices
    before building the bbox.
    """
    if not entries:
        return {}

    import torch
    from megapose.lib3d.camera_geometry import (
        get_K_crop_resize,
        project_points_robust,
    )

    pose_matrices: list[np.ndarray] = []
    valid_entries: list[dict[str, Any]] = []
    for entry in entries:
        pose_matrix = entry.get("pose_matrix")
        if pose_matrix is None:
            continue
        pose_matrices.append(np.asarray(pose_matrix, dtype=np.float32))
        valid_entries.append(entry)
    if not pose_matrices:
        return {}

    n = len(valid_entries)
    renderer = pose_estimator.coarse_model.renderer
    TCO = torch.as_tensor(np.stack(pose_matrices, axis=0), dtype=torch.float32)
    K_np = np.asarray(K, dtype=np.float32)
    K_batch = torch.as_tensor(
        np.repeat(K_np[None, :, :], n, axis=0), dtype=torch.float32
    )

    # Sample mesh points to determine per-hypothesis 2D support.
    # We keep the render geometry untouched and only adjust K_crop so the render
    # thumbnails stay centered on the physical object center across rotations.
    mesh_db = pose_estimator.coarse_model.mesh_db
    meshes = mesh_db.select([label] * n)
    pts = meshes.sample_points(2000, deterministic=True)  # [n, 2000, 3]

    center_object_np = np.zeros(3, dtype=np.float32)
    if object_center_object_m is not None:
        center_object_np = np.asarray(object_center_object_m, dtype=np.float32).reshape(3)
    center_object_3d = torch.as_tensor(
        np.repeat(center_object_np[None, :], n, axis=0),
        dtype=torch.float32,
        device=pts.device,
    )  # [n, 3]

    # Rotation-invariant support radius around the true object center. This keeps
    # the render scale stable while also centering every crop on the same physical
    # point of the part instead of the silhouette midpoint.
    sphere_radius = (pts - center_object_3d.unsqueeze(1)).norm(dim=-1).max(dim=1).values  # [n]
    ctr_h = torch.cat(
        [center_object_3d, torch.ones(n, 1, device=center_object_3d.device)], dim=-1
    )  # [n, 4]
    ctr_cam = torch.bmm(TCO[:, :3].reshape(n, 3, 4), ctr_h.unsqueeze(-1)).squeeze(-1)  # [n, 3]
    z_ctr = ctr_cam[:, 2].clamp(min=1e-3)  # [n], depth of object center
    sphere_px_w = 2.0 * sphere_radius * K_batch[:, 0, 0] / z_ctr  # [n]
    sphere_px_h = 2.0 * sphere_radius * K_batch[:, 1, 1] / z_ctr  # [n]

    # Compute camera-space z to filter behind-camera vertices before bbox.
    # project_points_robust clips z < 0.1 → extreme UV → inflated bbox.
    pts_h = torch.cat(
        [pts, torch.ones(n, pts.shape[1], 1, device=pts.device)], dim=-1
    )  # [n, 2000, 4]
    pts_cam = torch.bmm(TCO[:, :3].reshape(n, 3, 4), pts_h.transpose(1, 2)).transpose(1, 2)
    z_vals = pts_cam[..., 2]  # [n, 2000]
    in_front = z_vals > 1e-3

    uv_all = project_points_robust(pts, K_batch, TCO)  # [n, 2000, 2]
    center_uv = project_points_robust(center_object_3d.unsqueeze(1), K_batch, TCO).squeeze(1)  # [n, 2]

    render_h, render_w = int(resolution[0]), int(resolution[1])
    frame_h, frame_w = int(frame_size[0]), int(frame_size[1])
    aspect = render_w / render_h
    margin = 1.4

    crop_boxes_list: list[Any] = []
    for i in range(n):
        valid_mask = in_front[i]
        uv_i = uv_all[i][valid_mask] if valid_mask.sum() >= 4 else uv_all[i]
        cx_i = center_uv[i, 0]
        cy_i = center_uv[i, 1]
        half_w = torch.max(torch.abs(uv_i[:, 0] - cx_i))
        half_h = torch.max(torch.abs(uv_i[:, 1] - cy_i))
        bw_i = (2.0 * half_w) * margin
        bh_i = (2.0 * half_h) * margin

        # Enforce a minimum crop equal to the projected sphere around the true
        # object center so every orientation uses the same visual anchor.
        bw_i = torch.max(bw_i, sphere_px_w[i] * margin)
        bh_i = torch.max(bh_i, sphere_px_h[i] * margin)

        # Force aspect ratio to match render target.
        bw_f = torch.max(bw_i, bh_i * aspect)
        bh_f = torch.max(bh_i, bw_i / aspect)

        crop_boxes_list.append(
            torch.stack([cx_i - bw_f / 2, cy_i - bh_f / 2,
                         cx_i + bw_f / 2, cy_i + bh_f / 2]).unsqueeze(0)
        )

    crop_boxes = torch.cat(crop_boxes_list, dim=0)  # [n, 4]

    K_crop = get_K_crop_resize(
        K=K_batch,
        boxes=crop_boxes,
        orig_size=(frame_h, frame_w),
        crop_resize=(render_h, render_w),
    )  # [n, 3, 3] — K calibrated for the render-target resolution

    # Render with original TCO and K_crop. K_crop is centered on the projected
    # object center, so every thumbnail rotates around the same physical center.
    light_datas = [_make_feature_revealing_light_rig(modules) for _ in valid_entries]
    render_output = renderer.render(
        labels=[label] * n,
        TCO=TCO,
        K=K_crop,
        light_datas=light_datas,
        resolution=resolution,
        render_depth=True,
        render_mask=False,
        render_normals=True,
    )

    review: dict[str, np.ndarray] = {}
    if render_output.rgbs is not None:
        review["rgbs"] = render_output.rgbs.detach().cpu().numpy()[None, ...]
    if render_output.normals is not None:
        review["normals"] = render_output.normals.detach().cpu().numpy()[None, ...]
    if render_output.depths is not None:
        review["depths"] = render_output.depths.detach().cpu().numpy()[None, ...]
    return review


def _entry_hypothesis_id(entry: dict[str, Any]) -> int | None:
    raw_value = entry.get("hypothesis_id")
    if raw_value is None:
        return None
    try:
        text = str(raw_value).strip()
        if not text:
            return None
        return int(float(text))
    except Exception:
        return None


def _reorder_entries_by_hypothesis(
    entries: list[dict[str, Any]],
    hypothesis_ids: list[int],
    *,
    limit: int,
) -> list[dict[str, Any]]:
    if not entries:
        return []
    if not hypothesis_ids:
        return entries[: max(0, int(limit))]

    by_id: dict[int, dict[str, Any]] = {}
    for entry in entries:
        hyp_id = _entry_hypothesis_id(entry)
        if hyp_id is None or hyp_id in by_id:
            continue
        by_id[hyp_id] = entry

    ordered: list[dict[str, Any]] = []
    used_ids: set[int] = set()
    for hyp_id in hypothesis_ids:
        entry = by_id.get(int(hyp_id))
        if entry is None:
            continue
        ordered.append(entry)
        used_ids.add(int(hyp_id))
        if len(ordered) >= int(limit):
            return ordered[: int(limit)]

    for entry in entries:
        hyp_id = _entry_hypothesis_id(entry)
        if hyp_id is not None and hyp_id in used_ids:
            continue
        ordered.append(entry)
        if len(ordered) >= int(limit):
            break
    return ordered[: int(limit)]


def _collect_pose_entries(
    data_tc: Any,
    *,
    score_field: str,
    limit: int,
) -> list[dict[str, Any]]:
    if data_tc is None:
        return []
    infos = getattr(data_tc, "infos", None)
    poses = getattr(data_tc, "poses", None)
    if infos is None or poses is None:
        return []
    try:
        df = infos.reset_index(drop=True)
        poses_np = poses.detach().cpu().numpy()
    except Exception:
        return []

    entries: list[dict[str, Any]] = []
    for idx, (_, row) in enumerate(df.iterrows()):
        record = {str(key): _json_safe(val) for key, val in row.to_dict().items()}
        record["_row_index"] = idx
        if idx < len(poses_np):
            record["pose_matrix"] = np.asarray(poses_np[idx], dtype=np.float32)
        entries.append(record)

    def score_of(entry: dict[str, Any]) -> float:
        value = entry.get(score_field)
        try:
            score = float(value)
        except Exception:
            return float("-inf")
        return score if math.isfinite(score) else float("-inf")

    entries.sort(key=score_of, reverse=True)
    return entries[: max(0, int(limit))]


def _resize_tile_rgb(image: np.ndarray, *, max_width: int = 320, max_height: int = 220) -> np.ndarray:
    h, w = image.shape[:2]
    if h <= 0 or w <= 0:
        return image
    scale = min(float(max_width) / float(w), float(max_height) / float(h), 1.0)
    if scale >= 0.999:
        return image
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)


def _draw_pose_axes_on_image(
    image: np.ndarray,
    *,
    pose_matrix: np.ndarray,
    K: np.ndarray,
    object_center_object_m: np.ndarray | None = None,
    axis_length_m: float = 0.015,
    draw_labels: bool = True,
) -> np.ndarray:
    debug = image.copy()
    rotation = np.asarray(pose_matrix[:3, :3], dtype=np.float32)
    translation = np.asarray(pose_matrix[:3, 3], dtype=np.float32)
    center_object = np.zeros(3, dtype=np.float32)
    if object_center_object_m is not None:
        center_object = np.asarray(object_center_object_m, dtype=np.float32).reshape(3)
    origin_xyz_m = (rotation @ center_object) + translation
    origin_uv = _project_point_uv(origin_xyz_m, K)
    if origin_uv is None:
        return debug

    ox = int(round(origin_uv[0]))
    oy = int(round(origin_uv[1]))
    cv2.circle(debug, (ox, oy), 4, (0, 255, 255), -1, cv2.LINE_AA)

    axis_length = max(0.005, float(axis_length_m or 0.015))
    axis_defs = [
        ((0, 0, 255), rotation[:, 0], "X"),
        ((0, 255, 0), rotation[:, 1], "Y"),
        ((255, 120, 0), rotation[:, 2], "Z"),
    ]
    for color, axis_vec, axis_name in axis_defs:
        endpoint_xyz = origin_xyz_m + (np.asarray(axis_vec, dtype=np.float32) * axis_length)
        endpoint_uv = _project_point_uv(endpoint_xyz, K)
        if endpoint_uv is None:
            continue
        ex = int(round(endpoint_uv[0]))
        ey = int(round(endpoint_uv[1]))
        cv2.arrowedLine(debug, (ox, oy), (ex, ey), color, 2, cv2.LINE_AA, tipLength=0.18)
        if draw_labels:
            cv2.putText(
                debug,
                axis_name,
                (ex + 4, ey - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                2,
                cv2.LINE_AA,
            )
    return debug


def _render_pose_overlay_images(
    *,
    rgb: np.ndarray,
    K: np.ndarray,
    resolution: tuple[int, int],
    label: str,
    modules: dict[str, Any],
    pose_estimator: Any,
    entries: list[dict[str, Any]],
    object_center_object_m: np.ndarray | None = None,
    axis_length_m: float = 0.015,
) -> list[dict[str, Any]]:
    if not entries:
        return []

    import torch

    pose_matrices = []
    valid_entries: list[dict[str, Any]] = []
    for entry in entries:
        pose_matrix = entry.get("pose_matrix")
        if pose_matrix is None:
            continue
        pose_matrices.append(np.asarray(pose_matrix, dtype=np.float32))
        valid_entries.append(entry)
    if not pose_matrices:
        return []

    renderer = pose_estimator.coarse_model.renderer
    TCO = torch.as_tensor(np.stack(pose_matrices, axis=0), dtype=torch.float32)
    K_tensor = torch.as_tensor(
        np.repeat(np.asarray(K, dtype=np.float32)[None, :, :], len(valid_entries), axis=0),
        dtype=torch.float32,
    )
    light_datas = [_make_feature_revealing_light_rig(modules) for _ in valid_entries]
    render_output = renderer.render(
        labels=[label] * len(valid_entries),
        TCO=TCO,
        K=K_tensor,
        light_datas=light_datas,
        resolution=resolution,
        render_depth=True,
        render_mask=False,
        render_normals=False,
    )

    rendered_entries: list[dict[str, Any]] = []
    for idx, entry in enumerate(valid_entries):
        rendered_rgb = render_output.rgbs[idx].detach().cpu().permute(1, 2, 0).numpy()
        rendered_rgb = np.clip(np.round(rendered_rgb * 255.0), 0, 255).astype(np.uint8)
        rendered_depth = None
        if render_output.depths is not None:
            rendered_depth = render_output.depths[idx].detach().cpu().permute(1, 2, 0).numpy()
        overlay = make_mesh_overlay_on_rgb(rgb, rendered_rgb)
        overlay = _draw_pose_axes_on_image(
            overlay,
            pose_matrix=np.asarray(entry["pose_matrix"], dtype=np.float32),
            K=np.asarray(K, dtype=np.float32),
            object_center_object_m=object_center_object_m,
            axis_length_m=float(axis_length_m),
            draw_labels=True,
        )
        rendered_entries.append(
            {
                "entry": entry,
                "overlay": overlay,
                "rendered_depth": rendered_depth,
            }
        )
    return rendered_entries


def _save_pose_overlay_montage(
    *,
    rgb: np.ndarray,
    K: np.ndarray,
    resolution: tuple[int, int],
    label: str,
    modules: dict[str, Any],
    pose_estimator: Any,
    entries: list[dict[str, Any]],
    destination: Path,
    score_field: str,
    title_prefix: str,
    object_center_object_m: np.ndarray | None = None,
    axis_length_m: float = 0.015,
) -> Optional[Path]:
    rendered_entries = _render_pose_overlay_images(
        rgb=rgb,
        K=K,
        resolution=resolution,
        label=label,
        modules=modules,
        pose_estimator=pose_estimator,
        entries=entries,
        object_center_object_m=object_center_object_m,
        axis_length_m=axis_length_m,
    )
    if not rendered_entries:
        return None

    tiles: list[np.ndarray] = []
    for rendered in rendered_entries:
        entry = rendered["entry"]
        overlay = np.asarray(rendered["overlay"], dtype=np.uint8)
        rendered_depth = rendered.get("rendered_depth")
        tile = _resize_tile_rgb(overlay)
        score = entry.get(score_field)
        try:
            score_text = f"{float(score):.3f}"
        except Exception:
            score_text = "n/a"
        hyp = entry.get("hypothesis_id")
        hyp_text = "" if hyp is None else f" h{int(hyp)}"
        cv2.rectangle(tile, (0, 0), (tile.shape[1] - 1, tile.shape[0] - 1), (255, 215, 0), 2)
        cv2.putText(
            tile,
            f"{title_prefix}{hyp_text} {score_field}={score_text}",
            (8, max(18, int(tile.shape[0] * 0.10))),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 215, 0),
            1,
            cv2.LINE_AA,
        )
        if rendered_depth is not None:
            nonzero = rendered_depth[rendered_depth > 0]
            if nonzero.size:
                z_text = f"z~{float(np.median(nonzero)):.3f}m"
                cv2.putText(
                    tile,
                    z_text,
                    (8, max(36, int(tile.shape[0] * 0.18))),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.42,
                    (255, 215, 0),
                    1,
                    cv2.LINE_AA,
                )
        tiles.append(tile)

    if not tiles:
        return None

    tile_h = max(tile.shape[0] for tile in tiles)
    tile_w = max(tile.shape[1] for tile in tiles)
    cols = min(4, len(tiles))
    rows = int(math.ceil(len(tiles) / cols))
    montage = np.zeros((rows * tile_h, cols * tile_w, 3), dtype=np.uint8)
    for idx, tile in enumerate(tiles):
        row = idx // cols
        col = idx % cols
        y0 = row * tile_h
        x0 = col * tile_w
        montage[y0 : y0 + tile.shape[0], x0 : x0 + tile.shape[1]] = tile

    destination.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(destination), cv2.cvtColor(montage, cv2.COLOR_RGB2BGR))
    return destination


def _save_single_pose_overlay(
    *,
    rgb: np.ndarray,
    K: np.ndarray,
    resolution: tuple[int, int],
    label: str,
    modules: dict[str, Any],
    pose_estimator: Any,
    pose_matrix: np.ndarray,
    destination: Path,
    title_text: str = "",
    object_center_object_m: np.ndarray | None = None,
    axis_length_m: float = 0.015,
) -> Optional[Path]:
    rendered_entries = _render_pose_overlay_images(
        rgb=rgb,
        K=K,
        resolution=resolution,
        label=label,
        modules=modules,
        pose_estimator=pose_estimator,
        entries=[{"pose_matrix": np.asarray(pose_matrix, dtype=np.float32)}],
        object_center_object_m=object_center_object_m,
        axis_length_m=axis_length_m,
    )
    if not rendered_entries:
        return None

    overlay = np.asarray(rendered_entries[0]["overlay"], dtype=np.uint8)
    if title_text:
        cv2.putText(
            overlay,
            title_text,
            (12, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            (20, 20, 20),
            4,
            cv2.LINE_AA,
        )
        cv2.putText(
            overlay,
            title_text,
            (12, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(destination), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    return destination


def _save_bgr_image(
    bgr: np.ndarray,
    destination: Path,
) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(destination), bgr)
    return destination


def _save_json_file(
    destination: Path,
    payload: Any,
) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(_json_safe(payload), indent=2), encoding="utf-8")
    return destination


def _candidate_depth_stats(
    det: dict[str, Any],
    *,
    image_shape: tuple[int, int],
    depth_m: np.ndarray | None,
) -> dict[str, float | None]:
    if depth_m is None:
        return {"z_min_m": None, "z_max_m": None}
    try:
        selection_mask = make_selection_mask(
            image_shape,
            det.get("mask"),
            np.asarray(det.get("bbox_xyxy"), dtype=np.float32),
        )
    except Exception:
        return {"z_min_m": None, "z_max_m": None}
    if selection_mask.shape != depth_m.shape:
        return {"z_min_m": None, "z_max_m": None}
    valid_mask = selection_mask.astype(bool) & np.isfinite(depth_m) & (depth_m > 0.0)
    if not np.any(valid_mask):
        return {"z_min_m": None, "z_front_m": None, "z_max_m": None}
    z_vals = depth_m[valid_mask].astype(np.float64)
    sorted_z = np.sort(z_vals)
    support_points = max(3, int(math.ceil(len(sorted_z) * 0.02)))
    support_points = min(len(sorted_z), support_points)
    return {
        "z_min_m": float(sorted_z[0]),
        "z_front_m": float(sorted_z[support_points - 1]),
        "z_max_m": float(sorted_z[-1]),
    }


def _save_segmentation_candidates_overlay(
    bgr: np.ndarray,
    detections: list[dict[str, Any]],
    *,
    depth_m: np.ndarray | None,
    destination: Path,
) -> Path:
    canvas = bgr.copy()
    palette = [
        (0, 255, 255),
        (255, 180, 0),
        (255, 0, 255),
        (0, 200, 0),
        (0, 128, 255),
    ]
    image_shape = tuple(canvas.shape[:2])
    for det in detections:
        rank = int(det.get("rank", 0))
        color = palette[rank % len(palette)]
        bbox_xyxy = np.asarray(det.get("bbox_xyxy"), dtype=np.float32)
        selection_mask = make_selection_mask(image_shape, det.get("mask"), bbox_xyxy)
        if np.any(selection_mask):
            overlay = canvas.copy()
            overlay[selection_mask.astype(bool)] = color
            canvas = cv2.addWeighted(canvas, 0.86, overlay, 0.14, 0.0)
            contours, _ = cv2.findContours(
                (selection_mask.astype(np.uint8) * 255),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )
            cv2.drawContours(canvas, contours, -1, color, 2, cv2.LINE_AA)

        depth_stats = _candidate_depth_stats(det, image_shape=image_shape, depth_m=depth_m)
        z_front = depth_stats.get("z_front_m")
        z_text = "zfront n/a" if z_front is None else f"zfront {float(z_front):.3f}m"
        x1, y1, _, _ = [int(v) for v in bbox_xyxy.tolist()]
        text_org = (max(8, x1), max(18, y1 - 6))
        text = f"#{rank} s {float(det.get('score', 0.0)):.2f} {z_text}"
        cv2.putText(
            canvas,
            text,
            text_org,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (20, 20, 20),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            text,
            text_org,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            color,
            1,
            cv2.LINE_AA,
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(destination), canvas)
    return destination


def _save_camera_k_json(
    K: np.ndarray,
    resolution: tuple[int, int],
    destination: Path,
) -> Path:
    payload = {
        "K": np.asarray(K, dtype=np.float32).tolist(),
        "fx": float(K[0, 0]),
        "fy": float(K[1, 1]),
        "cx": float(K[0, 2]),
        "cy": float(K[1, 2]),
        "resolution": {
            "height": int(resolution[0]),
            "width": int(resolution[1]),
        },
    }
    return _save_json_file(destination, payload)


def _latest_refiner_pose_collection(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "infos") and hasattr(value, "poses"):
        return value
    if not isinstance(value, dict):
        return None

    best_iteration = -1
    best_value = None
    first_value = None
    for key, candidate in value.items():
        if not (hasattr(candidate, "infos") and hasattr(candidate, "poses")):
            continue
        if first_value is None:
            first_value = candidate
        iteration = -1
        text = str(key).strip()
        if "=" in text:
            try:
                iteration = int(text.split("=")[-1])
            except Exception:
                iteration = -1
        elif text.isdigit():
            iteration = int(text)
        if iteration >= best_iteration:
            best_iteration = iteration
            best_value = candidate
    return best_value if best_value is not None else first_value


def _save_megapose_debug_bundle(
    output_dir: Path,
    extra_data: dict[str, Any],
    *,
    rgb: np.ndarray,
    K: np.ndarray,
    resolution: tuple[int, int],
    label: str,
    modules: dict[str, Any],
    pose_estimator: Any,
    object_center_object_m: np.ndarray | None,
    axis_length_m: float,
    request_id: str | None,
) -> dict[str, str]:
    artifacts: dict[str, str] = {}
    coarse_dir = output_dir / "coarse"
    refined_dir = output_dir / "refined"
    coarse_dir.mkdir(parents=True, exist_ok=True)
    refined_dir.mkdir(parents=True, exist_ok=True)

    coarse_block = extra_data.get("coarse") or {}
    coarse_data = coarse_block.get("data") or {}
    topk_block = extra_data.get("coarse_filter") or {}
    topk_preds = topk_block.get("preds")
    topk_records = _serialize_pose_collection(topk_preds)
    refiner_all_block = extra_data.get("refiner_all_hypotheses") or {}
    refiner_preds = _latest_refiner_pose_collection(refiner_all_block.get("preds"))

    selected_hypothesis_ids: list[int] = []
    for record in topk_records:
        if not isinstance(record, dict):
            continue
        hyp_id = _entry_hypothesis_id(record)
        if hyp_id is None or hyp_id in selected_hypothesis_ids:
            continue
        selected_hypothesis_ids.append(hyp_id)
        if len(selected_hypothesis_ids) >= 5:
            break

    camera_k_path = _save_camera_k_json(
        K=np.asarray(K, dtype=np.float32),
        resolution=resolution,
        destination=coarse_dir / "camera_k.json",
    )
    artifacts["camera_k"] = str(camera_k_path)

    coarse_debug = coarse_data.get("debug") or {}
    crop_bank = coarse_debug.get("images_crop")
    render_bank = coarse_debug.get("renders")
    coarse_crop_path = _save_candidate_montage(
        crop_bank,
        coarse_dir / "top5_crops.png",
        title_prefix="id",
        candidate_indices=selected_hypothesis_ids,
        selected_indices=set(selected_hypothesis_ids),
        limit=5,
    )
    if coarse_crop_path is not None:
        artifacts["coarse_top5_crops"] = str(coarse_crop_path)

    coarse_entries = _reorder_entries_by_hypothesis(
        _collect_pose_entries(
            topk_preds,
            score_field="coarse_logit",
            limit=max(5, len(selected_hypothesis_ids)),
        ),
        selected_hypothesis_ids,
        limit=5,
    )

    # Attempt a fresh re-render of the top-5 coarse poses using per-hypothesis K_crop
    # so every render shows the object consistently fitted to the render frame.
    # Falls back to the coarse-model debug renders if the re-render cannot be done.
    coarse_render_path = None
    try:
        _render_size = getattr(getattr(pose_estimator, "coarse_model", None), "render_size", None)
        if _render_size is not None and coarse_entries:
            _proper_renders = _render_candidate_review_banks(
                entries=coarse_entries,
                label=label,
                K=K,
                resolution=tuple(_render_size),
                frame_size=tuple(resolution),
                modules=modules,
                pose_estimator=pose_estimator,
                object_center_object_m=object_center_object_m,
            )
            _proper_bank = _proper_renders.get("rgbs")
            if _proper_bank is not None:
                coarse_render_path = _save_candidate_montage(
                    _proper_bank,
                    coarse_dir / "top5_renders.png",
                    title_prefix="id",
                    candidate_indices=list(range(len(coarse_entries))),
                    title_indices=selected_hypothesis_ids,
                    selected_indices=set(range(len(coarse_entries))),
                    limit=5,
                )
    except Exception:
        pass

    # Fall back to the coarse-model debug renders if re-render failed or was unavailable.
    if coarse_render_path is None:
        coarse_render_path = _save_candidate_montage(
            render_bank,
            coarse_dir / "top5_renders.png",
            title_prefix="id",
            candidate_indices=selected_hypothesis_ids,
            selected_indices=set(selected_hypothesis_ids),
            limit=5,
        )
    if coarse_render_path is not None:
        artifacts["coarse_top5_renders"] = str(coarse_render_path)

    coarse_overlay_path = _save_pose_overlay_montage(
        rgb=rgb,
        K=K,
        resolution=resolution,
        label=label,
        modules=modules,
        pose_estimator=pose_estimator,
        entries=coarse_entries,
        destination=coarse_dir / "top5_pose_overlays.png",
        score_field="coarse_logit",
        title_prefix="coarse",
        object_center_object_m=object_center_object_m,
        axis_length_m=axis_length_m,
    )
    if coarse_overlay_path is not None:
        artifacts["coarse_top5_pose_overlays"] = str(coarse_overlay_path)

    refined_entries = _reorder_entries_by_hypothesis(
        _collect_pose_entries(
            refiner_preds,
            score_field="pose_logit",
            limit=max(5, len(selected_hypothesis_ids)),
        ),
        selected_hypothesis_ids,
        limit=5,
    )
    refined_overlay_path = _save_pose_overlay_montage(
        rgb=rgb,
        K=K,
        resolution=resolution,
        label=label,
        modules=modules,
        pose_estimator=pose_estimator,
        entries=refined_entries,
        destination=refined_dir / "top5_pose_overlays.png",
        score_field="pose_logit",
        title_prefix="refined",
        object_center_object_m=object_center_object_m,
        axis_length_m=axis_length_m,
    )
    if refined_overlay_path is not None:
        artifacts["refined_top5_pose_overlays"] = str(refined_overlay_path)

    _runtime_log(
        request_id,
        f"curated_bin_picking_artifacts_saved dir={output_dir}",
    )
    return artifacts


def save_point_cloud_ply(
    points: np.ndarray,
    colors_rgba: np.ndarray,
    destination: Path,
) -> Path | None:
    import trimesh

    if len(points) == 0:
        return None
    destination.parent.mkdir(parents=True, exist_ok=True)
    point_cloud = trimesh.points.PointCloud(vertices=points, colors=colors_rgba)
    point_cloud.export(str(destination))
    return destination


def make_pose_scene(
    mesh_path: Path,
    pose_matrix: np.ndarray,
    mesh_units: str,
    mesh_scale: float,
    object_center_object_m: np.ndarray | None = None,
    scene_points: np.ndarray | None = None,
    scene_colors_rgba: np.ndarray | None = None,
    segmented_points: np.ndarray | None = None,
) -> Any:
    import trimesh

    mesh = trimesh.load_mesh(mesh_path, force="mesh")
    unit_scale = {"m": 1.0, "mm": 0.001}[str(mesh_units).strip().lower()]
    mesh.apply_scale(unit_scale * float(mesh_scale))
    pose_matrix = np.asarray(pose_matrix, dtype=np.float64)
    if not np.isfinite(pose_matrix).all():
        pose_matrix = np.eye(4, dtype=np.float64)
    center_world = compute_object_center_world(
        mesh.vertices.copy(),
        pose_matrix,
        object_center_object=object_center_object_m,
    )
    mesh.apply_transform(pose_matrix)

    if hasattr(mesh.visual, "face_colors"):
        mesh.visual.face_colors = np.tile(
            np.array([[0, 0, 0, 180]], dtype=np.uint8),
            (len(mesh.faces), 1),
        )

    extent = float(max(np.max(mesh.extents), 0.05))
    axis_length = extent * 0.8
    axis_radius = max(axis_length * 0.02, 0.0015)
    center_radius = max(axis_length * 0.06, 0.003)
    center_pose = pose_matrix.astype(np.float64).copy()
    center_pose[:3, 3] = center_world

    axis_mesh = trimesh.creation.axis(
        origin_size=center_radius * 1.5,
        axis_radius=axis_radius,
        axis_length=axis_length,
        transform=center_pose,
    )
    center_sphere = trimesh.creation.icosphere(subdivisions=2, radius=center_radius)
    center_sphere.apply_translation(center_world)
    if hasattr(center_sphere.visual, "face_colors"):
        center_sphere.visual.face_colors = np.tile(
            np.array([[255, 255, 255, 255]], dtype=np.uint8),
            (len(center_sphere.faces), 1),
        )

    scene = trimesh.Scene()
    # trimesh serializes accessor min/max from vertex data; any non-finite
    # vertex leaks `Infinity`/`NaN` into the GLB JSON header, which browser
    # JSON.parse rejects. Filter defensively even though upstream callers
    # should already be clean.
    if scene_points is not None and len(scene_points) > 0 and scene_colors_rgba is not None:
        sp = np.asarray(scene_points)
        sc = np.asarray(scene_colors_rgba)
        finite = np.isfinite(sp).all(axis=1)
        sp = sp[finite]
        sc = sc[finite]
        if len(sp) > 0:
            scene_cloud = trimesh.points.PointCloud(vertices=sp, colors=sc)
            scene.add_geometry(scene_cloud, node_name="scene_point_cloud")
    if segmented_points is not None and len(segmented_points) > 0:
        seg = np.asarray(segmented_points)
        finite = np.isfinite(seg).all(axis=1)
        seg = seg[finite]
        if len(seg) > 0:
            segmented_colors = np.tile(
                np.array([[0, 200, 0, 255]], dtype=np.uint8),
                (len(seg), 1),
            )
            segmented_cloud = trimesh.points.PointCloud(
                vertices=seg,
                colors=segmented_colors,
            )
            scene.add_geometry(segmented_cloud, node_name="segmented_point_cloud")
    scene.add_geometry(mesh, node_name="posed_cad_mesh")
    scene.add_geometry(axis_mesh, node_name="pose_axes")
    scene.add_geometry(center_sphere, node_name="pose_center")
    return scene


def save_pose_scene_glb(
    mesh_path: Path,
    pose_matrix: np.ndarray,
    mesh_units: str,
    mesh_scale: float,
    destination: Path,
    object_center_object_m: np.ndarray | None = None,
    scene_points: np.ndarray | None = None,
    scene_colors_rgba: np.ndarray | None = None,
    segmented_points: np.ndarray | None = None,
) -> Path:
    scene = make_pose_scene(
        mesh_path=mesh_path,
        pose_matrix=pose_matrix,
        mesh_units=mesh_units,
        mesh_scale=mesh_scale,
        object_center_object_m=object_center_object_m,
        scene_points=scene_points,
        scene_colors_rgba=scene_colors_rgba,
        segmented_points=segmented_points,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    scene.export(str(destination))
    return destination


def save_multi_pose_scene_glb(
    mesh_path: Path,
    pose_matrices: list[np.ndarray | list[list[float]]],
    mesh_units: str,
    mesh_scale: float,
    destination: Path,
    object_center_object_m: np.ndarray | None = None,
) -> Path:
    import trimesh

    scene = trimesh.Scene()
    unit_scale = {"m": 1.0, "mm": 0.001}[str(mesh_units).strip().lower()]
    palette = [
        np.array([0, 0, 0, 180], dtype=np.uint8),
        np.array([0, 90, 180, 150], dtype=np.uint8),
        np.array([0, 150, 110, 150], dtype=np.uint8),
        np.array([180, 90, 0, 150], dtype=np.uint8),
        np.array([110, 0, 150, 150], dtype=np.uint8),
    ]

    for idx, raw_pose_matrix in enumerate(pose_matrices or []):
        pose_matrix = np.asarray(raw_pose_matrix, dtype=np.float64)
        if pose_matrix.shape != (4, 4) or not np.isfinite(pose_matrix).all():
            continue
        mesh = trimesh.load_mesh(mesh_path, force="mesh")
        if hasattr(mesh, "dump") and not hasattr(mesh, "vertices"):
            mesh = mesh.dump(concatenate=True)
        if not hasattr(mesh, "vertices") or len(mesh.vertices) == 0:
            continue
        mesh.apply_scale(unit_scale * float(mesh_scale))
        center_world = compute_object_center_world(
            mesh.vertices.copy(),
            pose_matrix,
            object_center_object=object_center_object_m,
        )
        mesh.apply_transform(pose_matrix)
        color_rgba = palette[idx % len(palette)]
        if hasattr(mesh.visual, "face_colors"):
            mesh.visual.face_colors = np.tile(color_rgba, (len(mesh.faces), 1))

        extent = float(max(np.max(mesh.extents), 0.05))
        axis_length = extent * 0.8
        axis_radius = max(axis_length * 0.02, 0.0015)
        center_radius = max(axis_length * 0.06, 0.003)
        center_pose = pose_matrix.copy()
        center_pose[:3, 3] = center_world

        axis_mesh = trimesh.creation.axis(
            origin_size=center_radius * 1.5,
            axis_radius=axis_radius,
            axis_length=axis_length,
            transform=center_pose,
        )
        center_sphere = trimesh.creation.icosphere(subdivisions=2, radius=center_radius)
        center_sphere.apply_translation(center_world)
        if hasattr(center_sphere.visual, "face_colors"):
            center_sphere.visual.face_colors = np.tile(
                np.array([[255, 255, 255, 255]], dtype=np.uint8),
                (len(center_sphere.faces), 1),
            )
        scene.add_geometry(mesh, node_name=f"posed_cad_mesh_{idx:02d}")
        scene.add_geometry(axis_mesh, node_name=f"pose_axes_{idx:02d}")
        scene.add_geometry(center_sphere, node_name=f"pose_center_{idx:02d}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    scene.export(str(destination))
    return destination


def make_render_mask(rendered_rgb: np.ndarray, rendered_depth: np.ndarray | None = None) -> np.ndarray:
    if rendered_depth is not None:
        depth = np.asarray(rendered_depth)
        if depth.ndim == 3:
            depth = depth[..., 0]
        mask = np.isfinite(depth) & (depth > 0)
        return mask.astype(np.uint8) * 255

    mask = np.any(rendered_rgb > 0, axis=-1)
    return mask.astype(np.uint8) * 255


def make_mesh_overlay_on_rgb(rgb: np.ndarray, rendered_rgb: np.ndarray) -> np.ndarray:
    mask = np.any(rendered_rgb > 0, axis=-1)
    overlay = rgb.copy().astype(np.float32)
    # Keep the overlay readable regardless of renderer polarity by blending
    # the actual rendered CAD appearance instead of forcing a fixed black tint.
    rendered_rgb_f = rendered_rgb.astype(np.float32)
    overlay[mask] = overlay[mask] * 0.45 + rendered_rgb_f[mask] * 0.55
    return np.clip(np.round(overlay), 0, 255).astype(np.uint8)


def make_contour_overlay_on_rgb(
    rgb: np.ndarray,
    rendered_rgb: np.ndarray,
    rendered_depth: np.ndarray | None = None,
) -> np.ndarray:
    mask = make_render_mask(rendered_rgb, rendered_depth)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    contour_overlay = rgb.copy()
    if contours:
        cv2.drawContours(
            contour_overlay,
            contours,
            contourIdx=-1,
            color=(0, 200, 0),
            thickness=2,
            lineType=cv2.LINE_AA,
        )
    return contour_overlay


def save_pose_visualizations(
    rgb: np.ndarray,
    pose_matrix: np.ndarray,
    K: np.ndarray,
    resolution: tuple[int, int],
    label: str,
    modules: dict[str, Any],
    pose_estimator: Any,
    mesh_overlay_path: Path,
    contour_overlay_path: Path,
    all_results_image_path: Path,
) -> dict[str, str]:
    import torch

    renderer = pose_estimator.coarse_model.renderer
    TCO = torch.as_tensor(pose_matrix, dtype=torch.float32).unsqueeze(0)
    K_tensor = torch.as_tensor(K, dtype=torch.float32).unsqueeze(0)
    light_datas = [_make_feature_revealing_light_rig(modules)]
    render_output = renderer.render(
        labels=[label],
        TCO=TCO,
        K=K_tensor,
        light_datas=light_datas,
        resolution=resolution,
        render_depth=True,
        render_mask=False,
        render_normals=False,
    )
    rendered_rgb = render_output.rgbs[0].detach().cpu().permute(1, 2, 0).numpy()
    rendered_rgb = np.clip(np.round(rendered_rgb * 255.0), 0, 255).astype(np.uint8)
    rendered_depth = None
    if render_output.depths is not None:
        rendered_depth = render_output.depths[0].detach().cpu().permute(1, 2, 0).numpy()

    mesh_overlay = make_mesh_overlay_on_rgb(rgb, rendered_rgb)
    contour_overlay = make_contour_overlay_on_rgb(rgb, rendered_rgb, rendered_depth)
    all_results = np.concatenate([rgb, contour_overlay, mesh_overlay], axis=1)

    mesh_overlay_path.parent.mkdir(parents=True, exist_ok=True)
    contour_overlay_path.parent.mkdir(parents=True, exist_ok=True)
    all_results_image_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(mesh_overlay_path), cv2.cvtColor(mesh_overlay, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(contour_overlay_path), cv2.cvtColor(contour_overlay, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(all_results_image_path), cv2.cvtColor(all_results, cv2.COLOR_RGB2BGR))
    return {
        "mesh_overlay_path": str(mesh_overlay_path),
        "contour_overlay_path": str(contour_overlay_path),
        "all_results_image_path": str(all_results_image_path),
    }


def save_pose_3d_assets(
    rgb: np.ndarray,
    depth_m: np.ndarray | None,
    K: np.ndarray,
    mask: np.ndarray | None,
    mesh_path: Path,
    pose_matrix: np.ndarray,
    mesh_units: str,
    mesh_scale: float,
    object_center_object_m: np.ndarray | None,
    scene_point_cloud_ply_path: Path,
    segmented_point_cloud_ply_path: Path,
    pose_scene_glb_path: Path,
    max_depth_m: float | None = None,
    segmented_points_override: np.ndarray | None = None,
) -> dict[str, str | None]:
    scene_point_cloud_path = None
    segmented_point_cloud_ply = None
    scene_points = None
    scene_colors = None
    segmented_points = None
    segmented_colors = None

    if depth_m is not None:
        scene_points, scene_colors = depth_to_point_cloud(
            rgb,
            depth_m,
            K,
            max_depth_m=max_depth_m,
        )
        if segmented_points_override is not None and len(segmented_points_override) > 0:
            seg_pts = np.asarray(segmented_points_override, dtype=np.float32)
            if seg_pts.ndim == 2 and seg_pts.shape[1] >= 3:
                segmented_points = np.ascontiguousarray(seg_pts[:, :3], dtype=np.float32)
                segmented_colors = np.tile(
                    np.array([[0, 220, 60, 255]], dtype=np.uint8),
                    (len(segmented_points), 1),
                )
        if segmented_points is None:
            segmented_points, segmented_colors = depth_to_point_cloud(
                rgb,
                depth_m,
                K,
                mask=mask,
                max_depth_m=max_depth_m,
            )
        scene_point_cloud_path = save_point_cloud_ply(
            scene_points,
            scene_colors,
            scene_point_cloud_ply_path,
        )
        segmented_point_cloud_ply = save_point_cloud_ply(
            segmented_points,
            segmented_colors,
            segmented_point_cloud_ply_path,
        )

    pose_scene_path = save_pose_scene_glb(
        mesh_path=mesh_path,
        pose_matrix=pose_matrix,
        mesh_units=mesh_units,
        mesh_scale=mesh_scale,
        object_center_object_m=object_center_object_m,
        destination=pose_scene_glb_path,
        scene_points=scene_points,
        scene_colors_rgba=scene_colors,
        segmented_points=segmented_points,
    )
    return {
        "scene_point_cloud_ply_path": None
        if scene_point_cloud_path is None
        else str(scene_point_cloud_path),
        "segmented_point_cloud_ply_path": None
        if segmented_point_cloud_ply is None
        else str(segmented_point_cloud_ply),
        "pose_scene_glb_path": str(pose_scene_path),
    }


def _build_output_dir(params: Dict[str, Any], request_id: str | None, label: str) -> Path | None:
    if not bool(params.get("save_outputs", False)):
        return None
    root_value = params.get("output_root") or DEFAULT_OUTPUT_ROOT
    root = resolve_workspace_path(root_value, workspace_root=WORKSPACE_ROOT)
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    suffix = _safe_slug(request_id or label)
    output_dir = root / timestamp / suffix
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def prepare_megapose_environment(source_root: Path, runtime_data_root: Path) -> Path:
    megapose_src_dir = _resolve_megapose_src_dir(source_root)
    runtime_data_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MEGAPOSE_DATA_DIR", str(runtime_data_root))
    os.environ.setdefault("HOME", str(Path.home()))
    conda_prefix = Path(os.environ.get("CONDA_PREFIX", source_root / ".conda-placeholder"))
    conda_prefix.mkdir(parents=True, exist_ok=True)
    os.environ["CONDA_PREFIX"] = str(conda_prefix)
    cuda_visible_devices = str(os.environ.get("CUDA_VISIBLE_DEVICES", "")).strip()
    if cuda_visible_devices.lower() == "auto":
        cuda_visible_devices = ""
    if cuda_visible_devices:
        try:
            [int(part.strip()) for part in cuda_visible_devices.split(",") if part.strip()]
        except Exception:
            cuda_visible_devices = ""
    if not cuda_visible_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    egl_visible_devices = str(os.environ.get("EGL_VISIBLE_DEVICES", "")).strip()
    if egl_visible_devices.lower() == "auto":
        egl_visible_devices = ""
    if egl_visible_devices:
        try:
            [int(part.strip()) for part in egl_visible_devices.split(",") if part.strip()]
        except Exception:
            egl_visible_devices = ""
    if not egl_visible_devices:
        os.environ["EGL_VISIBLE_DEVICES"] = "0"
    os.environ.setdefault("MEGAPOSE_PANDA3D_DISPLAY", "p3headlessgl")
    nvidia_egl_icd = Path("/usr/share/glvnd/egl_vendor.d/10_nvidia.json")
    if (
        platform.system().lower() == "linux"
        and nvidia_egl_icd.exists()
        and str(os.environ.get("CUDA_VISIBLE_DEVICES", "")).strip() not in ("", "auto")
    ):
        # Hybrid Intel+NVIDIA desktops can default EGL to Mesa for the X11 display
        # even when CUDA inference is running on NVIDIA. Force Panda3D's offscreen
        # renderer onto NVIDIA's surfaceless EGL path so render/readback stays on
        # the same GPU as MegaPose inference.
        os.environ.setdefault(
            "__EGL_VENDOR_LIBRARY_FILENAMES",
            str(nvidia_egl_icd),
        )
        os.environ.setdefault("EGL_PLATFORM", "surfaceless")
    if str(megapose_src_dir) not in sys.path:
        sys.path.insert(0, str(megapose_src_dir))
    return megapose_src_dir


def import_megapose_modules(source_root: Path, runtime_data_root: Path) -> dict[str, Any]:
    cache_key = str(source_root.resolve())
    with _MEGAPOSE_IMPORT_LOCK:
        cached = _IMPORTED_MODULES.get(cache_key)
        if cached is not None:
            return cached
        prepare_megapose_environment(source_root, runtime_data_root)
        from megapose.lib3d.symmetries import ContinuousSymmetry
        from megapose.datasets.object_dataset import RigidObject, RigidObjectDataset
        from megapose.inference.pose_estimator import PoseEstimator
        from megapose.inference.types import ObservationTensor
        from megapose.lib3d.rigid_mesh_database import MeshDataBase
        from megapose.panda3d_renderer import Panda3dLightData
        from megapose.panda3d_renderer.panda3d_batch_renderer import Panda3dBatchRenderer
        from megapose.training.pose_models_cfg import (
            check_update_config as check_update_config_pose,
        )
        from megapose.training.pose_models_cfg import create_model_pose
        from megapose.utils.models_compat import change_keys_of_older_models
        from megapose.utils.tensor_collection import PandasTensorCollection

        modules = {
            "ContinuousSymmetry": ContinuousSymmetry,
            "RigidObject": RigidObject,
            "RigidObjectDataset": RigidObjectDataset,
            "PoseEstimator": PoseEstimator,
            "ObservationTensor": ObservationTensor,
            "MeshDataBase": MeshDataBase,
            "Panda3dLightData": Panda3dLightData,
            "Panda3dBatchRenderer": Panda3dBatchRenderer,
            "check_update_config_pose": check_update_config_pose,
            "create_model_pose": create_model_pose,
            "change_keys_of_older_models": change_keys_of_older_models,
            "PandasTensorCollection": PandasTensorCollection,
        }
        _IMPORTED_MODULES[cache_key] = modules
        return modules


def load_cfg(path: Path) -> Any:
    import yaml
    from omegaconf import OmegaConf

    cfg = yaml.load(path.read_text(encoding="utf-8"), Loader=yaml.UnsafeLoader)
    if isinstance(cfg, dict):
        cfg = OmegaConf.load(path)
    return cfg


SUPPORTED_SO3_GRID_SIZES = (72, 512, 576, 4608)


def resolve_supported_so3_grid_size(value: Any) -> int:
    try:
        requested = int(value)
    except (TypeError, ValueError):
        return 72
    if requested <= 0:
        return 0
    if requested in SUPPORTED_SO3_GRID_SIZES:
        return requested
    return min(
        SUPPORTED_SO3_GRID_SIZES,
        key=lambda candidate: (abs(candidate - requested), candidate),
    )


def ensure_models_available(weights_root: Path, model_name: str) -> None:
    run_ids = {
        MODEL_CONFIGS[model_name]["coarse_run_id"],
        MODEL_CONFIGS[model_name]["refiner_run_id"],
    }
    missing = []
    for run_id in run_ids:
        run_dir = weights_root / run_id
        if not (run_dir / "config.yaml").exists() or not (run_dir / "checkpoint.pth.tar").exists():
            missing.append(run_id)
    if missing:
        raise FileNotFoundError("megapose_weights_missing")


def load_pose_model(
    run_id: str,
    *,
    weights_root: Path,
    renderer: Any,
    mesh_db_batched: Any,
    modules: dict[str, Any],
    device: Any,
    render_size: tuple[int, int] | None = None,
) -> Any:
    import torch

    run_dir = weights_root / run_id
    cfg = load_cfg(run_dir / "config.yaml")
    cfg = modules["check_update_config_pose"](cfg)
    if render_size is not None:
        cfg.render_size = [int(render_size[0]), int(render_size[1])]
    model = modules["create_model_pose"](cfg, renderer=renderer, mesh_db=mesh_db_batched)
    checkpoint = torch.load(run_dir / "checkpoint.pth.tar", map_location="cpu")
    state_dict = modules["change_keys_of_older_models"](checkpoint["state_dict"])
    model.load_state_dict(state_dict)
    model = model.to(device).eval()
    model.cfg = cfg
    model.config = cfg
    return model


def resolve_device(device_name: str, torch_module: Any) -> Any:
    if device_name == "auto":
        return torch_module.device("cuda" if torch_module.cuda.is_available() else "cpu")
    if device_name == "cuda" and not torch_module.cuda.is_available():
        raise RuntimeError("megapose_cuda_unavailable")
    return torch_module.device(device_name)


def resolve_renderer_workers(
    renderer_workers: Any,
    device: Any,
    *,
    allow_multiprocess: bool = False,
) -> int:
    try:
        requested_workers = int(renderer_workers)
    except (TypeError, ValueError):
        requested_workers = 0
    if requested_workers < 0:
        requested_workers = 0
    if device.type == "cpu":
        return 0
    # Panda3D offscreen depth readback has proven stable in-process on this stack,
    # but its worker-process path can return empty RAM images on Linux/GLX GPU runs.
    # Keep the default on the safe in-process renderer unless explicitly opted in.
    if not allow_multiprocess:
        return 0
    return requested_workers


def configure_cpu_runtime(
    cpu_threads: int,
    cpu_interop_threads: int,
    device: Any,
    torch_module: Any,
) -> int | None:
    if device.type != "cpu":
        return None
    cpu_count = os.cpu_count() or 1
    requested_threads = int(cpu_threads)
    if requested_threads <= 0:
        requested_threads = min(8, cpu_count)
    requested_threads = max(1, min(requested_threads, cpu_count))
    requested_interop_threads = max(1, int(cpu_interop_threads))
    os.environ["MKL_NUM_THREADS"] = str(requested_threads)
    os.environ["OMP_NUM_THREADS"] = str(requested_threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(requested_threads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(requested_threads)
    torch_module.set_num_threads(requested_threads)
    try:
        torch_module.set_num_interop_threads(requested_interop_threads)
    except RuntimeError:
        pass
    return requested_threads


def build_detections(
    label: str,
    bbox_xyxy: np.ndarray,
    score: float,
    pandas_tensor_collection_cls: Any,
) -> Any:
    import pandas as pd
    import torch

    infos = pd.DataFrame(
        {
            "label": [label],
            "batch_im_id": [0],
            "instance_id": [0],
            "score": [float(score)],
        }
    )
    bboxes = torch.as_tensor(bbox_xyxy, dtype=torch.float32).unsqueeze(0)
    return pandas_tensor_collection_cls(infos=infos, bboxes=bboxes)


def choose_model(model_name: str, has_depth: bool) -> str:
    if model_name != "auto":
        return model_name
    return "rgbd" if has_depth else "rgb-multi-hypothesis"


def resolve_requested_model(params: Dict[str, Any], has_depth: bool) -> str:
    requested_model = choose_model(str(params.get("model", "rgbd")).lower(), has_depth)
    if requested_model in MODEL_CONFIGS:
        return requested_model
    return "rgbd" if has_depth else "rgb-multi-hypothesis"


def resolve_model_bundle_config(
    selected_model: str,
    params: Dict[str, Any],
) -> tuple[dict[str, Any], str]:
    model_cfg = dict(MODEL_CONFIGS[selected_model])
    variant_suffix = "default"
    if bool(params.get("use_rgb_refiner_only", False)) and selected_model == "rgbd":
        model_cfg["refiner_run_id"] = MODEL_CONFIGS["rgb"]["refiner_run_id"]
        variant_suffix = "rgb_refiner_only"
    return model_cfg, variant_suffix


def _prewarm_frame_shape(params: Dict[str, Any]) -> tuple[int, int]:
    camera_data = (
        params.get("camera_data") if isinstance(params.get("camera_data"), dict) else {}
    )
    intrinsics = camera_data.get("intrinsics") if isinstance(camera_data, dict) else {}
    if not isinstance(intrinsics, dict):
        intrinsics = {}
    resolution = (
        _parse_intrinsics_resolution(camera_data.get("resolution"))
        or _parse_intrinsics_resolution(camera_data.get("image_size"))
        or _parse_intrinsics_resolution(intrinsics.get("resolution"))
        or _parse_intrinsics_resolution((params.get("intrinsics") or {}).get("resolution"))
    )
    if resolution:
        width, height = resolution
        return int(height), int(width)
    width = int(params.get("prewarm_width", 1280))
    height = int(params.get("prewarm_height", 720))
    return max(64, height), max(64, width)


def _warmup_estimator_inference(
    bundle: EstimatorBundle,
    assets: ResolvedObjectAssets,
    params: Dict[str, Any],
    model_name: str,
) -> None:
    frame_h, frame_w = _prewarm_frame_shape(params)
    rgb = np.zeros((frame_h, frame_w, 3), dtype=np.uint8)
    camera_data = build_camera_data(params, rgb.shape[:2])
    depth_m = None
    if MODEL_CONFIGS[model_name]["requires_depth"]:
        depth_m = np.full((frame_h, frame_w), 0.35, dtype=np.float32)
    box_w = max(48, int(frame_w * 0.12))
    box_h = max(48, int(frame_h * 0.14))
    x1 = max(0, (frame_w - box_w) // 2)
    y1 = max(0, (frame_h - box_h) // 2)
    x2 = min(frame_w, x1 + box_w)
    y2 = min(frame_h, y1 + box_h)
    bbox_xyxy = np.array([x1, y1, x2, y2], dtype=np.float32)
    selection_mask = np.zeros((frame_h, frame_w), dtype=bool)
    selection_mask[y1:y2, x1:x2] = True
    megapose_rgb, megapose_depth, _, _, _ = mask_observation_to_selected_object(
        rgb,
        depth_m,
        selection_mask,
        bbox_xyxy,
        K=np.asarray(camera_data["K"], dtype=np.float32),
    )
    detections_tc = build_detections(
        assets.label,
        bbox_xyxy,
        1.0,
        bundle.modules["PandasTensorCollection"],
    ).to(bundle.device)
    observation = bundle.modules["ObservationTensor"].from_numpy(
        megapose_rgb,
        megapose_depth,
        camera_data["K"],
    ).to(bundle.device)
    with bundle.pose_lock:
        bundle.pose_estimator.run_inference_pipeline(
            observation=observation,
            detections=detections_tc,
            n_refiner_iterations=1,
            n_pose_hypotheses=1,
            return_coarse_debug_data=False,
        )


def parse_mtl_asset_references(mtl_path: Path) -> list[str]:
    asset_keywords = {
        "map_kd",
        "map_ka",
        "map_ks",
        "map_ke",
        "map_bump",
        "bump",
        "disp",
        "decal",
        "norm",
        "refl",
    }
    assets: list[str] = []
    for line in mtl_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split()
        if not parts or parts[0].lower() not in asset_keywords:
            continue
        for token in reversed(parts[1:]):
            if not token.startswith("-"):
                assets.append(token)
                break
    return assets


def rewrite_mtl_with_local_assets(source_mtl: Path, destination_mtl: Path) -> None:
    asset_keywords = {
        "map_kd",
        "map_ka",
        "map_ks",
        "map_ke",
        "map_bump",
        "bump",
        "disp",
        "decal",
        "norm",
        "refl",
    }
    rewritten_lines: list[str] = []
    for line in source_mtl.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            rewritten_lines.append(line)
            continue
        parts = stripped.split()
        if not parts or parts[0].lower() not in asset_keywords:
            rewritten_lines.append(line)
            continue
        asset_idx = None
        for idx in range(len(parts) - 1, 0, -1):
            if not parts[idx].startswith("-"):
                asset_idx = idx
                break
        if asset_idx is None:
            rewritten_lines.append(line)
            continue
        parts[asset_idx] = Path(parts[asset_idx]).name
        rewritten_lines.append(" ".join(parts))
    destination_mtl.write_text("\n".join(rewritten_lines) + "\n", encoding="utf-8")


def resolve_obj_dependencies(obj_path: Path) -> list[Path]:
    dependencies: list[Path] = []
    source_dir = obj_path.parent
    for line in obj_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped.startswith("mtllib "):
            continue
        mtl_name = stripped.split(maxsplit=1)[1]
        mtl_path = source_dir / mtl_name
        if mtl_path.exists():
            dependencies.append(mtl_path)
            for asset_name in parse_mtl_asset_references(mtl_path):
                asset_path = source_dir / asset_name
                if asset_path.exists():
                    dependencies.append(asset_path)
    return dependencies


def _obj_file_has_normals(obj_path: Path) -> bool:
    try:
        with obj_path.open("r", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                if line.startswith("vn "):
                    return True
    except Exception:
        return False
    return False


def _export_recentered_trimesh_obj(
    source_mesh_path: Path,
    destination_obj_path: Path,
    frame_info_path: Path,
    *,
    export_config: dict[str, Any],
) -> Path:
    import trimesh
    from trimesh.exchange.obj import export_obj

    mesh = trimesh.load_mesh(
        source_mesh_path,
        force="mesh",
        process=bool(export_config.get("process", True)),
    )
    if hasattr(mesh, "dump") and not hasattr(mesh, "vertices"):
        mesh = mesh.dump(concatenate=True)
    if not hasattr(mesh, "vertices") or len(mesh.vertices) == 0:
        raise ValueError("megapose_mesh_vertices_missing")

    try:
        mesh.remove_unreferenced_vertices()
    except Exception:
        pass
    if bool(export_config.get("merge_vertices", True)):
        try:
            mesh.merge_vertices()
        except Exception:
            pass
    if bool(export_config.get("recompute_vertex_normals", True)):
        try:
            mesh.fix_normals()
        except Exception:
            pass
        _ = mesh.vertex_normals

    center_mesh_units = 0.5 * (mesh.bounds[0] + mesh.bounds[1])
    mesh.apply_translation(-center_mesh_units)
    obj_text = export_obj(
        mesh,
        include_normals=True,
        include_color=False,
        include_texture=False,
        return_texture=False,
    )

    destination_obj_path.parent.mkdir(parents=True, exist_ok=True)
    destination_obj_path.write_text(obj_text, encoding="utf-8")
    frame_info_path.write_text(
        json.dumps(
            {
                "source_mesh": str(source_mesh_path),
                "origin_adjustment_mesh_units": center_mesh_units.tolist(),
                "export_config": {str(k): _json_safe(v) for k, v in export_config.items()},
                "vertex_count": int(len(mesh.vertices)),
                "triangle_count": int(len(mesh.faces)),
                "has_vertex_normals": True,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return destination_obj_path


def recenter_obj_mesh(source_obj_path: Path, destination_obj_path: Path, frame_info_path: Path) -> Path:
    vertex_lines: list[tuple[int, np.ndarray]] = []
    mtllibs: list[str] = []
    lines = source_obj_path.read_text(encoding="utf-8").splitlines()
    for line_idx, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("mtllib "):
            mtllibs.append(stripped.split(maxsplit=1)[1])
        elif stripped.startswith("v "):
            parts = stripped.split()
            if len(parts) >= 4:
                vertex = np.array([float(parts[1]), float(parts[2]), float(parts[3])], dtype=np.float64)
                vertex_lines.append((line_idx, vertex))
    if not vertex_lines:
        raise ValueError("megapose_obj_vertices_missing")
    vertices = np.stack([vertex for _, vertex in vertex_lines], axis=0)
    center_mesh_units = 0.5 * (vertices.min(axis=0) + vertices.max(axis=0))
    rewritten_lines = list(lines)
    for line_idx, vertex in vertex_lines:
        centered = vertex - center_mesh_units
        rewritten_lines[line_idx] = f"v {centered[0]:.6f} {centered[1]:.6f} {centered[2]:.6f}"
    destination_obj_path.parent.mkdir(parents=True, exist_ok=True)
    destination_obj_path.write_text("\n".join(rewritten_lines) + "\n", encoding="utf-8")

    copied_assets: list[str] = []
    source_dir = source_obj_path.parent
    dest_dir = destination_obj_path.parent
    for mtllib_name in mtllibs:
        source_mtl = source_dir / mtllib_name
        if not source_mtl.exists():
            continue
        destination_mtl = dest_dir / mtllib_name
        rewrite_mtl_with_local_assets(source_mtl, destination_mtl)
        copied_assets.append(str(destination_mtl))
        for asset_name in parse_mtl_asset_references(source_mtl):
            source_asset = source_dir / asset_name
            if not source_asset.exists():
                continue
            destination_asset = dest_dir / Path(asset_name).name
            destination_asset.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_asset, destination_asset)
            copied_assets.append(str(destination_asset))
    frame_info = {
        "source_mesh": str(source_obj_path),
        "origin_adjustment_mesh_units": center_mesh_units.tolist(),
        "copied_assets": copied_assets,
    }
    frame_info_path.write_text(json.dumps(frame_info, indent=2), encoding="utf-8")
    return destination_obj_path


def convert_mesh_to_obj(
    mesh_path: Path,
    label: str,
    runtime_data_root: Path,
    *,
    export_config: dict[str, Any] | None = None,
) -> Path:
    mesh_dir = runtime_data_root / "meshes" / _safe_slug(label)
    mesh_dir.mkdir(parents=True, exist_ok=True)
    obj_path = mesh_dir / f"{_safe_slug(label)}.obj"
    frame_info_path = mesh_dir / "mesh_frame.json"
    export_config = dict(export_config or {})
    dependency_paths = [mesh_path]
    if mesh_path.suffix.lower() == ".obj":
        dependency_paths.extend(resolve_obj_dependencies(mesh_path))
    cached_export_config = None
    if frame_info_path.exists():
        try:
            cached_export_config = json.loads(frame_info_path.read_text(encoding="utf-8")).get("export_config")
        except Exception:
            cached_export_config = None
    if (
        obj_path.exists()
        and frame_info_path.exists()
        and cached_export_config == export_config
        and obj_path.stat().st_mtime >= max(path.stat().st_mtime for path in dependency_paths if path.exists())
    ):
        return obj_path
    if mesh_path.suffix.lower() == ".obj" and not bool(
        export_config.get("recompute_vertex_normals", True)
    ):
        return recenter_obj_mesh(mesh_path, obj_path, frame_info_path)
    return _export_recentered_trimesh_obj(
        mesh_path,
        obj_path,
        frame_info_path,
        export_config=export_config,
    )


def get_estimator_bundle(params: Dict[str, Any], assets: ResolvedObjectAssets, selected_model: str) -> EstimatorBundle:
    source_root = resolve_workspace_path(
        params.get("megapose_source_root") or DEFAULT_MEGAPOSE_SOURCE_ROOT,
        workspace_root=WORKSPACE_ROOT,
    )
    runtime_data_root = resolve_workspace_path(
        params.get("runtime_data_root") or DEFAULT_RUNTIME_DATA_ROOT,
        workspace_root=WORKSPACE_ROOT,
    )
    weights_root = resolve_workspace_path(
        params.get("megapose_weights_root") or DEFAULT_MEGAPOSE_WEIGHTS_ROOT,
        workspace_root=WORKSPACE_ROOT,
    )

    import torch

    device_name = str(params.get("device", "auto")).lower()
    device = resolve_device(device_name, torch)
    configure_cpu_runtime(
        int(params.get("cpu_threads", 0)),
        int(params.get("cpu_interop_threads", 1)),
        device,
        torch,
    )
    try:
        torch.multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    renderer_workers = resolve_renderer_workers(
        params.get("renderer_workers", 0),
        device,
        allow_multiprocess=bool(params.get("allow_renderer_workers", False)),
    )
    requested_coarse_grid_size = int(params.get("coarse_grid_size", 72))
    coarse_grid_size = resolve_supported_so3_grid_size(requested_coarse_grid_size)
    mesh_units = str(params.get("mesh_units", "mm"))
    mesh_scale = float(params.get("mesh_scale", 1.0))
    mesh_origin_strategy = "recenter_runtime_obj_v2"
    object_runtime_config = _load_object_runtime_config(assets.object_folder)
    mesh_export_config = _build_mesh_export_config(params, object_runtime_config)
    pose_render_size = _resolve_pose_render_size(params, object_runtime_config)
    symmetry_info = _build_symmetry_config(params, object_runtime_config)
    model_cfg, model_variant_suffix = resolve_model_bundle_config(selected_model, params)
    symmetry_cache_signature = (
        symmetry_info.get("type") or "",
        None
        if symmetry_info.get("axis") is None
        else tuple(round(float(v), 6) for v in np.asarray(symmetry_info["axis"]).tolist()),
        int(symmetry_info.get("samples", 64)),
    )
    cache_key = (
        str(source_root.resolve()),
        str(weights_root.resolve()),
        str(runtime_data_root.resolve()),
        str(assets.mesh_path.resolve()),
        assets.label,
        selected_model,
        model_variant_suffix,
        mesh_units,
        mesh_scale,
        mesh_origin_strategy,
        device.type,
        coarse_grid_size,
        renderer_workers,
        pose_render_size,
        tuple(sorted((str(k), _json_safe(v)) for k, v in mesh_export_config.items())),
        symmetry_cache_signature,
        _estimator_cache_scope(device, renderer_workers),
    )
    with _ESTIMATOR_CACHE_LOCK:
        cached = _ESTIMATOR_CACHE.get(cache_key)
        if cached is not None:
            return cached

        modules = import_megapose_modules(source_root, runtime_data_root)
        required_run_ids = {model_cfg["coarse_run_id"], model_cfg["refiner_run_id"]}
        missing = []
        for run_id in required_run_ids:
            run_dir = weights_root / run_id
            if not (run_dir / "config.yaml").exists() or not (run_dir / "checkpoint.pth.tar").exists():
                missing.append(run_id)
        if missing:
            raise FileNotFoundError("megapose_weights_missing")
        mesh_obj_path = convert_mesh_to_obj(
            assets.mesh_path,
            assets.label,
            runtime_data_root,
            export_config=mesh_export_config,
        )
        symmetries_continuous = []
        if symmetry_info.get("enabled") and symmetry_info.get("axis") is not None:
            symmetries_continuous = [
                modules["ContinuousSymmetry"](
                    offset=np.zeros(3, dtype=np.float32),
                    axis=np.asarray(symmetry_info["axis"], dtype=np.float32),
                )
            ]
        object_dataset = modules["RigidObjectDataset"](
            [
                modules["RigidObject"](
                    label=assets.label,
                    mesh_path=mesh_obj_path,
                    mesh_units=mesh_units,
                    scaling_factor=mesh_scale,
                    symmetries_continuous=symmetries_continuous,
                )
            ]
        )
        mesh_db = modules["MeshDataBase"].from_object_ds(object_dataset)
        mesh_db_batched = mesh_db.batched(
            n_sym=int(symmetry_info.get("samples", 64)),
        ).to(device)
        renderer = modules["Panda3dBatchRenderer"](
            object_dataset=object_dataset,
            n_workers=renderer_workers,
            preload_cache=False,
            split_objects=False,
            device=device,
        )
        coarse_model = load_pose_model(
            model_cfg["coarse_run_id"],
            weights_root=weights_root,
            renderer=renderer,
            mesh_db_batched=mesh_db_batched,
            modules=modules,
            device=device,
            render_size=pose_render_size,
        )
        refiner_model = load_pose_model(
            model_cfg["refiner_run_id"],
            weights_root=weights_root,
            renderer=renderer,
            mesh_db_batched=mesh_db_batched,
            modules=modules,
            device=device,
            render_size=pose_render_size,
        )
        pose_estimator = modules["PoseEstimator"](
            refiner_model=refiner_model,
            coarse_model=coarse_model,
            detector_model=None,
            depth_refiner=None,
            bsz_objects=8,
            bsz_images=128,
        ).to(device)
        if coarse_grid_size > 0:
            if coarse_grid_size != requested_coarse_grid_size:
                print(
                    "[megapose_bin_picking] coarse_grid_size "
                    f"{requested_coarse_grid_size} unsupported; using {coarse_grid_size}",
                    flush=True,
                )
            pose_estimator.load_SO3_grid(coarse_grid_size)

        bundle = EstimatorBundle(
            modules=modules,
            object_dataset=object_dataset,
            pose_estimator=pose_estimator,
            mesh_obj_path=mesh_obj_path,
            device=device,
            selected_model=selected_model,
            pose_lock=threading.Lock(),
            cache_key=cache_key,
            object_runtime_config=object_runtime_config,
        )
        _ESTIMATOR_CACHE[cache_key] = bundle
        return bundle


def prewarm_megapose_bin_picking(
    params: Dict[str, Any],
    *,
    force_bundle_warm: bool = False,
) -> Dict[str, Any]:
    raw_object_folder = _clean_path_string(params.get("object_folder"))
    if not raw_object_folder:
        raise ValueError("megapose_object_folder_missing")

    label_override = (
        str(params.get("label_override") or params.get("object_id") or "").strip() or None
    )
    object_resolve_started_at = time.perf_counter()
    assets = resolve_object_assets(
        raw_object_folder,
        workspace_root=WORKSPACE_ROOT,
        extra_roots=[WORKSPACE_ROOT / "data"],
        label_override=label_override,
        segmentation_model_override=params.get("segmentation_model_path"),
        mesh_extension_priority=list(
            params.get("mesh_extension_priority") or [".glb", ".obj", ".stl"]
        ),
    )
    _runtime_log(
        None,
        "prewarm_resolved_assets "
        f"label={assets.label} mesh={assets.mesh_path.name} seg={assets.segmentation_model_path.name} "
        f"dt={time.perf_counter() - object_resolve_started_at:.3f}s",
    )

    selected_model = resolve_requested_model(params, has_depth=True)

    frame_h, frame_w = _prewarm_frame_shape(params)
    segmentation_backend = resolve_segmentation_backend(
        str(params.get("segmentation_backend", "auto")),
        assets.segmentation_model_path,
    )
    segmentation_device = str(params.get("segmentation_device") or params.get("yolo_device") or params.get("device") or "cpu")
    segmentation_key = f"{segmentation_backend}:{assets.segmentation_model_path.resolve()}"

    segmentation_start = time.perf_counter()
    segmentation_model = get_segmentation_model(assets.segmentation_model_path, segmentation_backend)
    segmentation_load_seconds = time.perf_counter() - segmentation_start

    segmentation_warmup_seconds = 0.0
    with _PREWARM_CACHE_LOCK:
        segmentation_warmed = segmentation_key in (_SAM_WARMED if segmentation_backend == "sam" else _YOLO_WARMED)
    if not segmentation_warmed:
        segmentation_warmup_start = time.perf_counter()
        try:
            processing_max_width = int(params.get("processing_max_width", 1280) or 1280)
            processing_max_height = int(params.get("processing_max_height", 720) or 720)
            scale = min(
                float(processing_max_width) / float(frame_w),
                float(processing_max_height) / float(frame_h),
                1.0,
            )
            warmup_h = max(1, int(round(frame_h * scale)))
            warmup_w = max(1, int(round(frame_w * scale)))
            _warmup_segmentation_predict(
                segmentation_model,
                segmentation_backend,
                (warmup_h, warmup_w),
                float(params.get("yolo_conf", 0.2)),
                segmentation_device,
                int(params.get("yolo_imgsz", 1024) or 1024),
                bool(params.get("retina_masks", True)),
            )
            segmentation_warmup_seconds = time.perf_counter() - segmentation_warmup_start
            with _PREWARM_CACHE_LOCK:
                (_SAM_WARMED if segmentation_backend == "sam" else _YOLO_WARMED).add(segmentation_key)
        except Exception:
            segmentation_warmup_seconds = time.perf_counter() - segmentation_warmup_start

    import torch

    device_name = str(params.get("device", "auto")).lower()
    resolved_device = resolve_device(device_name, torch)
    bundle_seconds = 0.0
    bundle = None
    renderer_workers = resolve_renderer_workers(
        params.get("renderer_workers", 0),
        resolved_device,
        allow_multiprocess=bool(params.get("allow_renderer_workers", False)),
    )
    bundle_cache_scope = _estimator_cache_scope(resolved_device, renderer_workers)
    can_share_bundle = bool(force_bundle_warm) or bundle_cache_scope == "shared"
    if can_share_bundle:
        bundle_start = time.perf_counter()
        bundle = get_estimator_bundle(params, assets, selected_model)
        bundle_seconds = time.perf_counter() - bundle_start

    estimator_warmup_seconds = 0.0
    do_estimator_warmup = bool(params.get("prewarm_estimator_inference", False)) and bundle is not None
    estimator_warmed = False
    if bundle is not None:
        with _PREWARM_CACHE_LOCK:
            estimator_warmed = bundle.cache_key in _ESTIMATOR_WARMED
    if do_estimator_warmup and not estimator_warmed:
        estimator_warmup_start = time.perf_counter()
        try:
            _warmup_estimator_inference(bundle, assets, params, selected_model)
            estimator_warmup_seconds = time.perf_counter() - estimator_warmup_start
            with _PREWARM_CACHE_LOCK:
                _ESTIMATOR_WARMED.add(bundle.cache_key)
        except Exception:
            estimator_warmup_seconds = time.perf_counter() - estimator_warmup_start

    return {
        "status": "ok",
        "object_folder": str(assets.object_folder),
        "label": assets.label,
        "model_name": selected_model,
        "mesh_path": str(assets.mesh_path),
        "segmentation_model_path": str(assets.segmentation_model_path),
        "segmentation_backend": segmentation_backend,
        "device": str(getattr(bundle.device, "type", resolved_device) if bundle is not None else getattr(resolved_device, "type", resolved_device)),
        "renderer_workers": int(params.get("renderer_workers", 0) or 0),
        "estimator_bundle_scope": bundle_cache_scope,
        "timing": {
            "segmentation_model_load_seconds": float(segmentation_load_seconds),
            "segmentation_predict_warmup_seconds": float(segmentation_warmup_seconds),
            "bundle_warmup_seconds": float(bundle_seconds),
            "estimator_inference_warmup_seconds": float(estimator_warmup_seconds),
        },
        "cache_ready": True,
        "segmentation_model_loaded": segmentation_model is not None,
        "estimator_bundle_prewarmed": bool(bundle is not None),
    }


def run_megapose_bin_picking(
    *,
    bgr: np.ndarray,
    depth: np.ndarray | None,
    params: Dict[str, Any],
    request_id: str | None = None,
) -> Dict[str, Any]:
    run_started_at = time.perf_counter()
    timing_log = _BinPickingTimingLog.from_params(params, request_id)
    queue_wait_seconds = _coerce_float(params.get("_worker_queue_wait_seconds"), 0.0) or 0.0
    if queue_wait_seconds > 0.0:
        timing_log.span("vision_worker_queue_wait", run_started_at - queue_wait_seconds)
    _runtime_log(
        request_id,
        f"start bgr_shape={tuple(int(v) for v in bgr.shape)} depth_shape="
        f"{None if depth is None else tuple(int(v) for v in depth.shape)}",
    )
    setup_started_at = time.perf_counter()
    raw_object_folder = _clean_path_string(params.get("object_folder"))
    if not raw_object_folder:
        raise ValueError("megapose_object_folder_missing")

    label_override = (
        str(params.get("label_override") or params.get("object_id") or "").strip() or None
    )
    assets = resolve_object_assets(
        raw_object_folder,
        workspace_root=WORKSPACE_ROOT,
        extra_roots=[WORKSPACE_ROOT / "data"],
        label_override=label_override,
        segmentation_model_override=params.get("segmentation_model_path"),
        mesh_extension_priority=list(
            params.get("mesh_extension_priority") or [".glb", ".obj", ".stl"]
        ),
    )
    timing_log.span("resolve_object_assets", setup_started_at, object_folder=str(assets.object_folder))

    preprocess_started_at = time.perf_counter()
    color_order = str(params.get("input_color_order", "bgr")).lower()
    if color_order == "rgb":
        rgb = bgr.copy()
        bgr_frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    else:
        bgr_frame = bgr.copy()
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)

    # Camera-core publishes depth as float32 metres; only integer (raw-unit) depth
    # needs the depth_scale multiply. If a stale param/buffer ever feeds us the
    # wrong scale or a near-empty depth frame, the point cloud silently collapses,
    # so emit a loud, actionable warning with the depth statistics.
    _depth_raw_dtype = None if depth is None else str(getattr(depth, "dtype", ""))
    _depth_was_integer = bool(depth is not None and np.issubdtype(depth.dtype, np.integer))
    depth_m = normalize_depth_map(depth, float(params.get("depth_scale", 0.001)))
    if depth_m is not None:
        _finite = np.isfinite(depth_m)
        _pos = _finite & (depth_m > 0.0)
        _pos_count = int(np.count_nonzero(_pos))
        _total = int(depth_m.size) or 1
        if _pos_count > 0:
            _vals = depth_m[_pos]
            _dmin, _dmax, _dmed = float(_vals.min()), float(_vals.max()), float(np.median(_vals))
            # Plausible RealSense-class working range; if almost nothing falls in it,
            # the scale is almost certainly wrong (e.g. mm fed as metres, or vice versa).
            _plausible = int(np.count_nonzero((_vals >= 0.02) & (_vals <= 6.0)))
            if _plausible < max(1, int(0.01 * _pos_count)):
                print(
                    f"[{request_id}] WARNING depth looks mis-scaled or stale: "
                    f"dtype={_depth_raw_dtype} integer={_depth_was_integer} "
                    f"depth_scale_param={float(params.get('depth_scale', 0.001))} "
                    f"nonzero_px={_pos_count}/{_total} min={_dmin:.4f} med={_dmed:.4f} max={_dmax:.4f} "
                    f"plausible_in_[0.02,6]m={_plausible}. Point cloud / pose depth will be wrong; "
                    f"check camera-core depth units vs the task's depth_scale."
                )
        elif depth is not None:
            print(
                f"[{request_id}] WARNING depth frame is empty (no positive pixels): "
                f"dtype={_depth_raw_dtype} shape={None if depth is None else tuple(int(v) for v in depth.shape)}. "
                f"Likely a stale/rotated shared-memory buffer; the runtime will fall back to a flat synthetic plane."
            )
    depth_m = filter_depth_range(
        depth_m,
        min_depth_m=_coerce_float(params.get("depth_min_m"), None),
        max_depth_m=_coerce_float(params.get("depth_max_m"), None),
    )
    if depth_m is not None:
        _after = int(np.count_nonzero(np.isfinite(depth_m) & (depth_m > 0.0)))
        _before_min = _coerce_float(params.get("depth_min_m"), None)
        _before_max = _coerce_float(params.get("depth_max_m"), None)
        if _after == 0 and (_before_min or _before_max):
            print(
                f"[{request_id}] WARNING depth_range filter [min={_before_min}, max={_before_max}] "
                f"removed ALL depth pixels — the configured range likely doesn't match the actual scene "
                f"depth (e.g. depth in mm but range in m). Point cloud will be empty."
            )
    processing_max_width = int(params.get("processing_max_width", 1280) or 1280)
    processing_max_height = int(params.get("processing_max_height", 720) or 720)
    yolo_imgsz = int(params.get("yolo_imgsz", 1024) or 1024)
    retina_masks = bool(params.get("retina_masks", True))
    rgb, depth_m = resize_inputs_for_processing(
        rgb,
        depth_m,
        max_width=processing_max_width,
        max_height=processing_max_height,
    )
    camera_data = build_camera_data(params, rgb.shape[:2])
    model_name = resolve_requested_model(params, has_depth=depth_m is not None)
    timing_log.span(
        "preprocess_inputs",
        preprocess_started_at,
        rgb_shape=tuple(int(v) for v in rgb.shape),
        depth_present=depth_m is not None,
        model_name=model_name,
    )
    intr = camera_data.get("intrinsics") or {}
    _runtime_log(
        request_id,
        "camera_ready "
        f"model={model_name} depth={'yes' if depth_m is not None else 'no'} "
        f"fx={float(intr.get('fx', 0.0)):.2f} fy={float(intr.get('fy', 0.0)):.2f} "
        f"cx={float(intr.get('cx', 0.0)):.2f} cy={float(intr.get('cy', 0.0)):.2f}",
    )
    if MODEL_CONFIGS[model_name]["requires_depth"] and depth_m is None:
        raise ValueError("megapose_requires_depth")
    yolo_conf = float(params.get("yolo_conf", 0.2))
    segmentation_backend = resolve_segmentation_backend(
        str(params.get("segmentation_backend", "auto")),
        assets.segmentation_model_path,
    )
    segmentation_device = params.get("segmentation_device", params.get("yolo_device"))
    bundle_started_at = time.perf_counter()
    bundle = get_estimator_bundle(params, assets, model_name)
    object_runtime_config = bundle.object_runtime_config or {}
    annotation_meta = get_mesh_annotation_meta(
        bundle.mesh_obj_path,
        str(params.get("mesh_units", "mm")),
        float(params.get("mesh_scale", 1.0)),
    )
    _runtime_log(
        request_id,
        "estimator_ready "
        f"device={bundle.device.type} cache_key={bundle.cache_key} "
        f"dt={time.perf_counter() - bundle_started_at:.3f}s",
    )
    timing_log.span(
        "estimator_bundle_ready",
        bundle_started_at,
        device=str(bundle.device.type),
        model_name=model_name,
    )
    segmentation_started_at = time.perf_counter()
    detections = run_segmentation(
        rgb=rgb,
        model_path=assets.segmentation_model_path,
        backend=segmentation_backend,
        conf=yolo_conf,
        device=str(segmentation_device) if segmentation_device not in (None, "") else bundle.device.type,
        imgsz=yolo_imgsz,
        retina_masks=retina_masks,
    )
    _runtime_log(
        request_id,
        "segmentation_done "
        f"backend={segmentation_backend} detections={len(detections)} conf={yolo_conf:.3f} "
        f"dt={time.perf_counter() - segmentation_started_at:.3f}s",
    )
    timing_log.span(
        "segmentation",
        segmentation_started_at,
        backend=segmentation_backend,
        device=str(segmentation_device) if segmentation_device not in (None, "") else bundle.device.type,
        detections=len(detections),
        confidence=yolo_conf,
    )
    duplicate_filter_started_at = time.perf_counter()
    detections, duplicate_filter_info = suppress_nested_duplicate_detections(
        detections,
        image_shape=tuple(rgb.shape[:2]),
        depth_m=depth_m,
    )
    timing_log.span(
        "segmentation_duplicate_filter",
        duplicate_filter_started_at,
        kept=int(duplicate_filter_info.get("kept_count", len(detections))),
        suppressed=int(duplicate_filter_info.get("suppressed_count", 0)),
    )
    _runtime_log(
        request_id,
        "segmentation_duplicate_filter "
        f"kept={int(duplicate_filter_info.get('kept_count', len(detections)))} "
        f"suppressed={int(duplicate_filter_info.get('suppressed_count', 0))}",
    )
    image_area_px = float(rgb.shape[0] * rgb.shape[1])
    raw_min_area_px = params.get("selection_min_area_px")
    raw_max_area_px = params.get("selection_max_area_px")
    min_area_ratio = float(params.get("selection_min_area_ratio", 0.0) or 0.0)
    max_area_ratio = float(params.get("selection_max_area_ratio", 0.0) or 0.0)
    selection_min_area_px = max(
        0.0,
        float(
            (
                raw_min_area_px
                if raw_min_area_px is not None and float(raw_min_area_px) > 0.0
                else min_area_ratio * image_area_px
            )
            or 0.0
        ),
    )
    selection_max_area_px = max(
        0.0,
        float(
            (
                raw_max_area_px
                if raw_max_area_px is not None and float(raw_max_area_px) > 0.0
                else max_area_ratio * image_area_px
            )
            or 0.0
        ),
    )
    _bin_roi_raw = params.get("bin_roi") if isinstance(params, dict) else None
    _bin_roi_polygon_uv = None
    if isinstance(_bin_roi_raw, dict):
        _pts = _bin_roi_raw.get("obb_points_uv") or []
        if isinstance(_pts, list) and len(_pts) == 4:
            _bin_roi_polygon_uv = [[float(p[0]), float(p[1])] for p in _pts]
    candidate_select_started_at = time.perf_counter()
    candidate = select_detection_candidate(
        detections,
        image_shape=rgb.shape[:2],
        depth_m=depth_m,
        K=camera_data["K"],
        top_k=int(params.get("selection_top_k", 5)),
        min_confidence=float(params.get("selection_min_confidence", 0.2)),
        min_area_px=selection_min_area_px,
        max_area_px=selection_max_area_px,
        selection_mode=str(params.get("selection_mode", "high_confidence_highest_z")),
        high_confidence_gate=float(params.get("selection_high_confidence_gate", 0.8)),
        bin_roi_polygon_uv=_bin_roi_polygon_uv,
        excluded_ranks=[
            int(v)
            for v in (params.get("excluded_detection_ranks") or [])
            if str(v).strip()
        ],
    )
    timing_log.span(
        "candidate_selection",
        candidate_select_started_at,
        candidate_found=candidate is not None,
        selection_mode=str(params.get("selection_mode", "high_confidence_highest_z")),
    )
    if candidate is None:
        detection_summaries = [
            {
                "rank": int(det.get("rank", idx)),
                "score": float(det.get("score", 0.0)),
                "area_px": float(det.get("area_px", 0.0)),
                "bbox_xywh": [float(v) for v in det.get("bbox_xywh", [])],
            }
            for idx, det in enumerate(detections[:5])
        ]
        _runtime_log(
            request_id,
            "candidate_none "
            f"min_conf={float(params.get('selection_min_confidence', 0.2)):.3f} "
            f"min_area_px={selection_min_area_px:.1f} "
            f"max_area_px={selection_max_area_px:.1f} "
            f"top_k={int(params.get('selection_top_k', 5))} "
            f"detections={json.dumps(detection_summaries)}",
        )
        timing_log.summary(
            valid=False,
            error="megapose_no_candidate",
            total_detections=len(detections),
            timing_log_path=str(timing_log.path) if timing_log.path else None,
        )
        debug_paths: dict[str, str] = {}
        output_dir = _build_output_dir(params, request_id, assets.label)
        if output_dir is not None:
            try:
                debug_paths["raw_rgb"] = str(
                    _save_bgr_image(
                        bgr_frame,
                        output_dir / "raw_rgb.png",
                    )
                )
                debug_paths["segmentation_candidates"] = str(
                    _save_segmentation_candidates_overlay(
                        bgr_frame,
                        detections,
                        depth_m=depth_m,
                        destination=output_dir / "segmentation_candidates.png",
                    )
                )
            except Exception as exc:
                _runtime_log(
                    request_id,
                    f"candidate_none_artifacts_failed error={exc}",
                )
        return {
            "valid": False,
            "matches": [],
            "terminal": True,
            "error": "megapose_no_candidate",
            "debug_paths": debug_paths,
            "details": {
                "detections": len(detections),
                "selection_top_k": int(params.get("selection_top_k", 5)),
                "selection_min_confidence": float(params.get("selection_min_confidence", 0.2)),
                "selection_min_area_px": selection_min_area_px,
                "selection_max_area_px": selection_max_area_px,
                "top_detections": detection_summaries,
                "duplicate_filter": duplicate_filter_info,
                "debug_paths": debug_paths,
            },
        }

    n_refiner_iterations = int(params.get("refiner_iterations", 1))
    if n_refiner_iterations <= 0:
        n_refiner_iterations = MODEL_CONFIGS[model_name]["n_refiner_iterations"]
    save_megapose_debug = bool(params.get("save_megapose_debug", True))
    pose_hypotheses = int(
        params.get("pose_hypotheses", MODEL_CONFIGS[model_name]["n_pose_hypotheses"])
    )
    if pose_hypotheses <= 0:
        pose_hypotheses = MODEL_CONFIGS[model_name]["n_pose_hypotheses"]
    segmented_cluster_filter_enabled = bool(params.get("segmented_cluster_filter_enabled", True))
    segmented_cluster_distance_m = float(params.get("segmented_cluster_distance_m", 0.03) or 0.03)
    segmented_cluster_min_points = int(params.get("segmented_cluster_min_points", 48) or 48)
    segmented_cluster_min_ratio = float(params.get("segmented_cluster_min_ratio", 0.01) or 0.01)

    fallback_candidate = select_detection_candidate(
        detections,
        image_shape=rgb.shape[:2],
        depth_m=depth_m,
        K=camera_data["K"],
        top_k=int(params.get("selection_top_k", 5)),
        min_confidence=float(params.get("selection_min_confidence", 0.2)),
        min_area_px=selection_min_area_px,
        max_area_px=selection_max_area_px,
        high_confidence_gate=float(params.get("selection_high_confidence_gate", 0.8)),
        selection_mode=(
            "highest_segmentation_score"
            if str(params.get("selection_mode", "high_confidence_highest_z")).strip().lower()
            == "high_confidence_highest_z"
            else "high_confidence_highest_z"
        ),
        bin_roi_polygon_uv=_bin_roi_polygon_uv,
        excluded_ranks=[
            int(v)
            for v in (params.get("excluded_detection_ranks") or [])
            if str(v).strip()
        ],
    )

    def _run_candidate_inference(
        selected_candidate: dict[str, Any],
        *,
        stage_label: str,
    ) -> dict[str, Any]:
        bbox_xyxy_local = np.asarray(selected_candidate["bbox_xyxy"], dtype=np.float32)
        bbox_xywh_local = selected_candidate["bbox_xywh"]
        _runtime_log(
            request_id,
            "candidate_selected "
            f"stage={stage_label} "
            f"mode={str(selected_candidate.get('selection_mode') or 'unknown')} "
            f"rank={int(selected_candidate.get('rank', 0))} "
            f"score={float(selected_candidate.get('score', 0.0)):.4f} "
            f"area_px={float(selected_candidate.get('area_px', 0.0)):.1f} "
            f"bbox_xywh={json.dumps([float(v) for v in bbox_xywh_local])}",
        )
        candidate_stage_started_at = time.perf_counter()
        mask_started_at = time.perf_counter()
        selection_mask_local = make_selection_mask(
            rgb.shape[:2],
            selected_candidate.get("mask"),
            bbox_xyxy_local,
        )
        coarse_rgb_mask_dilation_base_px_local = 5.0
        refiner_rgb_mask_dilation_base_px_local = 0.0
        megapose_rgb_coarse_local, megapose_depth_local, selection_mask_local, depth_selection_mask_local, cluster_filter_info_local = mask_observation_to_selected_object(
            rgb,
            depth_m,
            selected_candidate.get("mask"),
            bbox_xyxy_local,
            K=np.asarray(camera_data["K"], dtype=np.float32),
            filter_depth_clusters=segmented_cluster_filter_enabled,
            cluster_distance_m=segmented_cluster_distance_m,
            min_cluster_points=segmented_cluster_min_points,
            min_cluster_ratio=segmented_cluster_min_ratio,
            rgb_mask_dilation_base_px=coarse_rgb_mask_dilation_base_px_local,
        )
        megapose_rgb_refiner_local = megapose_rgb_coarse_local
        if int(n_refiner_iterations) > 0:
            megapose_rgb_refiner_local, _, _, _, _ = mask_observation_to_selected_object(
                rgb,
                depth_m,
                selected_candidate.get("mask"),
                bbox_xyxy_local,
                K=np.asarray(camera_data["K"], dtype=np.float32),
                filter_depth_clusters=segmented_cluster_filter_enabled,
                cluster_distance_m=segmented_cluster_distance_m,
                min_cluster_points=segmented_cluster_min_points,
                min_cluster_ratio=segmented_cluster_min_ratio,
                rgb_mask_dilation_base_px=refiner_rgb_mask_dilation_base_px_local,
            )
        segmentation_contours_uv_local = extract_segmentation_contours(selection_mask_local)
        _runtime_log(
            request_id,
            "mask_ready "
            f"stage={stage_label} "
            f"mask_pixels={int(np.count_nonzero(selection_mask_local))} "
            f"coarse_rgb_dilation_base_px={coarse_rgb_mask_dilation_base_px_local:.1f} "
            f"refiner_rgb_dilation_base_px={refiner_rgb_mask_dilation_base_px_local:.1f} "
            f"contours={len(segmentation_contours_uv_local)} "
            f"dt={time.perf_counter() - mask_started_at:.3f}s",
        )
        timing_log.span(
            f"{stage_label}.mask_preparation",
            mask_started_at,
            mask_pixels=int(np.count_nonzero(selection_mask_local)),
            contours=len(segmentation_contours_uv_local),
        )
        if cluster_filter_info_local is not None:
            _runtime_log(
                request_id,
                "segmented_cluster_filter "
                f"stage={stage_label} "
                f"distance_m={float(cluster_filter_info_local.get('cluster_distance_m', segmented_cluster_distance_m)):.3f} "
                f"clusters={int(cluster_filter_info_local.get('cluster_count', 0))} "
                f"retained={int(cluster_filter_info_local.get('retained_cluster_count', 0))} "
                f"min_pts={int(cluster_filter_info_local.get('min_cluster_points', segmented_cluster_min_points))} "
                f"kept={int(cluster_filter_info_local.get('kept_cluster_points', 0))} "
                f"removed={int(cluster_filter_info_local.get('removed_points', 0))}",
            )
        observation_started_at = time.perf_counter()
        detections_tc_local = build_detections(
            assets.label,
            bbox_xyxy_local,
            float(selected_candidate.get("score", 0.0)),
            bundle.modules["PandasTensorCollection"],
        ).to(bundle.device)
        coarse_observation_local = bundle.modules["ObservationTensor"].from_numpy(
            megapose_rgb_coarse_local,
            megapose_depth_local,
            camera_data["K"],
        ).to(bundle.device)
        refiner_observation_local = bundle.modules["ObservationTensor"].from_numpy(
            megapose_rgb_refiner_local,
            megapose_depth_local,
            camera_data["K"],
        ).to(bundle.device)
        _runtime_log(
            request_id,
            "observation_ready "
            f"stage={stage_label} "
            f"depth_nonzero_px={int(np.count_nonzero(megapose_depth_local)) if megapose_depth_local is not None else 0} "
            f"depth_min_m={float(megapose_depth_local[megapose_depth_local > 0].min()) if megapose_depth_local is not None and np.any(megapose_depth_local > 0) else -1.0:.4f} "
            f"depth_max_m={float(megapose_depth_local[megapose_depth_local > 0].max()) if megapose_depth_local is not None and np.any(megapose_depth_local > 0) else -1.0:.4f} "
            f"dt={time.perf_counter() - observation_started_at:.3f}s",
        )
        timing_log.span(
            f"{stage_label}.observation_tensor",
            observation_started_at,
            depth_nonzero_px=int(np.count_nonzero(megapose_depth_local)) if megapose_depth_local is not None else 0,
        )

        _runtime_log(
            request_id,
            "inference_start "
            f"stage={stage_label} "
            f"refiner_iterations={n_refiner_iterations} "
            f"pose_hypotheses={pose_hypotheses}",
        )
        inference_start_local = time.perf_counter()
        _runtime_log(request_id, f"inference_lock_wait stage={stage_label}")
        with bundle.pose_lock:
            _runtime_log(request_id, f"inference_lock_acquired stage={stage_label}")
            if int(n_refiner_iterations) > 0:
                data_TCO_coarse_local, coarse_extra_data_local = bundle.pose_estimator.forward_coarse_model(
                    observation=coarse_observation_local,
                    detections=detections_tc_local,
                    return_debug_data=save_megapose_debug,
                )
                data_TCO_filtered_local = bundle.pose_estimator.filter_pose_estimates(
                    data_TCO_coarse_local,
                    top_K=pose_hypotheses,
                    filter_field="coarse_logit",
                )
                preds_local, refiner_extra_data_local = bundle.pose_estimator.forward_refiner(
                    refiner_observation_local,
                    data_TCO_filtered_local,
                    n_iterations=n_refiner_iterations,
                )
                data_TCO_refined_local = preds_local[f"iteration={n_refiner_iterations}"]
                data_TCO_scored_local, scoring_extra_data_local = bundle.pose_estimator.forward_scoring_model(
                    refiner_observation_local,
                    data_TCO_refined_local,
                )
                output_local = bundle.pose_estimator.filter_pose_estimates(
                    data_TCO_scored_local,
                    top_K=1,
                    filter_field="pose_logit",
                )
                extra_data_local = {
                    "coarse": {"preds": data_TCO_coarse_local, "data": coarse_extra_data_local},
                    "coarse_filter": {"preds": data_TCO_filtered_local},
                    "refiner_all_hypotheses": {"preds": preds_local, "data": refiner_extra_data_local},
                    "scoring": {"preds": data_TCO_scored_local, "data": scoring_extra_data_local},
                    "refiner": {"preds": output_local, "data": refiner_extra_data_local},
                }
            else:
                output_local, extra_data_local = bundle.pose_estimator.run_inference_pipeline(
                    observation=coarse_observation_local,
                    detections=detections_tc_local,
                    n_refiner_iterations=n_refiner_iterations,
                    n_pose_hypotheses=pose_hypotheses,
                    return_coarse_debug_data=save_megapose_debug,
                )
        inference_seconds_local = time.perf_counter() - inference_start_local
        timing_log.span(
            f"{stage_label}.pose_estimation",
            inference_start_local,
            refiner_iterations=int(n_refiner_iterations),
            pose_hypotheses=int(pose_hypotheses),
        )
        _runtime_log(
            request_id,
            f"inference_done stage={stage_label} dt={inference_seconds_local:.3f}s",
        )

        pose_matrix_local = output_local.poses[0].detach().cpu().numpy()
        initial_pose_matrix_local = pose_matrix_local.copy()
        coarse_input_pose_matrix_local = None
        coarse_filtered_local = (extra_data_local.get("coarse_filter") or {}).get("preds")
        if coarse_filtered_local is not None:
            coarse_poses_local = getattr(coarse_filtered_local, "poses", None)
            if coarse_poses_local is not None:
                try:
                    coarse_np_local = coarse_poses_local.detach().cpu().numpy()
                    if len(coarse_np_local):
                        coarse_input_pose_matrix_local = np.asarray(coarse_np_local[0], dtype=np.float32)
                except Exception:
                    coarse_input_pose_matrix_local = None

        timing_log.span(
            f"{stage_label}.candidate_total",
            candidate_stage_started_at,
            rank=int(selected_candidate.get("rank", 0)),
            score=float(selected_candidate.get("score", 0.0)),
        )
        return {
            "candidate": dict(selected_candidate),
            "bbox_xyxy": bbox_xyxy_local,
            "bbox_xywh": bbox_xywh_local,
            "megapose_depth": megapose_depth_local,
            "megapose_rgb": megapose_rgb_refiner_local,
            "selection_mask": selection_mask_local,
            "depth_selection_mask": depth_selection_mask_local,
            "segmentation_contours_uv": segmentation_contours_uv_local,
            "cluster_filter_info": cluster_filter_info_local,
            "output": output_local,
            "pose_matrix": pose_matrix_local,
            "initial_pose_matrix": initial_pose_matrix_local,
            "coarse_input_pose_matrix": coarse_input_pose_matrix_local,
            "extra_data": extra_data_local,
            "inference_seconds": inference_seconds_local,
        }

    candidate_run = _run_candidate_inference(candidate, stage_label="primary")
    candidate = candidate_run["candidate"]
    bbox_xyxy = candidate_run["bbox_xyxy"]
    bbox_xywh = candidate_run["bbox_xywh"]
    megapose_depth = candidate_run["megapose_depth"]
    selection_mask = candidate_run["selection_mask"]
    depth_selection_mask = candidate_run["depth_selection_mask"]
    segmentation_contours_uv = candidate_run["segmentation_contours_uv"]
    cluster_filter_info = candidate_run["cluster_filter_info"]
    output = candidate_run["output"]
    pose_matrix = candidate_run["pose_matrix"]
    initial_pose_matrix = candidate_run["initial_pose_matrix"]
    coarse_input_pose_matrix = candidate_run["coarse_input_pose_matrix"]
    extra_data = candidate_run["extra_data"]
    inference_seconds = float(candidate_run["inference_seconds"])
    rerank_result = None
    if bool(params.get("rerank_enabled", True)) and pose_hypotheses > 1:
        def _rerank_float_param(name: str, default: float) -> float:
            value = params.get(name, default)
            if value is None or value == "":
                return float(default)
            try:
                return float(value)
            except (TypeError, ValueError):
                return float(default)

        rerank_started_at = time.perf_counter()
        rerank_result = rerank_pose_hypotheses(
            hypotheses=((extra_data.get("scoring") or {}).get("preds") if isinstance(extra_data, dict) else None),
            selection_mask=selection_mask,
            depth_selection_mask=depth_selection_mask,
            depth_m=depth_m,
            K=np.asarray(camera_data["K"], dtype=np.float32),
            resolution=tuple(camera_data["resolution"]),
            label=assets.label,
            modules=bundle.modules,
            pose_estimator=bundle.pose_estimator,
            top_k=int(params.get("rerank_top_k", pose_hypotheses)),
            min_depth_overlap_pixels=int(params.get("rerank_min_depth_overlap_pixels", 64) or 64),
            depth_error_scale_m=_rerank_float_param("rerank_depth_error_scale_m", 0.004),
            mask_iou_weight=_rerank_float_param("rerank_mask_iou_weight", 0.30),
            mask_coverage_weight=_rerank_float_param("rerank_mask_coverage_weight", 0.12),
            depth_weight=_rerank_float_param("rerank_depth_weight", 0.28),
            network_prior_weight=_rerank_float_param("rerank_network_prior_weight", 1.0),
        )
        if rerank_result is not None:
            reranked_pose = np.asarray(rerank_result.get("selected_pose_matrix"), dtype=np.float32)
            if reranked_pose.shape == (4, 4):
                pose_matrix = reranked_pose
                _runtime_log(
                    request_id,
                    "rerank_selected "
                    f"hypothesis_id={rerank_result.get('selected_hypothesis_id')} "
                    f"score={float(rerank_result.get('selected_rerank_score', 0.0)):.4f}",
                )
        timing_log.span(
            "rerank_pose_hypotheses",
            rerank_started_at,
            applied=rerank_result is not None,
        )
    refiner_guard_info = _make_refiner_guard_info(
        params,
        object_runtime_config,
        np.asarray(annotation_meta.get("object_extents_m") or [0.0, 0.0, 0.0], dtype=np.float32),
    )
    refiner_guard_result: dict[str, Any] | None = None
    if refiner_guard_info.get("enabled") and coarse_input_pose_matrix is not None:
        refiner_guard_started_at = time.perf_counter()
        translation_delta_m = float(
            np.linalg.norm(
                np.asarray(pose_matrix[:3, 3], dtype=np.float64)
                - np.asarray(coarse_input_pose_matrix[:3, 3], dtype=np.float64)
            )
        )
        rotation_delta_deg = _rotation_angle_between(
            coarse_input_pose_matrix[:3, :3],
            pose_matrix[:3, :3],
        )
        object_center_object_m = np.asarray(
            annotation_meta.get("object_center_object_m") or [0.0, 0.0, 0.0],
            dtype=np.float32,
        ).reshape(3)
        refined_rotation = np.asarray(pose_matrix[:3, :3], dtype=np.float32)
        refined_translation = np.asarray(pose_matrix[:3, 3], dtype=np.float32)
        refined_center_camera = (refined_rotation @ object_center_object_m) + refined_translation
        refined_center_uv = _project_point_uv(
            np.asarray(refined_center_camera, dtype=np.float32),
            np.asarray(camera_data["K"], dtype=np.float32),
        )
        # Fill interior holes so cup/ring parts (where the CAD origin projects
        # through the cavity onto a background pixel) don't falsely trip the
        # guard. Re-extract the outer contour(s) and rasterize a solid mask.
        selection_mask_filled = selection_mask.astype(bool)
        try:
            mask_u8 = selection_mask.astype(np.uint8)
            if mask_u8.max() == 1:
                mask_u8 = mask_u8 * 255
            contours, _ = cv2.findContours(
                mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if contours:
                filled = np.zeros_like(mask_u8)
                cv2.drawContours(filled, contours, -1, 255, thickness=cv2.FILLED)
                selection_mask_filled = filled.astype(bool)
        except Exception:
            selection_mask_filled = selection_mask.astype(bool)

        refined_center_inside_mask = False
        if refined_center_uv is not None:
            try:
                center_u = int(round(float(refined_center_uv[0])))
                center_v = int(round(float(refined_center_uv[1])))
                if (
                    0 <= center_v < int(selection_mask_filled.shape[0])
                    and 0 <= center_u < int(selection_mask_filled.shape[1])
                ):
                    refined_center_inside_mask = bool(
                        selection_mask_filled[center_v, center_u]
                    )
            except Exception:
                refined_center_inside_mask = False
        refiner_guard_result = {
            "enabled": True,
            "max_translation_delta_m": float(refiner_guard_info["max_translation_delta_m"]),
            "max_rotation_delta_deg": float(refiner_guard_info["max_rotation_delta_deg"]),
            "observed_translation_delta_m": translation_delta_m,
            "observed_rotation_delta_deg": rotation_delta_deg,
            "fallback": refiner_guard_info.get("fallback") or "coarse_input",
            "trigger_mode": "refined_center_outside_segmentation_mask",
            "refined_center_camera_m": [float(v) for v in refined_center_camera.tolist()],
            "refined_center_uv": [float(v) for v in refined_center_uv] if refined_center_uv else None,
            "refined_center_inside_mask": bool(refined_center_inside_mask),
            "triggered": False,
        }
        if not refined_center_inside_mask:
            refiner_guard_result["triggered"] = True
            refiner_guard_result["trigger_reason"] = "refined_center_outside_segmentation_mask"
            fallback_applied = False
            fallback_bbox = None
            if fallback_candidate is not None:
                fallback_bbox = [float(v) for v in fallback_candidate.get("bbox_xywh", [])]
            if (
                fallback_candidate is not None
                and [float(v) for v in fallback_candidate.get("bbox_xywh", [])]
                != [float(v) for v in candidate.get("bbox_xywh", [])]
            ):
                _runtime_log(
                    request_id,
                    "refiner_guard_fallback "
                    "strategy=top_k_highest_z_segmentation "
                    f"refined_center_uv={json.dumps(refiner_guard_result['refined_center_uv'])} "
                    f"inside_mask={refined_center_inside_mask} "
                    f"fallback_bbox_xywh={json.dumps(fallback_bbox)} "
                    f"translation_delta_m={translation_delta_m:.4f} "
                    f"rotation_delta_deg={rotation_delta_deg:.2f}",
                )
                fallback_run = _run_candidate_inference(fallback_candidate, stage_label="fallback_top_k_highest_z")
                candidate = fallback_run["candidate"]
                bbox_xyxy = fallback_run["bbox_xyxy"]
                bbox_xywh = fallback_run["bbox_xywh"]
                megapose_depth = fallback_run["megapose_depth"]
                selection_mask = fallback_run["selection_mask"]
                depth_selection_mask = fallback_run["depth_selection_mask"]
                segmentation_contours_uv = fallback_run["segmentation_contours_uv"]
                cluster_filter_info = fallback_run["cluster_filter_info"]
                output = fallback_run["output"]
                pose_matrix = fallback_run["pose_matrix"]
                initial_pose_matrix = fallback_run["initial_pose_matrix"]
                coarse_input_pose_matrix = fallback_run["coarse_input_pose_matrix"]
                extra_data = fallback_run["extra_data"]
                inference_seconds = float(fallback_run["inference_seconds"])
                fallback_applied = True
                if rerank_result is not None:
                    rerank_result["superseded_by"] = "refiner_guard_fallback_candidate"
                refiner_guard_result["applied_pose_source"] = "segmentation_top_k_highest_z"
                refiner_guard_result["fallback_candidate_bbox_xywh"] = fallback_bbox
            elif str(refiner_guard_info.get("fallback") or "coarse_input") == "coarse_input":
                pose_matrix = coarse_input_pose_matrix.copy()
                if rerank_result is not None:
                    rerank_result["superseded_by"] = "refiner_guard_coarse_input"
                refiner_guard_result["applied_pose_source"] = "coarse_input"
                _runtime_log(
                    request_id,
                    "refiner_guard_fallback "
                    "strategy=coarse_input "
                    f"refined_center_uv={json.dumps(refiner_guard_result['refined_center_uv'])} "
                    f"inside_mask={refined_center_inside_mask} "
                    f"translation_delta_m={translation_delta_m:.4f} "
                    f"rotation_delta_deg={rotation_delta_deg:.2f}",
                )
            if not fallback_applied and "applied_pose_source" not in refiner_guard_result:
                refiner_guard_result["applied_pose_source"] = "refiner"
        else:
            refiner_guard_result["applied_pose_source"] = "refiner"
        timing_log.span(
            "refiner_guard",
            refiner_guard_started_at,
            triggered=bool(refiner_guard_result.get("triggered", False)),
            applied_pose_source=refiner_guard_result.get("applied_pose_source"),
        )
    depth_ray_refinement = None
    depth_ray_refinement_seconds = 0.0
    if bool(params.get("depth_ray_refinement", True)) and depth_m is not None:
        depth_points_started_at = time.perf_counter()
        observed_points = depth_to_point_cloud_points(
            megapose_depth if 'megapose_depth' in locals() else depth_m,
            np.asarray(camera_data["K"], dtype=np.float32),
            mask=depth_selection_mask,
            max_depth_m=_coerce_float(params.get("depth_max_m"), None),
        )
        timing_log.span(
            "depth_observed_point_cloud",
            depth_points_started_at,
            observed_points=int(len(observed_points)),
        )
        depth_ray_refinement_started_at = time.perf_counter()
        pose_matrix, depth_ray_refinement = refine_pose_distance_along_center_ray(
            mesh_path=bundle.mesh_obj_path,
            pose_matrix=pose_matrix,
            observed_points=observed_points,
            mesh_units=str(params.get("mesh_units", "mm")),
            mesh_scale=float(params.get("mesh_scale", 1.0)),
            iterations=int(params.get("depth_ray_refinement_iterations", 2) or 2),
            mesh_samples=int(params.get("depth_ray_refinement_mesh_samples", 2500) or 2500),
            observed_samples=int(params.get("depth_ray_refinement_observed_samples", 4000) or 4000),
            max_correspondence_m=float(
                params.get("depth_ray_refinement_max_correspondence_m", 0.02) or 0.02
            ),
            max_shift_m=float(params.get("depth_ray_refinement_max_shift_m", 0.05) or 0.05),
            front_layer_thickness_m=float(
                params.get("depth_ray_front_layer_thickness_m", 0.01) or 0.01
            ),
            center_keep_ratio=float(params.get("depth_ray_center_keep_ratio", 0.7) or 0.7),
        )
        if depth_ray_refinement is not None and not depth_ray_refinement.get("applied", False):
            fallback_pose_matrix, fallback_refinement = refine_pose_distance_from_rendered_depth_overlap(
                pose_matrix=pose_matrix,
                depth_m=depth_m,
                selection_mask=depth_selection_mask,
                K=np.asarray(camera_data["K"], dtype=np.float32),
                resolution=tuple(camera_data["resolution"]),
                label=assets.label,
                modules=bundle.modules,
                pose_estimator=bundle.pose_estimator,
                max_shift_m=float(params.get("depth_ray_refinement_max_shift_m", 0.05) or 0.05),
                front_layer_thickness_m=float(
                    params.get("depth_ray_front_layer_thickness_m", 0.01) or 0.01
                ),
                center_keep_ratio=float(params.get("depth_ray_center_keep_ratio", 0.7) or 0.7),
            )
            if fallback_refinement is not None:
                depth_ray_refinement["fallback"] = fallback_refinement
                if fallback_refinement.get("applied", False):
                    pose_matrix = fallback_pose_matrix
                    depth_ray_refinement["applied"] = True
                    depth_ray_refinement["method"] = fallback_refinement.get("method")
                    depth_ray_refinement["shift_along_camera_z_m"] = fallback_refinement.get(
                        "shift_along_camera_z_m", 0.0
                    )
        depth_ray_refinement_seconds = time.perf_counter() - depth_ray_refinement_started_at
        timing_log.span(
            "depth_ray_refinement",
            depth_ray_refinement_started_at,
            applied=bool(depth_ray_refinement.get("applied", False)) if depth_ray_refinement else False,
            shift_along_center_ray_m=(
                float(depth_ray_refinement.get("shift_along_center_ray_m", 0.0))
                if depth_ray_refinement
                else 0.0
            ),
        )
        if depth_ray_refinement is not None:
            _runtime_log(
                request_id,
                "depth_ray_refine "
                f"shift_along_center_ray_m={float(depth_ray_refinement.get('shift_along_center_ray_m', 0.0)):.4f} "
                f"shift_along_camera_z_m={float(depth_ray_refinement.get('shift_along_camera_z_m', 0.0)):.4f} "
                f"status={'applied' if depth_ray_refinement.get('applied') else str(depth_ray_refinement.get('skip_reason', 'not_applied'))} "
                f"dt={depth_ray_refinement_seconds:.3f}s",
            )

    rotation = pose_matrix[:3, :3]
    translation = pose_matrix[:3, 3]
    object_center_object_m = np.asarray(
        annotation_meta.get("object_center_object_m") or [0.0, 0.0, 0.0],
        dtype=np.float32,
    )
    object_center_camera = (rotation @ object_center_object_m) + translation
    axis_object = _normalize_axis(params.get("axis", [0.0, 0.0, 1.0]))
    axis_camera = rotation @ axis_object
    yaw_deg = _yaw_from_axis(axis_camera)
    quat_xyzw = rotation_matrix_to_quaternion_xyzw(rotation)
    center_uv = _project_point_uv(object_center_camera, camera_data["K"]) or [
        float(bbox_xywh[0] + (bbox_xywh[2] * 0.5)),
        float(bbox_xywh[1] + (bbox_xywh[3] * 0.5)),
    ]
    selected_depth_metrics = {
        "closest_region_distance_m": candidate.get("closest_region_distance_m"),
        "robust_region_distance_m": candidate.get("robust_region_distance_m"),
        "z_min_m": candidate.get("z_min_m"),
        "z_front_m": candidate.get("z_front_m"),
        "z_max_m": candidate.get("z_max_m"),
        "valid_depth_points": candidate.get("valid_depth_points"),
        "support_depth_points": candidate.get("support_depth_points"),
    }

    output_setup_started_at = time.perf_counter()
    output_dir = _build_output_dir(params, request_id, assets.label)
    debug_paths: dict[str, str] = {}
    visualization_paths: dict[str, str] = {}
    visualization_3d_paths: dict[str, str | None] = {}
    if output_dir is not None:
        artifacts_started_at = time.perf_counter()
        debug_paths["raw_rgb"] = str(
            _save_bgr_image(
                bgr_frame,
                output_dir / "raw_rgb.png",
            )
        )
        debug_paths["segmentation_candidates"] = str(
            _save_segmentation_candidates_overlay(
                bgr_frame,
                detections,
                depth_m=depth_m,
                destination=output_dir / "segmentation_candidates.png",
            )
        )
        try:
            curated_paths = _save_megapose_debug_bundle(
                output_dir,
                extra_data,
                rgb=rgb,
                K=np.asarray(camera_data["K"], dtype=np.float32),
                resolution=tuple(camera_data["resolution"]),
                label=assets.label,
                modules=bundle.modules,
                pose_estimator=bundle.pose_estimator,
                object_center_object_m=object_center_object_m,
                axis_length_m=float(annotation_meta.get("annotation_axis_length_m") or 0.015),
                request_id=request_id,
            )
            debug_paths.update(curated_paths)
        except Exception as exc:
            _runtime_log(request_id, f"curated_artifact_save_failed error={exc}")

        # Build per-match safety point clouds (camera frame) for orchestrator-side
        # collision avoidance. Downsampled so the JSON payload stays bounded.
        target_points_camera: list[list[float]] = []
        neighbor_points_camera: list[list[float]] = []
        safety_pcd_meta: dict[str, Any] = {
            "frame": "camera",
            "voxel_size_m": float(
                _coerce_float(params.get("safety_pcd_voxel_size_m"), 0.003) or 0.003
            ),
            "max_depth_m": _coerce_float(params.get("depth_max_m"), None),
            "max_points_per_cloud": int(
                _coerce_float(params.get("safety_pcd_max_points"), 20000) or 20000
            ),
        }
        try:
            if depth_m is not None and selection_mask is not None:
                K_safety = np.asarray(camera_data["K"], dtype=np.float32)
                target_mask = selection_mask.astype(bool)
                neighbor_mask = ~target_mask
                raw_target = depth_to_point_cloud_points(
                    depth_m,
                    K_safety,
                    mask=target_mask,
                    max_depth_m=safety_pcd_meta["max_depth_m"],
                )
                raw_neighbor = depth_to_point_cloud_points(
                    depth_m,
                    K_safety,
                    mask=neighbor_mask,
                    max_depth_m=safety_pcd_meta["max_depth_m"],
                )
                voxel = safety_pcd_meta["voxel_size_m"]
                tgt_ds = voxel_downsample_points(raw_target, voxel)
                nbr_ds = voxel_downsample_points(raw_neighbor, voxel)
                cap = safety_pcd_meta["max_points_per_cloud"]
                if cap > 0 and len(tgt_ds) > cap:
                    idx = np.random.default_rng(0).choice(len(tgt_ds), cap, replace=False)
                    tgt_ds = tgt_ds[np.sort(idx)]
                if cap > 0 and len(nbr_ds) > cap:
                    idx = np.random.default_rng(0).choice(len(nbr_ds), cap, replace=False)
                    nbr_ds = nbr_ds[np.sort(idx)]
                target_points_camera = tgt_ds.tolist()
                neighbor_points_camera = nbr_ds.tolist()
                safety_pcd_meta["target_point_count"] = len(target_points_camera)
                safety_pcd_meta["neighbor_point_count"] = len(neighbor_points_camera)
        except Exception as exc:
            _runtime_log(request_id, f"safety_pcd_build_failed error={exc}")

        if bool(params.get("save_pose_3d_assets", True)):
            try:
                visualization_3d_paths = save_pose_3d_assets(
                    rgb=rgb,
                    depth_m=depth_m,
                    K=np.asarray(camera_data["K"], dtype=np.float32),
                    mask=selection_mask,
                    mesh_path=bundle.mesh_obj_path,
                    pose_matrix=pose_matrix,
                    mesh_units=str(params.get("mesh_units", "mm")),
                    mesh_scale=float(params.get("mesh_scale", 1.0)),
                    object_center_object_m=object_center_object_m,
                    scene_point_cloud_ply_path=output_dir / "scene_point_cloud.ply",
                    segmented_point_cloud_ply_path=output_dir / "segmented_point_cloud.ply",
                    pose_scene_glb_path=output_dir / "pose_scene.glb",
                    max_depth_m=_coerce_float(params.get("depth_max_m"), None),
                )
                for key, value in visualization_3d_paths.items():
                    if value:
                        debug_paths[key] = str(value)
            except Exception as exc:
                _runtime_log(request_id, f"pose_3d_asset_save_failed error={exc}")

        final_pose_title = (
            f"final s {float(candidate.get('score', 0.0)):.3f} "
            f"z {float(pose_matrix[2, 3]):.3f}m"
        )
        try:
            final_pose_path = _save_single_pose_overlay(
                rgb=rgb,
                K=np.asarray(camera_data["K"], dtype=np.float32),
                resolution=tuple(camera_data["resolution"]),
                label=assets.label,
                modules=bundle.modules,
                pose_estimator=bundle.pose_estimator,
                pose_matrix=np.asarray(pose_matrix, dtype=np.float32),
                destination=output_dir / "final_pose.png",
                title_text=final_pose_title,
                object_center_object_m=object_center_object_m,
                axis_length_m=float(annotation_meta.get("annotation_axis_length_m") or 0.015),
            )
        except Exception as exc:
            _runtime_log(request_id, f"final_pose_save_failed error={exc}")
            final_pose_path = None
        if final_pose_path is not None:
            debug_paths["final_pose"] = str(final_pose_path)
            debug_paths["pose_annotated"] = str(final_pose_path)
        payload_path = output_dir / "pose.json"
        timing_log.span(
            "debug_artifacts_and_safety_pcd",
            artifacts_started_at,
            output_dir=str(output_dir),
            safety_target_points=len(target_points_camera),
            safety_neighbor_points=len(neighbor_points_camera),
        )
    else:
        payload_path = None
    timing_log.span(
        "output_setup",
        output_setup_started_at,
        save_outputs=output_dir is not None,
    )

    match = {
        "object_id": str(params.get("object_id") or assets.label),
        "label": assets.label,
        "method": "megapose_bin_picking",
        "camera_calibration_source": str(
            params.get("camera_calibration_source") or "module_params"
        ),
        "score": float(candidate.get("score", 0.0)),
        "bbox_xywh": [float(v) for v in bbox_xywh],
        "bbox_xyxy": [float(v) for v in bbox_xyxy.tolist()],
        "area_px": float(candidate.get("area_px", 0.0)),
        "rank": int(candidate.get("rank", 0)),
        "center_uv": center_uv,
        "center_xyz_m": [
            float(object_center_camera[0]),
            float(object_center_camera[1]),
            float(object_center_camera[2]),
        ],
        "depth_shift_along_center_ray_m": 0.0
        if depth_ray_refinement is None
        else float(depth_ray_refinement.get("shift_along_center_ray_m", 0.0)),
        "depth_shift_along_camera_z_m": 0.0
        if depth_ray_refinement is None
        else float(depth_ray_refinement.get("shift_along_camera_z_m", 0.0)),
        "depth_m": float(object_center_camera[2]),
        "initial_pose_origin_xyz_m": [
            float(initial_pose_matrix[0, 3]),
            float(initial_pose_matrix[1, 3]),
            float(initial_pose_matrix[2, 3]),
        ],
        "pose_origin_xyz_m": [float(translation[0]), float(translation[1]), float(translation[2])],
        "annotation_origin_xyz_m": [
            float(object_center_camera[0]),
            float(object_center_camera[1]),
            float(object_center_camera[2]),
        ],
        "selected_depth_metrics": selected_depth_metrics,
        "closest_region_distance_m": candidate.get("closest_region_distance_m"),
        "robust_region_distance_m": candidate.get("robust_region_distance_m"),
        "z_min_m": candidate.get("z_min_m"),
        "z_front_m": candidate.get("z_front_m"),
        "z_max_m": candidate.get("z_max_m"),
        "yaw_deg": None if yaw_deg is None else float(yaw_deg),
        "surface_normal_cam": [float(axis_camera[0]), float(axis_camera[1]), float(axis_camera[2])],
        "orientation_axis_camera": [float(axis_camera[0]), float(axis_camera[1]), float(axis_camera[2])],
        "quaternion_xyzw": [float(v) for v in quat_xyzw.tolist()],
        "pose_quat_xyzw": [float(v) for v in quat_xyzw.tolist()],
        "initial_pose_matrix": initial_pose_matrix.tolist(),
        "pose_matrix": pose_matrix.tolist(),
        "object_center_object_m": [float(v) for v in object_center_object_m.tolist()],
        "selection_mask_pixels": int(np.count_nonzero(selection_mask)),
        "segmentation_contour_uv": segmentation_contours_uv[0] if segmentation_contours_uv else [],
        "segmentation_contours_uv": segmentation_contours_uv,
        "segmented_cluster_filter": cluster_filter_info,
        "segmentation_duplicate_filter": duplicate_filter_info,
        "selection_pool": candidate.get("selection_pool", []),
        "model_name": model_name,
        "mesh_path": str(assets.mesh_path),
        "segmentation_model_path": str(assets.segmentation_model_path),
        "camera_intrinsics": camera_data.get("intrinsics"),
        "object_extents_m": annotation_meta.get("object_extents_m"),
        "annotation_axis_length_m": annotation_meta.get("annotation_axis_length_m"),
        "object_runtime_config": object_runtime_config,
        "rerank": rerank_result,
        "refiner_guard": refiner_guard_result,
        "depth_ray_refinement": depth_ray_refinement,
        "debug_paths": debug_paths,
        "visualization": visualization_paths,
        "visualization_3d": visualization_3d_paths,
        "safety_pcd": {
            **safety_pcd_meta,
            "target_points_camera_m": target_points_camera,
            "neighbor_points_camera_m": neighbor_points_camera,
        },
    }

    matches_out = [match]
    # --- All-poses mode --------------------------------------------------------
    # Run megapose on every remaining eligible detection (beyond the primary
    # candidate) and emit one match per successful pose. Minimal per-pose payload:
    # pose + bbox + contour + safety_pcd. Skips heavy viz/rerank.
    if isinstance(params, dict) and bool(params.get("all_poses_mode", False)):
        all_poses_started_at = time.perf_counter()
        try:
            # Compute eligibility the same way `select_detection_candidate` does,
            # minus the top_k truncation — we want all passing masks.
            _min_area_all = max(0.0, float(selection_min_area_px or 0.0))
            _max_area_all = (
                float(selection_max_area_px)
                if selection_max_area_px is not None and float(selection_max_area_px) > 0.0
                else None
            )
            _min_conf_all = float(params.get("selection_min_confidence", 0.2))
            _primary_rank = int(candidate.get("rank", -1))
            _eligible_extra = [
                d
                for d in detections
                if int(d.get("rank", -1)) != _primary_rank
                and float(d.get("score", 0.0)) >= _min_conf_all
                and float(d.get("area_px", 0.0)) >= _min_area_all
                and (_max_area_all is None or float(d.get("area_px", 0.0)) <= _max_area_all)
                and _detection_passes_bin_roi(d, _bin_roi_polygon_uv)
            ]
            _all_poses_cap = int(params.get("all_poses_max", 20) or 20)
            _eligible_extra = _eligible_extra[: max(0, _all_poses_cap)]
            _K_safety_all = np.asarray(camera_data["K"], dtype=np.float32)
            _voxel_all = float(
                _coerce_float(params.get("safety_pcd_voxel_size_m"), 0.003) or 0.003
            )
            _max_depth_all = _coerce_float(params.get("depth_max_m"), None)
            _cap_all = int(
                _coerce_float(params.get("safety_pcd_max_points"), 20000) or 20000
            )
            _runtime_log(
                request_id,
                f"all_poses_mode begin extra={len(_eligible_extra)} cap={_all_poses_cap}",
            )
            for _extra_cand in _eligible_extra:
                try:
                    _extra_run = _run_candidate_inference(_extra_cand, stage_label="all_poses")
                except Exception as _exc:
                    _runtime_log(
                        request_id,
                        f"all_poses_candidate_failed rank={int(_extra_cand.get('rank', -1))} err={_exc}",
                    )
                    continue
                _pose_m = _extra_run.get("pose_matrix")
                if _pose_m is None:
                    continue
                _pose_m = np.asarray(_pose_m, dtype=np.float32)
                _tx = float(_pose_m[0, 3])
                _ty = float(_pose_m[1, 3])
                _tz = float(_pose_m[2, 3])
                try:
                    _rot = _pose_m[:3, :3]
                    _trace = float(_rot[0, 0] + _rot[1, 1] + _rot[2, 2])
                    if _trace > 0:
                        _s = 0.5 / math.sqrt(_trace + 1.0)
                        _qw = 0.25 / _s
                        _qx = (_rot[2, 1] - _rot[1, 2]) * _s
                        _qy = (_rot[0, 2] - _rot[2, 0]) * _s
                        _qz = (_rot[1, 0] - _rot[0, 1]) * _s
                    else:
                        if _rot[0, 0] > _rot[1, 1] and _rot[0, 0] > _rot[2, 2]:
                            _s = 2.0 * math.sqrt(1.0 + _rot[0, 0] - _rot[1, 1] - _rot[2, 2])
                            _qw = (_rot[2, 1] - _rot[1, 2]) / _s
                            _qx = 0.25 * _s
                            _qy = (_rot[0, 1] + _rot[1, 0]) / _s
                            _qz = (_rot[0, 2] + _rot[2, 0]) / _s
                        elif _rot[1, 1] > _rot[2, 2]:
                            _s = 2.0 * math.sqrt(1.0 + _rot[1, 1] - _rot[0, 0] - _rot[2, 2])
                            _qw = (_rot[0, 2] - _rot[2, 0]) / _s
                            _qx = (_rot[0, 1] + _rot[1, 0]) / _s
                            _qy = 0.25 * _s
                            _qz = (_rot[1, 2] + _rot[2, 1]) / _s
                        else:
                            _s = 2.0 * math.sqrt(1.0 + _rot[2, 2] - _rot[0, 0] - _rot[1, 1])
                            _qw = (_rot[1, 0] - _rot[0, 1]) / _s
                            _qx = (_rot[0, 2] + _rot[2, 0]) / _s
                            _qy = (_rot[1, 2] + _rot[2, 1]) / _s
                            _qz = 0.25 * _s
                    _quat = [float(_qx), float(_qy), float(_qz), float(_qw)]
                except Exception:
                    _quat = [0.0, 0.0, 0.0, 1.0]
                _bbox_xywh_extra = _extra_run.get("bbox_xywh")
                _contours_extra = _extra_run.get("segmentation_contours_uv") or []
                _sel_mask_extra = _extra_run.get("selection_mask")
                _tgt_pts: list[list[float]] = []
                _nbr_pts: list[list[float]] = []
                if depth_m is not None and _sel_mask_extra is not None:
                    try:
                        _tgt_mask = _sel_mask_extra.astype(bool)
                        _nbr_mask = ~_tgt_mask
                        _tgt_raw = depth_to_point_cloud_points(
                            depth_m, _K_safety_all, mask=_tgt_mask, max_depth_m=_max_depth_all,
                        )
                        _nbr_raw = depth_to_point_cloud_points(
                            depth_m, _K_safety_all, mask=_nbr_mask, max_depth_m=_max_depth_all,
                        )
                        _tgt_ds = voxel_downsample_points(_tgt_raw, _voxel_all)
                        _nbr_ds = voxel_downsample_points(_nbr_raw, _voxel_all)
                        if _cap_all > 0 and len(_tgt_ds) > _cap_all:
                            _idx = np.random.default_rng(0).choice(len(_tgt_ds), _cap_all, replace=False)
                            _tgt_ds = _tgt_ds[np.sort(_idx)]
                        if _cap_all > 0 and len(_nbr_ds) > _cap_all:
                            _idx = np.random.default_rng(0).choice(len(_nbr_ds), _cap_all, replace=False)
                            _nbr_ds = _nbr_ds[np.sort(_idx)]
                        _tgt_pts = _tgt_ds.tolist()
                        _nbr_pts = _nbr_ds.tolist()
                    except Exception as _exc:
                        _runtime_log(
                            request_id,
                            f"all_poses_safety_pcd_failed rank={int(_extra_cand.get('rank', -1))} err={_exc}",
                        )
                matches_out.append(
                    {
                        "object_id": str(params.get("object_id") or assets.label),
                        "label": assets.label,
                        "method": "megapose_bin_picking",
                        "score": float(_extra_cand.get("score", 0.0)),
                        "rank": int(_extra_cand.get("rank", 0)),
                        "bbox_xywh": [float(v) for v in (_bbox_xywh_extra or [])],
                        "center_xyz_m": [_tx, _ty, _tz],
                        "pose_origin_xyz_m": [_tx, _ty, _tz],
                        "depth_m": _tz,
                        "quaternion_xyzw": _quat,
                        "pose_quat_xyzw": _quat,
                        "pose_matrix": _pose_m.tolist(),
                        "segmentation_contour_uv": _contours_extra[0] if _contours_extra else [],
                        "segmentation_contours_uv": _contours_extra,
                        "selection_mask_pixels": int(
                            np.count_nonzero(_sel_mask_extra)
                            if _sel_mask_extra is not None
                            else 0
                        ),
                        "area_px": float(_extra_cand.get("area_px", 0.0)),
                        "all_poses_match": True,
                        "debug_paths": debug_paths,
                        "visualization": visualization_paths,
                        "visualization_3d": visualization_3d_paths,
                        "safety_pcd": {
                            "frame": "camera",
                            "voxel_size_m": _voxel_all,
                            "max_depth_m": _max_depth_all,
                            "max_points_per_cloud": _cap_all,
                            "target_point_count": len(_tgt_pts),
                            "neighbor_point_count": len(_nbr_pts),
                            "target_points_camera_m": _tgt_pts,
                            "neighbor_points_camera_m": _nbr_pts,
                        },
                        "camera_intrinsics": camera_data.get("intrinsics"),
                        "mesh_path": str(assets.mesh_path),
                    }
                )
            if (
                output_dir is not None
                and bool(params.get("save_pose_3d_assets", True))
                and bool(params.get("render_all_poses_3d", True))
                and len(matches_out) > 1
            ):
                _pose_matrices_all = [
                    np.asarray(_m.get("pose_matrix"), dtype=np.float64)
                    for _m in matches_out
                    if isinstance(_m, dict) and _m.get("pose_matrix") is not None
                ]
                if len(_pose_matrices_all) > 1:
                    _all_pose_scene_path = output_dir / "pose_scene_all_poses.glb"
                    save_multi_pose_scene_glb(
                        mesh_path=bundle.mesh_obj_path,
                        pose_matrices=_pose_matrices_all,
                        mesh_units=str(params.get("mesh_units", "mm")),
                        mesh_scale=float(params.get("mesh_scale", 1.0)),
                        destination=_all_pose_scene_path,
                        object_center_object_m=object_center_object_m,
                    )
                    visualization_3d_paths["pose_scene_glb_path"] = str(_all_pose_scene_path)
                    visualization_3d_paths["pose_scene_all_poses_glb_path"] = str(
                        _all_pose_scene_path
                    )
                    debug_paths["pose_scene_glb_path"] = str(_all_pose_scene_path)
                    debug_paths["pose_scene_all_poses_glb_path"] = str(
                        _all_pose_scene_path
                    )
            _runtime_log(
                request_id,
                f"all_poses_mode end total_matches={len(matches_out)}",
            )
        except Exception as _exc:
            _runtime_log(request_id, f"all_poses_mode_failed err={_exc}")
        timing_log.span(
            "all_poses_mode",
            all_poses_started_at,
            total_matches=len(matches_out),
        )

    result = {
        "valid": True,
        "matches": matches_out,
        "match_count": int(len(matches_out)),
        "all_poses_mode": bool(
            isinstance(params, dict) and bool(params.get("all_poses_mode", False))
        ),
        "model_name": model_name,
        "object_folder": str(assets.object_folder),
        "camera_calibration_source": match["camera_calibration_source"],
        "selected_detection": {
            "rank": int(candidate.get("rank", 0)),
            "score": float(candidate.get("score", 0.0)),
            "area_px": float(candidate.get("area_px", 0.0)),
            "bbox_xywh": [float(v) for v in bbox_xywh],
        },
        "segmentation_duplicate_filter": duplicate_filter_info,
        "timing": {
            "inference_seconds": float(inference_seconds),
            "depth_ray_refinement_seconds": float(depth_ray_refinement_seconds),
            "total_detections": len(detections),
            "timing_log_path": str(timing_log.path) if timing_log.enabled and timing_log.path else None,
        },
        "depth_ray_refinement": depth_ray_refinement,
        "rerank": rerank_result,
        "debug_paths": debug_paths,
        "visualization": visualization_paths,
        "visualization_3d": visualization_3d_paths,
        "request_id": request_id,
    }
    z_front_log = (
        f"{float(candidate.get('z_front_m')):.4f}"
        if candidate.get("z_front_m") is not None
        else "null"
    )
    robust_region_distance_log = (
        f"{float(candidate.get('robust_region_distance_m')):.4f}"
        if candidate.get("robust_region_distance_m") is not None
        else "null"
    )
    _runtime_log(
        request_id,
        "done "
        f"score={float(candidate.get('score', 0.0)):.4f} "
        f"rerank={'yes' if rerank_result is not None else 'no'} "
        f"center_xyz_m={json.dumps(match['center_xyz_m'])} "
        f"pose_origin_z_m={float(translation[2]):.4f} "
        f"z_front_m={z_front_log} "
        f"robust_region_distance_m={robust_region_distance_log} "
        f"depth_shift_along_center_ray_m={float(match['depth_shift_along_center_ray_m']):.4f} "
        f"depth_shift_along_camera_z_m={float(match['depth_shift_along_camera_z_m']):.4f} "
        f"total_dt={time.perf_counter() - run_started_at:.3f}s",
    )
    if payload_path is not None:
        payload_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        result["pose_json_path"] = str(payload_path)
    extra_infos = getattr(output, "infos", None)
    if extra_infos is not None:
        try:
            result["pose_infos"] = extra_infos.to_dict(orient="records")
        except Exception:
            pass
    timing_summary = extra_data.get("timing_str") if isinstance(extra_data, dict) else None
    if timing_summary:
        result["timing"]["summary"] = timing_summary
    timing_log.summary(
        valid=True,
        match_count=int(len(matches_out)),
        total_detections=len(detections),
        timing_log_path=str(timing_log.path) if timing_log.path else None,
    )
    return result
