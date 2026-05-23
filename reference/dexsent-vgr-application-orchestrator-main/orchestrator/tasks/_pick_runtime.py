"""Shared runtime for pick-style orchestrator tasks."""

import json
import math
import re
import struct
import time
from dataclasses import dataclass
from multiprocessing import shared_memory
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from orchestrator.core.context import StationContext
from orchestrator.core.runs import RunState
from orchestrator.vision.calibration import build_station_camera_calibration

_TEMPLATE_BASED_MODULES = {
    "template_matching",
    "tamplate_matching_sift",
    "opt_sift",
    "feature_matching",
}
_BIN_PICKING_POSE_MODULES = {
    "megapose_bin_picking",
    "ppf_icp_bin_picking",
}
_BIN_PICKING_NO_CANDIDATE_ERRORS = {
    "megapose_no_candidate",
    "parallel_jaw_no_grasp_candidate",
    "ppf_icp_no_candidate",
}

_RESULT_SHM_HEADER = struct.Struct("<QQII")
_RESULT_SHM_HEADER_BYTES = 64
_RESULT_SHM_FLAG_VALID = 1


def _result_shm_name(request_id: str) -> str:
    return f"vgr_result_{str(request_id or '').replace('-', '_')}"


def _try_read_result_shm(request_id: str) -> Optional[Dict[str, Any]]:
    """Best-effort read of the per-request vision result SHM.

    The vision engine also publishes results over ZMQ, but large bin-picking
    payloads can be missed by the live subscriber. SHM is the durable path for
    exactly this request, so use it as a fallback while the request is active.
    """
    if not request_id:
        return None
    shm = None
    try:
        try:
            shm = shared_memory.SharedMemory(
                name=_result_shm_name(request_id),
                create=False,
                track=False,
            )
        except TypeError:
            shm = shared_memory.SharedMemory(
                name=_result_shm_name(request_id),
                create=False,
            )
        if len(shm.buf) < _RESULT_SHM_HEADER_BYTES:
            return None
        timestamp_ns, sequence_id, result_size, flags = _RESULT_SHM_HEADER.unpack_from(
            shm.buf, 0
        )
        if not (flags & _RESULT_SHM_FLAG_VALID) or result_size <= 0:
            return None
        end = _RESULT_SHM_HEADER_BYTES + int(result_size)
        if end > len(shm.buf):
            return None
        payload = json.loads(bytes(shm.buf[_RESULT_SHM_HEADER_BYTES:end]).decode("utf-8"))
        if not isinstance(payload, dict):
            return None
        return {
            "event": "VISION_RESULT",
            "request_id": payload.get("request_id") or request_id,
            "camera_id": payload.get("camera_id"),
            "module": payload.get("module"),
            "timestamp_ns": payload.get("timestamp_ns", timestamp_ns),
            "sequence_id": payload.get("sequence_id", sequence_id),
            "result": payload.get("result") or {},
            "process_time_ms": payload.get("process_time_ms"),
            "source": "result_shm",
        }
    except FileNotFoundError:
        return None
    except Exception:
        return None
    finally:
        if shm is not None:
            try:
                shm.close()
            except Exception:
                pass


@dataclass
class PickRuntimeHooks:
    load_runtime_state: Optional[
        Callable[
            [StationContext, RunState, Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any], int, Callable[..., None]],
            Any,
        ]
    ] = None
    rank_matches: Optional[
        Callable[
            [
                StationContext, RunState, Dict[str, Any], Dict[str, Any],
                Dict[str, Any], Dict[str, Any], Any, List[Dict[str, Any]],
                Callable[..., None],
            ],
            List[Dict[str, Any]],
        ]
    ] = None
    enrich_vision_only_match: Optional[
        Callable[
            [StationContext, RunState, Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any], Any, Dict[str, Any], Callable[..., None]],
            Optional[Dict[str, Any]],
        ]
    ] = None
    resolve_target_override: Optional[
        Callable[
            [StationContext, RunState, Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any], Any, int, int, Dict[str, Any], Dict[str, Any], Dict[str, Any], List[float], List[float], Callable[..., None]],
            Optional[Dict[str, Any]],
        ]
    ] = None
    resolve_orientation_overrides: Optional[
        Callable[
            [StationContext, RunState, Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any], Any, int, int, Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any], List[float]],
            Optional[Dict[str, Any]],
        ]
    ] = None
    resolve_place_plan: Optional[
        Callable[
            [StationContext, RunState, Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any], Any, Dict[str, Any], Callable[..., None]],
            Optional[Dict[str, Any]],
        ]
    ] = None


def _log(ctx: StationContext, run_id: Optional[str], event: str, **fields: Any) -> None:
    if not run_id:
        return
    payload = {"event": event, "timestamp_ns": time.time_ns()}
    payload.update(fields)
    ctx.runs.append_event(run_id, payload)


def _quat_xyzw_to_rpy_deg(quat_xyzw: List[float]) -> List[float]:
    if not isinstance(quat_xyzw, list) or len(quat_xyzw) < 4:
        return [0.0, 0.0, 0.0]
    x, y, z, w = [float(v) for v in quat_xyzw[:4]]
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm < 1e-12:
        return [0.0, 0.0, 0.0]
    x /= norm
    y /= norm
    z /= norm
    w /= norm
    sinr = 2.0 * (w * x + y * z)
    cosr = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr, cosr)
    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1.0:
        pitch = math.copysign(math.pi / 2.0, sinp)
    else:
        pitch = math.asin(sinp)
    siny = 2.0 * (w * z + x * y)
    cosy = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny, cosy)
    return [math.degrees(roll), math.degrees(pitch), math.degrees(yaw)]


def _pose_debug_summary(pose: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(pose, dict):
        return {}
    position = pose.get("position_m")
    quat = pose.get("quat_xyzw")
    summary: Dict[str, Any] = {
        "frame": pose.get("frame", "base"),
    }
    if isinstance(position, list) and len(position) >= 3:
        summary["position_m"] = [float(position[0]), float(position[1]), float(position[2])]
    if isinstance(quat, list) and len(quat) >= 4:
        quat_norm = _normalize_quat_xyzw([float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])])
        summary["quat_xyzw"] = quat_norm
        summary["rotation_rpy_deg"] = _quat_xyzw_to_rpy_deg(quat_norm)
    return summary


def _tool_tcp_pose_from_state(
    state: Optional[Dict[str, Any]],
    tcp_calibration: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    if not isinstance(state, dict):
        return None
    flange_pose = state.get("flange_pose")
    ee_pose = state.get("tcp_pose")
    base_pose = flange_pose if isinstance(flange_pose, dict) else ee_pose
    if not isinstance(base_pose, dict):
        return None
    try:
        base_to_tool_tcp = _apply_tcp_calibration_to_base(
            {
                "translation_m": list(base_pose.get("position_m") or [0.0, 0.0, 0.0]),
                "rotation_quat_xyzw": list(base_pose.get("quat_xyzw") or [0.0, 0.0, 0.0, 1.0]),
            },
            tcp_calibration,
        )
        quat = (
            list(ee_pose.get("quat_xyzw"))
            if isinstance(ee_pose, dict) and isinstance(ee_pose.get("quat_xyzw"), list)
            else base_to_tool_tcp.get("rotation_quat_xyzw")
        )
        return _pose_debug_summary(
            {
                "position_m": base_to_tool_tcp.get("translation_m"),
                "quat_xyzw": quat,
                "frame": "base",
            }
        )
    except Exception:
        return None


def _robot_state_debug_summary(
    state: Optional[Dict[str, Any]],
    tcp_calibration: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if not isinstance(state, dict):
        return {}
    summary: Dict[str, Any] = {
        "mode": state.get("mode"),
        "active_motion_id": state.get("active_motion_id"),
    }
    if isinstance(state.get("q"), list):
        summary["q"] = [float(v) for v in state["q"]]
    if isinstance(state.get("tcp_pose"), dict):
        summary["tcp_pose"] = _pose_debug_summary(state.get("tcp_pose"))
    if isinstance(state.get("flange_pose"), dict):
        summary["flange_pose"] = _pose_debug_summary(state.get("flange_pose"))
    tool_tcp_pose = _tool_tcp_pose_from_state(state, tcp_calibration)
    if tool_tcp_pose:
        summary["tool_tcp_pose"] = tool_tcp_pose
    if isinstance(state.get("custom_tcp_pose"), dict):
        summary["custom_tcp_pose"] = _pose_debug_summary(state.get("custom_tcp_pose"))
    return summary


def _timing_json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _timing_json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_timing_json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _timing_enabled(*configs: Dict[str, Any]) -> bool:
    for cfg in configs:
        if not isinstance(cfg, dict):
            continue
        for key in ("timing_log_enabled", "enable_timing_log", "bin_picking_timing"):
            if key in cfg:
                return bool(cfg.get(key))
        timing_cfg = cfg.get("timing")
        if isinstance(timing_cfg, dict) and "enabled" in timing_cfg:
            return bool(timing_cfg.get("enabled"))
    return False


class _PickIterationTiming:
    def __init__(
        self,
        *,
        enabled: bool,
        path: Path,
        run_id: str,
        request_id: str,
        cycle: int,
        attempt: int,
    ) -> None:
        self.enabled = bool(enabled)
        self.path = path
        self.run_id = run_id
        self.request_id = request_id
        self.cycle = int(cycle)
        self.attempt = int(attempt)
        self.started_perf = time.perf_counter()
        self.rows: List[Dict[str, Any]] = []

    def span(self, stage: str, started_perf: float, **fields: Any) -> None:
        if not self.enabled:
            return
        ended_perf = time.perf_counter()
        row = {
            "type": "orchestrator_stage",
            "run_id": self.run_id,
            "request_id": self.request_id,
            "cycle": self.cycle,
            "attempt": self.attempt,
            "stage": stage,
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
        row = {
            "type": "orchestrator_summary",
            "run_id": self.run_id,
            "request_id": self.request_id,
            "cycle": self.cycle,
            "attempt": self.attempt,
            "total_s": float(max(0.0, time.perf_counter() - self.started_perf)),
            "stage_totals_s": {
                str(item.get("stage")): float(item.get("duration_s", 0.0))
                for item in self.rows
            },
            "timestamp_ns": time.time_ns(),
        }
        row.update(fields)
        self._append(row)

    def _append(self, row: Dict[str, Any]) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(_timing_json_safe(row), separators=(",", ":")) + "\n")
        except Exception:
            self.enabled = False


def _resolve_task_payload(ctx: StationContext, state: RunState) -> Dict[str, Any]:
    payload = state.task or {}
    if isinstance(payload, dict):
        if "recipe" in payload and isinstance(payload["recipe"], dict):
            return payload["recipe"]
        return payload
    raise RuntimeError("task_payload_invalid")


def _robot_disabled(ctx: StationContext) -> bool:
    state = ctx.robot.get_state() or {}
    if "robot_disabled" in state:
        return bool(state.get("robot_disabled"))
    return not bool(ctx.runtime_state.get("robot_enabled", True))


def _vision_timeout_s(vision_cfg: Dict[str, Any], module: Optional[str]) -> float:
    try:
        timeout_s = float((vision_cfg or {}).get("timeout_s", 0.0) or 0.0)
    except (TypeError, ValueError):
        timeout_s = 0.0
    if timeout_s > 0.0:
        return timeout_s
    module_name = str(module or "").strip().lower()
    if module_name in _BIN_PICKING_POSE_MODULES:
        return 15.0
    return 5.0


def _infer_object_id_from_path(raw_path: str) -> str:
    if not raw_path:
        return ""
    parts = [part for part in re.split(r"[\\/]+", raw_path) if part]
    if not parts:
        return ""
    for key in ("objects", "object_library"):
        if key in parts:
            idx = len(parts) - 1 - parts[::-1].index(key)
            if idx + 1 < len(parts):
                return parts[idx + 1]
    return ""


def _clean_path_string(raw_path: Optional[str]) -> str:
    value = str(raw_path or "").strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        value = value[1:-1].strip()
    return value


def _resolve_templates_dir(
    ctx: StationContext,
    raw_path: Optional[str],
    process_id: Optional[str],
    object_id: Optional[str],
) -> str:
    resolved: Optional[Path] = None
    if raw_path:
        resolved = Path(raw_path)
        if not resolved.is_absolute():
            resolved = (ctx.data_root / raw_path).resolve()
        if resolved.exists():
            return str(resolved)
    obj_id = object_id or _infer_object_id_from_path(raw_path or "")
    if process_id and obj_id:
        process = ctx.processes.get(process_id)
        station_id = process.get("station_id") if process else None
        if station_id:
            candidate = (
                ctx.data_paths.process_objects_dir(station_id, process_id)
                / obj_id
                / "templates"
            )
            if not candidate.exists():
                try:
                    ctx.objects.list(process_id)
                except Exception:
                    pass
            if candidate.exists():
                return str(candidate)
    if resolved:
        return str(resolved)
    raise RuntimeError("missing_templates_dir")


def _run_pick_vision_only_session(
    ctx: StationContext,
    state: RunState,
    handle: Any,
    recipe: Dict[str, Any],
    hooks: Optional[PickRuntimeHooks] = None,
) -> None:
    run_id = state.run_id
    runtime_hooks = hooks or PickRuntimeHooks()
    vision_cfg = recipe.get("vision", {}) if isinstance(recipe, dict) else {}
    pick_cfg = (
        (recipe.get("robot", {}) or {}).get("pick", {})
        if isinstance(recipe.get("robot", {}), dict)
        else {}
    )
    camera_id = vision_cfg.get("camera_id")
    module = vision_cfg.get("module")
    if not camera_id or not module:
        raise RuntimeError("missing_vision_camera_or_module")

    module_params = dict(vision_cfg.get("params", {}))
    if vision_cfg.get("object_id") and "object_id" not in module_params:
        module_params["object_id"] = vision_cfg.get("object_id")
    if module in _TEMPLATE_BASED_MODULES:
        templates_dir = module_params.get("templates_dir") or vision_cfg.get(
            "templates_dir"
        )
        if templates_dir or vision_cfg.get("object_id") or module_params.get("object_id"):
            module_params["templates_dir"] = _resolve_templates_dir(
                ctx,
                templates_dir,
                state.process_id,
                vision_cfg.get("object_id") or module_params.get("object_id"),
            )
        if "templates_dir" not in module_params and state.process_id:
            try:
                objects = ctx.objects.list(state.process_id)
            except Exception:
                objects = []
            if objects:
                fallback_obj = objects[0].get("object_id")
                if fallback_obj:
                    module_params["templates_dir"] = _resolve_templates_dir(
                        ctx, None, state.process_id, fallback_obj
                    )
        if "templates_dir" not in module_params:
            raise RuntimeError("missing_templates_dir")
    if module in ("tamplate_matching_sift", "opt_sift"):
        module_params.setdefault("scene_scale", 0.6)
        module_params.setdefault("nfeatures", 550)
        module_params.setdefault("n_octave_layers", 2)
        module_params.setdefault("contrast_threshold", 0.06)
        module_params.setdefault("match_ratio", 0.6)
        module_params.setdefault("min_score", 0.55)
        module_params.setdefault("min_inliers", 10)
        module_params.setdefault("min_good_matches", 7)
        module_params.setdefault("max_good_matches", 50)
        module_params.setdefault("ransac_thresh", 1.5)
        module_params.setdefault("min_inlier_ratio", 0.28)
        module_params.setdefault("max_reproj_rmse_px", 7.0)
        module_params.setdefault("min_projected_area_px", 1800)
        module_params.setdefault("projected_bounds_margin_px", 5)
        module_params.setdefault("depth_window_px", 3)
        module_params.setdefault("compute_surface_normal", False)
        module_params.setdefault("enable_geometric_filters", False)
        module_params.setdefault("enable_temporal_filter", False)
        module_params.setdefault("enable_pose_smoothing", False)
        module_params.setdefault("enable_corner_smoothing", False)
        module_params.setdefault("hold_last_valid_frames", 0)
        module_params.setdefault("allow_partial_visibility", True)
        module_params.setdefault("min_visible_corners", 2)
        module_params.setdefault("min_visible_area_ratio", 0.15)
        module_params.setdefault("clip_bbox_to_image", True)
        module_params.setdefault("include_image", False)
        module_params.setdefault("image_on_match", False)

    filters = dict(vision_cfg.get("filters") or {})
    if "min_inliers" not in filters and "min_inliers" in module_params:
        filters["min_inliers"] = module_params["min_inliers"]
    if "depth_range_m" not in filters:
        depth_min = module_params.get("depth_min_m")
        depth_max = module_params.get("depth_max_m")
        if depth_min is not None or depth_max is not None:
            filters["depth_range_m"] = [depth_min or 0.0, depth_max or 10.0]
    filters.setdefault("steady_frames", 1 if bool(pick_cfg.get("fast_acquire", True)) else 2)
    filters.setdefault("smooth_frames", filters.get("steady_frames", 1))
    filters.setdefault("max_center_delta_px", 10.0)
    filters.setdefault("max_depth_delta_m", 0.02)
    filters.setdefault("max_yaw_delta_deg", 10.0)

    enable_shm_output = vision_cfg.get("enable_shm_output")
    if enable_shm_output is None:
        enable_shm_output = not bool(module_params.get("include_image", False))

    request_id = f"{state.run_id}-vision-only"
    runtime_state = None
    if callable(runtime_hooks.load_runtime_state):
        runtime_state = runtime_hooks.load_runtime_state(
            ctx,
            state,
            recipe,
            vision_cfg,
            recipe.get("robot", {}) if isinstance(recipe.get("robot", {}), dict) else {},
            pick_cfg,
            module_params,
            0,
            lambda *_args, **_kwargs: None,
        )
    payload = {
        "event": "VISION_START",
        "camera_id": camera_id,
        "module": module,
        "fps_limit": vision_cfg.get("fps_limit", 0),
        "process_mode": vision_cfg.get("process_mode", "continuous"),
        "params": module_params,
        "enable_shm_output": bool(enable_shm_output),
        "request_id": request_id,
    }
    station_calibration = build_station_camera_calibration(
        ctx.data_paths,
        state.station_id,
    )
    if station_calibration:
        payload["calibration"] = station_calibration
    shm_max_result_size = vision_cfg.get("shm_max_result_size")
    if shm_max_result_size:
        payload["shm_max_result_size"] = int(shm_max_result_size)

    handle.vision_request_id = request_id
    handle.last_vision_request_id = request_id
    ctx.run_manager.set_last_vision(run_id, request_id=request_id)
    _log(
        ctx,
        run_id,
        "ROBOT_DISABLED_VISION_ONLY",
        task_type=state.task_type,
        camera_id=camera_id,
        module=module,
        request_id=request_id,
    )
    vision_started = False
    try:
        ctx.vision.start_session(payload)
        vision_started = True
        match_bundle = _wait_for_match(
            ctx,
            request_id,
            _vision_timeout_s(vision_cfg, module),
            handle,
            filters,
            {"stable_count": 0, "history": []},
            lambda *_args, **_kwargs: None,
            pick_cfg=pick_cfg,
        )
        evt = match_bundle.get("event", {}) or {}
        match = match_bundle.get("match", {}) or {}
        if callable(runtime_hooks.enrich_vision_only_match):
            try:
                enriched_match = runtime_hooks.enrich_vision_only_match(
                    ctx,
                    state,
                    recipe,
                    vision_cfg,
                    recipe.get("robot", {}) if isinstance(recipe.get("robot", {}), dict) else {},
                    pick_cfg,
                    module_params,
                    runtime_state,
                    match,
                    lambda *_args, **_kwargs: None,
                )
                if isinstance(enriched_match, dict) and enriched_match:
                    match = enriched_match
            except Exception:
                pass
        ctx.run_manager.set_last_vision(
            run_id,
            request_id=request_id,
            frame_id=evt.get("frame_id"),
            timestamp_ns=evt.get("timestamp_ns"),
        )
        _log(
            ctx,
            run_id,
            "VISION_ONLY_MATCH",
            request_id=request_id,
            frame_id=evt.get("frame_id"),
            score=match.get("score"),
            match=match,
        )
    finally:
        if vision_started:
            try:
                ctx.vision.stop_session(request_id)
            except Exception:
                pass
        if handle.vision_request_id == request_id:
            handle.vision_request_id = None


def _add_vec(a: List[float], b: List[float]) -> List[float]:
    return [a[0] + b[0], a[1] + b[1], a[2] + b[2]]


def _is_finite_vec(v: Any, n: int) -> bool:
    if not isinstance(v, list) or len(v) < n:
        return False
    for i in range(n):
        try:
            if not math.isfinite(float(v[i])):
                return False
        except Exception:
            return False
    return True


def _quat_xyzw_to_wxyz(q: List[float]) -> List[float]:
    return [q[3], q[0], q[1], q[2]]


def _quat_wxyz_to_xyzw(q: List[float]) -> List[float]:
    return [q[1], q[2], q[3], q[0]]


def _quat_mul(a: List[float], b: List[float]) -> List[float]:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return [
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ]


def _quat_conj(q: List[float]) -> List[float]:
    return [q[0], -q[1], -q[2], -q[3]]


def _quat_rotate(q_xyzw: List[float], v: List[float]) -> List[float]:
    q = _quat_xyzw_to_wxyz(q_xyzw)
    vq = [0.0, v[0], v[1], v[2]]
    rq = _quat_mul(_quat_mul(q, vq), _quat_conj(q))
    return [rq[1], rq[2], rq[3]]


def _normalize_quat_xyzw(q: List[float]) -> List[float]:
    if not q or len(q) < 4:
        return [0.0, 0.0, 0.0, 1.0]
    norm = math.sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3])
    if norm < 1e-8:
        return [0.0, 0.0, 0.0, 1.0]
    return [q[0] / norm, q[1] / norm, q[2] / norm, q[3] / norm]


def _rpy_deg_to_quat_xyzw(
    roll_deg: float, pitch_deg: float, yaw_deg: float
) -> List[float]:
    # ZYX (yaw-pitch-roll) intrinsic -> quaternion
    roll = math.radians(roll_deg)
    pitch = math.radians(pitch_deg)
    yaw = math.radians(yaw_deg)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    return _normalize_quat_xyzw([qx, qy, qz, qw])


def _invert_transform(
    translation_m: List[float], rotation_quat_xyzw: List[float]
) -> (List[float], List[float]):
    quat = _normalize_quat_xyzw(rotation_quat_xyzw)
    q_inv = _quat_wxyz_to_xyzw(_quat_conj(_quat_xyzw_to_wxyz(quat)))
    t_rot = _quat_rotate(q_inv, translation_m)
    return [-t_rot[0], -t_rot[1], -t_rot[2]], q_inv


def _resolve_hand_eye(raw: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(raw, dict):
        return {
            "translation_m": [0.0, 0.0, 0.0],
            "rotation_quat_xyzw": [0.0, 0.0, 0.0, 1.0],
        }
    if isinstance(raw.get("hand_eye"), dict):
        raw = dict(raw["hand_eye"])
        if str(raw.get("parent_frame") or "").strip().lower() == "tool_tcp":
            raw.setdefault("hand_eye_frame", "camera_in_tcp")
        else:
            raw.setdefault("hand_eye_frame", "camera_in_gripper")
    translation = raw.get("translation_m") or [0.0, 0.0, 0.0]
    if not isinstance(translation, list) or len(translation) < 3:
        translation = [0.0, 0.0, 0.0]
    translation = [float(translation[0]), float(translation[1]), float(translation[2])]
    quat = raw.get("rotation_quat_xyzw") or raw.get("quat_xyzw") or [0.0, 0.0, 0.0, 0.0]
    quat = _normalize_quat_xyzw(
        list(quat) if isinstance(quat, list) else [0.0, 0.0, 0.0, 0.0]
    )
    if quat == [0.0, 0.0, 0.0, 1.0] and (
        raw.get("rotation_rpy_deg") or raw.get("rpy_deg")
    ):
        rpy = raw.get("rotation_rpy_deg") or raw.get("rpy_deg") or [0.0, 0.0, 0.0]
        if isinstance(rpy, list) and len(rpy) >= 3:
            quat = _rpy_deg_to_quat_xyzw(float(rpy[0]), float(rpy[1]), float(rpy[2]))
    frame = str(raw.get("hand_eye_frame") or raw.get("frame") or "base_to_camera").strip().lower()
    if frame in ("camera_in_gripper", "gripper_to_camera"):
        frame = "camera_in_gripper"
    return {
        "translation_m": (
            list(translation) if isinstance(translation, list) else [0.0, 0.0, 0.0]
        ),
        "rotation_quat_xyzw": quat,
        "hand_eye_frame": frame,
        "parent_frame": str(raw.get("parent_frame") or "").strip().lower(),
        "child_frame": str(raw.get("child_frame") or "").strip().lower(),
    }


def _load_station_handeye(
    ctx: StationContext, process_id: Optional[str]
) -> Optional[Dict[str, Any]]:
    if not process_id:
        return None
    process = ctx.processes.get(process_id)
    if not process:
        return None
    station_id = process.get("station_id")
    if not station_id:
        return None
    path = ctx.data_paths.station_calibration_dir(station_id) / "handeye.json"
    combined_path = ctx.data_paths.station_calibration_dir(station_id) / "tcp.json"
    if combined_path.exists():
        try:
            payload = json.loads(combined_path.read_text(encoding="utf-8"))
            if isinstance(payload, dict) and isinstance(payload.get("hand_eye"), dict):
                hand_eye = dict(payload["hand_eye"])
                hand_eye.setdefault("hand_eye_frame", "camera_in_gripper")
                return hand_eye
        except Exception:
            pass
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _resolve_runtime_handeye(
    ctx: StationContext,
    process_id: Optional[str],
    robot_cfg: Optional[Dict[str, Any]],
) -> Tuple[Dict[str, Any], str, str]:
    _ = robot_cfg if isinstance(robot_cfg, dict) else {}
    source_pref = "station"
    station_raw = _load_station_handeye(ctx, process_id) if process_id else None
    station_cfg = dict(station_raw) if isinstance(station_raw, dict) else {}
    if not station_cfg:
        raise RuntimeError("station_handeye_missing")
    chosen: Dict[str, Any] = dict(station_cfg)
    source_used = "station"

    if "hand_eye_frame" not in chosen and "frame" not in chosen:
        chosen = dict(chosen)
        # Backward-compatible default used in follow_object.
        chosen["hand_eye_frame"] = "gripper_to_camera"

    return chosen, source_pref, source_used


def _coerce_vec3(value: Any, default: Optional[List[float]] = None) -> List[float]:
    fallback = list(default) if isinstance(default, list) else [0.0, 0.0, 0.0]
    if not isinstance(value, (list, tuple)) or len(value) < 3:
        return fallback
    out: List[float] = []
    for idx in range(3):
        try:
            num = float(value[idx])
        except (TypeError, ValueError):
            return fallback
        if not math.isfinite(num):
            return fallback
        out.append(num)
    return out


def _has_tcp_calibration_fields(
    raw: Any,
    *,
    allow_transform_alias: bool = True,
) -> bool:
    if not isinstance(raw, dict):
        return False
    if isinstance(raw.get("custom_tcp"), dict):
        return True
    explicit = any(
        key in raw
        for key in (
            "tcp_offset_m",
            "tcp_offset_rpy_deg",
            "gripper_x_offset_m",
            "gripper_y_offset_m",
            "gripper_z_offset_m",
            "gripper_roll_offset_deg",
            "gripper_pitch_offset_deg",
            "gripper_yaw_offset_deg",
        )
    )
    if explicit:
        return True
    return bool(
        allow_transform_alias
        and any(key in raw for key in ("translation_m", "rotation_rpy_deg"))
    )


def _normalize_tcp_calibration(
    raw: Optional[Dict[str, Any]],
    *,
    source: str = "identity",
    allow_transform_alias: bool = True,
) -> Dict[str, Any]:
    payload = raw if isinstance(raw, dict) else {}
    # Compose tool_mount × custom_tcp to get full flange→tool_tcp transform
    if isinstance(payload.get("custom_tcp"), dict):
        custom_tcp = payload["custom_tcp"]
        tool_mount = payload.get("tool_mount") if isinstance(payload.get("tool_mount"), dict) else None

        if tool_mount:
            # Compose: T_flange_tcp = T_flange_gripperbase × T_gripperbase_tcp
            tm_trans = _coerce_vec3(tool_mount.get("translation_m"), [0.0, 0.0, 0.0])
            tm_rpy = _coerce_vec3(tool_mount.get("rotation_rpy_deg"), [0.0, 0.0, 0.0])
            tm_quat = _rpy_deg_to_quat_xyzw(tm_rpy[0], tm_rpy[1], tm_rpy[2])

            ct_trans = _coerce_vec3(custom_tcp.get("translation_m"), [0.0, 0.0, 0.0])
            ct_rpy = _coerce_vec3(custom_tcp.get("rotation_rpy_deg"), [0.0, 0.0, 0.0])
            ct_quat = _rpy_deg_to_quat_xyzw(ct_rpy[0], ct_rpy[1], ct_rpy[2])

            # Compose transforms
            composed = _compose_transform(
                {"translation_m": tm_trans, "rotation_quat_xyzw": tm_quat},
                {"translation_m": ct_trans, "rotation_quat_xyzw": ct_quat},
            )
            payload = {
                "translation_m": composed["translation_m"],
                "rotation_quat_xyzw": composed["rotation_quat_xyzw"],
                # Keep rpy as zeros - rotation_quat_xyzw will be used directly
                "rotation_rpy_deg": [0.0, 0.0, 0.0],
            }
        else:
            payload = dict(custom_tcp)
    translation = payload.get("tcp_offset_m")
    if translation is None and allow_transform_alias:
        translation = payload.get("translation_m")
    if translation is None and any(
        key in payload
        for key in ("gripper_x_offset_m", "gripper_y_offset_m", "gripper_z_offset_m")
    ):
        translation = [
            payload.get("gripper_x_offset_m", 0.0),
            payload.get("gripper_y_offset_m", 0.0),
            payload.get("gripper_z_offset_m", 0.0),
        ]

    rpy = payload.get("tcp_offset_rpy_deg")
    if rpy is None and allow_transform_alias:
        rpy = payload.get("rotation_rpy_deg")
    if rpy is None and any(
        key in payload
        for key in (
            "gripper_roll_offset_deg",
            "gripper_pitch_offset_deg",
            "gripper_yaw_offset_deg",
        )
    ):
        rpy = [
            payload.get("gripper_roll_offset_deg", 0.0),
            payload.get("gripper_pitch_offset_deg", 0.0),
            payload.get("gripper_yaw_offset_deg", 0.0),
        ]

    has_fields = _has_tcp_calibration_fields(
        payload,
        allow_transform_alias=allow_transform_alias,
    )
    translation_m = _coerce_vec3(translation, [0.0, 0.0, 0.0])
    rotation_rpy_deg = _coerce_vec3(rpy, [0.0, 0.0, 0.0])
    # Prefer existing quaternion from payload (e.g. from composed transform)
    payload_quat = payload.get("rotation_quat_xyzw")
    if payload_quat and len(payload_quat) == 4:
        rotation_quat_xyzw = _normalize_quat_xyzw(list(payload_quat))
    else:
        rotation_quat_xyzw = _rpy_deg_to_quat_xyzw(
            rotation_rpy_deg[0],
            rotation_rpy_deg[1],
            rotation_rpy_deg[2],
        )
    return {
        "tcp_offset_frame": str(payload.get("tcp_offset_frame") or "flange"),
        "tcp_offset_m": translation_m,
        "tcp_offset_rpy_deg": rotation_rpy_deg,
        "translation_m": translation_m,
        "rotation_rpy_deg": rotation_rpy_deg,
        "rotation_quat_xyzw": rotation_quat_xyzw,
        "grasp_up_axis": str(
            payload.get("grasp_up_axis")
            or payload.get("grasp_candidate_up_axis")
            or ""
        ).strip().lower(),
        "grasp_up_min_dot": _coerce_float(
            payload.get("grasp_up_min_dot")
            if payload.get("grasp_up_min_dot") is not None
            else payload.get("grasp_candidate_up_min_dot"),
            0.0,
        ),
        "source": source,
        "has_calibration": bool(has_fields),
    }


def _load_station_tcp_calibration(
    ctx: StationContext,
    process_id: Optional[str],
) -> Optional[Dict[str, Any]]:
    if not process_id:
        return None
    try:
        process = ctx.processes.get(process_id)
    except Exception:
        process = None
    station_id = (process or {}).get("station_id") or process_id
    if not station_id:
        return None
    try:
        path = ctx.data_paths.station_calibration_dir(station_id) / "tcp.json"
        if path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                return payload
    except Exception:
        return None
    return None


def _resolve_runtime_tcp_calibration(
    ctx: StationContext,
    process_id: Optional[str],
    hand_eye_raw: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, Any], str]:
    station_tcp = _load_station_tcp_calibration(ctx, process_id)
    if isinstance(station_tcp, dict) and _has_tcp_calibration_fields(station_tcp):
        return _normalize_tcp_calibration(station_tcp, source="station_tcp"), "station_tcp"
    if isinstance(hand_eye_raw, dict) and _has_tcp_calibration_fields(
        hand_eye_raw,
        allow_transform_alias=False,
    ):
        return (
            _normalize_tcp_calibration(
                hand_eye_raw,
                source="handeye",
                allow_transform_alias=False,
            ),
            "handeye",
        )
    return _normalize_tcp_calibration({}, source="identity"), "identity"


def _transform_from_pose_target(target: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "translation_m": list(target.get("position_m") or [0.0, 0.0, 0.0]),
        "rotation_quat_xyzw": list(
            target.get("quat_xyzw") or [0.0, 0.0, 0.0, 1.0]
        ),
    }


def _pose_target_from_transform(
    transform: Dict[str, Any],
    *,
    frame: str = "base",
) -> Dict[str, Any]:
    return {
        "position_m": list(transform.get("translation_m") or [0.0, 0.0, 0.0]),
        "quat_xyzw": _normalize_quat_xyzw(
            list(transform.get("rotation_quat_xyzw") or [0.0, 0.0, 0.0, 1.0])
        ),
        "frame": frame,
    }


def _tcp_calibration_transform(tcp_calibration: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    calibration = (
        tcp_calibration
        if isinstance(tcp_calibration, dict)
        else _normalize_tcp_calibration({}, source="identity")
    )
    return {
        "translation_m": _coerce_vec3(
            calibration.get("tcp_offset_m") or calibration.get("translation_m"),
            [0.0, 0.0, 0.0],
        ),
        "rotation_quat_xyzw": _normalize_quat_xyzw(
            list(
                calibration.get("rotation_quat_xyzw")
                or _rpy_deg_to_quat_xyzw(
                    *_coerce_vec3(
                        calibration.get("tcp_offset_rpy_deg")
                        or calibration.get("rotation_rpy_deg"),
                        [0.0, 0.0, 0.0],
                    )
                )
            )
        ),
    }


def _apply_tcp_calibration_to_base(
    base_to_flange: Dict[str, Any],
    tcp_calibration: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Return T_base_tcp = T_base_flange @ T_flange_tcp."""
    return _compose_transform(base_to_flange, _tcp_calibration_transform(tcp_calibration))


def _command_pose_for_desired_tcp_to_ee(
    desired_tcp_pose: Dict[str, Any],
    tcp_calibration: Optional[Dict[str, Any]],
    *,
    debug_log: Optional[Any] = None,
) -> Dict[str, Any]:
    """Convert a tool_tcp target to the robot command pose.

    The position needs the station TCP offset compensation, but the Franka
    command path already treats the requested quaternion as the visible tool
    orientation. Rotating the command quaternion by the mount offset creates
    the observed constant 45 degree yaw error.
    """
    target = dict(desired_tcp_pose or {})
    if not isinstance(tcp_calibration, dict) or not tcp_calibration.get("has_calibration"):
        return target

    # Convert tool_tcp to flange using TCP calibration inverse
    desired = _transform_from_pose_target(target)
    tcp_tf = _tcp_calibration_transform(tcp_calibration)
    inv_t, inv_q = _invert_transform(
        tcp_tf.get("translation_m", [0.0, 0.0, 0.0]),
        tcp_tf.get("rotation_quat_xyzw", [0.0, 0.0, 0.0, 1.0]),
    )

    # Get flange pose (tool_tcp @ inv(tcp_calibration))
    flange = _compose_transform(
        desired,
        {"translation_m": inv_t, "rotation_quat_xyzw": inv_q},
    )

    # Apply the full F_T_EE transform. The robot controller consumes EE poses,
    # so the pose sent over the wire must already include the flange->EE mount.
    f_t_ee = tcp_calibration.get("franka_f_t_ee", {})
    f_t_ee_trans = _coerce_vec3(f_t_ee.get("translation_m"), [0.0, 0.0, 0.1034])
    f_t_ee_quat = _normalize_quat_xyzw(
        list(
            f_t_ee.get("rotation_quat_xyzw")
            or _rpy_deg_to_quat_xyzw(
                *_coerce_vec3(f_t_ee.get("rotation_rpy_deg"), [0.0, 0.0, 0.0])
            )
        )
    )

    # Compose the full flange->EE offset.
    flange_with_ee_offset = _compose_transform(
        flange,
        {
            "translation_m": f_t_ee_trans,
            "rotation_quat_xyzw": f_t_ee_quat,
        },
    )

    result = _pose_target_from_transform(flange_with_ee_offset, frame=str(target.get("frame", "base")))
    result["quat_xyzw"] = _normalize_quat_xyzw(
        list(target.get("quat_xyzw") or [0.0, 0.0, 0.0, 1.0])
    )

    # Debug logging if provided
    if debug_log:
        debug_log("DEBUG_TCP_TO_EE_TRANSFORM",
            tcp_pos=desired.get("translation_m"),
            tcp_quat=desired.get("rotation_quat_xyzw"),
            flange_pos=flange.get("translation_m"),
            ee_pos=result.get("position_m"),
            ee_quat=result.get("quat_xyzw"),
            command_orientation_source="desired_tcp_pose",
        )

    return result


def _command_pose_for_desired_tcp(
    desired_tcp_pose: Dict[str, Any],
    tcp_calibration: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Convert a desired calibrated TCP pose into the robot command frame."""
    return _command_pose_for_desired_tcp_to_ee(desired_tcp_pose, tcp_calibration)


def _command_pose_for_tcp_position_with_flange_orientation(
    desired_tcp_pose: Dict[str, Any],
    tcp_calibration: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Command the SDK end-effector frame while treating target orientation as flange.

    Parallel-jaw grasp planning expresses jaw/approach axes in the robot
    end-effector frame. The target position is still the desired TCP/contact
    point, so only the calibrated translation offset is used to recover the
    flange position. Applying the TCP rotation here would rotate the already
    flange-relative grasp orientation a second time.
    """
    target = dict(desired_tcp_pose or {})
    if not isinstance(tcp_calibration, dict) or not tcp_calibration.get("has_calibration"):
        return target
    quat = _normalize_quat_xyzw(
        list(target.get("quat_xyzw") or [0.0, 0.0, 0.0, 1.0])
    ) or [0.0, 0.0, 0.0, 1.0]
    position = _coerce_vec3(target.get("position_m"), [0.0, 0.0, 0.0])
    tcp_offset = _coerce_vec3(
        tcp_calibration.get("tcp_offset_m") or tcp_calibration.get("translation_m"),
        [0.0, 0.0, 0.0],
    )
    flange_to_tcp_base = _quat_rotate(quat, tcp_offset)
    return {
        "position_m": [
            position[0] - flange_to_tcp_base[0],
            position[1] - flange_to_tcp_base[1],
            position[2] - flange_to_tcp_base[2],
        ],
        "quat_xyzw": quat,
        "frame": str(target.get("frame", "base")),
    }


def _quat_xyzw_to_matrix(quat_xyzw: List[float]) -> List[List[float]]:
    x, y, z, w = [float(v) for v in quat_xyzw[:4]]
    n = x * x + y * y + z * z + w * w
    if n < 1e-12:
        return [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    s = 2.0 / n
    xx, yy, zz = x * x * s, y * y * s, z * z * s
    xy, xz, yz = x * y * s, x * z * s, y * z * s
    wx, wy, wz = w * x * s, w * y * s, w * z * s
    return [
        [1.0 - (yy + zz), xy - wz, xz + wy],
        [xy + wz, 1.0 - (xx + zz), yz - wx],
        [xz - wy, yz + wx, 1.0 - (xx + yy)],
    ]


def _hand_eye_to_matrix(hand_eye: Dict[str, Any], frame: str) -> List[List[float]]:
    if frame not in ("camera_in_gripper", "gripper_to_camera"):
        raise RuntimeError(f"fusion_requires_gripper_to_camera_handeye:{frame}")
    translation = hand_eye.get("translation_m") or [0.0, 0.0, 0.0]
    quat = hand_eye.get("rotation_quat_xyzw") or [0.0, 0.0, 0.0, 1.0]
    R = _quat_xyzw_to_matrix(_normalize_quat_xyzw(list(quat)))
    return [
        [float(R[0][0]), float(R[0][1]), float(R[0][2]), float(translation[0])],
        [float(R[1][0]), float(R[1][1]), float(R[1][2]), float(translation[1])],
        [float(R[2][0]), float(R[2][1]), float(R[2][2]), float(translation[2])],
        [0.0, 0.0, 0.0, 1.0],
    ]


def _build_fused_camera_data(
    station_calibration: Optional[Dict[str, Any]],
    fused_camera: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    camera_data = {}
    if isinstance(station_calibration, dict):
        camera_data = dict(station_calibration.get("camera_data") or {})
    fused = dict(fused_camera or {})
    K = fused.get("K")
    width = int(fused.get("width") or 0)
    height = int(fused.get("height") or 0)
    if K:
        camera_data["K"] = K
    if width > 0 and height > 0:
        camera_data["resolution"] = [height, width]
    intrinsics = dict(camera_data.get("intrinsics") or {})
    if K and len(K) >= 3:
        intrinsics["fx"] = float(K[0][0])
        intrinsics["fy"] = float(K[1][1])
        intrinsics["cx"] = float(K[0][2])
        intrinsics["cy"] = float(K[1][2])
    if width > 0 and height > 0:
        intrinsics["resolution"] = {"width": width, "height": height}
    if intrinsics:
        camera_data["intrinsics"] = intrinsics
    camera_data["depth_scale_m_per_unit"] = float(fused.get("depth_scale", 1.0) or 1.0)
    camera_data.setdefault("source", "capture_pose_fusion")
    return camera_data


def _camera_core_control_url(ctx: StationContext) -> str:
    cfg = ctx.config.get("camera_core", {}) if ctx and ctx.config else {}
    return str(cfg.get("calibration_control_url", "http://127.0.0.1:8210")).rstrip("/")


def _request_camera_core_fusion(
    ctx: StationContext,
    payload: Dict[str, Any],
    timeout_s: float,
) -> Dict[str, Any]:
    url = f"{_camera_core_control_url(ctx)}/fusion/capture_pose"
    data = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    req = Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(req, timeout=float(timeout_s)) as resp:
            body = resp.read().decode("utf-8")
        return json.loads(body) if body else {"status": "ok"}
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"camera_core_fusion_http_{exc.code}:{detail or exc.reason}") from exc
    except URLError as exc:
        raise RuntimeError(f"camera_core_fusion_unreachable:{exc}") from exc


def _fusion_output_dir(
    ctx: StationContext,
    run_id: Optional[str],
    cycle: int,
    attempt: int,
) -> str:
    base = (
        ctx.data_paths.runs
        if ctx and getattr(ctx, "data_paths", None) is not None
        else Path("data/runs")
    )
    safe_run_id = str(run_id or "adhoc-run").strip() or "adhoc-run"
    return str(base / safe_run_id / "fusion" / f"cycle_{cycle:03d}_attempt_{attempt:03d}")


def _attach_fusion_visualization(
    match: Dict[str, Any],
    fusion_result: Optional[Dict[str, Any]],
    *,
    frame_id: Optional[str] = None,
    requested_camera_id: Optional[str] = None,
) -> Dict[str, Any]:
    if not isinstance(match, dict):
        return match
    depth_source = {
        "frame_id": str(frame_id or "").strip() or None,
        "camera_id": str(requested_camera_id or "").strip() or None,
        "mode": "raw",
    }
    frame_id_text = str(frame_id or "").strip()
    camera_id_text = str(requested_camera_id or "").strip()
    if frame_id_text:
        depth_source["camera_id"] = frame_id_text.split(":", 1)[0] or depth_source["camera_id"]
    if depth_source["camera_id"] and str(depth_source["camera_id"]).endswith("_fused"):
        depth_source["mode"] = "fused"
    match["depth_source"] = depth_source
    if not isinstance(fusion_result, dict):
        return match
    fused_paths = fusion_result.get("visualization_3d")
    if not isinstance(fused_paths, dict) or not fused_paths:
        return match
    debug_paths = dict(match.get("debug_paths") or {})
    visualization_3d = dict(match.get("visualization_3d") or {})
    for key, value in fused_paths.items():
        if not value:
            continue
        visualization_3d[key] = value
        debug_paths[key] = value
    match["visualization_3d"] = visualization_3d
    match["debug_paths"] = debug_paths
    match["fusion"] = {
        "status": str(fusion_result.get("status") or "ok"),
        "source_camera_id": fusion_result.get("source_camera_id"),
        "virtual_camera_id": fusion_result.get("virtual_camera_id"),
        "render_source": fusion_result.get("render_source"),
        "render_source_points": fusion_result.get("render_source_points"),
        "captures": fusion_result.get("captures"),
        "depth_valid_px": fusion_result.get("depth_valid_px"),
        "elapsed_s": fusion_result.get("elapsed_s"),
        "consumed_camera_id": depth_source.get("camera_id"),
        "consumed_mode": depth_source.get("mode"),
    }
    return match


def _quat_mul_xyzw(a: List[float], b: List[float]) -> List[float]:
    return _quat_wxyz_to_xyzw(_quat_mul(_quat_xyzw_to_wxyz(a), _quat_xyzw_to_wxyz(b)))


def _normalize_vec(v: List[float]) -> Optional[List[float]]:
    if not v or len(v) < 3:
        return None
    norm = math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
    if norm < 1e-8:
        return None
    return [v[0] / norm, v[1] / norm, v[2] / norm]


def _dot(a: List[float], b: List[float]) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _match_ok(match: Dict[str, Any], filters: Dict[str, Any]) -> (bool, str):
    inliers_raw = match.get("inliers")
    inliers = float(inliers_raw) if inliers_raw is not None else None
    score_raw = match.get("score")
    score = float(score_raw) if score_raw is not None else None
    area_raw = match.get("area_px")
    area_px = float(area_raw) if area_raw is not None else None
    x_scale_raw = match.get("x_scale")
    y_scale_raw = match.get("y_scale")
    x_scale = float(x_scale_raw) if x_scale_raw is not None else None
    y_scale = float(y_scale_raw) if y_scale_raw is not None else None
    bbox = match.get("bbox_xywh") or []
    if area_px is None and isinstance(bbox, list) and len(bbox) >= 4:
        try:
            bw = float(bbox[2])
            bh = float(bbox[3])
            if bw > 0 and bh > 0:
                area_px = bw * bh
        except (TypeError, ValueError):
            area_px = None

    min_inliers = float(filters.get("min_inliers", 30))
    min_score = float(filters.get("min_score", 0))
    min_area_px = float(filters.get("min_area_px", 8000))
    min_scale = float(filters.get("min_scale", 0.4))
    max_scale = float(filters.get("max_scale", 1.6))
    min_bbox_px = float(filters.get("min_bbox_px", 20))

    if inliers is not None and inliers < min_inliers:
        return False, "inliers_low"
    if score is not None and score < min_score:
        return False, "score_low"
    if area_px is not None and area_px < min_area_px:
        return False, "area_low"
    if (
        x_scale is not None
        and y_scale is not None
        and (x_scale < min_scale or y_scale < min_scale)
    ):
        return False, "scale_low"
    if (
        x_scale is not None
        and y_scale is not None
        and (x_scale > max_scale or y_scale > max_scale)
    ):
        return False, "scale_high"
    if isinstance(bbox, list) and len(bbox) >= 4:
        if bbox[2] < min_bbox_px or bbox[3] < min_bbox_px:
            return False, "bbox_small"
    return True, "ok"


def _normalize_angle_deg(angle: float) -> float:
    value = (angle + 180.0) % 360.0 - 180.0
    return value


def _wrap_yaw_90(yaw_deg: float) -> float:
    # Wrap to [-90, 90] by folding 180-deg symmetry.
    wrapped = _normalize_angle_deg(yaw_deg)
    if wrapped > 90.0:
        wrapped -= 180.0
    elif wrapped < -90.0:
        wrapped += 180.0
    return wrapped


def _normalize_quat_xyzw(q: List[float]) -> List[float]:
    if not q or len(q) != 4:
        return [0.0, 0.0, 0.0, 1.0]
    norm = math.sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3])
    if norm < 1e-8:
        return [0.0, 0.0, 0.0, 1.0]
    return [q[0] / norm, q[1] / norm, q[2] / norm, q[3] / norm]


def _cross(a: List[float], b: List[float]) -> List[float]:
    return [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]


def _project_to_plane(v: List[float], n: List[float]) -> List[float]:
    dot = v[0] * n[0] + v[1] * n[1] + v[2] * n[2]
    return [v[0] - dot * n[0], v[1] - dot * n[1], v[2] - dot * n[2]]


def _quat_from_axes(
    x_axis: List[float], y_axis: List[float], z_axis: List[float]
) -> List[float]:
    m00, m01, m02 = x_axis[0], y_axis[0], z_axis[0]
    m10, m11, m12 = x_axis[1], y_axis[1], z_axis[1]
    m20, m21, m22 = x_axis[2], y_axis[2], z_axis[2]
    trace = m00 + m11 + m22
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (m21 - m12) / s
        qy = (m02 - m20) / s
        qz = (m10 - m01) / s
    elif m00 > m11 and m00 > m22:
        s = math.sqrt(1.0 + m00 - m11 - m22) * 2.0
        qw = (m21 - m12) / s
        qx = 0.25 * s
        qy = (m01 + m10) / s
        qz = (m02 + m20) / s
    elif m11 > m22:
        s = math.sqrt(1.0 + m11 - m00 - m22) * 2.0
        qw = (m02 - m20) / s
        qx = (m01 + m10) / s
        qy = 0.25 * s
        qz = (m12 + m21) / s
    else:
        s = math.sqrt(1.0 + m22 - m00 - m11) * 2.0
        qw = (m10 - m01) / s
        qx = (m02 + m20) / s
        qy = (m12 + m21) / s
        qz = 0.25 * s
    return _normalize_quat_xyzw([qx, qy, qz, qw])


def _parse_axis(axis: str) -> Optional[Dict[str, Any]]:
    if not axis:
        return None
    axis = axis.strip().lower()
    sign = 1.0
    if axis.startswith("-"):
        sign = -1.0
        axis = axis[1:]
    if axis in ("x", "y", "z"):
        return {"axis": axis, "sign": sign}
    return None


def _scale_vec(v: List[float], scale: float) -> List[float]:
    return [v[0] * scale, v[1] * scale, v[2] * scale]


def _coerce_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        num = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(num):
        return default
    return num


def _normalize_parallel_jaw_family_label(value: Any) -> Optional[str]:
    raw = str(value or "").strip().lower()
    if raw in {"internal", "inside", "inner"}:
        return "internal"
    if raw in {"external", "outside", "outer"}:
        return "external"
    return None


def _issue_parallel_jaw_gripper_command(
    ctx: StationContext,
    action: str,
    width_m: Optional[float],
    force_n: Optional[float] = None,
) -> None:
    action_name = str(action or "open").strip().lower()
    if action_name == "close":
        ctx.robot.close_gripper(width_m, force_n=force_n)
        return
    ctx.robot.open_gripper(width_m, force_n=force_n)


def _resolve_parallel_jaw_widths(
    pick_cfg: Dict[str, Any],
    runtime_target_data: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    target = runtime_target_data if isinstance(runtime_target_data, dict) else {}
    nominal_width = _coerce_float(target.get("opening_width_m"), None)
    family_label = _normalize_parallel_jaw_family_label(
        target.get("grasp_family_label")
    )
    family_mode = "internal" if family_label == "internal" else "external"
    legacy_open_offset = _coerce_float(
        pick_cfg.get("parallel_jaw_pregrasp_open_offset_m"),
        _coerce_float(pick_cfg.get("parallel_jaw_open_offset_m"), 0.02),
    )
    legacy_close_offset = _coerce_float(
        pick_cfg.get("parallel_jaw_grasp_close_offset_m"),
        _coerce_float(pick_cfg.get("parallel_jaw_close_offset_m"), -0.005),
    )
    open_offset = _coerce_float(
        pick_cfg.get(f"parallel_jaw_{family_mode}_pregrasp_open_offset_m"),
        legacy_open_offset,
    ) or 0.0
    close_offset = _coerce_float(
        pick_cfg.get(f"parallel_jaw_{family_mode}_grasp_close_offset_m"),
        legacy_close_offset,
    ) or 0.0
    fallback_open = _coerce_float(
        pick_cfg.get("parallel_jaw_open_width_m"),
        None,
    )
    fallback_close = _coerce_float(
        pick_cfg.get("parallel_jaw_close_width_m"),
        None,
    )
    external_force_enabled = bool(pick_cfg.get("parallel_jaw_external_force_enabled", False))
    external_force_n = _coerce_float(
        pick_cfg.get("parallel_jaw_external_force_n"),
        40.0,
    )
    internal_force_enabled = bool(pick_cfg.get("parallel_jaw_internal_force_enabled", False))
    internal_force_n = _coerce_float(
        pick_cfg.get("parallel_jaw_internal_force_n"),
        40.0,
    )

    open_width = fallback_open
    close_width = fallback_close
    pregrasp_action = "open"
    grasp_action = "close"
    grasp_force_n: Optional[float] = None

    if family_mode == "internal":
        pregrasp_action = "close"
        grasp_action = "open"
        open_width = fallback_close
        close_width = fallback_open
        if nominal_width is not None:
            open_width = max(0.0, nominal_width - open_offset)
            close_width = max(0.0, nominal_width - close_offset)
        # Inner grasps expand to the target width with an open/move command.
        # Force is only meaningful when explicitly enabled.
        grasp_force_n = float(internal_force_n or 0.0) if internal_force_enabled else None
    else:
        if nominal_width is not None:
            open_width = max(0.0, nominal_width + open_offset)
            close_width = max(0.0, nominal_width + close_offset)
        # On Franka, close_gripper(width, force_n=None) uses a force grasp. For
        # position-only external grasps, pass zero force so the adapter uses move().
        grasp_force_n = float(external_force_n or 0.0) if external_force_enabled else 0.0

    return {
        "nominal_width_m": nominal_width,
        "pregrasp_open_width_m": open_width,
        "grasp_close_width_m": close_width,
        "pregrasp_open_offset_m": open_offset,
        "grasp_close_offset_m": close_offset,
        "grasp_family_label": family_label,
        "grasp_family_mode": family_mode,
        "pregrasp_action": pregrasp_action,
        "grasp_action": grasp_action,
        "grasp_force_n": grasp_force_n,
    }


def _axis_angle_quat(axis: List[float], angle_rad: float) -> List[float]:
    axis_n = _normalize_vec(axis)
    if not axis_n:
        return [0.0, 0.0, 0.0, 1.0]
    half = angle_rad / 2.0
    s = math.sin(half)
    return [axis_n[0] * s, axis_n[1] * s, axis_n[2] * s, math.cos(half)]


def _quat_from_two_vectors(a: List[float], b: List[float]) -> List[float]:
    a_n = _normalize_vec(a)
    b_n = _normalize_vec(b)
    if not a_n or not b_n:
        return [0.0, 0.0, 0.0, 1.0]
    dot = max(-1.0, min(1.0, a_n[0] * b_n[0] + a_n[1] * b_n[1] + a_n[2] * b_n[2]))
    if dot > 0.999999:
        return [0.0, 0.0, 0.0, 1.0]
    if dot < -0.999999:
        axis = _normalize_vec([0.0, -a_n[2], a_n[1]]) or [1.0, 0.0, 0.0]
        return _axis_angle_quat(axis, math.pi)
    axis = [
        a_n[1] * b_n[2] - a_n[2] * b_n[1],
        a_n[2] * b_n[0] - a_n[0] * b_n[2],
        a_n[0] * b_n[1] - a_n[1] * b_n[0],
    ]
    s = math.sqrt((1.0 + dot) * 2.0)
    inv_s = 1.0 / s
    return [axis[0] * inv_s, axis[1] * inv_s, axis[2] * inv_s, s * 0.5]


def _yaw_quat_xyzw(yaw_rad: float) -> List[float]:
    half = yaw_rad / 2.0
    return [0.0, 0.0, math.sin(half), math.cos(half)]


def _apply_yaw(seed: List[float], yaw_rad: float, yaw_frame: str) -> List[float]:
    yaw_quat = _yaw_quat_xyzw(yaw_rad)
    if yaw_frame == "base":
        return _quat_mul_xyzw(yaw_quat, seed)
    return _quat_mul_xyzw(seed, yaw_quat)


def _is_identity_quat(q: List[float], tol: float = 1e-4) -> bool:
    if len(q) != 4:
        return True
    return (
        abs(q[0]) < tol
        and abs(q[1]) < tol
        and abs(q[2]) < tol
        and abs(q[3] - 1.0) < tol
    )


def _get_pose_quat(pose: Dict[str, Any]) -> Optional[List[float]]:
    if not pose:
        return None
    target = pose.get("tcp_pose") or pose.get("tcp") or pose
    quat = target.get("quat_xyzw")
    if isinstance(quat, list) and len(quat) == 4:
        return quat
    return None


def _pose_missing(pose: Optional[Dict[str, Any]]) -> bool:
    if not pose:
        return True
    if pose.get("joints"):
        return False
    target = pose.get("tcp") or pose.get("tcp_pose") or pose
    return not target.get("position_m")


def _config_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"1", "true", "yes", "on", "enabled"}:
            return True
        if text in {"0", "false", "no", "off", "disabled"}:
            return False
    return bool(default)


def _resolve_optional_named_pose(
    pose_index: Dict[str, Dict[str, Any]],
    raw_pose: Any,
    raw_name: Any,
    *,
    error_code: str,
) -> tuple[Optional[Dict[str, Any]], str]:
    pose_name = str(raw_name or "").strip()
    pose_inline = raw_pose if isinstance(raw_pose, dict) else None
    if not pose_name and isinstance(pose_inline, dict):
        pose_name = str(
            pose_inline.get("pose_name") or pose_inline.get("name") or ""
        ).strip()
    if pose_name:
        pose = pose_index.get(pose_name)
        if isinstance(pose, dict):
            return dict(pose), pose_name
        if pose_inline is None:
            raise RuntimeError(error_code)
    if isinstance(pose_inline, dict) and not _pose_missing(pose_inline):
        pose = dict(pose_inline)
        if pose_name and not pose.get("name"):
            pose["name"] = pose_name
        return pose, pose_name or str(pose.get("name") or "").strip()
    return None, pose_name


def _compose_transform(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    ra = a.get("rotation_quat_xyzw", [0.0, 0.0, 0.0, 1.0])
    rb = b.get("rotation_quat_xyzw", [0.0, 0.0, 0.0, 1.0])
    ta = a.get("translation_m", [0.0, 0.0, 0.0])
    tb = b.get("translation_m", [0.0, 0.0, 0.0])
    r = _quat_wxyz_to_xyzw(_quat_mul(_quat_xyzw_to_wxyz(ra), _quat_xyzw_to_wxyz(rb)))
    t = _add_vec(ta, _quat_rotate(ra, tb))
    return {"translation_m": t, "rotation_quat_xyzw": r}


def _transform_point(pt: List[float], transform: Dict[str, Any]) -> List[float]:
    rot = transform.get("rotation_quat_xyzw", [0.0, 0.0, 0.0, 1.0])
    trans = transform.get("translation_m", [0.0, 0.0, 0.0])
    return _add_vec(_quat_rotate(rot, pt), trans)


def _move_to_pose(
    ctx: StationContext,
    run_id: Optional[str],
    pose: Dict[str, Any],
    default_profile: str,
    *,
    prefer_cartesian: bool = False,
    pose_index: Optional[Dict[str, Dict[str, Any]]] = None,
    tcp_calibration: Optional[Dict[str, Any]] = None,
) -> None:
    profile = pose.get("profile", default_profile)
    target = pose.get("tcp_pose") or pose.get("tcp") or pose
    position = target.get("position_m")
    quat = target.get("quat_xyzw", [0.0, 0.0, 0.0, 1.0])
    robot_state_before = ctx.robot.get_state() or {}
    robot_state_summary = _robot_state_debug_summary(robot_state_before, tcp_calibration)
    _log(
        ctx,
        run_id,
        "PICK_PLACE_MOVE_TO_POSE_BEGIN",
        profile=profile,
        prefer_cartesian=prefer_cartesian,
        target=_pose_debug_summary(target),
        state=robot_state_summary,
    )
    robot_mode = str(robot_state_before.get("mode") or "").strip().upper()
    robot_connected = robot_state_before.get("connected")
    if robot_connected is False or robot_mode in {"ERROR", "DISCONNECTED"}:
        reason = f"robot_not_ready: mode={robot_mode or 'UNKNOWN'}"
        _log(
            ctx,
            run_id,
            "PICK_PLACE_ROBOT_NOT_READY",
            stage="move_to_pose",
            reason=reason,
            state=robot_state_summary,
        )
        raise RuntimeError(reason)

    strategy = str(
        pose.get("move_strategy")
        or ("cartesian" if prefer_cartesian else "joint")
    ).strip().lower()
    # Prefer joints if available - they are the ground truth for where the robot was recorded
    if "joints" in pose:
        ctx.robot.movej(tuple(pose["joints"]), profile)
        _log(
            ctx,
            run_id,
            "PICK_PLACE_MOVE_TO_POSE_COMPLETE",
            profile=profile,
            prefer_cartesian=prefer_cartesian,
            target=_pose_debug_summary(target),
            state=_robot_state_debug_summary(ctx.robot.get_state() or {}, tcp_calibration),
        )
        return
    # Fall back to Cartesian if no joints available
    if strategy == "joint" or position is None:
        raise RuntimeError("pose_has_no_movement_target")
    if strategy == "ik_nullspace" and position is not None:
        preferred_name = str(pose.get("preferred_pose_name") or "").strip()
        preferred_pose = (pose_index or {}).get(preferred_name)
        raw_preferred_joints = (
            preferred_pose.get("joints") if isinstance(preferred_pose, dict) else None
        )
        preferred_joints = (
            tuple(raw_preferred_joints)
            if isinstance(raw_preferred_joints, list) and len(raw_preferred_joints) == 7
            else None
        )
        if preferred_name and preferred_joints is None:
            _log(
                ctx,
                run_id,
                "PICK_PLACE_PREFERRED_POSE_SKIPPED",
                stage="move_to_pose",
                preferred_pose_name=preferred_name,
                reason="missing_or_invalid_joints",
            )
        state = ctx.robot.get_state() or {}
        seed_joints = state.get("q") or state.get("joints") or None
        ctx.robot.move_tcp_ik(
            {
                "position_m": position,
                "quat_xyzw": quat,
                "frame": target.get("frame", "base"),
            },
            profile,
            seed_joints=tuple(seed_joints) if isinstance(seed_joints, list) and len(seed_joints) == 7 else None,
            preferred_joints=preferred_joints,
            position_tolerance_m=float(pose.get("ik_position_tolerance_m", 0.002)),
            orientation_tolerance_deg=float(pose.get("ik_orientation_tolerance_deg", 2.0)),
            approximate_position_tolerance_m=float(
                pose.get("ik_approximate_position_tolerance_m", 0.015)
            ),
            approximate_orientation_tolerance_deg=float(
                pose.get("ik_approximate_orientation_tolerance_deg", 3.0)
            ),
            max_iterations=int(pose.get("ik_max_iterations", 300)),
        )
        _log(
            ctx,
            run_id,
            "PICK_PLACE_MOVE_TO_POSE_COMPLETE",
            profile=profile,
            prefer_cartesian=prefer_cartesian,
            target=_pose_debug_summary(target),
            state=_robot_state_debug_summary(ctx.robot.get_state() or {}, tcp_calibration),
        )
        return
    if (prefer_cartesian or strategy == "cartesian") and position is not None:
        # Note: Can't access run_id here, so just use conversion_applied flag to track
        # The position has already been converted if tcp_calibration was available
        ctx.robot.movel(
            {
                "position_m": position,
                "quat_xyzw": quat,
                "frame": target.get("frame", "base"),
            },
            profile,
        )
        _log(
            ctx,
            run_id,
            "PICK_PLACE_MOVE_TO_POSE_COMPLETE",
            profile=profile,
            prefer_cartesian=prefer_cartesian,
            target=_pose_debug_summary(target),
            state=_robot_state_debug_summary(ctx.robot.get_state() or {}, tcp_calibration),
        )
        return
    if "joints" in pose:
        ctx.robot.movej(tuple(pose["joints"]), profile)
        _log(
            ctx,
            run_id,
            "PICK_PLACE_MOVE_TO_POSE_COMPLETE",
            profile=profile,
            prefer_cartesian=prefer_cartesian,
            target=_pose_debug_summary(target),
            state=_robot_state_debug_summary(ctx.robot.get_state() or {}, tcp_calibration),
        )
        return
    if position is None:
        raise RuntimeError("pose_missing_position")
    ctx.robot.movel(
        {
            "position_m": position,
            "quat_xyzw": quat,
            "frame": target.get("frame", "base"),
        },
        profile,
    )
    _log(
        ctx,
        run_id,
        "PICK_PLACE_MOVE_TO_POSE_COMPLETE",
        profile=profile,
        prefer_cartesian=prefer_cartesian,
        target=_pose_debug_summary(target),
        state=_robot_state_debug_summary(ctx.robot.get_state() or {}, tcp_calibration),
    )


def _pose_tcp_target(pose: Dict[str, Any]) -> Dict[str, Any]:
    target = pose.get("tcp_pose") or pose.get("tcp") or pose
    if not isinstance(target, dict):
        return {}
    return target


def _vec_distance_m(a: Any, b: Any) -> Optional[float]:
    if not (_is_finite_vec(a, 3) and _is_finite_vec(b, 3)):
        return None
    dx = float(a[0]) - float(b[0])
    dy = float(a[1]) - float(b[1])
    dz = float(a[2]) - float(b[2])
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def _quat_angle_error_deg(a: Any, b: Any) -> Optional[float]:
    if not (_is_finite_vec(a, 4) and _is_finite_vec(b, 4)):
        return None
    qa = _normalize_quat_xyzw([float(a[0]), float(a[1]), float(a[2]), float(a[3])])
    qb = _normalize_quat_xyzw([float(b[0]), float(b[1]), float(b[2]), float(b[3])])
    dot = abs(qa[0] * qb[0] + qa[1] * qb[1] + qa[2] * qb[2] + qa[3] * qb[3])
    dot = max(-1.0, min(1.0, dot))
    return math.degrees(2.0 * math.acos(dot))


def _wait_for_tcp_target(
    ctx: StationContext,
    handle: Any,
    run_id: Optional[str],
    pose: Dict[str, Any],
    *,
    timeout_s: float,
    position_tolerance_m: float,
    orientation_tolerance_deg: float,
    tcp_calibration: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    target = _pose_tcp_target(pose)
    target_position = target.get("position_m")
    target_quat = target.get("quat_xyzw")

    if not _is_finite_vec(target_position, 3):
        return ctx.robot.get_state().get("tcp_pose") or {}

    deadline = time.monotonic() + max(0.0, timeout_s)
    last_tcp: Dict[str, Any] = {}
    last_position_error = None
    last_orientation_error = None
    first_iteration = True
    while True:
        _ensure_stop(handle)
        state = ctx.robot.get_state() or {}
        last_tcp = state.get("tcp_pose") or {}
        flange_pose = state.get("flange_pose") or last_tcp
        current_position = flange_pose.get("position_m")
        current_quat = last_tcp.get("quat_xyzw") or flange_pose.get("quat_xyzw")

        # Compare position in calibrated tool_tcp space, but compare orientation
        # in the robot-command orientation frame. Applying the mount rotation to
        # orientation here recreates the constant 45 degree yaw offset.
        compare_position = current_position
        compare_quat = current_quat
        tcp_calib_used = False
        if isinstance(tcp_calibration, dict) and tcp_calibration.get("has_calibration"):
            flange_quat = flange_pose.get("quat_xyzw")
            if _is_finite_vec(flange_quat, 4) and _is_finite_vec(current_position, 3):
                base_to_flange = {
                    "translation_m": list(current_position),
                    "rotation_quat_xyzw": list(flange_quat),
                }
                base_to_tcp = _apply_tcp_calibration_to_base(base_to_flange, tcp_calibration)
                compare_position = base_to_tcp.get("translation_m")
                tcp_calib_used = True

        last_position_error = _vec_distance_m(compare_position, target_position)
        last_orientation_error = _quat_angle_error_deg(compare_quat, target_quat)
        position_ok = (
            last_position_error is not None
            and last_position_error <= max(0.0, position_tolerance_m)
        )
        orientation_ok = (
            not _is_finite_vec(target_quat, 4)
            or last_orientation_error is None
            or last_orientation_error <= max(0.0, orientation_tolerance_deg)
        )
        if first_iteration:
            first_iteration = False
            _log(
                ctx,
                run_id,
                "PICK_PLACE_WAIT_FOR_TCP_TARGET",
                target=_pose_debug_summary(target),
                state=_robot_state_debug_summary(state, tcp_calibration),
                compare_pose=_pose_debug_summary(
                    {
                        "position_m": compare_position,
                        "quat_xyzw": compare_quat,
                        "frame": "base",
                    }
                ),
                tcp_calib_used=tcp_calib_used,
                tcp_offset_m=(
                    list(tcp_calibration.get("tcp_offset_m"))
                    if isinstance(tcp_calibration, dict) and isinstance(tcp_calibration.get("tcp_offset_m"), list)
                    else None
                ),
                position_error_m=last_position_error,
                orientation_error_deg=last_orientation_error,
                position_tolerance_m=position_tolerance_m,
                orientation_tolerance_deg=orientation_tolerance_deg,
            )
        if position_ok and orientation_ok:
            _log(
                ctx,
                run_id,
                "PICK_PLACE_WAIT_FOR_TCP_TARGET_REACHED",
                target=_pose_debug_summary(target),
                compare_pose=_pose_debug_summary(
                    {
                        "position_m": compare_position,
                        "quat_xyzw": compare_quat,
                        "frame": "base",
                    }
                ),
                state=_robot_state_debug_summary(state, tcp_calibration),
                tcp_calib_used=tcp_calib_used,
                position_error_m=last_position_error,
                orientation_error_deg=last_orientation_error,
            )
            return last_tcp
        if time.monotonic() >= deadline:
            _log(
                ctx,
                run_id,
                "PICK_PLACE_WAIT_FOR_TCP_TARGET_TIMEOUT",
                target=_pose_debug_summary(target),
                compare_pose=_pose_debug_summary(
                    {
                        "position_m": compare_position,
                        "quat_xyzw": compare_quat,
                        "frame": "base",
                    }
                ),
                state=_robot_state_debug_summary(state, tcp_calibration),
                tcp_calib_used=tcp_calib_used,
                position_error_m=last_position_error,
                orientation_error_deg=last_orientation_error,
                position_tolerance_m=position_tolerance_m,
                orientation_tolerance_deg=orientation_tolerance_deg,
            )
            raise RuntimeError(
                "capture_pose_not_reached: "
                f"position_error_m={last_position_error} "
                f"orientation_error_deg={last_orientation_error}"
            )
        time.sleep(0.05)


def _movel_with_log(
    ctx: StationContext,
    run_id: Optional[str],
    cycle: int,
    label: str,
    target: Dict[str, Any],
    profile: str,
    tcp_calibration: Optional[Dict[str, Any]] = None,
) -> None:
    _log(
        ctx,
        run_id,
        "PICK_PLACE_MOVE_BEGIN",
        cycle=cycle,
        stage=label,
        profile=profile,
        target=_pose_debug_summary(target),
        state=_robot_state_debug_summary(ctx.robot.get_state() or {}, tcp_calibration),
        api="movel",
    )
    try:
        ctx.robot.movel(target, profile)
        _log(
            ctx,
            run_id,
            "PICK_PLACE_MOVE_COMPLETE",
            cycle=cycle,
            stage=label,
            profile=profile,
            target=_pose_debug_summary(target),
            state=_robot_state_debug_summary(ctx.robot.get_state() or {}, tcp_calibration),
            api="movel",
        )
    except Exception as exc:
        _log(
            ctx,
            run_id,
            "PICK_PLACE_IK_FAILED",
            cycle=cycle,
            stage=label,
            target=target,
            error=str(exc),
        )
        raise


def _move_tcp_target_with_log(
    ctx: StationContext,
    run_id: Optional[str],
    cycle: int,
    label: str,
    target: Dict[str, Any],
    profile: str,
    *,
    strategy: str = "cartesian",
    pose_index: Optional[Dict[str, Dict[str, Any]]] = None,
    preferred_pose_name: str = "",
    options: Optional[Dict[str, Any]] = None,
    tcp_calibration: Optional[Dict[str, Any]] = None,
) -> None:
    strategy = str(strategy or "cartesian").strip().lower()
    options = options if isinstance(options, dict) else {}
    _log(
        ctx,
        run_id,
        "PICK_PLACE_MOVE_BEGIN",
        cycle=cycle,
        stage=label,
        profile=profile,
        target=_pose_debug_summary(target),
        state=_robot_state_debug_summary(ctx.robot.get_state() or {}, tcp_calibration),
        move_strategy=strategy,
        preferred_pose_name=preferred_pose_name,
        api="move_tcp_ik" if strategy == "ik_nullspace" else "movel",
    )
    try:
        if strategy == "ik_nullspace":
            preferred_joints = None
            preferred_name = str(preferred_pose_name or "").strip()
            if preferred_name:
                preferred_pose = (pose_index or {}).get(preferred_name)
                if isinstance(preferred_pose, dict):
                    raw_joints = preferred_pose.get("joints")
                    if isinstance(raw_joints, list) and len(raw_joints) == 7:
                        preferred_joints = tuple(raw_joints)
            if preferred_name and preferred_joints is None:
                _log(
                    ctx,
                    run_id,
                    "PICK_PLACE_PREFERRED_POSE_SKIPPED",
                    cycle=cycle,
                    stage=label,
                    preferred_pose_name=preferred_name,
                    reason="missing_or_invalid_joints",
                )
            state = ctx.robot.get_state() or {}
            seed_joints = state.get("q") or state.get("joints") or None
            ctx.robot.move_tcp_ik(
                target,
                profile,
                seed_joints=(
                    tuple(seed_joints)
                    if isinstance(seed_joints, list) and len(seed_joints) == 7
                    else None
                ),
                preferred_joints=preferred_joints,
                position_tolerance_m=float(
                    options.get("ik_position_tolerance_m", 0.002)
                ),
                orientation_tolerance_deg=float(
                    options.get("ik_orientation_tolerance_deg", 2.0)
                ),
                approximate_position_tolerance_m=float(
                    options.get("ik_approximate_position_tolerance_m", 0.015)
                ),
                approximate_orientation_tolerance_deg=float(
                    options.get("ik_approximate_orientation_tolerance_deg", 3.0)
                ),
                max_iterations=int(options.get("ik_max_iterations", 300)),
            )
            _log(
                ctx,
                run_id,
                "PICK_PLACE_MOVE_COMPLETE",
                cycle=cycle,
                stage=label,
                profile=profile,
                target=_pose_debug_summary(target),
                state=_robot_state_debug_summary(ctx.robot.get_state() or {}, tcp_calibration),
                move_strategy=strategy,
                preferred_pose_name=preferred_pose_name,
                api="move_tcp_ik",
            )
            return
        _movel_with_log(
            ctx,
            run_id,
            cycle,
            label,
            target,
            profile,
            tcp_calibration,
        )
    except Exception as exc:
        if strategy == "ik_nullspace":
            _log(
                ctx,
                run_id,
                "PICK_PLACE_IK_FAILED",
                cycle=cycle,
                stage=label,
                target=target,
                move_strategy=strategy,
                error=str(exc),
            )
        raise


def _sleep_with_stop(handle: Any, duration_s: float) -> None:
    if duration_s <= 1e-6:
        return
    if handle.stop_event.wait(duration_s):
        raise RuntimeError("run_stopped")


def _resolve_pose_target(
    pose: Dict[str, Any],
    fallback_quat_xyzw: Optional[List[float]] = None,
) -> Dict[str, Any]:
    target = pose.get("tcp_pose") or pose.get("tcp") or pose
    quat = target.get("quat_xyzw")
    if isinstance(quat, list) and len(quat) >= 4:
        quat_xyzw = _normalize_quat_xyzw(
            [float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])]
        )
    else:
        quat_xyzw = _normalize_quat_xyzw(
            list(fallback_quat_xyzw or [0.0, 0.0, 0.0, 1.0])
        )
    return {
        "position_m": target.get("position_m"),
        "quat_xyzw": quat_xyzw,
        "frame": target.get("frame", "base"),
    }


def _execute_intermediate_regrasp(
    ctx: StationContext,
    handle: Any,
    run_id: Optional[str],
    cycle: int,
    default_profile: str,
    gripper_type: str,
    use_vacuum_gripper: bool,
    carry_orientation_xyzw: List[float],
    vacuum_pick_wait_s: float,
    pregrasp_open_width_m: Optional[float],
    grasp_close_width_m: Optional[float],
    plan: Dict[str, Any],
) -> Dict[str, Any]:
    regrasp = (
        plan.get("intermediate_regrasp")
        if isinstance(plan.get("intermediate_regrasp"), dict)
        else {}
    )
    if not regrasp:
        return {"carry_orientation": list(carry_orientation_xyzw)}

    place_pose = regrasp.get("place_pose") or {}
    pick_pose = regrasp.get("pick_pose") or {}
    if _pose_missing(place_pose):
        raise RuntimeError("intermediate_regrasp_place_pose_missing")
    if _pose_missing(pick_pose):
        raise RuntimeError("intermediate_regrasp_pick_pose_missing")

    place_profile = str(regrasp.get("place_profile") or default_profile).strip() or default_profile
    pick_profile = str(regrasp.get("pick_profile") or default_profile).strip() or default_profile
    place_open_width_m = _coerce_float(
        regrasp.get("place_open_width_m"),
        pregrasp_open_width_m,
    )
    pick_open_width_m = _coerce_float(
        regrasp.get("pick_open_width_m"),
        place_open_width_m if place_open_width_m is not None else pregrasp_open_width_m,
    )
    pick_close_width_m = _coerce_float(
        regrasp.get("pick_close_width_m"),
        grasp_close_width_m,
    )
    place_approach_offset = regrasp.get("place_approach_offset_m", [0.0, 0.0, 0.08])
    place_retreat_offset = regrasp.get("place_retreat_offset_m", place_approach_offset)
    pick_approach_offset = regrasp.get("pick_approach_offset_m", [0.0, 0.0, 0.08])
    pick_retreat_offset = regrasp.get("pick_retreat_offset_m", pick_approach_offset)
    settle_time_s = max(0.0, float(regrasp.get("settle_time_s", 0.0) or 0.0))

    _log(
        ctx,
        run_id,
        "PICK_PLACE_INTERMEDIATE_START",
        cycle=cycle,
        strategy_name=plan.get("strategy_name"),
        mode=plan.get("mode"),
        place_pose_name=regrasp.get("place_pose_name"),
        pick_pose_name=regrasp.get("pick_pose_name"),
    )

    place_target = _resolve_pose_target(place_pose, carry_orientation_xyzw)
    place_pos = place_target.get("position_m")
    place_quat = place_target.get("quat_xyzw", list(carry_orientation_xyzw))
    place_frame = place_target.get("frame", "base")
    if place_pos:
        place_approach = _add_vec(place_pos, place_approach_offset)
        _log(
            ctx,
            run_id,
            "PICK_PLACE_INTERMEDIATE_PLACE_APPROACH",
            cycle=cycle,
            target=place_approach,
            pose_name=regrasp.get("place_pose_name"),
        )
        _movel_with_log(
            ctx,
            run_id,
            cycle,
            "intermediate_place_approach",
            {
                "position_m": place_approach,
                "quat_xyzw": place_quat,
                "frame": place_frame,
            },
            place_profile,
            tcp_calibration,
        )
        _ensure_stop(handle)
        _log(
            ctx,
            run_id,
            "PICK_PLACE_INTERMEDIATE_PLACE_MOVE",
            cycle=cycle,
            target=place_pos,
            pose_name=regrasp.get("place_pose_name"),
        )
        _movel_with_log(
            ctx,
            run_id,
            cycle,
            "intermediate_place",
            {
                "position_m": place_pos,
                "quat_xyzw": place_quat,
                "frame": place_frame,
            },
            place_profile,
            tcp_calibration,
        )
    else:
        _log(
            ctx,
            run_id,
            "PICK_PLACE_INTERMEDIATE_PLACE_MOVE",
            cycle=cycle,
            pose_name=regrasp.get("place_pose_name"),
        )
        _move_to_pose(
            ctx,
            run_id,
            place_pose,
            place_profile,
            prefer_cartesian=True,
            tcp_calibration=tcp_calibration,
        )
    _ensure_stop(handle)
    _log(
        ctx,
        run_id,
        "PICK_PLACE_GRIPPER_OFF",
        cycle=cycle,
        stage="intermediate_place",
        gripper_type=gripper_type,
        width_m=place_open_width_m if not use_vacuum_gripper else None,
    )
    if use_vacuum_gripper:
        ctx.robot.open_gripper()
    else:
        ctx.robot.open_gripper(place_open_width_m)
    _ensure_stop(handle)
    _sleep_with_stop(handle, settle_time_s)
    if place_pos:
        place_retreat = _add_vec(place_pos, place_retreat_offset)
        _log(
            ctx,
            run_id,
            "PICK_PLACE_INTERMEDIATE_PLACE_RETREAT",
            cycle=cycle,
            target=place_retreat,
            pose_name=regrasp.get("place_pose_name"),
        )
        _movel_with_log(
            ctx,
            run_id,
            cycle,
            "intermediate_place_retreat",
            {
                "position_m": place_retreat,
                "quat_xyzw": place_quat,
                "frame": place_frame,
            },
            place_profile,
            tcp_calibration,
        )
        _ensure_stop(handle)

    if use_vacuum_gripper:
        ctx.robot.open_gripper()
    else:
        ctx.robot.open_gripper(pick_open_width_m)
    _ensure_stop(handle)

    pick_target = _resolve_pose_target(pick_pose, place_quat)
    pick_pos = pick_target.get("position_m")
    pick_quat = pick_target.get("quat_xyzw", place_quat)
    pick_frame = pick_target.get("frame", "base")
    if pick_pos:
        pick_approach = _add_vec(pick_pos, pick_approach_offset)
        _log(
            ctx,
            run_id,
            "PICK_PLACE_INTERMEDIATE_PICK_APPROACH",
            cycle=cycle,
            target=pick_approach,
            pose_name=regrasp.get("pick_pose_name"),
        )
        _movel_with_log(
            ctx,
            run_id,
            cycle,
            "intermediate_pick_approach",
            {
                "position_m": pick_approach,
                "quat_xyzw": pick_quat,
                "frame": pick_frame,
            },
            pick_profile,
            tcp_calibration,
        )
        _ensure_stop(handle)
        _log(
            ctx,
            run_id,
            "PICK_PLACE_INTERMEDIATE_PICK_MOVE",
            cycle=cycle,
            target=pick_pos,
            pose_name=regrasp.get("pick_pose_name"),
        )
        _movel_with_log(
            ctx,
            run_id,
            cycle,
            "intermediate_pick",
            {
                "position_m": pick_pos,
                "quat_xyzw": pick_quat,
                "frame": pick_frame,
            },
            pick_profile,
            tcp_calibration,
        )
    else:
        _log(
            ctx,
            run_id,
            "PICK_PLACE_INTERMEDIATE_PICK_MOVE",
            cycle=cycle,
            pose_name=regrasp.get("pick_pose_name"),
        )
        _move_to_pose(
            ctx,
            run_id,
            pick_pose,
            pick_profile,
            prefer_cartesian=True,
            tcp_calibration=tcp_calibration,
        )
    _ensure_stop(handle)

    _log(
        ctx,
        run_id,
        "PICK_PLACE_GRIPPER_ON",
        cycle=cycle,
        stage="intermediate_pick",
        gripper_type=gripper_type,
        width_m=pick_close_width_m if not use_vacuum_gripper else None,
    )
    if use_vacuum_gripper:
        ctx.robot.close_gripper()
        _ensure_stop(handle)
        _sleep_with_stop(handle, vacuum_pick_wait_s)
    else:
        ctx.robot.close_gripper(pick_close_width_m)
        _ensure_stop(handle)
    if pick_pos:
        pick_retreat = _add_vec(pick_pos, pick_retreat_offset)
        _log(
            ctx,
            run_id,
            "PICK_PLACE_INTERMEDIATE_PICK_RETREAT",
            cycle=cycle,
            target=pick_retreat,
            pose_name=regrasp.get("pick_pose_name"),
        )
        _movel_with_log(
            ctx,
            run_id,
            cycle,
            "intermediate_pick_retreat",
            {
                "position_m": pick_retreat,
                "quat_xyzw": pick_quat,
                "frame": pick_frame,
            },
            pick_profile,
            tcp_calibration,
        )
        _ensure_stop(handle)

    current_tcp = ctx.robot.get_state().get("tcp_pose") or {}
    carry_orientation = current_tcp.get("quat_xyzw")
    if not (isinstance(carry_orientation, list) and len(carry_orientation) >= 4):
        carry_orientation = pick_quat
    carry_orientation = _normalize_quat_xyzw(
        [
            float(carry_orientation[0]),
            float(carry_orientation[1]),
            float(carry_orientation[2]),
            float(carry_orientation[3]),
        ]
    )
    _log(
        ctx,
        run_id,
        "PICK_PLACE_INTERMEDIATE_DONE",
        cycle=cycle,
        strategy_name=plan.get("strategy_name"),
        carry_orientation=carry_orientation,
    )
    return {"carry_orientation": carry_orientation}


def _ensure_stop(handle: Any) -> None:
    if handle.stop_event.is_set():
        raise RuntimeError("run_stopped")


def _range_ok(value: Optional[float], rng: Optional[List[float]]) -> bool:
    if value is None or rng is None:
        return True
    if not isinstance(rng, list) or len(rng) < 2:
        return True
    lo, hi = rng[0], rng[1]
    try:
        return float(lo) <= float(value) <= float(hi)
    except (TypeError, ValueError):
        return True


def _workspace_ok(position: List[float], workspace: Dict[str, Any]) -> bool:
    if not workspace:
        return True
    min_xyz = workspace.get("min_xyz_m")
    max_xyz = workspace.get("max_xyz_m")
    if isinstance(min_xyz, list) and len(min_xyz) >= 3:
        if (
            position[0] < min_xyz[0]
            or position[1] < min_xyz[1]
            or position[2] < min_xyz[2]
        ):
            return False
    if isinstance(max_xyz, list) and len(max_xyz) >= 3:
        if (
            position[0] > max_xyz[0]
            or position[1] > max_xyz[1]
            or position[2] > max_xyz[2]
        ):
            return False
    return True


def _match_filters(match: Dict[str, Any], filters: Dict[str, Any]) -> Optional[str]:
    if not filters:
        return None
    inliers = match.get("inliers")
    score = match.get("score")
    area = match.get("area_px")
    depth = match.get("depth_m")
    method = match.get("method")

    min_inliers = filters.get("min_inliers")
    if min_inliers is not None and inliers is not None and inliers < int(min_inliers):
        return "min_inliers"
    min_score = filters.get("min_score")
    if method == "edge_template":
        min_score = filters.get("edge_min_score", min_score)
    if min_score is not None and score is not None and float(score) < float(min_score):
        return "min_score"
    min_area = filters.get("min_area_px")
    if min_area is not None and area is not None and float(area) < float(min_area):
        return "min_area_px"
    max_area = filters.get("max_area_px")
    if max_area is not None and area is not None and float(area) > float(max_area):
        return "max_area_px"
    if not _range_ok(depth, filters.get("depth_range_m")):
        return "depth_range"
    return None


def _yaw_delta(a: Optional[float], b: Optional[float]) -> float:
    if a is None or b is None:
        return 0.0
    delta = (a - b + 180.0) % 360.0 - 180.0
    return abs(delta)


def _smooth_match(history: List[Dict[str, Any]]) -> Dict[str, Any]:
    latest = dict(history[-1])
    centers = [m.get("center_xyz_m") for m in history if m.get("center_xyz_m")]
    centers_uv = [m.get("center_uv") for m in history if m.get("center_uv")]
    yaws = [m.get("yaw_deg") for m in history if m.get("yaw_deg") is not None]
    normals = [
        m.get("surface_normal_cam") for m in history if m.get("surface_normal_cam")
    ]
    if centers:
        xs = [c[0] for c in centers]
        ys = [c[1] for c in centers]
        zs = [c[2] for c in centers]
        latest["center_xyz_m"] = [
            float(sorted(xs)[len(xs) // 2]),
            float(sorted(ys)[len(ys) // 2]),
            float(sorted(zs)[len(zs) // 2]),
        ]
    if centers_uv:
        us = [c[0] for c in centers_uv]
        vs = [c[1] for c in centers_uv]
        latest["center_uv"] = [
            float(sorted(us)[len(us) // 2]),
            float(sorted(vs)[len(vs) // 2]),
        ]
    if yaws:
        latest["yaw_deg"] = float(sorted(yaws)[len(yaws) // 2])
    if normals:
        nx = ny = nz = 0.0
        count = 0
        for normal in normals:
            if not (isinstance(normal, list) and len(normal) >= 3):
                continue
            norm = math.sqrt(
                normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]
            )
            if norm < 1e-6:
                continue
            nx += normal[0] / norm
            ny += normal[1] / norm
            nz += normal[2] / norm
            count += 1
        if count > 0:
            norm = math.sqrt(nx * nx + ny * ny + nz * nz)
            if norm >= 1e-6:
                latest["surface_normal_cam"] = [nx / norm, ny / norm, nz / norm]
    return latest


def _stable_match(
    match: Dict[str, Any],
    filters: Dict[str, Any],
    track: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    steady_frames = int(filters.get("steady_frames", 1)) if filters else 1
    if steady_frames <= 1:
        return match
    max_center_delta_px = float(filters.get("max_center_delta_px", 8.0))
    max_depth_delta_m = float(filters.get("max_depth_delta_m", 0.01))
    max_yaw_delta_deg = float(filters.get("max_yaw_delta_deg", 5.0))
    smooth_frames = int(filters.get("smooth_frames", steady_frames))

    center = match.get("center_uv") or [0.0, 0.0]
    depth = match.get("depth_m")
    yaw = match.get("yaw_deg")
    last = track.get("last_match")
    if last:
        last_center = last.get("center_uv") or [0.0, 0.0]
        last_depth = last.get("depth_m")
        last_yaw = last.get("yaw_deg")
        delta_px = abs(center[0] - last_center[0]) + abs(center[1] - last_center[1])
        delta_depth = abs((depth or 0.0) - (last_depth or 0.0))
        delta_yaw = _yaw_delta(yaw, last_yaw)
        if (
            delta_px <= max_center_delta_px
            and delta_depth <= max_depth_delta_m
            and delta_yaw <= max_yaw_delta_deg
        ):
            track["stable_count"] = track.get("stable_count", 0) + 1
        else:
            track["stable_count"] = 1
    else:
        track["stable_count"] = 1
    track["last_match"] = match
    history = track.setdefault("history", [])
    history.append(match)
    if smooth_frames > 0 and len(history) > smooth_frames:
        del history[0 : len(history) - smooth_frames]
    if track["stable_count"] >= steady_frames:
        return _smooth_match(history)
    return None


def _reset_track(track: Dict[str, Any]) -> None:
    track["stable_count"] = 0
    track["last_match"] = None
    track["history"] = []


def _rank_segmentation_candidates(
    matches: List[Dict[str, Any]],
    pick_cfg: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Sort matches so the safest / topmost / most-isolated one is tried first.

    Operates on camera-frame data (no base transform needed). Returns a NEW list
    sorted ascending by rank tuple. Missing signals fall through as neutral.

    Ranking tuple (all ascending, so smaller is better):
      1. -top_score:       larger = higher in bin (= negated camera-frame z) → sorts first.
      2. -isolation_score: larger mean distance to nearest neighbor pts → more isolated first.
      3. -visibility:      larger selection_mask_pixels → more visible first.

    All-poses mode: if pick_cfg["_all_poses_ranker"] is a callable, it receives
    (matches, pick_cfg) and returns a reordered list. This is how bin_picking.py
    injects the "pickable + central" selection when all_poses_mode is on.
    """
    if not isinstance(matches, list) or not matches:
        return matches or []
    cfg = pick_cfg if isinstance(pick_cfg, dict) else {}
    ranker = cfg.get("_all_poses_ranker")
    if callable(ranker) and bool(cfg.get("all_poses_mode", False)):
        try:
            ranked = ranker(matches, cfg)
            if isinstance(ranked, list) and ranked:
                return ranked
        except Exception:
            pass  # Fall through to default ranking on error.
    enabled = bool(cfg.get("segmentation_safety_ranking_enabled", True))
    if not enabled:
        ranked_matches = list(matches)
        candidate_ranker = cfg.get("_segmentation_pickable_ranker")
        if callable(candidate_ranker):
            try:
                candidate_ranked = candidate_ranker(ranked_matches, cfg)
                if isinstance(candidate_ranked, list):
                    return candidate_ranked
            except Exception:
                pass
        return ranked_matches

    def _center_z(m: Dict[str, Any]) -> float:
        # In camera frame, smaller z = closer to camera = typically higher in bin.
        origin = m.get("pose_origin_xyz_m") or m.get("center_xyz_m") or [0.0, 0.0, 0.0]
        try:
            return float(origin[2])
        except (TypeError, IndexError, ValueError):
            return 0.0

    def _isolation_score(m: Dict[str, Any]) -> float:
        safety_pcd = m.get("safety_pcd") if isinstance(m, dict) else None
        if not isinstance(safety_pcd, dict):
            return 0.0
        target = safety_pcd.get("target_points_camera_m") or []
        neighbors = safety_pcd.get("neighbor_points_camera_m") or []
        if not target or not neighbors:
            return 0.0
        # Use target centroid as a cheap isolation proxy; avoid O(N*M) KD-tree.
        cx = sum(float(p[0]) for p in target) / len(target)
        cy = sum(float(p[1]) for p in target) / len(target)
        cz = sum(float(p[2]) for p in target) / len(target)
        min_d_sq = float("inf")
        for p in neighbors:
            try:
                dx = float(p[0]) - cx
                dy = float(p[1]) - cy
                dz = float(p[2]) - cz
                d_sq = dx * dx + dy * dy + dz * dz
                if d_sq < min_d_sq:
                    min_d_sq = d_sq
            except (TypeError, IndexError, ValueError):
                continue
        if not math.isfinite(min_d_sq):
            return 0.0
        return math.sqrt(min_d_sq)

    def _visibility(m: Dict[str, Any]) -> float:
        try:
            return float(m.get("selection_mask_pixels") or 0.0)
        except (TypeError, ValueError):
            return 0.0

    decorated = []
    for idx, m in enumerate(matches):
        key = (
            _center_z(m),              # smaller z first (= higher in bin for top-down cam)
            -_isolation_score(m),      # more isolated first
            -_visibility(m),           # more visible first
            idx,                        # stable tie-break
        )
        decorated.append((key, m))
    decorated.sort(key=lambda kv: kv[0])
    ranked_matches = [m for _, m in decorated]
    candidate_ranker = cfg.get("_segmentation_pickable_ranker")
    if callable(candidate_ranker):
        try:
            candidate_ranked = candidate_ranker(ranked_matches, cfg)
            if isinstance(candidate_ranked, list):
                return candidate_ranked
        except Exception:
            pass
    return ranked_matches


def _wait_for_match(
    ctx: StationContext,
    request_id: str,
    timeout_s: float,
    handle: Any,
    filters: Dict[str, Any],
    track: Dict[str, Any],
    log_debug,
    pick_cfg: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Wait until a stable, filter-compliant 3D match is available.

    The function consumes vision events and only returns when the top candidate:
    - has valid `center_xyz_m`,
    - passes recipe-level filters (`_match_filters`),
    - passes temporal stability gating (`_stable_match`).

    Before iterating, matches are re-ordered via `_rank_segmentation_candidates`
    so the safest / topmost / most-isolated target is tried first.
    """
    deadline = time.time() + timeout_s if timeout_s and timeout_s > 0 else None
    consumed_frame_ids: set[str] = set()
    while True:
        _ensure_stop(handle)
        chunk = 1.0
        if deadline is not None:
            remaining = deadline - time.time()
            if remaining <= 0:
                raise RuntimeError("vision_timeout")
            chunk = min(chunk, remaining)
        evt = ctx.vision_results.wait_for_result(request_id, chunk, handle.stop_event)
        if not evt:
            evt = _try_read_result_shm(request_id)
        if not evt:
            # Only fall back to the exact request cache entry here.
            # Older sibling retry attempts (for example `...-0-3` while waiting on
            # `...-0-4`) can carry terminal bin-picking no-candidate results. If we
            # accept alias/prefix matches, a live inference can be aborted by a stale
            # retry result and immediately retried again.
            cached_evt = ctx.vision_cache.get_latest(request_id)
            if cached_evt:
                cached_frame_id = str(cached_evt.get("frame_id") or "")
                if cached_frame_id and cached_frame_id not in consumed_frame_ids:
                    evt = cached_evt
        if not evt:
            continue
        evt_frame_id = str(evt.get("frame_id") or "")
        if evt_frame_id:
            consumed_frame_ids.add(evt_frame_id)
        result = evt.get("result", {})
        if result.get("terminal") and result.get("error"):
            error = str(result.get("error"))
            detail = str(result.get("error_detail") or "").strip()
            if not detail and isinstance(result.get("details"), dict) and result.get("details"):
                try:
                    detail = json.dumps(
                        result.get("details"),
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                except Exception:
                    detail = str(result.get("details"))
            raise RuntimeError(f"{error}: {detail}" if detail else error)
        matches = result.get("matches") or []
        if result.get("valid") and matches:
            matches = _rank_segmentation_candidates(matches, pick_cfg)
            all_stable: List[Dict[str, Any]] = []
            for match in matches:
                if not match.get("center_xyz_m"):
                    continue
                reason = _match_filters(match, filters)
                if reason:
                    log_debug(
                        "match_reject",
                        reason=reason,
                        match=match,
                        frame_id=evt.get("frame_id"),
                        request_id=evt.get("request_id") or request_id,
                        timestamp_ns=evt.get("timestamp_ns"),
                    )
                    continue
                stable = _stable_match(match, filters, track)
                if stable:
                    all_stable.append(stable)
            if all_stable:
                return {"match": all_stable[0], "event": evt, "all_stable_matches": all_stable}
        # keep waiting until a valid match


def _collect_multi_scan(
    ctx: StationContext,
    request_id: str,
    timeout_s: float,
    handle: Any,
    filters: Dict[str, Any],
    track: Dict[str, Any],
    log_debug,
    base_match: Dict[str, Any],
    base_evt: Dict[str, Any],
    pick_cfg: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Aggregate several stable detections and smooth them into one pick target."""
    max_frames = int(pick_cfg.get("multi_scan_frames", 8))
    min_frames = int(pick_cfg.get("multi_scan_min_frames", 3))
    window_s = float(pick_cfg.get("multi_scan_window_s", 0.8))
    reset_dist_m = float(pick_cfg.get("multi_scan_reset_dist_m", 0.02))
    matches: List[Dict[str, Any]] = [base_match]
    last_evt = base_evt
    deadline = time.time() + window_s if window_s and window_s > 0 else None
    while len(matches) < max_frames:
        if handle.stop_event.is_set():
            break
        chunk = 0.25
        if deadline is not None:
            remaining = deadline - time.time()
            if remaining <= 0:
                break
            chunk = min(chunk, remaining)
        evt = ctx.vision_results.wait_for_result(request_id, chunk, handle.stop_event)
        if not evt:
            continue
        last_evt = evt
        result = evt.get("result", {})
        matches_list = result.get("matches") or []
        if not result.get("valid") or not matches_list:
            continue
        for match in matches_list:
            if not match.get("center_xyz_m"):
                continue
            ok, reason = _match_ok(match, filters)
            if not ok:
                log_debug(
                    "match_reject",
                    reason=reason,
                    match=match,
                    frame_id=evt.get("frame_id"),
                    request_id=evt.get("request_id") or request_id,
                    timestamp_ns=evt.get("timestamp_ns"),
                )
                _reset_track(track)
                continue
            stable = _stable_match(match, filters, track)
            if not stable:
                continue
            if matches:
                prev = matches[-1].get("center_xyz_m")
                curr = stable.get("center_xyz_m")
                if prev and curr:
                    dx = curr[0] - prev[0]
                    dy = curr[1] - prev[1]
                    dz = curr[2] - prev[2]
                    if math.sqrt(dx * dx + dy * dy + dz * dz) > reset_dist_m:
                        matches = [stable]
                        continue
            matches.append(stable)
            break
    if len(matches) < min_frames:
        return base_match, base_evt
    return _smooth_match(matches), last_evt


def _evt_timestamp_s(evt: Dict[str, Any]) -> float:
    if isinstance(evt, dict):
        ts_ns = evt.get("timestamp_ns")
        if ts_ns is not None:
            try:
                return float(ts_ns) / 1e9
            except Exception:
                pass
    return time.time()


def _estimate_linear_velocity(
    samples: List[Tuple[float, List[float]]],
) -> Tuple[List[float], float]:
    """Least-squares velocity estimate from (time_s, position_m) samples."""
    if len(samples) < 2:
        return [0.0, 0.0, 0.0], 0.0
    t0 = float(samples[0][0])
    t_vals = [float(s[0]) - t0 for s in samples]
    px = [float(s[1][0]) for s in samples]
    py = [float(s[1][1]) for s in samples]
    pz = [float(s[1][2]) for s in samples]

    n = float(len(samples))
    sum_t = sum(t_vals)
    sum_tt = sum(t * t for t in t_vals)
    denom = n * sum_tt - sum_t * sum_t

    window_s = max(0.0, float(t_vals[-1] - t_vals[0]))
    if abs(denom) < 1e-9:
        dt = max(1e-6, window_s)
        vx = (px[-1] - px[0]) / dt
        vy = (py[-1] - py[0]) / dt
        vz = (pz[-1] - pz[0]) / dt
        return [vx, vy, vz], window_s

    sum_x = sum(px)
    sum_y = sum(py)
    sum_z = sum(pz)
    sum_tx = sum(t * x for t, x in zip(t_vals, px))
    sum_ty = sum(t * y for t, y in zip(t_vals, py))
    sum_tz = sum(t * z for t, z in zip(t_vals, pz))

    vx = (n * sum_tx - sum_t * sum_x) / denom
    vy = (n * sum_ty - sum_t * sum_y) / denom
    vz = (n * sum_tz - sum_t * sum_z) / denom
    return [vx, vy, vz], window_s


def run_pick_place_core(
    ctx: StationContext,
    state: RunState,
    handle: Any,
    hooks: Optional[PickRuntimeHooks] = None,
) -> None:
    """Execute one or more pick/place cycles.

    This pipeline handles both:
    - classic static pick/place, and
    - pallatizing-style conveyor interception when enabled in task config.
    """
    recipe = _resolve_task_payload(ctx, state)
    run_id = state.run_id
    runtime_hooks = hooks or PickRuntimeHooks()

    vision_cfg = recipe.get("vision", {})
    robot_cfg = recipe.get("robot", {})
    pick_cfg = robot_cfg.get("pick", {})
    timing_log_enabled = _timing_enabled(
        recipe,
        vision_cfg,
        vision_cfg.get("params", {}) if isinstance(vision_cfg, dict) else {},
        robot_cfg,
        pick_cfg,
    )
    timing_log_path = ctx.data_root / "vision" / "runs" / str(run_id) / "bin_picking_timing.jsonl"
    camera_id = vision_cfg.get("camera_id")
    module = vision_cfg.get("module")
    if not camera_id or not module:
        raise RuntimeError("missing_vision_camera_or_module")
    if _robot_disabled(ctx):
        _run_pick_vision_only_session(ctx, state, handle, recipe, hooks=runtime_hooks)
        return

    def _canonical_gripper_type(value: Any) -> str:
        raw = str(value or "").strip().lower()
        if raw in {"vacuum", "suction", "vacume"}:
            return "vacuum"
        return "parallel_jaw"

    gripper_type = _canonical_gripper_type(
        pick_cfg.get("gripper_type") or robot_cfg.get("gripper_type")
    )
    use_vacuum_gripper = gripper_type == "vacuum"
    try:
        vacuum_pick_wait_s = max(
            0.0,
            float(pick_cfg.get("vacuum_pick_wait_s", 0.3 if use_vacuum_gripper else 0.0)),
        )
    except (TypeError, ValueError):
        vacuum_pick_wait_s = 0.3 if use_vacuum_gripper else 0.0

    pall_cfg = (
        recipe.get("pallatizing", {})
        or robot_cfg.get("pallatizing", {})
        or pick_cfg.get("pallatizing", {})
    )
    pallatizing_mode = bool(
        state.task_type in ("pallatizing", "palletizing")
        or (isinstance(pall_cfg, dict) and pall_cfg.get("enabled"))
    )
    # Conveyor prediction parameters are consumed only when pallatizing mode is on.
    velocity_sample_s = float(
        (pall_cfg or {}).get("velocity_sample_s", 1.0 if pallatizing_mode else 0.0)
    )
    velocity_min_samples = max(2, int((pall_cfg or {}).get("velocity_min_samples", 4)))
    velocity_max_samples = max(
        velocity_min_samples, int((pall_cfg or {}).get("velocity_max_samples", 20))
    )
    prediction_horizon_s = max(
        0.0, float((pall_cfg or {}).get("prediction_horizon_s", 0.35))
    )
    max_object_speed_mps = max(
        0.0, float((pall_cfg or {}).get("max_object_speed_mps", 1.5))
    )
    max_prediction_displacement_m = max(
        0.0, float((pall_cfg or {}).get("max_prediction_displacement_m", 0.25))
    )
    conveyor_dynamic_pick_enabled = bool(
        (pall_cfg or {}).get("dynamic_pick_enabled", pallatizing_mode)
    )
    conveyor_velocity_deadband_mps = max(
        0.0, float((pall_cfg or {}).get("velocity_deadband_mps", 0.005))
    )
    conveyor_pre_pick_lead_s = max(
        0.0, float((pall_cfg or {}).get("pre_pick_lead_s", 0.6))
    )
    conveyor_pick_lead_s = max(0.0, float((pall_cfg or {}).get("pick_lead_s", 1.1)))
    conveyor_retreat_lead_s = max(
        0.0, float((pall_cfg or {}).get("retreat_lead_s", 1.3))
    )
    conveyor_pre_pick_profile = str(
        (pall_cfg or {}).get("pre_pick_profile", "")
    ).strip()
    conveyor_pick_profile = str((pall_cfg or {}).get("pick_profile", "")).strip()
    conveyor_retreat_profile = str((pall_cfg or {}).get("retreat_profile", "")).strip()
    fusion_raw = {}
    for candidate in (
        vision_cfg.get("fusion"),
        robot_cfg.get("fusion"),
        pick_cfg.get("fusion"),
        recipe.get("fusion"),
    ):
        if isinstance(candidate, dict) and candidate:
            fusion_raw = candidate
            break
    fusion_explicitly_disabled = any(
        isinstance(cfg, dict)
        and (
            ("use_fusion" in cfg and cfg.get("use_fusion") is False)
            or (
                isinstance(cfg.get("fusion"), dict)
                and cfg["fusion"].get("enabled") is False
            )
        )
        for cfg in (vision_cfg, robot_cfg, pick_cfg, recipe)
    )
    fusion_enabled = bool(
        str(module or "").strip().lower() in _BIN_PICKING_POSE_MODULES
        and not fusion_explicitly_disabled
        and (
            vision_cfg.get("use_fusion", False)
            or robot_cfg.get("use_fusion", False)
            or pick_cfg.get("use_fusion", False)
            or recipe.get("use_fusion", False)
            or fusion_raw.get("enabled", False)
        )
    )
    fusion_virtual_camera_id = str(
        fusion_raw.get("virtual_camera_id") or f"{camera_id}_fused"
    )
    fusion_push_addr = str(fusion_raw.get("push_addr") or "tcp://127.0.0.1:5556")
    fusion_publish_repeats = max(1, int(fusion_raw.get("publish_repeats", 4)))
    fusion_publish_interval_s = max(
        0.0, float(fusion_raw.get("publish_interval_s", 0.03))
    )
    fusion_fallback_to_live = bool(fusion_raw.get("fallback_to_live", True))
    fusion_timeout_s = max(
        30.0,
        float(
            fusion_raw.get(
                "timeout_s",
                max(
                    45.0,
                    float(vision_cfg.get("timeout_s", 15.0) or 15.0) * 3.0,
                ),
            )
        ),
    )
    pose_index: Dict[str, Dict[str, Any]] = {}
    if state.process_id:
        for pose in ctx.poses.list(state.process_id):
            if pose.get("name"):
                pose_index[pose["name"]] = pose
    capture_pose = robot_cfg.get("capture_pose") or {}
    place_pose = robot_cfg.get("place_pose") or {}
    capture_name = robot_cfg.get("capture_pose_name") or capture_pose.get("name") or ""
    place_name = robot_cfg.get("place_pose_name") or place_pose.get("name") or ""
    if capture_name and capture_name in pose_index:
        capture_pose = pose_index[capture_name]
    if place_name and place_name in pose_index:
        place_pose = pose_index[place_name]
    capture_pose = dict(capture_pose)
    capture_pose.setdefault(
        "move_strategy",
        robot_cfg.get("capture_move_strategy")
        or robot_cfg.get("move_strategy")
        or "cartesian",
    )
    capture_pose.setdefault(
        "preferred_pose_name",
        robot_cfg.get("capture_preferred_pose_name")
        or robot_cfg.get("preferred_pose_name")
        or "",
    )
    intermediate_pose, intermediate_pose_name = _resolve_optional_named_pose(
        pose_index,
        robot_cfg.get("intermediate_pose")
        or robot_cfg.get("pick_intermediate_pose")
        or pick_cfg.get("intermediate_pose"),
        robot_cfg.get("intermediate_pose_name")
        or robot_cfg.get("pick_intermediate_pose_name")
        or pick_cfg.get("intermediate_pose_name"),
        error_code="missing_intermediate_pose",
    )
    use_intermediate_grasp_pose = _config_bool(
        robot_cfg.get(
            "use_intermediate_grasp_pose",
            robot_cfg.get("use_intermediate_pose", True),
        ),
        True,
    )
    if not use_intermediate_grasp_pose:
        intermediate_pose = None
        intermediate_pose_name = ""
    if intermediate_pose is not None:
        intermediate_pose.setdefault(
            "move_strategy",
            robot_cfg.get("intermediate_move_strategy")
            or pick_cfg.get("intermediate_move_strategy")
            or robot_cfg.get("capture_move_strategy")
            or robot_cfg.get("move_strategy")
            or "cartesian",
        )
        intermediate_pose.setdefault(
            "preferred_pose_name",
            robot_cfg.get("intermediate_preferred_pose_name")
            or pick_cfg.get("intermediate_preferred_pose_name")
            or robot_cfg.get("capture_preferred_pose_name")
            or robot_cfg.get("preferred_pose_name")
            or "",
        )
    if _pose_missing(capture_pose) or _pose_missing(place_pose):
        raise RuntimeError("missing_capture_or_place_pose")

    hand_eye_raw = robot_cfg.get("hand_eye") or vision_cfg.get("hand_eye") or pick_cfg.get("hand_eye") or {}
    tcp_calibration, tcp_source = _resolve_runtime_tcp_calibration(ctx, state.process_id, hand_eye_raw)
    _log(ctx, run_id, "DEBUG_TCP_CALIBRATION", tcp_source=tcp_source, has_calibration=tcp_calibration.get("has_calibration"), tcp_offset=tcp_calibration.get("tcp_offset_m"))

    default_profile = robot_cfg.get("default_profile", "normal")
    repeat = bool(robot_cfg.get("repeat", True))
    max_cycles = int(robot_cfg.get("max_cycles", 0))
    cycle = 0
    debug = bool(
        recipe.get("debug") or vision_cfg.get("debug") or robot_cfg.get("debug")
    )
    filters = dict(vision_cfg.get("filters") or {})
    params = vision_cfg.get("params", {})
    if "min_inliers" not in filters and "min_inliers" in params:
        filters["min_inliers"] = params["min_inliers"]
    if "depth_range_m" not in filters:
        depth_min = params.get("depth_min_m")
        depth_max = params.get("depth_max_m")
        if depth_min is not None or depth_max is not None:
            filters["depth_range_m"] = [depth_min or 0.0, depth_max or 10.0]
    filters.setdefault("steady_frames", 2)
    filters.setdefault("smooth_frames", filters.get("steady_frames", 2))
    filters.setdefault("max_center_delta_px", 10.0)
    filters.setdefault("max_depth_delta_m", 0.02)
    filters.setdefault("max_yaw_delta_deg", 10.0)
    # Default to faster acquisition for pick/place so the robot does not wait
    # too long at capture pose when the object is already visible.
    fast_acquire = bool(pick_cfg.get("fast_acquire", True))
    if fast_acquire:
        filters["steady_frames"] = 1
        filters["smooth_frames"] = 1
    raw_max_attempts = int(robot_cfg.get("max_pick_attempts", 0) or 0)
    unlimited_attempts = raw_max_attempts <= 0
    max_attempts = max(1, raw_max_attempts)
    retry_on_ik = bool(robot_cfg.get("retry_on_ik", unlimited_attempts or max_attempts > 1))
    no_candidate_retry_delay_s = max(
        0.0, float(robot_cfg.get("no_candidate_retry_delay_s", 0.5))
    )
    capture_settle_s = max(
        0.0,
        float(
            robot_cfg.get(
                "capture_settle_s",
                pick_cfg.get(
                    "capture_settle_s",
                    vision_cfg.get("capture_settle_s", 0.3),
                ),
            )
        ),
    )
    capture_arrival_timeout_s = max(
        0.0,
        float(
            robot_cfg.get(
                "capture_arrival_timeout_s",
                pick_cfg.get("capture_arrival_timeout_s", 10.0),
            )
            or 10.0
        ),
    )
    capture_position_tolerance_m = max(
        0.0,
        float(
            robot_cfg.get(
                "capture_position_tolerance_m",
                pick_cfg.get("capture_position_tolerance_m", 0.004),
            )
            or 0.004
        ),
    )
    capture_orientation_tolerance_deg = max(
        0.0,
        float(
            robot_cfg.get(
                "capture_orientation_tolerance_deg",
                pick_cfg.get("capture_orientation_tolerance_deg", 5.0),
            )
            or 5.0
        ),
    )
    stay_at_capture_on_no_candidate = bool(
        robot_cfg.get("stay_at_capture_on_no_candidate", True)
    )
    workspace = robot_cfg.get("workspace") or {}
    retry_errors = {
        str(value).strip()
        for value in (
            list(robot_cfg.get("retry_errors") or [])
            + list(vision_cfg.get("retry_errors") or [])
        )
        if str(value).strip()
    }

    def _log_debug(stage: str, **fields: Any) -> None:
        if not debug:
            return
        payload = {"stage": stage}
        payload.update(fields)
        _log(ctx, run_id, "PICK_PLACE_DEBUG", **payload)

    def _log_fusion(stage: str, **fields: Any) -> None:
        payload = {"stage": stage}
        payload.update(fields)
        _log(ctx, run_id, "PICK_PLACE_FUSION", **payload)
        if debug:
            _log(ctx, run_id, "PICK_PLACE_DEBUG", **payload)

    def _should_retry(exc: Exception) -> bool:
        msg = str(exc)
        code = msg.split(":", 1)[0].strip()
        if msg == "run_stopped":
            return False
        if "ik_failed" in msg:
            return retry_on_ik
        if code in ("missing_3d_target", "vision_timeout"):
            return True
        if code in retry_errors:
            return True
        return False

    while True:
        # Cycle boundary: move to capture, run vision, then execute pick/place once.
        _log(ctx, run_id, "PICK_PLACE_CAPTURE_MOVE", cycle=cycle)
        _move_to_pose(
            ctx,
            run_id,
            capture_pose,
            default_profile,
            prefer_cartesian=True,
            pose_index=pose_index,
            tcp_calibration=tcp_calibration,
        )
        _ensure_stop(handle)
        capture_tcp_pose = _wait_for_tcp_target(
            ctx,
            handle,
            run_id,
            capture_pose,
            timeout_s=capture_arrival_timeout_s,
            position_tolerance_m=capture_position_tolerance_m,
            orientation_tolerance_deg=capture_orientation_tolerance_deg,
            tcp_calibration=tcp_calibration,
        )
        _log(
            ctx,
            run_id,
            "PICK_PLACE_CAPTURE_REACHED",
            cycle=cycle,
            tcp_pose=capture_tcp_pose,
            position_tolerance_m=capture_position_tolerance_m,
            orientation_tolerance_deg=capture_orientation_tolerance_deg,
        )
        _sleep_with_stop(handle, capture_settle_s)

        module_params = dict(vision_cfg.get("params", {}))
        if timing_log_enabled:
            module_params["timing_log_enabled"] = True
            module_params["timing_log_path"] = str(timing_log_path)
            module_params["timing_run_id"] = str(run_id)
        if vision_cfg.get("object_id") and "object_id" not in module_params:
            module_params["object_id"] = vision_cfg.get("object_id")
        if module in _TEMPLATE_BASED_MODULES:
            templates_dir = module_params.get("templates_dir") or vision_cfg.get(
                "templates_dir"
            )
            if (
                templates_dir
                or vision_cfg.get("object_id")
                or module_params.get("object_id")
            ):
                module_params["templates_dir"] = _resolve_templates_dir(
                    ctx,
                    templates_dir,
                    state.process_id,
                    vision_cfg.get("object_id") or module_params.get("object_id"),
                )
            if "templates_dir" not in module_params:
                if state.process_id:
                    try:
                        objects = ctx.objects.list(state.process_id)
                    except Exception:
                        objects = []
                    if objects:
                        fallback_obj = objects[0].get("object_id")
                        if fallback_obj:
                            module_params["templates_dir"] = _resolve_templates_dir(
                                ctx, None, state.process_id, fallback_obj
                            )
            if "templates_dir" not in module_params:
                raise RuntimeError("missing_templates_dir")
        # For new pick/pallatizing tasks created from UI, module params can be sparse.
        # Apply robust defaults only when keys are missing so explicit user tuning still wins.
        if module in ("tamplate_matching_sift", "opt_sift"):
            module_params.setdefault("scene_scale", 0.6)
            module_params.setdefault("nfeatures", 550)
            module_params.setdefault("n_octave_layers", 2)
            module_params.setdefault("contrast_threshold", 0.06)
            module_params.setdefault("match_ratio", 0.6)
            module_params.setdefault("min_score", 0.55)
            module_params.setdefault("min_inliers", 10)
            module_params.setdefault("min_good_matches", 7)
            module_params.setdefault("max_good_matches", 50)
            module_params.setdefault("ransac_thresh", 1.5)
            module_params.setdefault("min_inlier_ratio", 0.28)
            module_params.setdefault("max_reproj_rmse_px", 7.0)
            module_params.setdefault("min_projected_area_px", 1800)
            module_params.setdefault("projected_bounds_margin_px", 5)
            module_params.setdefault("depth_window_px", 3)
            module_params.setdefault("compute_surface_normal", False)
            module_params.setdefault("enable_geometric_filters", False)
            module_params.setdefault("enable_temporal_filter", False)
            module_params.setdefault("enable_pose_smoothing", False)
            module_params.setdefault("enable_corner_smoothing", False)
            module_params.setdefault("hold_last_valid_frames", 0)
            module_params.setdefault("allow_partial_visibility", True)
            module_params.setdefault("min_visible_corners", 2)
            module_params.setdefault("min_visible_area_ratio", 0.15)
            module_params.setdefault("clip_bbox_to_image", True)
            module_params.setdefault("include_image", False)
            module_params.setdefault("image_on_match", False)

        runtime_state = None
        if callable(runtime_hooks.load_runtime_state):
            runtime_state = runtime_hooks.load_runtime_state(
                ctx,
                state,
                recipe,
                vision_cfg,
                robot_cfg,
                pick_cfg,
                module_params,
                cycle,
                _log_debug,
            )

        enable_shm_output = vision_cfg.get("enable_shm_output")
        if enable_shm_output is None:
            enable_shm_output = not bool(module_params.get("include_image", False))
        min_frame_timestamp_ns = time.time_ns()
        payload_base = {
            "event": "VISION_START",
            "camera_id": fusion_virtual_camera_id if fusion_enabled else camera_id,
            "module": module,
            "fps_limit": vision_cfg.get("fps_limit", 0),
            "process_mode": vision_cfg.get("process_mode", "continuous"),
            "params": module_params,
            "enable_shm_output": bool(enable_shm_output),
            "min_frame_timestamp_ns": min_frame_timestamp_ns,
        }
        station_calibration = build_station_camera_calibration(
            ctx.data_paths,
            state.station_id,
        )
        if station_calibration:
            payload_base["calibration"] = station_calibration
        shm_max_result_size = vision_cfg.get("shm_max_result_size")
        if shm_max_result_size:
            payload_base["shm_max_result_size"] = int(shm_max_result_size)

        timeout_s = _vision_timeout_s(vision_cfg, module)
        attempt = 0
        success = False
        while unlimited_attempts or attempt < max_attempts:
            attempt += 1
            request_id = (
                f"{state.run_id}-vision-{cycle}-{attempt}"
                if unlimited_attempts or max_attempts > 1
                else f"{state.run_id}-vision-{cycle}"
            )
            handle.vision_request_id = request_id
            handle.last_vision_request_id = request_id
            ctx.run_manager.set_last_vision(run_id, request_id=request_id)
            ctx.run_manager.set_phase(
                run_id,
                "pose_estimation",
                {"cycle": cycle, "attempt": attempt, "request_id": request_id},
            )
            iter_timing = _PickIterationTiming(
                enabled=timing_log_enabled,
                path=timing_log_path,
                run_id=str(run_id),
                request_id=request_id,
                cycle=cycle,
                attempt=attempt,
            )
            track = {"stable_count": 0, "history": []}
            vision_started = False
            fusion_result = None
            payload = dict(payload_base)
            payload_params = dict(module_params)
            if isinstance(runtime_state, dict):
                excluded_ranks = [
                    int(rank)
                    for rank in (runtime_state.get("rejected_detection_ranks") or [])
                    if str(rank).strip()
                ]
                if excluded_ranks:
                    payload_params["excluded_detection_ranks"] = excluded_ranks
            payload["params"] = payload_params
            payload["request_id"] = request_id
            try:
                _log(
                    ctx,
                    run_id,
                    "PICK_PLACE_VISION_START",
                    request_id=request_id,
                    cycle=cycle,
                    attempt=attempt,
                )
                _log(
                    ctx,
                    run_id,
                    "VISION_SESSION_START",
                    request_id=request_id,
                    cycle=cycle,
                    attempt=attempt,
                )
                if fusion_enabled:
                    try:
                        hand_eye_raw, _pref, _used = _resolve_runtime_handeye(
                            ctx, state.process_id, robot_cfg
                        )
                        hand_eye = _resolve_hand_eye(hand_eye_raw)
                        hand_eye_frame = (
                            str(hand_eye.get("hand_eye_frame") or "camera_in_gripper")
                            .strip()
                            .lower()
                        )
                        T_grip_cam = _hand_eye_to_matrix(hand_eye, hand_eye_frame)
                        _log_fusion(
                            "fusion_start",
                            cycle=cycle,
                            attempt=attempt,
                            source_camera_id=camera_id,
                            virtual_camera_id=fusion_virtual_camera_id,
                            publish_repeats=fusion_publish_repeats,
                        )
                        fusion_result = _request_camera_core_fusion(
                            ctx,
                            {
                                "source_camera_id": camera_id,
                                "virtual_camera_id": fusion_virtual_camera_id,
                                "capture_pose": capture_pose,
                                "t_grip_cam": T_grip_cam,
                                "camera_data": _build_fused_camera_data(
                                    station_calibration,
                                    None,
                                ),
                                "calib_version": 1,
                                "publish_repeats": fusion_publish_repeats,
                                "publish_interval_s": fusion_publish_interval_s,
                                "settle_frames": max(
                                    0, int(fusion_raw.get("settle_frames", 2))
                                ),
                                "recv_timeout_ms": max(
                                    100,
                                    int(fusion_raw.get("recv_timeout_ms", 5000)),
                                ),
                                "bus_addr": str(
                                    fusion_raw.get("bus_addr") or "tcp://127.0.0.1:5555"
                                ),
                                "topic": str(fusion_raw.get("topic") or "camera"),
                                "capture": fusion_raw.get("capture")
                                if isinstance(fusion_raw.get("capture"), dict)
                                else {},
                                "pipeline": {
                                    **(
                                        fusion_raw.get("pipeline")
                                        if isinstance(fusion_raw.get("pipeline"), dict)
                                        else {}
                                    ),
                                    "out_dir": _fusion_output_dir(
                                        ctx,
                                        run_id,
                                        cycle,
                                        attempt,
                                    ),
                                },
                            },
                            timeout_s=fusion_timeout_s,
                        )
                        payload["camera_id"] = fusion_virtual_camera_id
                        _log_fusion(
                            "fusion_ready",
                            cycle=cycle,
                            attempt=attempt,
                            virtual_camera_id=fusion_virtual_camera_id,
                            depth_valid_px=int(fusion_result.get("depth_valid_px", 0) or 0),
                            captures=int(fusion_result.get("captures", 0) or 0),
                        )
                    except Exception as exc:
                        _log_fusion(
                            "fusion_failed",
                            cycle=cycle,
                            attempt=attempt,
                            error=str(exc),
                        )
                        print(
                            "[pick_runtime] fusion_failed "
                            f"run_id={run_id} cycle={cycle} attempt={attempt} "
                            f"request_id={request_id} camera_id={camera_id} "
                            f"virtual_camera_id={fusion_virtual_camera_id} error={exc}"
                        )
                        if not fusion_fallback_to_live:
                            raise
                        payload["camera_id"] = camera_id
                        _log_fusion(
                            "fusion_fallback_live",
                            cycle=cycle,
                            attempt=attempt,
                            request_id=request_id,
                            camera_id=camera_id,
                        )
                        print(
                            "[pick_runtime] fusion_fallback_live "
                            f"run_id={run_id} cycle={cycle} attempt={attempt} "
                            f"request_id={request_id} camera_id={camera_id}"
                        )
                vision_start_started_at = time.perf_counter()
                ctx.vision.start_session(payload)
                vision_started = True
                iter_timing.span(
                    "vision_session_start_request",
                    vision_start_started_at,
                    module=module,
                    camera_id=payload.get("camera_id"),
                )

                _log(ctx, run_id, "PICK_PLACE_LOOKING", cycle=cycle, attempt=attempt)

                match = None
                match_evt = None
                target_base = None
                object_center_base = None
                runtime_target_data = None

                while True:
                    wait_match_started_at = time.perf_counter()
                    match_bundle = _wait_for_match(
                        ctx,
                        request_id,
                        timeout_s,
                        handle,
                        filters,
                        track,
                        _log_debug,
                        pick_cfg=pick_cfg,
                    )
                    iter_timing.span(
                        "vision_wait_for_match",
                        wait_match_started_at,
                        frame_id=(match_bundle.get("event", {}) or {}).get("frame_id"),
                    )
                    match = match_bundle.get("match", {})
                    match_evt = match_bundle.get("event", {})
                    all_stable_matches = match_bundle.get("all_stable_matches") or [match]
                    validate_started_at = time.perf_counter()
                    ok, reason = _match_ok(match, filters)
                    if not ok:
                        iter_timing.span(
                            "vision_match_validation",
                            validate_started_at,
                            accepted=False,
                            reason=reason,
                        )
                        _log_debug(
                            "match_reject",
                            cycle=cycle,
                            attempt=attempt,
                            reason=reason,
                            match=match,
                            frame_id=match_evt.get("frame_id"),
                            request_id=match_evt.get("request_id") or request_id,
                            timestamp_ns=match_evt.get("timestamp_ns"),
                        )
                        _reset_track(track)
                        continue
                    iter_timing.span(
                        "vision_match_validation",
                        validate_started_at,
                        accepted=True,
                    )
                    target_cam = match.get("center_xyz_m")
                    if not target_cam:
                        raise RuntimeError("missing_3d_target")

                    pose_transform_started_at = time.perf_counter()
                    hand_eye_raw, hand_eye_source_pref, hand_eye_source_used = (
                        _resolve_runtime_handeye(ctx, state.process_id, robot_cfg)
                    )
                    hand_eye = _resolve_hand_eye(hand_eye_raw)
                    hand_eye_frame = (
                        str(hand_eye.get("hand_eye_frame") or "gripper_to_camera")
                        .strip()
                        .lower()
                    )

                    robot_state_now = ctx.robot.get_state() or {}
                    tcp_pose = (
                        robot_state_now.get("flange_pose")
                        if isinstance(robot_state_now.get("flange_pose"), dict)
                        else robot_state_now.get("tcp_pose")
                    ) or {}
                    base_to_ee = {
                        "translation_m": tcp_pose.get("position_m", [0.0, 0.0, 0.0]),
                        "rotation_quat_xyzw": tcp_pose.get(
                            "quat_xyzw", [0.0, 0.0, 0.0, 1.0]
                        ),
                    }
                    tcp_calibration, tcp_calibration_source = (
                        _resolve_runtime_tcp_calibration(
                            ctx,
                            state.process_id,
                            hand_eye_raw,
                        )
                    )
                    base_to_tcp = _apply_tcp_calibration_to_base(
                        base_to_ee,
                        tcp_calibration,
                    )

                    if hand_eye_frame in ("base_to_camera", "base"):
                        base_to_cam = hand_eye
                    elif (
                        hand_eye_frame in ("camera_in_tcp", "tcp_to_camera", "tool_tcp_to_camera")
                        or str(hand_eye.get("parent_frame") or "").strip().lower() == "tool_tcp"
                    ):
                        base_to_cam = _compose_transform(base_to_tcp, hand_eye)
                    else:
                        base_to_cam = _compose_transform(base_to_ee, hand_eye)
                    object_center_base = _transform_point(target_cam, base_to_cam)
                    target_base = list(object_center_base)
                    target_base_raw = list(object_center_base)
                    velocity_samples: List[Tuple[float, List[float]]] = []
                    velocity_base_mps = [0.0, 0.0, 0.0]
                    if (
                        pallatizing_mode
                        and velocity_sample_s > 1e-3
                        and _is_finite_vec(target_base, 3)
                    ):
                        velocity_samples.append(
                            (
                                _evt_timestamp_s(match_evt),
                                [target_base[0], target_base[1], target_base[2]],
                            )
                        )
                        sample_deadline = time.time() + velocity_sample_s
                        while len(velocity_samples) < velocity_max_samples:
                            _ensure_stop(handle)
                            remaining = sample_deadline - time.time()
                            if remaining <= 0:
                                break
                            evt_next = ctx.vision_results.wait_for_result(
                                request_id,
                                min(0.2, remaining),
                                handle.stop_event,
                            )
                            if not evt_next:
                                continue
                            result_next = evt_next.get("result", {})
                            matches_next = result_next.get("matches") or []
                            if not result_next.get("valid") or not matches_next:
                                continue
                            accepted_next: Optional[Dict[str, Any]] = None
                            for cand in matches_next:
                                if not cand.get("center_xyz_m"):
                                    continue
                                ok_next, _ = _match_ok(cand, filters)
                                if not ok_next:
                                    continue
                                stable_next = _stable_match(cand, filters, track)
                                if stable_next:
                                    accepted_next = stable_next
                                    break
                            if not accepted_next:
                                continue
                            target_cam_next = accepted_next.get("center_xyz_m")
                            if not _is_finite_vec(target_cam_next, 3):
                                continue
                            target_base_next = _transform_point(
                                target_cam_next, base_to_cam
                            )
                            if not _is_finite_vec(target_base_next, 3):
                                continue
                            target_base = [
                                float(target_base_next[0]),
                                float(target_base_next[1]),
                                float(target_base_next[2]),
                            ]
                            match = accepted_next
                            match_evt = evt_next
                            velocity_samples.append(
                                (
                                    _evt_timestamp_s(evt_next),
                                    [target_base[0], target_base[1], target_base[2]],
                                )
                            )
                            if len(velocity_samples) >= velocity_max_samples:
                                break

                        if len(velocity_samples) >= velocity_min_samples:
                            velocity_base_mps, sample_window_s = (
                                _estimate_linear_velocity(velocity_samples)
                            )
                            speed_mps = math.sqrt(
                                velocity_base_mps[0] * velocity_base_mps[0]
                                + velocity_base_mps[1] * velocity_base_mps[1]
                                + velocity_base_mps[2] * velocity_base_mps[2]
                            )
                            if (
                                max_object_speed_mps > 0
                                and speed_mps > max_object_speed_mps
                                and speed_mps > 1e-9
                            ):
                                s = max_object_speed_mps / speed_mps
                                velocity_base_mps = [
                                    velocity_base_mps[0] * s,
                                    velocity_base_mps[1] * s,
                                    velocity_base_mps[2] * s,
                                ]
                                speed_mps = max_object_speed_mps
                            prediction_displacement = [
                                velocity_base_mps[0] * prediction_horizon_s,
                                velocity_base_mps[1] * prediction_horizon_s,
                                velocity_base_mps[2] * prediction_horizon_s,
                            ]
                            disp_norm = math.sqrt(
                                prediction_displacement[0] * prediction_displacement[0]
                                + prediction_displacement[1]
                                * prediction_displacement[1]
                                + prediction_displacement[2]
                                * prediction_displacement[2]
                            )
                            if (
                                max_prediction_displacement_m > 0
                                and disp_norm > max_prediction_displacement_m
                                and disp_norm > 1e-9
                            ):
                                s = max_prediction_displacement_m / disp_norm
                                prediction_displacement = [
                                    prediction_displacement[0] * s,
                                    prediction_displacement[1] * s,
                                prediction_displacement[2] * s,
                                ]
                            target_base = _add_vec(target_base, prediction_displacement)
                            _log_debug(
                                "pallatizing_velocity",
                                cycle=cycle,
                                attempt=attempt,
                                sample_count=len(velocity_samples),
                                sample_window_s=sample_window_s,
                                velocity_base_mps=velocity_base_mps,
                                speed_mps=speed_mps,
                                prediction_horizon_s=prediction_horizon_s,
                                prediction_displacement_m=prediction_displacement,
                                target_base_raw=target_base_raw,
                                target_base_predicted=target_base,
                            )
                    # Try each ranked candidate from this frame for grasp selection.
                    # On no-grasp for candidate[i], fall back to candidate[i+1].
                    # Only raise parallel_jaw_no_grasp_candidate when all are exhausted.
                    # For candidate[0] (the primary match) preserve the pallatizing-
                    # adjusted target_base computed above; recompute fresh for others.
                    _primary_center = list(target_base)
                    _candidate_accepted = False
                    if (
                        callable(runtime_hooks.resolve_target_override)
                        and str(module or "").strip().lower() in _BIN_PICKING_POSE_MODULES
                    ):
                        ctx.run_manager.set_phase(
                            run_id,
                            "grasp_planning",
                            {
                                "cycle": cycle,
                                "attempt": attempt,
                                "request_id": match_evt.get("request_id") or request_id,
                                "frame_id": match_evt.get("frame_id"),
                                "candidate_count": len(all_stable_matches),
                            },
                        )
                    for _cand_idx, _cand_match in enumerate(all_stable_matches):
                        _cand_cam = _cand_match.get("center_xyz_m")
                        if not _cand_cam:
                            continue
                        if _cand_idx == 0:
                            _cand_center = _primary_center
                            _cand_target = _primary_center
                        else:
                            _cand_center = _transform_point(_cand_cam, base_to_cam)
                            _cand_target = list(_cand_center)
                        _cand_runtime_data = None
                        if callable(runtime_hooks.resolve_target_override):
                            _cand_override = runtime_hooks.resolve_target_override(
                                ctx,
                                state,
                                recipe,
                                vision_cfg,
                                robot_cfg,
                                pick_cfg,
                                module_params,
                                runtime_state,
                                cycle,
                                attempt,
                                _cand_match,
                                base_to_cam,
                                base_to_ee,
                                _cand_center,
                                _cand_target,
                                _log_debug,
                            )
                            if isinstance(_cand_override, dict):
                                if _cand_override.get("_no_grasp_candidate"):
                                    _log_debug(
                                        "grasp_fallback_skip",
                                        cycle=cycle,
                                        attempt=attempt,
                                        frame_id=match_evt.get("frame_id"),
                                        request_id=match_evt.get("request_id") or request_id,
                                        match=_cand_match,
                                        reason="no_grasp_candidate",
                                    )
                                    continue
                                _ov_target = _cand_override.get("target_base")
                                _ov_center = _cand_override.get("object_center_base")
                                _cand_runtime_data = _cand_override.get("runtime_target_data")
                                _ov_match = _cand_override.get("match")
                                if _is_finite_vec(_ov_center, 3):
                                    _cand_center = list(_ov_center)
                                if _is_finite_vec(_ov_target, 3):
                                    _cand_target = list(_ov_target)
                                if isinstance(_ov_match, dict):
                                    _cand_match = _ov_match
                        if not _workspace_ok(_cand_target, workspace):
                            _log_debug(
                                "workspace_reject",
                                cycle=cycle,
                                attempt=attempt,
                                target_base=_cand_target,
                                workspace=workspace,
                            )
                            continue
                        # Candidate accepted — promote to the active match.
                        match = _cand_match
                        object_center_base = _cand_center
                        target_base = _cand_target
                        runtime_target_data = _cand_runtime_data
                        _candidate_accepted = True
                        break

                    if not _candidate_accepted:
                        iter_timing.span(
                            "target_transform_and_workspace",
                            pose_transform_started_at,
                            workspace_ok=False,
                        )
                        # Emit a non-debug event so the vision monitor always shows
                        # the estimated frame even when no valid grasp was found.
                        _log(
                            ctx,
                            run_id,
                            "PICK_PLACE_VISION_REJECTED",
                            frame_id=match_evt.get("frame_id"),
                            request_id=match_evt.get("request_id") or request_id,
                            vision_timestamp_ns=match_evt.get("timestamp_ns"),
                            cycle=cycle,
                            attempt=attempt,
                            reject_reason="no_grasp_candidate",
                            match=match,
                        )
                        _log_debug(
                            "vision_match",
                            cycle=cycle,
                            attempt=attempt,
                            frame_id=match_evt.get("frame_id"),
                            timestamp_ns=match_evt.get("timestamp_ns"),
                            match=match,
                            intrinsics=vision_cfg.get("params", {}).get("intrinsics"),
                            depth_window_px=vision_cfg.get("params", {}).get(
                                "depth_window_px"
                            ),
                            rejected=True,
                            reject_reason="no_grasp_candidate",
                        )
                        ctx.run_manager.set_last_vision(
                            run_id,
                            request_id=match_evt.get("request_id") or request_id,
                            frame_id=match_evt.get("frame_id"),
                            timestamp_ns=match_evt.get("timestamp_ns"),
                        )
                        _reset_track(track)
                        raise RuntimeError("parallel_jaw_no_grasp_candidate")

                    iter_timing.span(
                        "target_transform_and_workspace",
                        pose_transform_started_at,
                        workspace_ok=True,
                    )
                    if isinstance(match, dict):
                        match = dict(match)
                        match["center_base_m"] = list(object_center_base)
                        match["object_base_m"] = list(object_center_base)
                        match["target_base_m"] = list(target_base)
                    cam_origin = base_to_cam.get("translation_m", [0.0, 0.0, 0.0])
                    cam_x_axis = _quat_rotate(
                        base_to_cam.get("rotation_quat_xyzw", [0.0, 0.0, 0.0, 1.0]),
                        [1.0, 0.0, 0.0],
                    )
                    cam_y_axis = _quat_rotate(
                        base_to_cam.get("rotation_quat_xyzw", [0.0, 0.0, 0.0, 1.0]),
                        [0.0, 1.0, 0.0],
                    )
                    cam_z_axis = _quat_rotate(
                        base_to_cam.get("rotation_quat_xyzw", [0.0, 0.0, 0.0, 1.0]),
                        [0.0, 0.0, 1.0],
                    )
                    cam_to_target = [
                        target_base[0] - cam_origin[0],
                        target_base[1] - cam_origin[1],
                        target_base[2] - cam_origin[2],
                    ]
                    _log_debug(
                        "vision_match",
                        cycle=cycle,
                        attempt=attempt,
                        frame_id=match_evt.get("frame_id"),
                        timestamp_ns=match_evt.get("timestamp_ns"),
                        match=match,
                        intrinsics=vision_cfg.get("params", {}).get("intrinsics"),
                        depth_window_px=vision_cfg.get("params", {}).get(
                            "depth_window_px"
                        ),
                    )
                    _log_debug(
                        "transform",
                        cycle=cycle,
                        attempt=attempt,
                        target_cam=target_cam,
                        hand_eye_frame=hand_eye_frame,
                        base_to_ee=base_to_ee,
                        base_to_tcp=base_to_tcp,
                        hand_eye=hand_eye,
                        hand_eye_source_pref=hand_eye_source_pref,
                        hand_eye_source_used=hand_eye_source_used,
                        tcp_calibration=tcp_calibration,
                        tcp_calibration_source=tcp_calibration_source,
                        base_to_cam=base_to_cam,
                        object_center_base=object_center_base,
                        target_base=target_base,
                        target_base_raw=target_base_raw,
                        pallatizing_mode=pallatizing_mode,
                        conveyor_velocity_base_mps=velocity_base_mps,
                        prediction_horizon_s=(
                            prediction_horizon_s if pallatizing_mode else 0.0
                        ),
                        pick_contact=(
                            match.get("pick_contact")
                            if isinstance(match, dict)
                            else None
                        ),
                        cam_origin=cam_origin,
                        cam_axes_base={
                            "x": cam_x_axis,
                            "y": cam_y_axis,
                            "z": cam_z_axis,
                        },
                        cam_to_target=cam_to_target,
                    )
                    break

                match = _attach_fusion_visualization(match, fusion_result)
                _log(
                    ctx,
                    run_id,
                    "PICK_PLACE_MATCH",
                    match=match,
                    frame_id=match_evt.get("frame_id"),
                    request_id=match_evt.get("request_id") or request_id,
                    vision_timestamp_ns=match_evt.get("timestamp_ns"),
                    cycle=cycle,
                    attempt=attempt,
                )
                scan_mode = str(pick_cfg.get("scan_mode", "single")).lower()
                if scan_mode == "multi":
                    match, match_evt = _collect_multi_scan(
                        ctx,
                        request_id,
                        timeout_s,
                        handle,
                        filters,
                        track,
                        _log_debug,
                        match,
                        match_evt,
                        pick_cfg,
                    )
                    match = _attach_fusion_visualization(match, fusion_result)
                    _log(
                        ctx,
                        run_id,
                        "PICK_PLACE_MULTI_SCAN",
                        match=match,
                        frame_id=match_evt.get("frame_id"),
                        request_id=match_evt.get("request_id") or request_id,
                        vision_timestamp_ns=match_evt.get("timestamp_ns"),
                        cycle=cycle,
                        attempt=attempt,
                    )
                _log(
                    ctx,
                    run_id,
                    "VISION_MATCH",
                    frame_id=match_evt.get("frame_id"),
                    request_id=match_evt.get("request_id") or request_id,
                    vision_timestamp_ns=match_evt.get("timestamp_ns"),
                    cycle=cycle,
                    attempt=attempt,
                )
                ctx.run_manager.set_last_vision(
                    run_id,
                    request_id=match_evt.get("request_id") or request_id,
                    frame_id=match_evt.get("frame_id"),
                    timestamp_ns=match_evt.get("timestamp_ns"),
                )

                stop_vision_started_at = time.perf_counter()
                try:
                    ctx.vision.stop_session(request_id)
                except Exception:
                    pass
                iter_timing.span("vision_session_stop", stop_vision_started_at)
                vision_started = False
                handle.vision_request_id = None

                planning_started_at = time.perf_counter()
                ctx.run_manager.set_phase(
                    run_id,
                    "motion_planning",
                    {
                        "cycle": cycle,
                        "attempt": attempt,
                        "request_id": match_evt.get("request_id") or request_id,
                        "frame_id": match_evt.get("frame_id"),
                    },
                )
                orientation_overrides = {}
                if callable(runtime_hooks.resolve_orientation_overrides):
                    orientation_overrides = (
                        runtime_hooks.resolve_orientation_overrides(
                            ctx,
                            state,
                            recipe,
                            vision_cfg,
                            robot_cfg,
                            pick_cfg,
                            module_params,
                            runtime_state,
                            cycle,
                            attempt,
                            match,
                            base_to_cam,
                            base_to_ee,
                            capture_pose,
                            capture_tcp_pose,
                        )
                        or {}
                    )

                align_with_surface = bool(
                    pick_cfg.get("align_with_surface", False)
                ) or bool(orientation_overrides.get("force_align_with_surface"))
                approach_offset = pick_cfg.get("approach_offset_m", [0.0, 0.0, 0.1])
                retreat_offset = pick_cfg.get("retreat_offset_m", approach_offset)
                grasp_offset = pick_cfg.get("grasp_offset_m", [0.0, 0.0, 0.0])
                handeye_gripper_offset_local = [
                    float(hand_eye_raw.get("gripper_x_offset_m", 0.0) or 0.0),
                    float(hand_eye_raw.get("gripper_y_offset_m", 0.0) or 0.0),
                    float(hand_eye_raw.get("gripper_z_offset_m", 0.0) or 0.0),
                ]
                tcp_calibration_active = bool(
                    isinstance(tcp_calibration, dict)
                    and tcp_calibration.get("has_calibration")
                )
                default_use_handeye_gripper_offset = (
                    gripper_type == "parallel_jaw" and not tcp_calibration_active
                )
                use_handeye_gripper_offset = bool(
                    pick_cfg.get(
                        "use_handeye_gripper_offset",
                        default_use_handeye_gripper_offset,
                    )
                )
                default_pick_z_offset = (
                    0.0
                    if gripper_type == "parallel_jaw" or tcp_calibration_active
                    else hand_eye_raw.get("gripper_z_offset_m", 0.0)
                )
                pick_z_offset = float(
                    pick_cfg.get(
                        "pick_z_offset_m",
                        default_pick_z_offset,
                    )
                )
                approach_normal_only = bool(pick_cfg.get("approach_normal_only", True))
                retreat_normal_only = bool(pick_cfg.get("retreat_normal_only", True))
                approach_mode = str(
                    pick_cfg.get(
                        "approach_mode",
                        "normal" if align_with_surface else "base",
                    )
                ).lower()
                retreat_mode = str(
                    pick_cfg.get(
                        "retreat_mode",
                        approach_mode,
                    )
                ).lower()

                orientation_mode = str(pick_cfg.get("orientation_mode", "")).lower()
                if not orientation_mode:
                    orientation_mode = str(
                        orientation_overrides.get("orientation_mode_default") or ""
                    ).lower()
                base_orientation = pick_cfg.get("orientation_quat_xyzw") or [
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                ]
                capture_orientation = (
                    _get_pose_quat(capture_pose)
                    or capture_tcp_pose.get("quat_xyzw")
                    or base_to_tcp.get("rotation_quat_xyzw", [0.0, 0.0, 0.0, 1.0])
                )
                vision_pose_quat_cam = _normalize_quat_xyzw(
                    match.get("pose_quat_xyzw")
                    or match.get("quaternion_xyzw")
                    or [0.0, 0.0, 0.0, 1.0]
                )
                vision_pose_quat_base = _quat_mul_xyzw(
                    base_to_cam.get("rotation_quat_xyzw", [0.0, 0.0, 0.0, 1.0]),
                    vision_pose_quat_cam,
                )
                vision_pose_offset = _normalize_quat_xyzw(
                    pick_cfg.get("vision_pose_quat_offset_xyzw")
                    or [0.0, 0.0, 0.0, 1.0]
                )

                yaw_deg = orientation_overrides.get("yaw_deg")
                if yaw_deg is None:
                    yaw_deg = match.get("yaw_deg")
                yaw_offset_deg = float(pick_cfg.get("yaw_offset_deg", 0.0))
                yaw_sign = float(pick_cfg.get("yaw_sign", 1.0))
                yaw_auto_sign = bool(pick_cfg.get("yaw_auto_sign", True))
                # For surface-aligned picking, default to smart yaw folding to avoid
                # 180-degree axis flips from 2D matcher ambiguity.
                smart_yaw = bool(
                    pick_cfg.get(
                        "smart_yaw",
                        orientation_overrides.get(
                            "smart_yaw_default", align_with_surface
                        ),
                    )
                )
                freeze_roll_pitch = bool(pick_cfg.get("freeze_roll_pitch", False))
                # Decoupling is mainly useful when using surface normals.
                yaw_decouple_surface = bool(
                    pick_cfg.get(
                        "yaw_decouple_surface",
                        orientation_overrides.get(
                            "yaw_decouple_surface_default", align_with_surface
                        ),
                    )
                )
                yaw_snap_deg = float(pick_cfg.get("yaw_snap_deg", 0.0))
                yaw_round_decimals = pick_cfg.get("yaw_round_decimals")
                yaw_frame_default = (
                    "base"
                    if (align_with_surface and yaw_decouple_surface)
                    else "tool"
                )
                yaw_frame_default = str(
                    orientation_overrides.get("yaw_frame_default", yaw_frame_default)
                ).lower()
                yaw_frame = str(pick_cfg.get("yaw_frame", yaw_frame_default)).lower()
                if yaw_frame not in ("base", "tool"):
                    yaw_frame = yaw_frame_default
                surface_align_axis_default = str(
                    orientation_overrides.get("surface_align_axis_default", "auto")
                ).lower()
                surface_align_axis = str(
                    pick_cfg.get("surface_align_axis", surface_align_axis_default)
                ).lower()
                if yaw_deg is not None:
                    if yaw_snap_deg > 0:
                        yaw_deg = round(float(yaw_deg) / yaw_snap_deg) * yaw_snap_deg
                    elif yaw_round_decimals is not None:
                        try:
                            yaw_decimals = int(yaw_round_decimals)
                        except (TypeError, ValueError):
                            yaw_decimals = None
                        if yaw_decimals is not None:
                            yaw_deg = round(float(yaw_deg), yaw_decimals)
                yaw_sign_effective = yaw_sign
                if yaw_auto_sign and yaw_frame == "base":
                    cam_z = _quat_rotate(
                        base_to_cam.get("rotation_quat_xyzw", [0.0, 0.0, 0.0, 1.0]),
                        [0.0, 0.0, 1.0],
                    )
                    if isinstance(cam_z, list) and len(cam_z) >= 3 and cam_z[2] < 0:
                        yaw_sign_effective *= -1.0
                yaw_effective_deg = _normalize_angle_deg(
                    (yaw_deg or 0.0) * yaw_sign_effective + yaw_offset_deg
                )
                pick_yaw_deg = (
                    _wrap_yaw_90(yaw_effective_deg) if smart_yaw else yaw_effective_deg
                )
                place_yaw_comp_deg = (
                    _normalize_angle_deg(pick_yaw_deg - yaw_effective_deg)
                    if smart_yaw
                    else 0.0
                )
                yaw_rad = math.radians(pick_yaw_deg)
                orientation_override = orientation_overrides.get("orientation_quat_xyzw")
                if isinstance(orientation_override, list) and len(orientation_override) >= 4:
                    orientation_override = _normalize_quat_xyzw(
                        [
                            float(orientation_override[0]),
                            float(orientation_override[1]),
                            float(orientation_override[2]),
                            float(orientation_override[3]),
                        ]
                    )
                else:
                    orientation_override = None
                _log_debug(
                    "orientation_inputs",
                    cycle=cycle,
                    attempt=attempt,
                    orientation_mode=orientation_mode,
                    yaw_deg=yaw_deg,
                    yaw_effective_deg=yaw_effective_deg,
                    yaw_rad=yaw_rad,
                    yaw_offset_deg=yaw_offset_deg,
                    yaw_sign=yaw_sign,
                    yaw_sign_effective=yaw_sign_effective,
                    yaw_auto_sign=yaw_auto_sign,
                    yaw_round_decimals=yaw_round_decimals,
                    yaw_frame=yaw_frame,
                    surface_align_axis=surface_align_axis,
                    base_orientation=base_orientation,
                    capture_orientation=capture_orientation,
                    smart_yaw=smart_yaw,
                    yaw_decouple_surface=yaw_decouple_surface,
                    pick_yaw_deg=pick_yaw_deg,
                    place_yaw_comp_deg=place_yaw_comp_deg,
                    freeze_roll_pitch=freeze_roll_pitch,
                    orientation_override=orientation_override,
                    vision_pose_quat_cam=vision_pose_quat_cam,
                    vision_pose_quat_base=vision_pose_quat_base,
                    runtime_target_data=runtime_target_data,
                )

                normal_cam = orientation_overrides.get("normal_cam")
                if normal_cam is None:
                    normal_cam = match.get("surface_normal_cam") or match.get("normal_cam")
                normal_base = orientation_overrides.get("normal_base")
                if _is_finite_vec(normal_base, 3):
                    normal_base = list(normal_base)
                else:
                    normal_base = None
                if normal_base is None and (
                    align_with_surface
                    and isinstance(normal_cam, list)
                    and len(normal_cam) >= 3
                ):
                    normal_base = _normalize_vec(
                        _quat_rotate(
                            base_to_cam.get("rotation_quat_xyzw", [0.0, 0.0, 0.0, 1.0]),
                            normal_cam,
                        )
                    )
                if align_with_surface:
                    _log_debug(
                        "surface_normal",
                        cycle=cycle,
                        attempt=attempt,
                        normal_cam=normal_cam,
                        normal_base=normal_base,
                    )
                    if normal_base:
                        _log_debug(
                            "surface_align",
                            cycle=cycle,
                            attempt=attempt,
                            yaw_frame=yaw_frame,
                            surface_align_axis=surface_align_axis,
                        )
                if orientation_override:
                    orientation = orientation_override
                elif freeze_roll_pitch:
                    orientation = _apply_yaw([0.0, 0.0, 0.0, 1.0], yaw_rad, "base")
                elif align_with_surface and normal_base:
                    if yaw_frame == "base" or yaw_decouple_surface:
                        axis_choice = _parse_axis(surface_align_axis)
                        if surface_align_axis == "auto" or axis_choice is None:
                            seed = capture_orientation
                            tool_axes = {
                                "x": _quat_rotate(seed, [1.0, 0.0, 0.0]),
                                "y": _quat_rotate(seed, [0.0, 1.0, 0.0]),
                                "z": _quat_rotate(seed, [0.0, 0.0, 1.0]),
                            }
                            best_axis = max(
                                tool_axes,
                                key=lambda k: abs(_dot(tool_axes[k], normal_base)),
                            )
                            axis_choice = {
                                "axis": best_axis,
                                "sign": (
                                    1.0
                                    if _dot(tool_axes[best_axis], normal_base) >= 0
                                    else -1.0
                                ),
                            }
                        align_axis = axis_choice["axis"]
                        align_sign = axis_choice["sign"]
                        if debug:
                            _log_debug(
                                "surface_align_choice",
                                cycle=cycle,
                                attempt=attempt,
                                align_axis=align_axis,
                                align_sign=align_sign,
                            )
                        target_normal = [
                            normal_base[0] * align_sign,
                            normal_base[1] * align_sign,
                            normal_base[2] * align_sign,
                        ]
                        if yaw_decouple_surface:
                            # Decouple yaw from normal estimation while preserving surface alignment:
                            # use base-frame heading as the in-plane reference (if yaw is available).
                            if yaw_deg is not None:
                                x_ref = [math.cos(yaw_rad), math.sin(yaw_rad), 0.0]
                            else:
                                x_seed = _quat_rotate(
                                    capture_orientation, [1.0, 0.0, 0.0]
                                )
                                x_ref = _project_to_plane(x_seed, target_normal)
                                x_ref = _normalize_vec(x_ref) or [1.0, 0.0, 0.0]
                        else:
                            x_ref = [math.cos(yaw_rad), math.sin(yaw_rad), 0.0]
                        if align_axis == "y":
                            y_axis = target_normal
                            x_proj = _project_to_plane(x_ref, y_axis)
                            x_axis = _normalize_vec(x_proj)
                            if not x_axis:
                                fallback = (
                                    [1.0, 0.0, 0.0]
                                    if abs(y_axis[0]) < 0.9
                                    else [0.0, 0.0, 1.0]
                                )
                                x_axis = _normalize_vec(
                                    _project_to_plane(fallback, y_axis)
                                ) or [1.0, 0.0, 0.0]
                            z_axis = _normalize_vec(_cross(x_axis, y_axis)) or [
                                0.0,
                                0.0,
                                1.0,
                            ]
                            x_axis = _normalize_vec(_cross(y_axis, z_axis)) or x_axis
                            orientation = _quat_from_axes(x_axis, y_axis, z_axis)
                        elif align_axis == "x":
                            x_axis = target_normal
                            y_proj = _project_to_plane(x_ref, x_axis)
                            y_axis = _normalize_vec(y_proj)
                            if not y_axis:
                                fallback = (
                                    [0.0, 1.0, 0.0]
                                    if abs(x_axis[1]) < 0.9
                                    else [0.0, 0.0, 1.0]
                                )
                                y_axis = _normalize_vec(
                                    _project_to_plane(fallback, x_axis)
                                ) or [0.0, 1.0, 0.0]
                            z_axis = _normalize_vec(_cross(x_axis, y_axis)) or [
                                0.0,
                                0.0,
                                1.0,
                            ]
                            y_axis = _normalize_vec(_cross(z_axis, x_axis)) or y_axis
                            orientation = _quat_from_axes(x_axis, y_axis, z_axis)
                        else:
                            z_axis = target_normal
                            x_proj = _project_to_plane(x_ref, z_axis)
                            x_axis = _normalize_vec(x_proj)
                            if not x_axis:
                                fallback = (
                                    [1.0, 0.0, 0.0]
                                    if abs(z_axis[0]) < 0.9
                                    else [0.0, 1.0, 0.0]
                                )
                                x_axis = _normalize_vec(
                                    _project_to_plane(fallback, z_axis)
                                ) or [1.0, 0.0, 0.0]
                            y_axis = _normalize_vec(_cross(z_axis, x_axis)) or [
                                0.0,
                                1.0,
                                0.0,
                            ]
                            x_axis = _normalize_vec(_cross(y_axis, z_axis)) or x_axis
                            orientation = _quat_from_axes(x_axis, y_axis, z_axis)
                    else:
                        q_align = _quat_from_two_vectors([0.0, 0.0, 1.0], normal_base)
                        if yaw_decouple_surface:
                            orientation = q_align
                        else:
                            q_yaw = _axis_angle_quat(normal_base, yaw_rad)
                            orientation = _quat_mul_xyzw(q_yaw, q_align)
                elif orientation_mode == "vision_pose":
                    orientation = _quat_mul_xyzw(
                        vision_pose_quat_base,
                        vision_pose_offset,
                    )
                elif orientation_mode == "fixed":
                    orientation = base_orientation
                elif orientation_mode == "capture_pose":
                    orientation = capture_orientation
                elif orientation_mode == "fixed_with_yaw":
                    orientation = _apply_yaw(base_orientation, yaw_rad, yaw_frame)
                elif orientation_mode == "vision_yaw":
                    orientation = _apply_yaw(capture_orientation, yaw_rad, yaw_frame)
                else:
                    use_base = not _is_identity_quat(base_orientation)
                    if yaw_deg is not None:
                        seed = base_orientation if use_base else capture_orientation
                        orientation = _apply_yaw(seed, yaw_rad, yaw_frame)
                    else:
                        orientation = (
                            base_orientation if use_base else capture_orientation
                        )
                pick_profile = pick_cfg.get("profile", default_profile)

                approach_vec = approach_offset
                retreat_vec = retreat_offset
                approach_dir = None
                if align_with_surface and normal_base:
                    preserve_surface_direction = (
                        isinstance(runtime_target_data, dict)
                        and str(runtime_target_data.get("grasp_type") or "").strip().lower()
                        == "parallel_jaw_pair"
                    )
                    to_cam = _normalize_vec(
                        [-cam_to_target[0], -cam_to_target[1], -cam_to_target[2]]
                    )
                    approach_dir = _normalize_vec(normal_base)
                    if approach_dir and not preserve_surface_direction:
                        if to_cam and _dot(approach_dir, to_cam) < 0:
                            approach_dir = _scale_vec(approach_dir, -1.0)
                        elif not to_cam and _dot(approach_dir, [0.0, 0.0, 1.0]) < 0:
                            approach_dir = _scale_vec(approach_dir, -1.0)
                    if approach_mode == "normal" and approach_dir:
                        base_offset = [0.0, 0.0, 0.0]
                        if (
                            isinstance(approach_offset, list)
                            and len(approach_offset) >= 3
                        ):
                            if not approach_normal_only:
                                base_offset = [
                                    approach_offset[0],
                                    approach_offset[1],
                                    0.0,
                                ]
                            normal_dist = float(approach_offset[2])
                        else:
                            normal_dist = 0.0
                        normal_sign = -1.0 if preserve_surface_direction else 1.0
                        approach_vec = _add_vec(
                            base_offset, _scale_vec(approach_dir, normal_sign * abs(normal_dist))
                        )
                    if retreat_mode == "normal" and approach_dir:
                        base_offset = [0.0, 0.0, 0.0]
                        if (
                            isinstance(retreat_offset, list)
                            and len(retreat_offset) >= 3
                        ):
                            if not retreat_normal_only:
                                base_offset = [
                                    retreat_offset[0],
                                    retreat_offset[1],
                                    0.0,
                                ]
                            normal_dist = float(retreat_offset[2])
                        else:
                            normal_dist = 0.0
                        normal_sign = -1.0 if preserve_surface_direction else 1.0
                        retreat_vec = _add_vec(
                            base_offset, _scale_vec(approach_dir, normal_sign * abs(normal_dist))
                        )

                pick_offset_vec = [0.0, 0.0, 0.0]
                if use_handeye_gripper_offset and any(
                    abs(float(v)) > 1e-9 for v in handeye_gripper_offset_local
                ):
                    # Keep the robot origin/finger base behind the selected
                    # grasp center while preserving the full pregrasp distance.
                    pick_offset_vec = _scale_vec(
                        _quat_rotate(orientation, handeye_gripper_offset_local),
                        -1.0,
                    )
                if abs(pick_z_offset) > 1e-9:
                    if approach_dir:
                        pick_offset_vec = _add_vec(
                            pick_offset_vec,
                            _scale_vec(approach_dir, pick_z_offset),
                        )
                    else:
                        pick_offset_vec = _add_vec(
                            pick_offset_vec,
                            [0.0, 0.0, pick_z_offset],
                        )

                target_pick = _add_vec(target_base, pick_offset_vec)
                approach_nominal = _add_vec(target_pick, approach_vec)
                grasp_nominal = _add_vec(target_pick, grasp_offset)
                retreat_nominal = _add_vec(target_pick, retreat_vec)

                conv_speed_mps = math.sqrt(
                    velocity_base_mps[0] * velocity_base_mps[0]
                    + velocity_base_mps[1] * velocity_base_mps[1]
                    + velocity_base_mps[2] * velocity_base_mps[2]
                )
                dynamic_pick_active = (
                    pallatizing_mode
                    and conveyor_dynamic_pick_enabled
                    and conv_speed_mps >= conveyor_velocity_deadband_mps
                )
                plan_t_s = time.time()

                def _lead_point(point_m: List[float], lead_s: float) -> List[float]:
                    if not dynamic_pick_active:
                        return list(point_m)
                    return [
                        point_m[0] + velocity_base_mps[0] * lead_s,
                        point_m[1] + velocity_base_mps[1] * lead_s,
                        point_m[2] + velocity_base_mps[2] * lead_s,
                    ]

                approach = _lead_point(approach_nominal, conveyor_pre_pick_lead_s)
                grasp = _lead_point(grasp_nominal, conveyor_pick_lead_s)
                retreat = _lead_point(retreat_nominal, conveyor_retreat_lead_s)

                # Log all pick pose states for debugging
                _log(
                    ctx,
                    run_id,
                    "DEBUG_PICK_POSES",
                    cycle=cycle,
                    attempt=attempt,
                    target_base=target_base,
                    pick_offset_vec=pick_offset_vec,
                    target_pick=target_pick,
                    approach_offset=approach_offset,
                    approach_vec=approach_vec,
                    approach_nominal=approach_nominal,
                    grasp_nominal=grasp_nominal,
                    retreat_nominal=retreat_nominal,
                    grasp_offset=grasp_offset,
                    retreat_offset=retreat_offset,
                    retreat_vec=retreat_vec,
                    approach_with_lead=approach,
                    grasp_with_lead=grasp,
                    retreat_with_lead=retreat,
                    orientation_quat=orientation,
                )

                def _tcp_and_command_target(point_m: List[float]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
                    desired_tcp_pose = {
                        "position_m": list(point_m),
                        "quat_xyzw": _normalize_quat_xyzw(list(orientation)),
                        "frame": "base",
                    }
                    command_pose = _command_pose_for_desired_tcp(
                        desired_tcp_pose,
                        tcp_calibration,
                    )
                    return desired_tcp_pose, command_pose

                approach_tcp_pose, approach_command_pose = _tcp_and_command_target(approach)
                grasp_tcp_pose, grasp_command_pose = _tcp_and_command_target(grasp)
                retreat_tcp_pose, retreat_command_pose = _tcp_and_command_target(retreat)
                pick_profile_approach = conveyor_pre_pick_profile or pick_profile
                pick_profile_grasp = conveyor_pick_profile or pick_profile
                pick_profile_retreat = conveyor_retreat_profile or pick_profile
                approach_move_strategy = str(
                    pick_cfg.get("parallel_jaw_approach_move_strategy")
                    or pick_cfg.get("approach_move_strategy")
                    or robot_cfg.get("pick_approach_move_strategy")
                    or ("ik_nullspace" if gripper_type == "parallel_jaw" else "cartesian")
                ).strip().lower()
                approach_preferred_pose_name = str(
                    pick_cfg.get("parallel_jaw_approach_preferred_pose_name")
                    or pick_cfg.get("approach_preferred_pose_name")
                    or robot_cfg.get("pick_approach_preferred_pose_name")
                    or robot_cfg.get("capture_preferred_pose_name")
                    or robot_cfg.get("preferred_pose_name")
                    or ""
                ).strip()
                gripper_widths = _resolve_parallel_jaw_widths(
                    pick_cfg,
                    runtime_target_data if isinstance(runtime_target_data, dict) else None,
                )
                pregrasp_open_width_m = gripper_widths.get("pregrasp_open_width_m")
                grasp_close_width_m = gripper_widths.get("grasp_close_width_m")
                pregrasp_action = str(
                    gripper_widths.get("pregrasp_action") or "open"
                ).strip().lower()
                grasp_action = str(
                    gripper_widths.get("grasp_action") or "close"
                ).strip().lower()
                raw_grasp_force_n = gripper_widths.get("grasp_force_n")
                grasp_force_n = (
                    None
                    if raw_grasp_force_n is None
                    else _coerce_float(raw_grasp_force_n, None)
                )
                if intermediate_pose is not None:
                    intermediate_profile = str(
                        robot_cfg.get("intermediate_profile")
                        or pick_cfg.get("intermediate_profile")
                        or intermediate_pose.get("profile")
                        or pick_profile_approach
                    )
                    _log(
                        ctx,
                        run_id,
                        "PICK_PLACE_INTERMEDIATE_MOVE",
                        cycle=cycle,
                        pose_name=intermediate_pose_name,
                        profile=intermediate_profile,
                        target=_pose_debug_summary(
                            intermediate_pose.get("tcp_pose")
                            or intermediate_pose.get("tcp")
                            or intermediate_pose
                        ),
                    )
                    robot_stage_started_at = time.perf_counter()
                    _move_to_pose(
                        ctx,
                        run_id,
                        intermediate_pose,
                        intermediate_profile,
                        prefer_cartesian=True,
                        pose_index=pose_index,
                        tcp_calibration=tcp_calibration,
                    )
                    iter_timing.span(
                        "robot_command_intermediate_pose",
                        robot_stage_started_at,
                        pose_name=intermediate_pose_name,
                        profile=intermediate_profile,
                    )
                    _ensure_stop(handle)
                if use_vacuum_gripper:
                    ctx.robot.open_gripper()
                else:
                    pregrasp_force_n = (
                        0.0 if pregrasp_action == "close" else None
                    )
                    _log(
                        ctx,
                        run_id,
                        "PICK_PLACE_GRIPPER_PREGRASP",
                        cycle=cycle,
                        stage="pregrasp",
                        gripper_type=gripper_type,
                        action=pregrasp_action,
                        width_m=pregrasp_open_width_m,
                        force_n=pregrasp_force_n,
                    )
                    _issue_parallel_jaw_gripper_command(
                        ctx,
                        pregrasp_action,
                        pregrasp_open_width_m,
                        force_n=pregrasp_force_n,
                    )
                iter_timing.span(
                    "pick_pose_planning_and_pregrasp",
                    planning_started_at,
                    gripper_type=gripper_type,
                )
                _log_debug(
                    "parallel_jaw_widths",
                    cycle=cycle,
                    attempt=attempt,
                    gripper_type=gripper_type,
                    widths=gripper_widths,
                    runtime_target_data=runtime_target_data,
                )
                _log(
                    ctx,
                    run_id,
                    "PICK_PLACE_APPROACH",
                    target=approach,
                    cycle=cycle,
                    desired_tcp_pose=approach_tcp_pose,
                    command_pose=approach_command_pose,
                    command_orientation_frame="tcp",
                    tcp_calibration=tcp_calibration,
                    tcp_calibration_source=tcp_calibration_source,
                )
                _log_debug(
                    "approach",
                    cycle=cycle,
                    attempt=attempt,
                    orientation=orientation,
                    approach=approach,
                    grasp=grasp,
                    retreat=retreat,
                    pick_offset=pick_offset_vec,
                    pick_z_offset_m=pick_z_offset,
                    handeye_gripper_offset_local_m=handeye_gripper_offset_local,
                    use_handeye_gripper_offset=use_handeye_gripper_offset,
                    tcp_calibration_active=tcp_calibration_active,
                    approach_mode=approach_mode,
                    retreat_mode=retreat_mode,
                    approach_offset=approach_vec,
                    retreat_offset=retreat_vec,
                    pick_profile=pick_profile,
                    pallatizing_mode=pallatizing_mode,
                    dynamic_pick_active=dynamic_pick_active,
                    conveyor_velocity_base_mps=velocity_base_mps,
                    conveyor_speed_mps=conv_speed_mps,
                    pre_pick_lead_s=conveyor_pre_pick_lead_s,
                    pick_lead_s=conveyor_pick_lead_s,
                    retreat_lead_s=conveyor_retreat_lead_s,
                    approach_nominal=approach_nominal,
                    grasp_nominal=grasp_nominal,
                    retreat_nominal=retreat_nominal,
                    approach_dynamic=approach,
                    grasp_dynamic=grasp,
                    retreat_dynamic=retreat,
                    tcp_calibration=tcp_calibration,
                    tcp_calibration_source=tcp_calibration_source,
                    command_orientation_frame="tcp",
                    approach_command_pose=approach_command_pose,
                    grasp_command_pose=grasp_command_pose,
                    retreat_command_pose=retreat_command_pose,
                    pick_profile_approach=pick_profile_approach,
                    pick_profile_grasp=pick_profile_grasp,
                    pick_profile_retreat=pick_profile_retreat,
                    approach_move_strategy=approach_move_strategy,
                    approach_preferred_pose_name=approach_preferred_pose_name,
                )
                robot_stage_started_at = time.perf_counter()
                _move_tcp_target_with_log(
                    ctx,
                    run_id,
                    cycle,
                    "approach",
                    approach_command_pose,
                    pick_profile_approach,
                    strategy=approach_move_strategy,
                    pose_index=pose_index,
                    preferred_pose_name=approach_preferred_pose_name,
                    options=pick_cfg,
                    tcp_calibration=tcp_calibration,
                )
                iter_timing.span(
                    "robot_command_approach",
                    robot_stage_started_at,
                    target=approach,
                    desired_tcp_pose=approach_tcp_pose,
                    command_pose=approach_command_pose,
                    tcp_calibration=tcp_calibration,
                    tcp_calibration_source=tcp_calibration_source,
                    profile=pick_profile_approach,
                    move_strategy=approach_move_strategy,
                    preferred_pose_name=approach_preferred_pose_name,
                )
                _ensure_stop(handle)

                if use_vacuum_gripper:
                    gripper_started_at = time.perf_counter()
                    _log(
                        ctx,
                        run_id,
                        "PICK_PLACE_GRIPPER_ON",
                        cycle=cycle,
                        stage="pre_pick",
                        gripper_type=gripper_type,
                    )
                    ctx.robot.close_gripper()
                    iter_timing.span("robot_gripper_pre_pick", gripper_started_at, gripper_type=gripper_type)
                    _ensure_stop(handle)

                if dynamic_pick_active:
                    elapsed_s = max(0.0, time.time() - plan_t_s)
                    grasp = _lead_point(grasp_nominal, conveyor_pick_lead_s + elapsed_s)
                    grasp_tcp_pose, grasp_command_pose = _tcp_and_command_target(grasp)

                _log(
                    ctx,
                    run_id,
                    "PICK_PLACE_GRASP",
                    target=grasp,
                    cycle=cycle,
                    desired_tcp_pose=grasp_tcp_pose,
                    command_pose=grasp_command_pose,
                    tcp_calibration=tcp_calibration,
                    tcp_calibration_source=tcp_calibration_source,
                )
                robot_stage_started_at = time.perf_counter()
                _movel_with_log(
                    ctx,
                    run_id,
                    cycle,
                    "grasp",
                    grasp_command_pose,
                    pick_profile_grasp,
                    tcp_calibration,
                )
                iter_timing.span(
                    "robot_command_grasp",
                    robot_stage_started_at,
                    target=grasp,
                    desired_tcp_pose=grasp_tcp_pose,
                    command_pose=grasp_command_pose,
                    tcp_calibration=tcp_calibration,
                    tcp_calibration_source=tcp_calibration_source,
                    profile=pick_profile_grasp,
                )
                _ensure_stop(handle)

                if use_vacuum_gripper:
                    _log(
                        ctx,
                        run_id,
                        "PICK_PLACE_GRIPPER_HOLD",
                        cycle=cycle,
                        stage="pick",
                        gripper_type=gripper_type,
                    )
                    if vacuum_pick_wait_s > 1e-6:
                        _log(
                            ctx,
                            run_id,
                            "PICK_PLACE_VACUUM_WAIT",
                            cycle=cycle,
                            stage="pick",
                            gripper_type=gripper_type,
                            duration_s=vacuum_pick_wait_s,
                        )
                        if handle.stop_event.wait(vacuum_pick_wait_s):
                            raise RuntimeError("run_stopped")
                else:
                    gripper_started_at = time.perf_counter()
                    _log(
                        ctx,
                        run_id,
                        "PICK_PLACE_GRIPPER_ON",
                        cycle=cycle,
                        stage="pick",
                        gripper_type=gripper_type,
                        action=grasp_action,
                        width_m=grasp_close_width_m,
                        force_n=grasp_force_n,
                    )
                    _issue_parallel_jaw_gripper_command(
                        ctx,
                        grasp_action,
                        grasp_close_width_m,
                        force_n=grasp_force_n,
                    )
                    iter_timing.span(
                        "robot_gripper_pick",
                        gripper_started_at,
                        gripper_type=gripper_type,
                        action=grasp_action,
                        width_m=grasp_close_width_m,
                    )
                    _ensure_stop(handle)

                if dynamic_pick_active:
                    elapsed_s = max(0.0, time.time() - plan_t_s)
                    retreat = _lead_point(
                        retreat_nominal, conveyor_retreat_lead_s + elapsed_s
                    )
                    retreat_tcp_pose, retreat_command_pose = _tcp_and_command_target(retreat)

                _log(
                    ctx,
                    run_id,
                    "PICK_PLACE_RETREAT",
                    target=retreat,
                    cycle=cycle,
                    desired_tcp_pose=retreat_tcp_pose,
                    command_pose=retreat_command_pose,
                    tcp_calibration=tcp_calibration,
                    tcp_calibration_source=tcp_calibration_source,
                )
                robot_stage_started_at = time.perf_counter()
                _movel_with_log(
                    ctx,
                    run_id,
                    cycle,
                    "retreat",
                    retreat_command_pose,
                    pick_profile_retreat,
                    tcp_calibration,
                )
                iter_timing.span(
                    "robot_command_retreat",
                    robot_stage_started_at,
                    target=retreat,
                    desired_tcp_pose=retreat_tcp_pose,
                    command_pose=retreat_command_pose,
                    tcp_calibration=tcp_calibration,
                    tcp_calibration_source=tcp_calibration_source,
                    profile=pick_profile_retreat,
                )
                _ensure_stop(handle)
                _log(ctx, run_id, "PICK_DONE", cycle=cycle)

                place_cfg = robot_cfg.get("place", {})
                place_plan = {}
                if callable(runtime_hooks.resolve_place_plan):
                    place_plan = (
                        runtime_hooks.resolve_place_plan(
                            ctx,
                            state,
                            recipe,
                            vision_cfg,
                            robot_cfg,
                            pick_cfg,
                            module_params,
                            runtime_state,
                            {
                                "cycle": cycle,
                                "attempt": attempt,
                                "match": match,
                                "runtime_target_data": runtime_target_data,
                                "orientation": orientation,
                                "pose_index": pose_index,
                                "capture_pose": capture_pose,
                                "intermediate_pose": intermediate_pose,
                                "intermediate_pose_name": intermediate_pose_name,
                                "place_pose": place_pose,
                                "default_profile": default_profile,
                                "gripper_type": gripper_type,
                                "use_vacuum_gripper": use_vacuum_gripper,
                                "pregrasp_open_width_m": pregrasp_open_width_m,
                                "grasp_close_width_m": grasp_close_width_m,
                            },
                            _log_debug,
                        )
                        or {}
                    )
                if not isinstance(place_plan, dict):
                    place_plan = {}

                # Log the place_plan resolution
                _log(
                    ctx,
                    run_id,
                    "DEBUG_PLACE_PLAN_RESOLVED",
                    cycle=cycle,
                    attempt=attempt,
                    place_plan_strategy_name=place_plan.get("strategy_name"),
                    place_plan_mode=place_plan.get("mode"),
                    place_plan_has_place_pose=isinstance(place_plan.get("place_pose"), dict),
                    place_plan_place_pose=place_plan.get("place_pose"),
                    place_pose_from_config_tcp=place_pose.get("tcp_pose", place_pose.get("tcp")),
                    place_pose_name_in_config="place_pose_name" in robot_cfg,
                )

                intermediate_regrasp = (
                    place_plan.get("intermediate_regrasp")
                    if isinstance(place_plan.get("intermediate_regrasp"), dict)
                    else None
                )
                if intermediate_regrasp:
                    _log_debug(
                        "place_plan",
                        cycle=cycle,
                        attempt=attempt,
                        place_plan=place_plan,
                    )
                    regrasp_result = _execute_intermediate_regrasp(
                        ctx,
                        handle,
                        run_id,
                        cycle,
                        default_profile,
                        gripper_type,
                        use_vacuum_gripper,
                        orientation,
                        vacuum_pick_wait_s,
                        pregrasp_open_width_m,
                        grasp_close_width_m,
                        place_plan,
                    )
                    carry_orientation = regrasp_result.get("carry_orientation")
                    if _is_finite_vec(carry_orientation, 4):
                        orientation = _normalize_quat_xyzw(list(carry_orientation))

                place_pose_effective = (
                    place_plan.get("place_pose")
                    if isinstance(place_plan.get("place_pose"), dict)
                    else place_pose
                )
                place_profile = place_plan.get("place_profile") or place_cfg.get(
                    "profile", default_profile
                )
                place_open_width_m = _coerce_float(
                    place_plan.get("place_open_width_m"),
                    _coerce_float(
                        place_cfg.get("parallel_jaw_open_width_m"),
                        pregrasp_open_width_m,
                    ),
                )
                place_approach_offset = place_plan.get("place_approach_offset_m") or place_cfg.get(
                    "approach_offset_m",
                    pick_cfg.get("approach_offset_m", [0.0, 0.0, 0.1]),
                )
                place_retreat_offset = place_plan.get("place_retreat_offset_m") or place_cfg.get(
                    "retreat_offset_m", place_approach_offset
                )

                place_target = (
                    place_pose_effective.get("tcp")
                    or place_pose_effective.get("tcp_pose")
                    or {}
                )
                place_pos = place_target.get("position_m")
                place_frame = place_target.get("frame", "base")
                place_quat = place_target.get("quat_xyzw")
                if not (isinstance(place_quat, list) and len(place_quat) >= 4):
                    place_quat = None
                place_quat_override = place_plan.get("place_quat_xyzw")
                if (
                    isinstance(place_quat_override, list)
                    and len(place_quat_override) >= 4
                ):
                    place_quat = _normalize_quat_xyzw(
                        [
                            float(place_quat_override[0]),
                            float(place_quat_override[1]),
                            float(place_quat_override[2]),
                            float(place_quat_override[3]),
                        ]
                    )
                force_place_yaw_deg = place_plan.get("force_place_yaw_deg", None)
                if force_place_yaw_deg is None:
                    force_place_yaw_deg = place_cfg.get("force_yaw_deg", None)
                if force_place_yaw_deg is not None:
                    place_quat = _rpy_deg_to_quat_xyzw(
                        0.0, 0.0, float(force_place_yaw_deg) + place_yaw_comp_deg
                    )
                elif place_quat is None:
                    place_quat = _normalize_quat_xyzw(orientation)
                else:
                    place_quat = _normalize_quat_xyzw(place_quat)

                # Log all place pose states for debugging
                _log(
                    ctx,
                    run_id,
                    "DEBUG_PLACE_POSES",
                    cycle=cycle,
                    attempt=attempt,
                    place_pose_effective_name=place_pose_effective.get("name"),
                    place_pose_effective_tcp_pos=place_target.get("position_m"),
                    place_pos=place_pos,
                    place_approach_offset=place_approach_offset,
                    place_retreat_offset=place_retreat_offset,
                    place_frame=place_frame,
                    place_quat=place_quat,
                    place_approach_offset_source="place_plan" if place_plan.get("place_approach_offset_m") else "place_cfg",
                    place_retreat_offset_source="place_plan" if place_plan.get("place_retreat_offset_m") else "place_cfg",
                )

                if place_pos:
                    place_approach = _add_vec(place_pos, place_approach_offset)
                    place_retreat = _add_vec(place_pos, place_retreat_offset)
                    # Log the computed place approach/actual/retreat positions
                    _log(
                        ctx,
                        run_id,
                        "DEBUG_PLACE_POSITIONS",
                        cycle=cycle,
                        attempt=attempt,
                        place_pos_base=place_pos,
                        place_approach_computed=place_approach,
                        place_retreat_computed=place_retreat,
                        place_approach_offset_applied=place_approach_offset,
                        place_retreat_offset_applied=place_retreat_offset,
                    )
                    # Convert tool_tcp pose to Franka EE pose for movel command
                    place_approach_tcp_pose = {
                        "position_m": place_approach,
                        "quat_xyzw": place_quat,
                        "frame": place_frame,
                    }
                    place_approach_ee = _command_pose_for_desired_tcp_to_ee(
                        place_approach_tcp_pose,
                        tcp_calibration,
                    )
                    _log(
                        ctx,
                        run_id,
                        "PICK_PLACE_PLACE_APPROACH",
                        target=place_approach_ee.get("position_m"),
                        cycle=cycle,
                    )
                    robot_stage_started_at = time.perf_counter()
                    _movel_with_log(
                        ctx,
                        run_id,
                        cycle,
                        "place_approach",
                        place_approach_ee,
                        place_profile,
                        tcp_calibration,
                    )
                    iter_timing.span(
                        "robot_command_place_approach",
                        robot_stage_started_at,
                        target=place_approach_ee.get("position_m"),
                        profile=place_profile,
                    )
                    _ensure_stop(handle)

                    place_move_tcp_pose = {
                        "position_m": place_pos,
                        "quat_xyzw": place_quat,
                        "frame": place_frame,
                    }
                    place_move_ee = _command_pose_for_desired_tcp_to_ee(
                        place_move_tcp_pose,
                        tcp_calibration,
                    )
                    _log(
                        ctx,
                        run_id,
                        "PICK_PLACE_PLACE_MOVE",
                        target=place_move_ee.get("position_m"),
                        cycle=cycle,
                    )
                    robot_stage_started_at = time.perf_counter()
                    _movel_with_log(
                        ctx,
                        run_id,
                        cycle,
                        "place",
                        place_move_ee,
                        place_profile,
                        tcp_calibration,
                    )
                    iter_timing.span(
                        "robot_command_place",
                        robot_stage_started_at,
                        target=place_move_ee.get("position_m"),
                        profile=place_profile,
                    )
                    _ensure_stop(handle)

                    _log(
                        ctx,
                        run_id,
                        "PICK_PLACE_GRIPPER_OFF",
                        cycle=cycle,
                        stage="place",
                        gripper_type=gripper_type,
                        width_m=place_open_width_m if not use_vacuum_gripper else None,
                    )
                    if use_vacuum_gripper:
                        gripper_started_at = time.perf_counter()
                        ctx.robot.open_gripper()
                        iter_timing.span("robot_gripper_place", gripper_started_at, gripper_type=gripper_type)
                    else:
                        gripper_started_at = time.perf_counter()
                        ctx.robot.open_gripper(place_open_width_m)
                        iter_timing.span(
                            "robot_gripper_place",
                            gripper_started_at,
                            gripper_type=gripper_type,
                            width_m=place_open_width_m,
                        )
                    _ensure_stop(handle)

                    place_retreat_tcp_pose = {
                        "position_m": place_retreat,
                        "quat_xyzw": place_quat,
                        "frame": place_frame,
                    }
                    place_retreat_ee = _command_pose_for_desired_tcp_to_ee(
                        place_retreat_tcp_pose,
                        tcp_calibration,
                    )
                    _log(
                        ctx,
                        run_id,
                        "PICK_PLACE_PLACE_RETREAT",
                        target=place_retreat_ee.get("position_m"),
                        cycle=cycle,
                    )
                    robot_stage_started_at = time.perf_counter()
                    _movel_with_log(
                        ctx,
                        run_id,
                        cycle,
                        "place_retreat",
                        place_retreat_ee,
                        place_profile,
                        tcp_calibration,
                    )
                    iter_timing.span(
                        "robot_command_place_retreat",
                        robot_stage_started_at,
                        target=place_retreat_ee.get("position_m"),
                        profile=place_profile,
                    )
                    _ensure_stop(handle)
                    _log(ctx, run_id, "PLACE_DONE", cycle=cycle)
                else:
                    _log(ctx, run_id, "PICK_PLACE_PLACE_MOVE", cycle=cycle)
                    robot_stage_started_at = time.perf_counter()
                    _move_to_pose(
                        ctx,
                        run_id,
                        place_pose_effective,
                        place_profile,
                        prefer_cartesian=True,
                        tcp_calibration=tcp_calibration,
                    )
                    iter_timing.span(
                        "robot_command_place_pose",
                        robot_stage_started_at,
                        profile=place_profile,
                    )
                    _ensure_stop(handle)
                    _log(
                        ctx,
                        run_id,
                        "PICK_PLACE_GRIPPER_OFF",
                        cycle=cycle,
                        stage="place",
                        gripper_type=gripper_type,
                        width_m=place_open_width_m if not use_vacuum_gripper else None,
                    )
                    if use_vacuum_gripper:
                        gripper_started_at = time.perf_counter()
                        ctx.robot.open_gripper()
                        iter_timing.span("robot_gripper_place", gripper_started_at, gripper_type=gripper_type)
                    else:
                        gripper_started_at = time.perf_counter()
                        ctx.robot.open_gripper(place_open_width_m)
                        iter_timing.span(
                            "robot_gripper_place",
                            gripper_started_at,
                            gripper_type=gripper_type,
                            width_m=place_open_width_m,
                        )
                    _log(ctx, run_id, "PLACE_DONE", cycle=cycle)

                _log(ctx, run_id, "PICK_PLACE_DONE", cycle=cycle)
                iter_timing.summary(
                    status="success",
                    match_frame_id=match_evt.get("frame_id") if isinstance(match_evt, dict) else None,
                    timing_log_path=str(timing_log_path),
                )
                success = True
                break
            except Exception as exc:
                if str(exc) == "run_stopped":
                    raise
                error_code = str(exc).split(":", 1)[0].strip()
                retry = (unlimited_attempts or attempt < max_attempts) and _should_retry(exc)
                _log(
                    ctx,
                    run_id,
                    "PICK_PLACE_ATTEMPT_FAILED",
                    cycle=cycle,
                    attempt=attempt,
                    error=str(exc),
                    retry=retry,
                )
                iter_timing.summary(
                    status="failed",
                    error=str(exc),
                    retry=retry,
                    timing_log_path=str(timing_log_path),
                )
                if retry:
                    _log(ctx, run_id, "PICK_PLACE_RETRY", cycle=cycle, attempt=attempt)
                    if error_code in _BIN_PICKING_NO_CANDIDATE_ERRORS:
                        if no_candidate_retry_delay_s > 1e-6:
                            _ensure_stop(handle)
                            if handle.stop_event.wait(no_candidate_retry_delay_s):
                                raise RuntimeError("run_stopped")
                        if not stay_at_capture_on_no_candidate:
                            _move_to_pose(
                                ctx,
                                run_id,
                                capture_pose,
                                default_profile,
                                prefer_cartesian=True,
                                pose_index=pose_index,
                                tcp_calibration=tcp_calibration,
                            )
                    else:
                        _move_to_pose(
                            ctx,
                            run_id,
                            capture_pose,
                            default_profile,
                            prefer_cartesian=True,
                            pose_index=pose_index,
                            tcp_calibration=tcp_calibration,
                        )
                    _ensure_stop(handle)
                    capture_tcp_pose = _wait_for_tcp_target(
                        ctx,
                        handle,
                        run_id,
                        capture_pose,
                        timeout_s=capture_arrival_timeout_s,
                        position_tolerance_m=capture_position_tolerance_m,
                        orientation_tolerance_deg=capture_orientation_tolerance_deg,
                        tcp_calibration=tcp_calibration,
                    )
                    _log(
                        ctx,
                        run_id,
                        "PICK_PLACE_CAPTURE_REACHED",
                        cycle=cycle,
                        attempt=attempt,
                        retry=True,
                        tcp_pose=capture_tcp_pose,
                        position_tolerance_m=capture_position_tolerance_m,
                        orientation_tolerance_deg=capture_orientation_tolerance_deg,
                    )
                    continue
                raise
            finally:
                if vision_started:
                    try:
                        ctx.vision.stop_session(request_id)
                    except Exception:
                        pass
                if handle.vision_request_id == request_id:
                    handle.vision_request_id = None

        if not success:
            raise RuntimeError("pick_attempts_exhausted")

        cycle += 1
        if not repeat:
            break
        if max_cycles > 0 and cycle >= max_cycles:
            break


def run_pallatizing(ctx: StationContext, state: RunState, handle: Any) -> None:
    # Reuse pick/place pipeline; run_pick_place_demo enables conveyor velocity
    # sampling and intercept when task_type is pallatizing/palletizing.
    run_pick_place_demo(ctx, state, handle)
