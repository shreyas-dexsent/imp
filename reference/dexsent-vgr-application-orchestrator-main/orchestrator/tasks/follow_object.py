"""Implementation for `orchestrator.tasks.follow_object`."""

import logging
import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from orchestrator.core.context import StationContext
from orchestrator.core.runs import RunState
from orchestrator.tasks.pick_place import (
    _compose_transform,
    _quat_rotate,
    _resolve_hand_eye,
    _resolve_runtime_handeye,
    _resolve_task_payload,
    _resolve_templates_dir,
    _transform_point,
)
from orchestrator.vision.calibration import build_station_camera_calibration

log = logging.getLogger("follow_object")


def _log(ctx: StationContext, run_id: Optional[str], event: str, **fields: Any) -> None:
    if not run_id:
        return
    payload = {"event": event, "timestamp_ns": time.time_ns()}
    payload.update(fields)
    ctx.runs.append_event(run_id, payload)


def _ensure_stop(handle: Any) -> None:
    stop_event = getattr(handle, "stop_event", None)
    if stop_event is not None and stop_event.is_set():
        raise RuntimeError("run_stopped")


def _as_float(v: Any, default: float) -> float:
    try:
        x = float(v)
        if math.isfinite(x):
            return x
    except Exception:
        pass
    return float(default)


def _as_bool(v: Any, default: bool) -> bool:
    if v is None:
        return default
    return bool(v)


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


def _norm3(v: List[float]) -> float:
    return math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _normalize_angle_deg(angle: float) -> float:
    return (float(angle) + 180.0) % 360.0 - 180.0


def _wrap_yaw_90(yaw_deg: float) -> float:
    wrapped = _normalize_angle_deg(yaw_deg)
    if wrapped > 90.0:
        wrapped -= 180.0
    elif wrapped < -90.0:
        wrapped += 180.0
    return wrapped


def _alpha_from_tau(dt: float, tau: float) -> float:
    if tau <= 1e-6:
        return 1.0
    return _clamp(dt / (tau + dt), 0.0, 1.0)


def _extract_pose_from_state(state: dict, key: str) -> Tuple[List[float], List[float]]:
    entry = state.get(key) or {}
    pos = entry.get("position_m") or [0.0, 0.0, 0.0]
    quat = entry.get("quat_xyzw") or [0.0, 0.0, 0.0, 1.0]
    if not _is_finite_vec(pos, 3):
        pos = [0.0, 0.0, 0.0]
    if not _is_finite_vec(quat, 4):
        quat = [0.0, 0.0, 0.0, 1.0]
    return [float(pos[0]), float(pos[1]), float(pos[2])], [
        float(quat[0]),
        float(quat[1]),
        float(quat[2]),
        float(quat[3]),
    ]


def _tcp_pose(robot: Any) -> Tuple[List[float], List[float]]:
    return _extract_pose_from_state(robot.get_state() or {}, "tcp_pose")


def _flange_pose(robot: Any) -> Tuple[List[float], List[float]]:
    state = robot.get_state() or {}
    # Use flange_pose if available (Franka adapter exposes it); fall back to tcp_pose.
    if state.get("flange_pose"):
        return _extract_pose_from_state(state, "flange_pose")
    return _extract_pose_from_state(state, "tcp_pose")


def _lookup_pose_at_time(
    history: deque[Tuple[float, List[float], List[float]]],
    target_epoch_s: Optional[float],
    fallback_pos: List[float],
    fallback_quat: List[float],
) -> Tuple[List[float], List[float], float]:
    if target_epoch_s is None or not history:
        return list(fallback_pos), list(fallback_quat), 0.0
    best_pos = list(fallback_pos)
    best_quat = list(fallback_quat)
    best_dt = float("inf")
    for t_s, p, q in history:
        dt = abs(t_s - target_epoch_s)
        if dt < best_dt:
            best_dt = dt
            best_pos = list(p)
            best_quat = list(q)
    if not math.isfinite(best_dt):
        best_dt = 0.0
    return best_pos, best_quat, float(best_dt)


def _quat_norm(q: List[float]) -> List[float]:
    if len(q) != 4:
        return [0.0, 0.0, 0.0, 1.0]
    n = math.sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3])
    if n < 1e-9:
        return [0.0, 0.0, 0.0, 1.0]
    return [q[0] / n, q[1] / n, q[2] / n, q[3] / n]


def _quat_from_rpy_rad(roll: float, pitch: float, yaw: float) -> List[float]:
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    qw = cr * cp * cy + sr * sp * sy
    return _quat_norm([qx, qy, qz, qw])


def _quat_to_yaw_deg(q: List[float]) -> float:
    x, y, z, w = q
    siny = 2.0 * (w * z + x * y)
    cosy = 1.0 - 2.0 * (y * y + z * z)
    return math.degrees(math.atan2(siny, cosy))


def _build_target_quat(
    roll_deg: float, pitch_deg: float, yaw_deg: float
) -> List[float]:
    return _quat_from_rpy_rad(
        math.radians(roll_deg), math.radians(pitch_deg), math.radians(yaw_deg)
    )


@dataclass
class _Cfg:
    mode: str
    rate_hz: float
    control_mode: str
    hover_height_m: float
    align_with_surface: bool
    max_target_age_s: float
    dropout_grace_s: float
    clear_target_on_lost: bool
    lost_behavior: str
    max_vel_mps: float
    max_yaw_vel_dps: float
    gain_pos: float
    max_step_xy_m: float
    max_step_z_m: float
    velocity_kp: float
    velocity_ki: float
    velocity_kd: float
    yaw_velocity_kp: float
    yaw_velocity_ki: float
    yaw_velocity_kd: float
    reach_error_enter_m: float
    reach_error_release_m: float
    reach_velocity_kp: float
    reach_velocity_ki: float
    reach_velocity_kd: float
    reach_max_vel_mps: float
    reach_yaw_velocity_kp: float
    reach_yaw_velocity_ki: float
    reach_yaw_velocity_kd: float
    reach_max_yaw_vel_dps: float
    track_velocity_kp: float
    track_velocity_ki: float
    track_velocity_kd: float
    track_max_vel_mps: float
    track_yaw_velocity_kp: float
    track_yaw_velocity_ki: float
    track_yaw_velocity_kd: float
    track_max_yaw_vel_dps: float
    velocity_ff_gain: float
    pos_deadband_m: float
    pos_release_m: float
    yaw_deadband_deg: float
    yaw_release_deg: float
    ema_alpha_pos: float
    ema_alpha_yaw: float
    simple_ema_enabled: bool
    yaw_mode: str
    yaw_fixed_deg: float
    yaw_blend_alpha: float
    yaw_sign: float
    yaw_offset_deg: float
    smart_yaw: bool
    roll_fixed_deg: float
    pitch_fixed_deg: float
    predictive_history_n: int
    predictive_lookahead_s: float
    log_interval_s: float


@dataclass
class _Runtime:
    measured_target_base: Optional[List[float]] = None
    measured_target_cam: Optional[List[float]] = None
    measured_yaw_deg: Optional[float] = None
    measured_normal_base: Optional[List[float]] = None
    measured_epoch_s: float = 0.0
    measured_frame_id: str = ""
    derived_tf_fps_hz: Optional[float] = None
    last_derived_tf_update_s: float = 0.0
    last_derived_tf_ts_ns: int = 0
    ema_target_base: Optional[List[float]] = None
    ema_yaw_deg: Optional[float] = None
    last_cmd_yaw_deg: Optional[float] = None
    yaw_anchor_deg: Optional[float] = None
    settled_hold: bool = False
    result_fps_hz: Optional[float] = None
    predict_hist: deque[Tuple[float, List[float]]] = field(
        default_factory=lambda: deque(maxlen=120)
    )
    prev_err: Optional[List[float]] = None
    prev_yaw_err: Optional[float] = None
    int_err: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    int_yaw_err: float = 0.0
    control_phase: str = "reach"


def _parse_cfg(follow_cfg: Dict[str, Any]) -> _Cfg:
    """Normalize/validate follow-task config and apply compatibility defaults."""
    mode_raw = str(follow_cfg.get("mode", "predictive")).strip().lower()
    mode = "direct" if mode_raw == "direct" else "predictive"

    cm_raw = (
        str(follow_cfg.get("control_mode", follow_cfg.get("control", "auto")))
        .strip()
        .lower()
    )
    if cm_raw in ("velocity", "vel", "cartesian_velocity", "auto"):
        control_mode = "velocity"
    else:
        control_mode = "position"

    yaw_mode = str(follow_cfg.get("yaw_mode", "vision")).strip().lower()
    if yaw_mode in ("match_initial", "initial_match"):
        yaw_mode = "initial"
    elif yaw_mode in ("copy", "match", "match_vision"):
        yaw_mode = "vision"
    if yaw_mode not in ("vision", "blend", "fixed", "initial", "off"):
        yaw_mode = "vision"

    rate_hz = max(1.0, _as_float(follow_cfg.get("rate_hz", 150.0), 150.0))
    hover_height_m = _as_float(follow_cfg.get("hover_height_m", 0.30), 0.30)
    align_with_surface = _as_bool(follow_cfg.get("align_with_surface", False), False)

    # Backward compatibility: older configs used no_match_timeout_s.
    max_target_age_raw = follow_cfg.get("max_target_age_s", None)
    if max_target_age_raw is None:
        max_target_age_raw = follow_cfg.get("no_match_timeout_s", 0.2)
    max_target_age_s = max(0.02, _as_float(max_target_age_raw, 0.2))
    dropout_grace_s = max(0.0, _as_float(follow_cfg.get("dropout_grace_s", 0.03), 0.03))
    keep_last_target = _as_bool(follow_cfg.get("keep_last_target", False), False)
    clear_target_on_lost = _as_bool(
        follow_cfg.get("clear_target_on_lost", not keep_last_target),
        not keep_last_target,
    )
    lost_behavior = (
        str(follow_cfg.get("lost_behavior", "stop_velocity")).strip().lower()
    )
    if lost_behavior not in ("stop_velocity", "hold_current"):
        lost_behavior = "stop_velocity"

    max_vel_mps = max(0.0, _as_float(follow_cfg.get("max_vel_mps", 0.12), 0.12))
    max_yaw_vel_dps = max(0.0, _as_float(follow_cfg.get("max_yaw_vel_dps", 45.0), 45.0))
    velocity_kp = max(0.0, _as_float(follow_cfg.get("velocity_kp", 1.2), 1.2))
    velocity_ki = max(0.0, _as_float(follow_cfg.get("velocity_ki", 0.0), 0.0))
    velocity_kd = max(0.0, _as_float(follow_cfg.get("velocity_kd", 0.0), 0.0))
    yaw_velocity_kp = max(0.0, _as_float(follow_cfg.get("yaw_velocity_kp", 1.0), 1.0))
    yaw_velocity_ki = max(0.0, _as_float(follow_cfg.get("yaw_velocity_ki", 0.0), 0.0))
    yaw_velocity_kd = max(0.0, _as_float(follow_cfg.get("yaw_velocity_kd", 0.0), 0.0))
    reach_error_enter_m = max(
        0.0, _as_float(follow_cfg.get("reach_error_enter_m", 0.06), 0.06)
    )
    reach_error_release_m = max(
        reach_error_enter_m,
        _as_float(
            follow_cfg.get("reach_error_release_m", reach_error_enter_m * 1.5),
            reach_error_enter_m * 1.5,
        ),
    )
    reach_velocity_kp = max(
        0.0, _as_float(follow_cfg.get("reach_velocity_kp", velocity_kp), velocity_kp)
    )
    reach_velocity_ki = max(
        0.0, _as_float(follow_cfg.get("reach_velocity_ki", velocity_ki), velocity_ki)
    )
    reach_velocity_kd = max(
        0.0, _as_float(follow_cfg.get("reach_velocity_kd", velocity_kd), velocity_kd)
    )
    reach_max_vel_mps = max(
        0.0, _as_float(follow_cfg.get("reach_max_vel_mps", max_vel_mps), max_vel_mps)
    )
    reach_yaw_velocity_kp = max(
        0.0,
        _as_float(
            follow_cfg.get("reach_yaw_velocity_kp", yaw_velocity_kp), yaw_velocity_kp
        ),
    )
    reach_yaw_velocity_ki = max(
        0.0,
        _as_float(
            follow_cfg.get("reach_yaw_velocity_ki", yaw_velocity_ki), yaw_velocity_ki
        ),
    )
    reach_yaw_velocity_kd = max(
        0.0,
        _as_float(
            follow_cfg.get("reach_yaw_velocity_kd", yaw_velocity_kd), yaw_velocity_kd
        ),
    )
    reach_max_yaw_vel_dps = max(
        0.0,
        _as_float(
            follow_cfg.get("reach_max_yaw_vel_dps", max_yaw_vel_dps), max_yaw_vel_dps
        ),
    )
    track_velocity_kp = max(
        0.0, _as_float(follow_cfg.get("track_velocity_kp", velocity_kp), velocity_kp)
    )
    track_velocity_ki = max(
        0.0, _as_float(follow_cfg.get("track_velocity_ki", velocity_ki), velocity_ki)
    )
    track_velocity_kd = max(
        0.0, _as_float(follow_cfg.get("track_velocity_kd", velocity_kd), velocity_kd)
    )
    track_max_vel_mps = max(
        0.0, _as_float(follow_cfg.get("track_max_vel_mps", max_vel_mps), max_vel_mps)
    )
    track_yaw_velocity_kp = max(
        0.0,
        _as_float(
            follow_cfg.get("track_yaw_velocity_kp", yaw_velocity_kp), yaw_velocity_kp
        ),
    )
    track_yaw_velocity_ki = max(
        0.0,
        _as_float(
            follow_cfg.get("track_yaw_velocity_ki", yaw_velocity_ki), yaw_velocity_ki
        ),
    )
    track_yaw_velocity_kd = max(
        0.0,
        _as_float(
            follow_cfg.get("track_yaw_velocity_kd", yaw_velocity_kd), yaw_velocity_kd
        ),
    )
    track_max_yaw_vel_dps = max(
        0.0,
        _as_float(
            follow_cfg.get("track_max_yaw_vel_dps", max_yaw_vel_dps), max_yaw_vel_dps
        ),
    )
    velocity_ff_gain = max(0.0, _as_float(follow_cfg.get("velocity_ff_gain", 0.0), 0.0))
    gain_pos = max(
        0.0, _as_float(follow_cfg.get("gain_pos", follow_cfg.get("gain", 0.35)), 0.35)
    )
    max_step_xy_m = max(0.0, _as_float(follow_cfg.get("max_step_xy_m", 0.004), 0.004))
    max_step_z_m = max(0.0, _as_float(follow_cfg.get("max_step_z_m", 0.002), 0.002))

    pos_deadband_m = max(0.0, _as_float(follow_cfg.get("pos_deadband_m", 0.008), 0.008))
    pos_release_m = max(
        pos_deadband_m,
        _as_float(
            follow_cfg.get("pos_release_m", pos_deadband_m * 1.5), pos_deadband_m * 1.5
        ),
    )
    yaw_deadband_deg = max(0.0, _as_float(follow_cfg.get("yaw_deadband_deg", 1.2), 1.2))
    yaw_release_deg = max(
        yaw_deadband_deg,
        _as_float(
            follow_cfg.get("yaw_release_deg", yaw_deadband_deg * 1.5),
            yaw_deadband_deg * 1.5,
        ),
    )

    simple_ema_enabled = _as_bool(follow_cfg.get("simple_ema_enabled", True), True)
    ema_alpha_pos = _clamp(
        _as_float(follow_cfg.get("ema_alpha_pos", 0.12), 0.12), 0.0, 1.0
    )
    ema_alpha_yaw = _clamp(
        _as_float(follow_cfg.get("ema_alpha_yaw", 0.12), 0.12), 0.0, 1.0
    )
    predictive_history_n = int(
        max(2, min(30, _as_float(follow_cfg.get("predictive_history_n", 5), 5)))
    )
    predictive_lookahead_s = max(
        0.0, _as_float(follow_cfg.get("predictive_lookahead_s", 0.0), 0.0)
    )
    log_interval_s = _clamp(
        _as_float(follow_cfg.get("log_interval_s", 0.2), 0.2), 0.05, 2.0
    )

    return _Cfg(
        mode=mode,
        rate_hz=rate_hz,
        control_mode=control_mode,
        hover_height_m=hover_height_m,
        align_with_surface=align_with_surface,
        max_target_age_s=max_target_age_s,
        dropout_grace_s=dropout_grace_s,
        clear_target_on_lost=clear_target_on_lost,
        lost_behavior=lost_behavior,
        max_vel_mps=max_vel_mps,
        max_yaw_vel_dps=max_yaw_vel_dps,
        gain_pos=gain_pos,
        max_step_xy_m=max_step_xy_m,
        max_step_z_m=max_step_z_m,
        velocity_kp=velocity_kp,
        velocity_ki=velocity_ki,
        velocity_kd=velocity_kd,
        yaw_velocity_kp=yaw_velocity_kp,
        yaw_velocity_ki=yaw_velocity_ki,
        yaw_velocity_kd=yaw_velocity_kd,
        reach_error_enter_m=reach_error_enter_m,
        reach_error_release_m=reach_error_release_m,
        reach_velocity_kp=reach_velocity_kp,
        reach_velocity_ki=reach_velocity_ki,
        reach_velocity_kd=reach_velocity_kd,
        reach_max_vel_mps=reach_max_vel_mps,
        reach_yaw_velocity_kp=reach_yaw_velocity_kp,
        reach_yaw_velocity_ki=reach_yaw_velocity_ki,
        reach_yaw_velocity_kd=reach_yaw_velocity_kd,
        reach_max_yaw_vel_dps=reach_max_yaw_vel_dps,
        track_velocity_kp=track_velocity_kp,
        track_velocity_ki=track_velocity_ki,
        track_velocity_kd=track_velocity_kd,
        track_max_vel_mps=track_max_vel_mps,
        track_yaw_velocity_kp=track_yaw_velocity_kp,
        track_yaw_velocity_ki=track_yaw_velocity_ki,
        track_yaw_velocity_kd=track_yaw_velocity_kd,
        track_max_yaw_vel_dps=track_max_yaw_vel_dps,
        velocity_ff_gain=velocity_ff_gain,
        pos_deadband_m=pos_deadband_m,
        pos_release_m=pos_release_m,
        yaw_deadband_deg=yaw_deadband_deg,
        yaw_release_deg=yaw_release_deg,
        ema_alpha_pos=ema_alpha_pos,
        ema_alpha_yaw=ema_alpha_yaw,
        simple_ema_enabled=simple_ema_enabled,
        yaw_mode=yaw_mode,
        yaw_fixed_deg=_as_float(follow_cfg.get("yaw_fixed_deg", 0.0), 0.0),
        yaw_blend_alpha=_clamp(
            _as_float(follow_cfg.get("yaw_blend", 0.25), 0.25), 0.0, 1.0
        ),
        yaw_sign=_as_float(follow_cfg.get("yaw_sign", -1.0), -1.0),
        yaw_offset_deg=_as_float(follow_cfg.get("yaw_offset_deg", 0.0), 0.0),
        smart_yaw=_as_bool(follow_cfg.get("smart_yaw", True), True),
        roll_fixed_deg=_as_float(follow_cfg.get("roll_fixed_deg", -180.0), -180.0),
        pitch_fixed_deg=_as_float(follow_cfg.get("pitch_fixed_deg", 0.0), 0.0),
        predictive_history_n=predictive_history_n,
        predictive_lookahead_s=predictive_lookahead_s,
        log_interval_s=log_interval_s,
    )


def _compute_yaw_cmd_deg(
    cfg: _Cfg, rt: _Runtime, vision_yaw_deg: Optional[float], robot_yaw_deg: float
) -> float:
    if cfg.yaw_mode == "fixed":
        yaw = _normalize_angle_deg(cfg.yaw_fixed_deg)
    elif cfg.yaw_mode == "off":
        yaw = robot_yaw_deg
    elif cfg.yaw_mode == "initial":
        if vision_yaw_deg is None:
            yaw = (
                rt.last_cmd_yaw_deg
                if rt.last_cmd_yaw_deg is not None
                else robot_yaw_deg
            )
        else:
            if rt.yaw_anchor_deg is None:
                rt.yaw_anchor_deg = robot_yaw_deg - vision_yaw_deg
            yaw = vision_yaw_deg + (rt.yaw_anchor_deg or 0.0)
    else:
        if vision_yaw_deg is None:
            yaw = (
                rt.last_cmd_yaw_deg
                if rt.last_cmd_yaw_deg is not None
                else robot_yaw_deg
            )
        else:
            if cfg.yaw_mode == "blend":
                base = (
                    rt.last_cmd_yaw_deg
                    if rt.last_cmd_yaw_deg is not None
                    else robot_yaw_deg
                )
                yaw = (
                    1.0 - cfg.yaw_blend_alpha
                ) * base + cfg.yaw_blend_alpha * vision_yaw_deg
            else:
                yaw = vision_yaw_deg
    yaw = _normalize_angle_deg(yaw)
    rt.last_cmd_yaw_deg = yaw
    return yaw


def run_follow_object(ctx: StationContext, state: RunState, handle: Any) -> None:
    """Run closed-loop object following.

    Data path:
    vision result -> camera/base transform -> filtered target -> robot servo command.

    Control path:
    - optional predictive target extrapolation,
    - optional EMA smoothing,
    - phase-based velocity PID (`reach`/`track`) or position stepping fallback.
    """
    run_id = getattr(handle, "run_id", None) or getattr(state, "run_id", None)
    task = _resolve_task_payload(ctx, state)
    vision_cfg = task.get("vision", {}) if isinstance(task, dict) else {}
    robot_cfg = task.get("robot", {}) if isinstance(task, dict) else {}
    follow_cfg = robot_cfg.get("follow", {}) if isinstance(robot_cfg, dict) else {}

    cfg = _parse_cfg(follow_cfg if isinstance(follow_cfg, dict) else {})

    camera_id = str(vision_cfg.get("camera_id") or "").strip()
    if not camera_id:
        raise RuntimeError("missing_camera_id")
    module = str(vision_cfg.get("module") or "tamplate_matching_sift").strip()
    module_params = dict(vision_cfg.get("params") or {})

    if module in ("feature_matching", "tamplate_matching_sift", "opt_sift"):
        module_params["templates_dir"] = _resolve_templates_dir(
            ctx,
            module_params.get("templates_dir"),
            state.process_id,
            vision_cfg.get("object_id") or module_params.get("object_id"),
        )

    enable_shm_output = vision_cfg.get("enable_shm_output")
    if enable_shm_output is None:
        # Follow loop consumes ZMQ results directly; disable per-frame result SHM by default.
        enable_shm_output = False

    request_id = f"{run_id}-vision"
    payload = {
        "event": "VISION_START",
        "request_id": request_id,
        "camera_id": camera_id,
        "module": module,
        "fps_limit": vision_cfg.get("fps_limit", 0),
        "process_mode": vision_cfg.get("process_mode", "continuous"),
        "params": module_params,
        "enable_shm_output": bool(enable_shm_output),
    }
    station_calibration = build_station_camera_calibration(
        ctx.data_paths,
        state.station_id,
    )
    if station_calibration:
        payload["calibration"] = station_calibration

    # Optional capture pose move before enabling continuous follow loop.
    capture_pose = (
        robot_cfg.get("capture_pose") if isinstance(robot_cfg, dict) else None
    )
    if capture_pose:
        try:
            from orchestrator.tasks.pick_place import _move_to_pose  # type: ignore

            _log(ctx, run_id, "FOLLOW_CAPTURE_MOVE")
            _move_to_pose(ctx, capture_pose, robot_cfg.get("default_profile", "normal"))
        except Exception:
            pass

    # Hand-eye is resolved at runtime from station calibration (preferred source).
    hand_eye_raw, hand_eye_source_pref, hand_eye_source_used = _resolve_runtime_handeye(
        ctx, state.process_id, robot_cfg if isinstance(robot_cfg, dict) else {}
    )
    hand_eye = _resolve_hand_eye(hand_eye_raw)
    hand_eye_frame = (
        str(hand_eye.get("hand_eye_frame") or "gripper_to_camera").strip().lower()
    )

    # Start vision session once and keep consuming latest cached frame results.
    _log(ctx, run_id, "FOLLOW_V3_VISION_START", request_id=request_id)
    ctx.vision.start_session(payload)
    handle.vision_request_id = request_id
    ctx.run_manager.set_last_vision(run_id, request_id=request_id)

    robot_supports_vel = hasattr(ctx.robot, "servo_tcp_velocity")
    actual_mode = (
        "velocity"
        if cfg.control_mode == "velocity" and robot_supports_vel
        else "position"
    )
    vel_profile = str(
        follow_cfg.get("velocity_profile", follow_cfg.get("profile", "normal"))
        or "normal"
    )
    pos_profile = str(
        follow_cfg.get("position_profile", follow_cfg.get("profile", "slow")) or "slow"
    )
    # Velocity servo uses mode 5. Position/cartesian commands use mode 0.
    enter_mode = 5 if actual_mode == "velocity" else 0
    if hasattr(ctx.robot, "set_mode"):
        ctx.robot.set_mode(enter_mode)
        # Allow controller mode switch to settle before set_state/servo commands.
        time.sleep(0.05)
    if hasattr(ctx.robot, "set_state"):
        ctx.robot.set_state(0)
    _log(
        ctx,
        run_id,
        "FOLLOW_V3_SERVO_MODE",
        mode=enter_mode,
        rate_hz=cfg.rate_hz,
        control_mode=actual_mode,
        hand_eye_frame=hand_eye_frame,
        hand_eye_source_pref=hand_eye_source_pref,
        hand_eye_source_used=hand_eye_source_used,
    )

    rt = _Runtime()
    last_frame_id = ""
    last_no_target_log = 0.0
    last_tf_log_s = 0.0
    last_cmd_log_s = 0.0
    last_vision_meta_update_s = 0.0
    last_loop_t = time.perf_counter()
    last_result_arrival_t: Optional[float] = None
    last_result_ts_ns: Optional[int] = None
    last_result_seq_id: int = 0
    robot_pose_history: deque[Tuple[float, List[float], List[float]]] = deque(
        maxlen=900
    )
    last_vision_has_matches = False
    last_vision_reject_reason: Optional[str] = None
    last_top_match_has_xyz = False
    last_top_match_score: Optional[float] = None

    try:
        while True:
            _ensure_stop(handle)
            loop_t0 = time.perf_counter()
            now_s = time.time()
            dt = max(1e-4, loop_t0 - last_loop_t)
            last_loop_t = loop_t0

            if hand_eye_frame == "camera_in_flange":
                cur_p, cur_q = _flange_pose(ctx.robot)
            else:
                cur_p, cur_q = _tcp_pose(ctx.robot)
            robot_pose_history.append((now_s, list(cur_p), list(cur_q)))

            evt = ctx.vision_cache.get_latest(request_id) or {}
            if isinstance(evt, dict):
                frame_id = str(evt.get("frame_id") or "")
                seq_raw = evt.get("sequence_id")
                ts_raw = evt.get("timestamp_ns")
                try:
                    seq_id = int(seq_raw) if seq_raw is not None else 0
                except Exception:
                    seq_id = 0
                try:
                    ts_ns = int(ts_raw) if ts_raw is not None else None
                except Exception:
                    ts_ns = None

                is_new_result = False
                if seq_id > 0:
                    if seq_id > last_result_seq_id:
                        is_new_result = True
                    elif (
                        isinstance(ts_ns, int)
                        and isinstance(last_result_ts_ns, int)
                        and ts_ns > last_result_ts_ns
                        and (ts_ns - last_result_ts_ns) >= int(5e9)
                    ):
                        # Camera sequence reset after reconnect.
                        is_new_result = True
                elif frame_id and frame_id != last_frame_id:
                    is_new_result = True

                if is_new_result:
                    # Process each frame_id exactly once to avoid duplicate control updates.
                    last_frame_id = frame_id
                    if seq_id > 0:
                        last_result_seq_id = seq_id
                    if (now_s - last_vision_meta_update_s) >= cfg.log_interval_s:
                        ctx.run_manager.set_last_vision(
                            run_id,
                            request_id=request_id,
                            frame_id=frame_id,
                            timestamp_ns=evt.get("timestamp_ns"),
                        )
                        last_vision_meta_update_s = now_s
                    result = (
                        evt.get("result") if isinstance(evt.get("result"), dict) else {}
                    )
                    matches = (
                        result.get("matches")
                        if isinstance(result.get("matches"), list)
                        else []
                    )
                    last_vision_has_matches = bool(matches)
                    last_vision_reject_reason = result.get("reject_reason")
                    vision_epoch_s = float(ts_ns) / 1e9 if ts_ns is not None else None

                    produced_ns = result.get("produced_timestamp_ns")
                    produced_latency_ms = None
                    if ts_ns is not None and produced_ns is not None:
                        produced_latency_ms = max(
                            0.0, (float(produced_ns) - float(ts_ns)) / 1e6
                        )

                    vision_age_s = (
                        (now_s - vision_epoch_s) if vision_epoch_s is not None else None
                    )
                    # Estimate result FPS from source frame timestamps (not loop arrival time),
                    # so it tracks actual camera/result cadence and avoids >camera-rate spikes.
                    if isinstance(ts_ns, int):
                        if last_result_ts_ns is not None and ts_ns > last_result_ts_ns:
                            dt_result = (ts_ns - last_result_ts_ns) / 1e9
                            if dt_result > 1e-6:
                                inst_fps = 1.0 / dt_result
                                if 0.0 < inst_fps < 200.0:
                                    if rt.result_fps_hz is None:
                                        rt.result_fps_hz = inst_fps
                                    else:
                                        rt.result_fps_hz = (
                                            0.8 * rt.result_fps_hz + 0.2 * inst_fps
                                        )
                        last_result_ts_ns = ts_ns
                    elif last_result_arrival_t is not None:
                        dt_result = loop_t0 - last_result_arrival_t
                        if dt_result > 1e-6:
                            inst_fps = 1.0 / dt_result
                            if 0.0 < inst_fps < 200.0:
                                if rt.result_fps_hz is None:
                                    rt.result_fps_hz = inst_fps
                                else:
                                    rt.result_fps_hz = (
                                        0.8 * rt.result_fps_hz + 0.2 * inst_fps
                                    )
                    last_result_arrival_t = loop_t0

                    top = matches[0] if matches else {}
                    last_top_match_has_xyz = (
                        _is_finite_vec(top.get("center_xyz_m"), 3) if matches else False
                    )
                    try:
                        last_top_match_score = (
                            float(top.get("score"))
                            if matches and top.get("score") is not None
                            else None
                        )
                    except Exception:
                        last_top_match_score = None
                    if matches:
                        match = matches[0]
                        target_cam = match.get("center_xyz_m")
                        if _is_finite_vec(target_cam, 3):
                            # Reconstruct base target at the same timestamp as vision frame.
                            tf_pos, tf_quat, tf_pose_dt_s = _lookup_pose_at_time(
                                robot_pose_history, vision_epoch_s, cur_p, cur_q
                            )
                            if hand_eye_frame in ("base_to_camera", "base"):
                                base_to_cam = hand_eye
                            else:
                                base_to_ee = {
                                    "translation_m": tf_pos,
                                    "rotation_quat_xyzw": tf_quat,
                                }
                                base_to_cam = _compose_transform(base_to_ee, hand_eye)

                            target_base = _transform_point(target_cam, base_to_cam)
                            if _is_finite_vec(target_base, 3):
                                target_base = [
                                    float(target_base[0]),
                                    float(target_base[1]),
                                    float(target_base[2]),
                                ]
                                normal_base = None
                                normal_cam = match.get(
                                    "surface_normal_cam"
                                ) or match.get("normal_cam")
                                if cfg.align_with_surface and _is_finite_vec(
                                    normal_cam, 3
                                ):
                                    try:
                                        n_base = _quat_rotate(
                                            base_to_cam.get(
                                                "rotation_quat_xyzw",
                                                [0.0, 0.0, 0.0, 1.0],
                                            ),
                                            normal_cam,
                                        )
                                        nn = _norm3(
                                            [
                                                float(n_base[0]),
                                                float(n_base[1]),
                                                float(n_base[2]),
                                            ]
                                        )
                                        if nn > 1e-9:
                                            normal_base = [
                                                n_base[0] / nn,
                                                n_base[1] / nn,
                                                n_base[2] / nn,
                                            ]
                                    except Exception:
                                        normal_base = None

                                if cfg.align_with_surface and _is_finite_vec(
                                    normal_base, 3
                                ):
                                    target_base = [
                                        target_base[0]
                                        + normal_base[0] * cfg.hover_height_m,
                                        target_base[1]
                                        + normal_base[1] * cfg.hover_height_m,
                                        target_base[2]
                                        + normal_base[2] * cfg.hover_height_m,
                                    ]
                                else:
                                    target_base[2] += cfg.hover_height_m

                                raw_yaw = match.get("yaw_deg")
                                vision_yaw = None
                                if raw_yaw is not None:
                                    try:
                                        y = _normalize_angle_deg(
                                            float(raw_yaw) * cfg.yaw_sign
                                            + cfg.yaw_offset_deg
                                        )
                                        if cfg.smart_yaw:
                                            y = _wrap_yaw_90(y)
                                        vision_yaw = y
                                    except Exception:
                                        vision_yaw = None

                                rt.measured_target_base = target_base
                                rt.measured_target_cam = [
                                    float(target_cam[0]),
                                    float(target_cam[1]),
                                    float(target_cam[2]),
                                ]
                                rt.measured_yaw_deg = vision_yaw
                                rt.measured_normal_base = (
                                    normal_base
                                    if _is_finite_vec(normal_base, 3)
                                    else None
                                )
                                rt.measured_epoch_s = now_s
                                rt.measured_frame_id = frame_id

                                if isinstance(ts_ns, int):
                                    if rt.last_derived_tf_ts_ns > 0 and ts_ns > rt.last_derived_tf_ts_ns:
                                        dt_tf = (ts_ns - rt.last_derived_tf_ts_ns) / 1e9
                                        if dt_tf > 1e-6:
                                            inst_tf_fps = 1.0 / dt_tf
                                            if (
                                                rt.result_fps_hz is not None
                                                and rt.result_fps_hz > 0
                                            ):
                                                inst_tf_fps = min(
                                                    inst_tf_fps, rt.result_fps_hz * 1.15
                                                )
                                            if 0.0 < inst_tf_fps < 200.0:
                                                if rt.derived_tf_fps_hz is None:
                                                    rt.derived_tf_fps_hz = inst_tf_fps
                                                else:
                                                    rt.derived_tf_fps_hz = (
                                                        0.8 * rt.derived_tf_fps_hz
                                                        + 0.2 * inst_tf_fps
                                                    )
                                    if rt.last_derived_tf_ts_ns <= 0 or ts_ns > rt.last_derived_tf_ts_ns:
                                        rt.last_derived_tf_ts_ns = ts_ns
                                elif rt.last_derived_tf_update_s > 0.0:
                                    dt_tf = now_s - rt.last_derived_tf_update_s
                                    if dt_tf > 1e-6:
                                        inst_tf_fps = 1.0 / dt_tf
                                        if 0.0 < inst_tf_fps < 200.0:
                                            if rt.derived_tf_fps_hz is None:
                                                rt.derived_tf_fps_hz = inst_tf_fps
                                            else:
                                                rt.derived_tf_fps_hz = (
                                                    0.8 * rt.derived_tf_fps_hz
                                                    + 0.2 * inst_tf_fps
                                                )
                                rt.last_derived_tf_update_s = now_s

                                if (now_s - last_tf_log_s) >= cfg.log_interval_s:
                                    _log(
                                        ctx,
                                        run_id,
                                        "FOLLOW_V3_TF",
                                        frame_id=frame_id,
                                        hand_eye_frame=hand_eye_frame,
                                        base_to_cam_translation_m=base_to_cam.get(
                                            "translation_m"
                                        ),
                                        base_to_cam_quat_xyzw=base_to_cam.get(
                                            "rotation_quat_xyzw"
                                        ),
                                        tf_robot_tcp_m=tf_pos,
                                        tf_robot_quat_xyzw=tf_quat,
                                        tf_pose_dt_s=tf_pose_dt_s,
                                        object_cam_m=rt.measured_target_cam,
                                        object_base_m=rt.measured_target_base,
                                        measured_yaw_deg=vision_yaw,
                                        measured_normal_base=rt.measured_normal_base,
                                        result_fps_hz=rt.result_fps_hz,
                                        derived_tf_fps_hz=rt.derived_tf_fps_hz,
                                        vision_age_s=vision_age_s,
                                    )
                                    last_tf_log_s = now_s
                        elif rt.measured_target_base is not None:
                            # Keep existing 3D target alive briefly when 2D match exists but depth is missing.
                            rt.measured_epoch_s = now_s
                            rt.measured_frame_id = frame_id

            # Direct-drive mode: keep commanding from the latest computed target.
            # This avoids frequent stop/start when some vision frames miss.
            target_age_s = None
            if rt.measured_epoch_s > 0:
                target_age_s = max(0.0, now_s - rt.measured_epoch_s)
            target_fresh = rt.measured_target_base is not None and (
                target_age_s is None
                or target_age_s <= (cfg.max_target_age_s + cfg.dropout_grace_s)
            )
            if not target_fresh and cfg.clear_target_on_lost:
                rt.measured_target_base = None
                rt.measured_target_cam = None
                rt.measured_yaw_deg = None
                rt.measured_normal_base = None
                rt.ema_target_base = None
                rt.ema_yaw_deg = None
                rt.predict_hist.clear()

            if target_fresh:
                # Pure direct feed from latest measured target, optionally EMA-smoothed.
                raw_tgt_p = list(rt.measured_target_base)
                pred_vel = [0.0, 0.0, 0.0]
                if rt.measured_epoch_s > 0:
                    rt.predict_hist.append((rt.measured_epoch_s, list(raw_tgt_p)))

                # Predictive target based on last N measured targets.
                if cfg.mode == "predictive" and len(rt.predict_hist) >= 2:
                    hist_n = max(2, min(cfg.predictive_history_n, len(rt.predict_hist)))
                    h = list(rt.predict_hist)[-hist_n:]
                    t0, p0 = h[0]
                    t1, p1 = h[-1]
                    dt_hist = max(1e-6, t1 - t0)
                    pred_vel = [
                        (p1[0] - p0[0]) / dt_hist,
                        (p1[1] - p0[1]) / dt_hist,
                        (p1[2] - p0[2]) / dt_hist,
                    ]
                    if cfg.predictive_lookahead_s > 0.0:
                        lookahead_s = cfg.predictive_lookahead_s
                    else:
                        lookahead_s = max(0.0, min(0.2, dt_hist / max(1, len(h) - 1)))
                    raw_tgt_p = [
                        p1[0] + pred_vel[0] * lookahead_s,
                        p1[1] + pred_vel[1] * lookahead_s,
                        p1[2] + pred_vel[2] * lookahead_s,
                    ]
                raw_yaw_src = rt.measured_yaw_deg
                if cfg.simple_ema_enabled:
                    if rt.ema_target_base is None:
                        rt.ema_target_base = list(raw_tgt_p)
                    else:
                        a_pos = _clamp(cfg.ema_alpha_pos, 0.0, 1.0)
                        rt.ema_target_base = [
                            (1.0 - a_pos) * rt.ema_target_base[0]
                            + a_pos * raw_tgt_p[0],
                            (1.0 - a_pos) * rt.ema_target_base[1]
                            + a_pos * raw_tgt_p[1],
                            (1.0 - a_pos) * rt.ema_target_base[2]
                            + a_pos * raw_tgt_p[2],
                        ]
                    tgt_p = list(rt.ema_target_base)

                    if raw_yaw_src is not None:
                        if rt.ema_yaw_deg is None:
                            rt.ema_yaw_deg = float(raw_yaw_src)
                        else:
                            a_yaw = _clamp(cfg.ema_alpha_yaw, 0.0, 1.0)
                            dy = _normalize_angle_deg(
                                float(raw_yaw_src) - rt.ema_yaw_deg
                            )
                            rt.ema_yaw_deg = _normalize_angle_deg(
                                rt.ema_yaw_deg + a_yaw * dy
                            )
                    yaw_src = (
                        rt.ema_yaw_deg if rt.ema_yaw_deg is not None else raw_yaw_src
                    )
                else:
                    rt.ema_target_base = None
                    rt.ema_yaw_deg = None
                    tgt_p = raw_tgt_p
                    yaw_src = raw_yaw_src

                robot_yaw = _quat_to_yaw_deg(cur_q)
                yaw_cmd = _compute_yaw_cmd_deg(cfg, rt, yaw_src, robot_yaw)
                tgt_q = _build_target_quat(
                    cfg.roll_fixed_deg, cfg.pitch_fixed_deg, yaw_cmd
                )

                if actual_mode == "velocity":
                    err = [
                        tgt_p[0] - cur_p[0],
                        tgt_p[1] - cur_p[1],
                        tgt_p[2] - cur_p[2],
                    ]
                    err_norm = _norm3(err)
                    yaw_err = _normalize_angle_deg(yaw_cmd - robot_yaw)

                    prev_phase = rt.control_phase
                    if rt.control_phase == "track":
                        if err_norm >= cfg.reach_error_release_m:
                            rt.control_phase = "reach"
                    else:
                        if err_norm <= cfg.reach_error_enter_m:
                            rt.control_phase = "track"
                    if rt.control_phase != prev_phase:
                        # Prevent integrator carry-over when switching between reach/track gains.
                        rt.int_err = [0.0, 0.0, 0.0]
                        rt.int_yaw_err = 0.0
                        rt.prev_err = None
                        rt.prev_yaw_err = None

                    if rt.control_phase == "reach":
                        kp_lin = cfg.reach_velocity_kp
                        ki_lin = cfg.reach_velocity_ki
                        kd_lin = cfg.reach_velocity_kd
                        max_lin = cfg.reach_max_vel_mps
                        kp_yaw = cfg.reach_yaw_velocity_kp
                        ki_yaw = cfg.reach_yaw_velocity_ki
                        kd_yaw = cfg.reach_yaw_velocity_kd
                        max_yaw = cfg.reach_max_yaw_vel_dps
                    else:
                        kp_lin = cfg.track_velocity_kp
                        ki_lin = cfg.track_velocity_ki
                        kd_lin = cfg.track_velocity_kd
                        max_lin = cfg.track_max_vel_mps
                        kp_yaw = cfg.track_yaw_velocity_kp
                        ki_yaw = cfg.track_yaw_velocity_ki
                        kd_yaw = cfg.track_yaw_velocity_kd
                        max_yaw = cfg.track_max_yaw_vel_dps

                    # Simple proportional velocity with speed clamp + deadband.
                    if rt.settled_hold:
                        if err_norm >= cfg.pos_release_m:
                            rt.settled_hold = False
                    elif err_norm <= cfg.pos_deadband_m:
                        rt.settled_hold = True

                    lin = [0.0, 0.0, 0.0]
                    if not rt.settled_hold:
                        derr = [0.0, 0.0, 0.0]
                        if rt.prev_err is not None and dt > 1e-6:
                            derr = [
                                (err[0] - rt.prev_err[0]) / dt,
                                (err[1] - rt.prev_err[1]) / dt,
                                (err[2] - rt.prev_err[2]) / dt,
                            ]
                        # Integrator with anti-windup (limit integral contribution to 60% of max velocity).
                        if ki_lin > 1e-9:
                            i_term_limit_lin = 0.6 * max_lin
                            i_cap_lin = i_term_limit_lin / ki_lin
                            rt.int_err = [
                                _clamp(
                                    rt.int_err[0] + err[0] * dt, -i_cap_lin, i_cap_lin
                                ),
                                _clamp(
                                    rt.int_err[1] + err[1] * dt, -i_cap_lin, i_cap_lin
                                ),
                                _clamp(
                                    rt.int_err[2] + err[2] * dt, -i_cap_lin, i_cap_lin
                                ),
                            ]
                        else:
                            rt.int_err = [0.0, 0.0, 0.0]
                        lin = [
                            err[0] * kp_lin + rt.int_err[0] * ki_lin + derr[0] * kd_lin,
                            err[1] * kp_lin + rt.int_err[1] * ki_lin + derr[1] * kd_lin,
                            err[2] * kp_lin + rt.int_err[2] * ki_lin + derr[2] * kd_lin,
                        ]
                        if cfg.velocity_ff_gain > 1e-9:
                            lin = [
                                lin[0] + pred_vel[0] * cfg.velocity_ff_gain,
                                lin[1] + pred_vel[1] * cfg.velocity_ff_gain,
                                lin[2] + pred_vel[2] * cfg.velocity_ff_gain,
                            ]
                        v = _norm3(lin)
                        if v > max_lin and v > 1e-9:
                            s = max_lin / v
                            lin = [lin[0] * s, lin[1] * s, lin[2] * s]
                    else:
                        rt.int_err = [0.0, 0.0, 0.0]

                    yaw_derr = 0.0
                    if rt.prev_yaw_err is not None and dt > 1e-6:
                        dy = _normalize_angle_deg(yaw_err - rt.prev_yaw_err)
                        yaw_derr = dy / dt
                    if ki_yaw > 1e-9 and not rt.settled_hold:
                        i_term_limit_yaw = 0.6 * max_yaw
                        i_cap_yaw = i_term_limit_yaw / ki_yaw
                        rt.int_yaw_err = _clamp(
                            rt.int_yaw_err + yaw_err * dt, -i_cap_yaw, i_cap_yaw
                        )
                    else:
                        rt.int_yaw_err = 0.0
                    yaw_vel = _clamp(
                        yaw_err * kp_yaw + rt.int_yaw_err * ki_yaw + yaw_derr * kd_yaw,
                        -max_yaw,
                        max_yaw,
                    )
                    if abs(yaw_err) <= cfg.yaw_deadband_deg:
                        yaw_vel = 0.0
                    rt.prev_err = list(err)
                    rt.prev_yaw_err = float(yaw_err)

                    ctx.robot.servo_tcp_velocity(
                        {
                            "linear_mps": lin,
                            "angular_dps": [0.0, 0.0, yaw_vel],
                            "frame": "base",
                        },
                        vel_profile,
                    )
                    if (now_s - last_cmd_log_s) >= cfg.log_interval_s:
                        _log(
                            ctx,
                            run_id,
                            "ROBOT_CMD",
                            cmd_mode="velocity",
                            linear_mps=lin,
                            cmd_speed_mps=_norm3(lin),
                            yaw_velocity_dps=yaw_vel,
                            yaw_error_deg=yaw_err,
                            yaw_robot_deg=robot_yaw,
                            target_base_m=[
                                float(tgt_p[0]),
                                float(tgt_p[1]),
                                float(tgt_p[2]),
                            ],
                            robot_base_m=[
                                float(cur_p[0]),
                                float(cur_p[1]),
                                float(cur_p[2]),
                            ],
                            error_base_m=[float(err[0]), float(err[1]), float(err[2])],
                            error_norm_m=err_norm,
                            yaw_cmd_deg=round(yaw_cmd, 3),
                            ff_linear_mps=[
                                float(pred_vel[0] * cfg.velocity_ff_gain),
                                float(pred_vel[1] * cfg.velocity_ff_gain),
                                float(pred_vel[2] * cfg.velocity_ff_gain),
                            ],
                            control_phase=rt.control_phase,
                            settled_hold=rt.settled_hold,
                            frame_id=rt.measured_frame_id,
                            vision_age_s=(
                                (now_s - rt.measured_epoch_s)
                                if rt.measured_epoch_s > 0
                                else None
                            ),
                        )
                        last_cmd_log_s = now_s
                else:
                    # Step-limited position control to avoid overshoot / SDK errors from large jumps.
                    err = [
                        tgt_p[0] - cur_p[0],
                        tgt_p[1] - cur_p[1],
                        tgt_p[2] - cur_p[2],
                    ]
                    step = [
                        err[0] * cfg.gain_pos,
                        err[1] * cfg.gain_pos,
                        err[2] * cfg.gain_pos,
                    ]
                    xy = math.sqrt(step[0] * step[0] + step[1] * step[1])
                    if xy > cfg.max_step_xy_m and xy > 1e-9:
                        s = cfg.max_step_xy_m / xy
                        step[0] *= s
                        step[1] *= s
                    step[2] = _clamp(step[2], -cfg.max_step_z_m, cfg.max_step_z_m)
                    nxt = [cur_p[0] + step[0], cur_p[1] + step[1], cur_p[2] + step[2]]
                    ctx.robot.servo_tcp(
                        {
                            "position_m": [float(nxt[0]), float(nxt[1]), float(nxt[2])],
                            "quat_xyzw": tgt_q,
                            "frame": "base",
                        },
                        pos_profile,
                    )
                    if (now_s - last_cmd_log_s) >= cfg.log_interval_s:
                        _log(
                            ctx,
                            run_id,
                            "ROBOT_CMD",
                            cmd_mode="position",
                            cmd_step_m=_norm3(step),
                            target_base_m=[float(nxt[0]), float(nxt[1]), float(nxt[2])],
                            target_goal_base_m=[
                                float(tgt_p[0]),
                                float(tgt_p[1]),
                                float(tgt_p[2]),
                            ],
                            robot_base_m=[
                                float(cur_p[0]),
                                float(cur_p[1]),
                                float(cur_p[2]),
                            ],
                            error_base_m=[float(err[0]), float(err[1]), float(err[2])],
                            error_norm_m=_norm3(err),
                            yaw_cmd_deg=round(yaw_cmd, 3),
                            frame_id=rt.measured_frame_id,
                            vision_age_s=(
                                (now_s - rt.measured_epoch_s)
                                if rt.measured_epoch_s > 0
                                else None
                            ),
                        )
                        last_cmd_log_s = now_s
            else:
                rt.settled_hold = False
                rt.prev_err = None
                rt.prev_yaw_err = None
                rt.int_err = [0.0, 0.0, 0.0]
                rt.int_yaw_err = 0.0
                rt.control_phase = "reach"

                if (now_s - last_no_target_log) >= cfg.log_interval_s:
                    _log(
                        ctx,
                        run_id,
                        "FOLLOW_V3_NO_TARGET",
                        frame_id=rt.measured_frame_id or last_frame_id,
                        target_is_fresh=False,
                        measured_age_s=target_age_s,
                        max_target_age_s=cfg.max_target_age_s,
                        dropout_grace_s=cfg.dropout_grace_s,
                        last_vision_has_matches=last_vision_has_matches,
                        last_top_match_has_xyz=last_top_match_has_xyz,
                        last_top_match_score=last_top_match_score,
                        last_vision_reject_reason=last_vision_reject_reason,
                        robot_base_m=[
                            float(cur_p[0]),
                            float(cur_p[1]),
                            float(cur_p[2]),
                        ],
                        lost_behavior=cfg.lost_behavior,
                    )
                    last_no_target_log = now_s

                if actual_mode == "velocity" and cfg.lost_behavior == "stop_velocity":
                    ctx.robot.servo_tcp_velocity(
                        {
                            "linear_mps": [0.0, 0.0, 0.0],
                            "angular_dps": [0.0, 0.0, 0.0],
                            "frame": "base",
                        },
                        vel_profile,
                    )
                elif actual_mode == "position":
                    ctx.robot.servo_tcp(
                        {"position_m": cur_p, "quat_xyzw": cur_q, "frame": "base"},
                        pos_profile,
                    )

            dt_target = 1.0 / max(1e-6, cfg.rate_hz)
            elapsed = time.perf_counter() - loop_t0
            if dt_target > elapsed:
                time.sleep(dt_target - elapsed)

    finally:
        try:
            ctx.vision.stop_session(request_id)
        except Exception:
            pass
        if getattr(handle, "vision_request_id", None) == request_id:
            handle.vision_request_id = None
        try:
            if actual_mode == "velocity" and hasattr(ctx.robot, "servo_tcp_velocity"):
                ctx.robot.servo_tcp_velocity(
                    {
                        "linear_mps": [0.0, 0.0, 0.0],
                        "angular_dps": [0.0, 0.0, 0.0],
                        "frame": "base",
                    },
                    vel_profile,
                )
        except Exception:
            pass
        try:
            if hasattr(ctx.robot, "set_mode"):
                ctx.robot.set_mode(0)
            if hasattr(ctx.robot, "set_state"):
                ctx.robot.set_state(0)
        except Exception:
            pass
