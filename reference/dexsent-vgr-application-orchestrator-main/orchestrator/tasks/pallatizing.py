"""Implementation for `orchestrator.tasks.pallatizing`."""

import math
import time
from typing import Any, Dict, List, Tuple

from orchestrator.core.context import StationContext
from orchestrator.core.runs import RunState
from orchestrator.tasks.pick_place import (
    _add_vec,
    _apply_yaw,
    _compose_transform,
    _ensure_stop,
    _estimate_linear_velocity,
    _evt_timestamp_s,
    _get_pose_quat,
    _is_finite_vec,
    _log,
    _match_ok,
    _move_to_pose,
    _normalize_quat_xyzw,
    _pose_missing,
    _quat_rotate,
    _robot_disabled,
    _run_pick_vision_only_session,
    _vision_timeout_s,
    _resolve_hand_eye,
    _resolve_runtime_handeye,
    _resolve_task_payload,
    _resolve_templates_dir,
    _rpy_deg_to_quat_xyzw,
    _transform_point,
    _wait_for_match,
    _workspace_ok,
)
from orchestrator.vision.calibration import build_station_camera_calibration


def _norm3(v: List[float]) -> float:
    return math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])


def _median(values: List[float]) -> float:
    vals = sorted(float(v) for v in values)
    n = len(vals)
    if n <= 0:
        return 0.0
    mid = n // 2
    if (n % 2) == 1:
        return vals[mid]
    return 0.5 * (vals[mid - 1] + vals[mid])


def _estimate_stable_depth_m(
    depth_samples_m: List[float],
    min_samples: int,
    mad_scale: float,
    min_tol_m: float,
) -> float:
    vals = [
        float(v)
        for v in depth_samples_m
        if isinstance(v, (int, float)) and math.isfinite(float(v))
    ]
    if not vals:
        return 0.0
    center = _median(vals)
    if len(vals) < max(1, int(min_samples)):
        return center
    abs_dev = [abs(v - center) for v in vals]
    mad = _median(abs_dev)
    tol = max(float(min_tol_m), float(mad_scale) * mad)
    if tol <= 0.0:
        return center
    inliers = [v for v in vals if abs(v - center) <= tol]
    if len(inliers) >= max(1, int(min_samples)):
        return _median(inliers)
    return center


def _validate_track_window(
    samples: List[Tuple[float, List[float]]],
    yaw_samples: List[float],
    min_samples: int,
    max_fit_residual_m: float,
    max_fit_z_residual_m: float,
    max_yaw_residual_deg: float,
) -> Tuple[bool, str, Dict[str, Any]]:
    """Check whether recent track samples are physically consistent.

    The goal is to reject garbage bursts (false matches) before robot motion:
    - fit residual in XY/Z must stay within limits,
    - yaw samples must remain self-consistent.
    """
    if len(samples) < max(2, int(min_samples)):
        return False, "insufficient_samples", {"count": len(samples)}
    vel, _window_s = _estimate_linear_velocity(samples)
    t0 = float(samples[0][0])
    p0 = samples[0][1]
    res_xy: List[float] = []
    res_z: List[float] = []
    for ts, p in samples:
        dt = float(ts) - t0
        pred = [
            float(p0[0]) + vel[0] * dt,
            float(p0[1]) + vel[1] * dt,
            float(p0[2]) + vel[2] * dt,
        ]
        dx = float(p[0]) - pred[0]
        dy = float(p[1]) - pred[1]
        dz = float(p[2]) - pred[2]
        res_xy.append(math.sqrt(dx * dx + dy * dy))
        res_z.append(abs(dz))
    med_xy = _median(res_xy)
    med_z = _median(res_z)
    if med_xy > float(max_fit_residual_m):
        return (
            False,
            "fit_residual_xy_high",
            {"median_xy_residual_m": med_xy, "median_z_residual_m": med_z},
        )
    if med_z > float(max_fit_z_residual_m):
        return (
            False,
            "fit_residual_z_high",
            {"median_xy_residual_m": med_xy, "median_z_residual_m": med_z},
        )

    yaws = [
        float(v)
        for v in yaw_samples
        if isinstance(v, (int, float)) and math.isfinite(float(v))
    ]
    if len(yaws) >= max(2, int(min_samples)):
        center = _estimate_stable_yaw_deg(yaws, 180.0, 2)
        yaw_res = [abs(_angle_delta_deg(v, center)) for v in yaws]
        med_yaw = _median(yaw_res)
        if med_yaw > float(max_yaw_residual_deg):
            return (
                False,
                "yaw_residual_high",
                {
                    "median_xy_residual_m": med_xy,
                    "median_z_residual_m": med_z,
                    "median_yaw_residual_deg": med_yaw,
                },
            )
        return (
            True,
            "ok",
            {
                "median_xy_residual_m": med_xy,
                "median_z_residual_m": med_z,
                "median_yaw_residual_deg": med_yaw,
            },
        )
    return True, "ok", {"median_xy_residual_m": med_xy, "median_z_residual_m": med_z}


def _lerp(a: List[float], b: List[float], t: float) -> List[float]:
    t = max(0.0, min(1.0, t))
    return [
        a[0] + (b[0] - a[0]) * t,
        a[1] + (b[1] - a[1]) * t,
        a[2] + (b[2] - a[2]) * t,
    ]


def _sleep_rate(rate_hz: float, tick_t: float) -> None:
    if rate_hz <= 1e-3:
        return
    dt = 1.0 / rate_hz
    rem = dt - (time.perf_counter() - tick_t)
    if rem > 1e-4:
        time.sleep(rem)


def _quat_dot(a: List[float], b: List[float]) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3]


def _quat_slerp(a: List[float], b: List[float], t: float) -> List[float]:
    ta = max(0.0, min(1.0, float(t)))
    qa = _normalize_quat_xyzw(list(a))
    qb = _normalize_quat_xyzw(list(b))
    dot = _quat_dot(qa, qb)
    if dot < 0.0:
        qb = [-qb[0], -qb[1], -qb[2], -qb[3]]
        dot = -dot
    if dot > 0.9995:
        out = [
            qa[0] + ta * (qb[0] - qa[0]),
            qa[1] + ta * (qb[1] - qa[1]),
            qa[2] + ta * (qb[2] - qa[2]),
            qa[3] + ta * (qb[3] - qa[3]),
        ]
        return _normalize_quat_xyzw(out)
    theta_0 = math.acos(max(-1.0, min(1.0, dot)))
    sin_theta_0 = math.sin(theta_0)
    if abs(sin_theta_0) < 1e-9:
        return qa
    theta = theta_0 * ta
    sin_theta = math.sin(theta)
    s0 = math.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0
    out = [
        s0 * qa[0] + s1 * qb[0],
        s0 * qa[1] + s1 * qb[1],
        s0 * qa[2] + s1 * qb[2],
        s0 * qa[3] + s1 * qb[3],
    ]
    return _normalize_quat_xyzw(out)


def _normalize_angle_deg(angle_deg: float) -> float:
    return ((float(angle_deg) + 180.0) % 360.0) - 180.0


def _quat_to_rpy_deg(quat_xyzw: List[float]) -> List[float]:
    x, y, z, w = [
        float(quat_xyzw[0]),
        float(quat_xyzw[1]),
        float(quat_xyzw[2]),
        float(quat_xyzw[3]),
    ]
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1.0:
        pitch = math.copysign(math.pi / 2.0, sinp)
    else:
        pitch = math.asin(sinp)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return [math.degrees(roll), math.degrees(pitch), math.degrees(yaw)]


def _pose_debug(
    position_m: List[float], quat_xyzw: List[float], frame: str = "base"
) -> Dict[str, Any]:
    qn = _normalize_quat_xyzw(list(quat_xyzw))
    rpy = _quat_to_rpy_deg(qn)
    return {
        "position_m": [
            float(position_m[0]),
            float(position_m[1]),
            float(position_m[2]),
        ],
        "quat_xyzw": [float(qn[0]), float(qn[1]), float(qn[2]), float(qn[3])],
        "rpy_deg": [float(rpy[0]), float(rpy[1]), float(rpy[2])],
        "frame": str(frame or "base"),
    }


def _wrap_yaw_90(yaw_deg: float) -> float:
    wrapped = _normalize_angle_deg(yaw_deg)
    if wrapped > 90.0:
        wrapped -= 180.0
    elif wrapped < -90.0:
        wrapped += 180.0
    return wrapped


def _angle_delta_deg(a_deg: float, b_deg: float) -> float:
    return _normalize_angle_deg(float(a_deg) - float(b_deg))


def _circular_mean_deg(values_deg: List[float]) -> float:
    if not values_deg:
        return 0.0
    s = 0.0
    c = 0.0
    for v in values_deg:
        r = math.radians(float(v))
        s += math.sin(r)
        c += math.cos(r)
    if abs(s) < 1e-9 and abs(c) < 1e-9:
        return _normalize_angle_deg(values_deg[-1])
    return _normalize_angle_deg(math.degrees(math.atan2(s, c)))


def _yaw_from_match(match: Dict[str, Any], axis_correction: bool) -> float:
    yaw = float(match.get("yaw_deg", 0.0))
    yaw = _normalize_angle_deg(yaw)
    if not axis_correction:
        return yaw

    # Correct occasional 90-degree ambiguity when homography axis assignment flips.
    tw = float(match.get("template_width", 0.0) or 0.0)
    th = float(match.get("template_height", 0.0) or 0.0)
    dw = float(match.get("detected_width", 0.0) or 0.0)
    dh = float(match.get("detected_height", 0.0) or 0.0)
    if tw > 1e-6 and th > 1e-6 and dw > 1e-6 and dh > 1e-6:
        t_ratio = max(1e-6, tw / th)
        curr_ratio = max(1e-6, dw / dh)
        swap_ratio = max(1e-6, dh / dw)
        curr_err = abs(math.log(curr_ratio / t_ratio))
        swap_err = abs(math.log(swap_ratio / t_ratio))
        if swap_err + 1e-6 < curr_err:
            yaw = _normalize_angle_deg(yaw + 90.0)
    return yaw


def _estimate_stable_yaw_deg(
    yaw_samples: List[float], reject_deg: float, min_samples: int
) -> float:
    vals = [
        float(v)
        for v in yaw_samples
        if isinstance(v, (int, float)) and math.isfinite(float(v))
    ]
    if not vals:
        return 0.0
    vals = [_normalize_angle_deg(v) for v in vals]
    if len(vals) == 1:
        return vals[0]

    center = _circular_mean_deg(vals)
    tol = max(1.0, float(reject_deg))
    keep = [v for v in vals if abs(_angle_delta_deg(v, center)) <= tol]
    if len(keep) >= max(2, int(min_samples)):
        center = _circular_mean_deg(keep)
    else:
        keep = vals

    unwrapped = [center + _angle_delta_deg(v, center) for v in keep]
    unwrapped.sort()
    mid = len(unwrapped) // 2
    if len(unwrapped) % 2 == 1:
        med = unwrapped[mid]
    else:
        med = 0.5 * (unwrapped[mid - 1] + unwrapped[mid])
    return _normalize_angle_deg(med)


def _move_to_capture_pose(
    ctx: StationContext,
    capture_pose: Dict[str, Any],
    profile: str,
    mode: str = "auto",
) -> None:
    """
    Pallatizing-specific capture return:
    - Forces requested profile (ignores pose.profile), and
    - Can prefer linear TCP move to avoid slow joint-only returns.
    """
    move_mode = str(mode or "auto").strip().lower()
    tcp_target = capture_pose.get("tcp") or capture_pose.get("tcp_pose") or {}
    tcp_pos = tcp_target.get("position_m")
    tcp_quat = tcp_target.get("quat_xyzw", [0.0, 0.0, 0.0, 1.0])
    tcp_frame = tcp_target.get("frame", "base")

    use_tcp = move_mode in ("auto", "tcp", "linear", "movel")
    if use_tcp and _is_finite_vec(tcp_pos, 3):
        ctx.robot.movel(
            {
                "position_m": [float(tcp_pos[0]), float(tcp_pos[1]), float(tcp_pos[2])],
                "quat_xyzw": [
                    float(tcp_quat[0]),
                    float(tcp_quat[1]),
                    float(tcp_quat[2]),
                    float(tcp_quat[3]),
                ],
                "frame": str(tcp_frame or "base"),
            },
            profile,
        )
        return

    if "joints" in capture_pose:
        ctx.robot.movej(tuple(capture_pose["joints"]), profile)
        return

    # Final fallback keeps compatibility with existing pose shapes.
    _move_to_pose(ctx, capture_pose, profile)


def run_pallatizing(ctx: StationContext, state: RunState, handle: Any) -> None:
    """Conveyor pick/place with velocity-aware interception.

    Compared to plain pick/place, this task:
    - gathers a short track window,
    - estimates object velocity,
    - predicts intercept timing/position,
    - supports continuous (servo) pickup trajectories.
    """
    recipe = _resolve_task_payload(ctx, state)
    run_id = state.run_id
    vision_cfg = recipe.get("vision", {}) if isinstance(recipe, dict) else {}
    robot_cfg = recipe.get("robot", {}) if isinstance(recipe, dict) else {}
    pick_cfg = robot_cfg.get("pick", {}) if isinstance(robot_cfg, dict) else {}
    pall_cfg = recipe.get("pallatizing", {}) if isinstance(recipe, dict) else {}
    camera_id = vision_cfg.get("camera_id")
    module = vision_cfg.get("module")
    if not camera_id or not module:
        raise RuntimeError("missing_vision_camera_or_module")
    if _robot_disabled(ctx):
        _run_pick_vision_only_session(ctx, state, handle, recipe)
        return

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
    if _pose_missing(capture_pose) or _pose_missing(place_pose):
        raise RuntimeError("missing_capture_or_place_pose")

    default_profile = str(robot_cfg.get("default_profile", "normal"))
    capture_profile = str(
        (pall_cfg or {}).get(
            "capture_profile", robot_cfg.get("capture_profile", default_profile)
        )
    )
    capture_return_profile = str(
        (pall_cfg or {}).get("capture_return_profile", capture_profile)
    )
    capture_return_mode = str((pall_cfg or {}).get("capture_return_mode", "auto"))
    repeat = bool(robot_cfg.get("repeat", True))
    max_cycles = int(robot_cfg.get("max_cycles", 0))
    workspace = robot_cfg.get("workspace") or {}
    timeout_s = _vision_timeout_s(vision_cfg, module)

    filters = dict(vision_cfg.get("filters") or {})
    params = dict(vision_cfg.get("params") or {})
    if "min_inliers" in params:
        try:
            filters["min_inliers"] = max(
                int(filters.get("min_inliers", 0)), int(params.get("min_inliers"))
            )
        except Exception:
            pass
    if "min_score" in params:
        try:
            filters["min_score"] = max(
                float(filters.get("min_score", 0.0)), float(params.get("min_score"))
            )
        except Exception:
            pass
    if "min_area_px" in params:
        try:
            filters["min_area_px"] = max(
                float(filters.get("min_area_px", 0.0)), float(params.get("min_area_px"))
            )
        except Exception:
            pass
    if "depth_range_m" not in filters:
        depth_min = params.get("depth_min_m")
        depth_max = params.get("depth_max_m")
        if depth_min is not None or depth_max is not None:
            filters["depth_range_m"] = [depth_min or 0.0, depth_max or 10.0]
    filters.setdefault("steady_frames", 1)
    filters.setdefault("smooth_frames", 1)
    debug = bool(
        recipe.get("debug") or vision_cfg.get("debug") or robot_cfg.get("debug")
    )

    def _log_debug(stage: str, **fields: Any) -> None:
        if not debug:
            return
        payload = {"stage": stage}
        payload.update(fields)
        _log(ctx, run_id, "PALLATIZING_DEBUG", **payload)

    velocity_sample_s = float((pall_cfg or {}).get("velocity_sample_s", 0.35))
    velocity_min_samples = max(2, int((pall_cfg or {}).get("velocity_min_samples", 3)))
    velocity_max_samples = max(
        velocity_min_samples, int((pall_cfg or {}).get("velocity_max_samples", 12))
    )
    lock_z_velocity = bool((pall_cfg or {}).get("lock_z_velocity", True))
    validation_samples = max(2, int((pall_cfg or {}).get("validation_samples", 12)))
    validation_min_samples = max(
        2, int((pall_cfg or {}).get("validation_min_samples", 6))
    )
    validation_max_fit_residual_m = max(
        0.001, float((pall_cfg or {}).get("validation_max_fit_residual_m", 0.008))
    )
    validation_max_fit_z_residual_m = max(
        0.0005, float((pall_cfg or {}).get("validation_max_fit_z_residual_m", 0.004))
    )
    validation_max_yaw_residual_deg = max(
        1.0, float((pall_cfg or {}).get("validation_max_yaw_residual_deg", 20.0))
    )
    z_min_samples = max(2, int((pall_cfg or {}).get("z_min_samples", 3)))
    z_outlier_mad_scale = max(
        0.1, float((pall_cfg or {}).get("z_outlier_mad_scale", 3.0))
    )
    z_outlier_min_tol_m = max(
        1e-4, float((pall_cfg or {}).get("z_outlier_min_tol_m", 0.003))
    )
    z_pick_bias_m = float((pall_cfg or {}).get("z_pick_bias_m", 0.0))
    prediction_horizon_s = max(
        0.0, float((pall_cfg or {}).get("prediction_horizon_s", 1.0))
    )
    max_object_speed_mps = max(
        0.0, float((pall_cfg or {}).get("max_object_speed_mps", 1.5))
    )
    max_prediction_displacement_m = max(
        0.0, float((pall_cfg or {}).get("max_prediction_displacement_m", 0.25))
    )
    velocity_scale = max(0.1, float((pall_cfg or {}).get("velocity_scale", 1.0)))
    lag_compensation_m = max(
        0.0, float((pall_cfg or {}).get("lag_compensation_m", 0.0))
    )
    yaw_axis_correction = bool((pall_cfg or {}).get("yaw_axis_correction", False))
    yaw_outlier_reject_deg = max(
        5.0, float((pall_cfg or {}).get("yaw_outlier_reject_deg", 35.0))
    )
    yaw_min_samples = max(2, int((pall_cfg or {}).get("yaw_min_samples", 3)))
    dynamic_pick_enabled = bool((pall_cfg or {}).get("dynamic_pick_enabled", True))
    velocity_deadband_mps = max(
        0.0, float((pall_cfg or {}).get("velocity_deadband_mps", 0.005))
    )
    pre_pick_lead_s = max(0.0, float((pall_cfg or {}).get("pre_pick_lead_s", 0.5)))
    pick_lead_s = max(0.0, float((pall_cfg or {}).get("pick_lead_s", 1.0)))
    retreat_lead_s = max(0.0, float((pall_cfg or {}).get("retreat_lead_s", 1.2)))
    servo_rate_hz = max(5.0, float((pall_cfg or {}).get("servo_rate_hz", 40.0)))
    servo_approach_window_s = max(
        0.08, float((pall_cfg or {}).get("servo_approach_window_s", 0.28))
    )
    servo_pick_window_s = max(
        0.08, float((pall_cfg or {}).get("servo_pick_window_s", 0.45))
    )
    servo_retreat_window_s = max(
        0.08, float((pall_cfg or {}).get("servo_retreat_window_s", 0.35))
    )
    close_gripper_ratio = max(
        0.0, min(1.0, float((pall_cfg or {}).get("close_gripper_ratio", 0.55)))
    )
    yaw_blend_during_pick = bool((pall_cfg or {}).get("yaw_blend_during_pick", True))
    yaw_align_window_s_cfg = (pall_cfg or {}).get("yaw_align_window_s", None)
    align_yaw_after_pick = bool((pall_cfg or {}).get("align_yaw_after_pick", True))
    pre_pick_profile = str(
        (pall_cfg or {}).get("pre_pick_profile", "")
        or pick_cfg.get("profile", default_profile)
    )
    pick_profile = str(
        (pall_cfg or {}).get("pick_profile", "")
        or pick_cfg.get("profile", default_profile)
    )
    retreat_profile = str(
        (pall_cfg or {}).get("retreat_profile", "")
        or pick_cfg.get("profile", default_profile)
    )

    cycle = 0
    while True:
        # Cycle starts from capture pose so each estimate uses a known camera viewpoint.
        _log(ctx, run_id, "PICK_PLACE_CAPTURE_MOVE", cycle=cycle)
        _move_to_capture_pose(
            ctx, capture_pose, capture_return_profile, capture_return_mode
        )
        _ensure_stop(handle)

        module_params = dict(params)
        module_params["templates_dir"] = _resolve_templates_dir(
            ctx,
            module_params.get("templates_dir") or vision_cfg.get("templates_dir"),
            state.process_id,
            vision_cfg.get("object_id") or module_params.get("object_id"),
        )
        payload = {
            "event": "VISION_START",
            "request_id": f"{state.run_id}-vision-{cycle}",
            "camera_id": camera_id,
            "module": module,
            "fps_limit": vision_cfg.get("fps_limit", 0),
            "process_mode": vision_cfg.get("process_mode", "continuous"),
            "params": module_params,
            "enable_shm_output": True,
        }
        station_calibration = build_station_camera_calibration(
            ctx.data_paths,
            state.station_id,
        )
        if station_calibration:
            payload["calibration"] = station_calibration
        request_id = payload["request_id"]
        handle.vision_request_id = request_id
        ctx.vision.start_session(payload)
        _log(ctx, run_id, "PICK_PLACE_LOOKING", cycle=cycle)

        try:
            track = {"stable_count": 0, "history": []}
            while True:
                mb = _wait_for_match(
                    ctx,
                    request_id,
                    timeout_s,
                    handle,
                    filters,
                    track,
                    lambda *_a, **_k: None,
                )
                match = mb.get("match", {})
                match_evt = mb.get("event", {})
                ok, _reason = _match_ok(match, filters)
                if not ok:
                    _log_debug(
                        "match_reject",
                        cycle=cycle,
                        reason=_reason,
                        frame_id=match_evt.get("frame_id"),
                        match=match,
                    )
                    continue
                target_cam = match.get("center_xyz_m")
                if not _is_finite_vec(target_cam, 3):
                    _log_debug(
                        "match_reject",
                        cycle=cycle,
                        reason="missing_3d_target",
                        frame_id=match_evt.get("frame_id"),
                        match=match,
                    )
                    continue
                _log_debug(
                    "vision_match",
                    cycle=cycle,
                    frame_id=match_evt.get("frame_id"),
                    match=match,
                    filters=filters,
                )

                hand_eye_raw, _pref, _used = _resolve_runtime_handeye(
                    ctx, state.process_id, robot_cfg
                )
                hand_eye = _resolve_hand_eye(hand_eye_raw)
                robot_state = ctx.robot.get_state()
                frame = (
                    str(hand_eye.get("hand_eye_frame") or "gripper_to_camera")
                    .strip()
                    .lower()
                )
                _ee_key = (
                    "flange_pose"
                    if frame == "camera_in_flange" and robot_state.get("flange_pose")
                    else "tcp_pose"
                )
                _ee = robot_state.get(_ee_key) or {}
                base_to_ee = {
                    "translation_m": _ee.get("position_m", [0.0, 0.0, 0.0]),
                    "rotation_quat_xyzw": _ee.get(
                        "quat_xyzw", [0.0, 0.0, 0.0, 1.0]
                    ),
                }
                base_to_cam = (
                    hand_eye
                    if frame in ("base_to_camera", "base")
                    else _compose_transform(base_to_ee, hand_eye)
                )
                target_base = _transform_point(target_cam, base_to_cam)
                if not _workspace_ok(target_base, workspace):
                    continue

                velocity_samples: List[Tuple[float, List[float]]] = [
                    (_evt_timestamp_s(match_evt), list(target_base))
                ]
                yaw_samples: List[float] = [_yaw_from_match(match, yaw_axis_correction)]
                velocity_base_mps = [0.0, 0.0, 0.0]
                if velocity_sample_s > 1e-3:
                    # Build a short track buffer for robust velocity/yaw estimation.
                    deadline = time.time() + velocity_sample_s
                    while (
                        len(velocity_samples) < velocity_max_samples
                        and time.time() < deadline
                    ):
                        _ensure_stop(handle)
                        evt2 = ctx.vision_results.wait_for_result(
                            request_id, 0.12, handle.stop_event
                        )
                        if not evt2:
                            continue
                        r2 = evt2.get("result", {})
                        m2 = (r2.get("matches") or [None])[0]
                        if (
                            not r2.get("valid")
                            or not m2
                            or not _is_finite_vec(m2.get("center_xyz_m"), 3)
                        ):
                            continue
                        ok2, reason2 = _match_ok(m2, filters)
                        if not ok2:
                            _log_debug(
                                "track_sample_reject",
                                cycle=cycle,
                                reason=reason2,
                                frame_id=evt2.get("frame_id"),
                                match=m2,
                            )
                            continue
                        t2 = _transform_point(m2["center_xyz_m"], base_to_cam)
                        velocity_samples.append(
                            (
                                _evt_timestamp_s(evt2),
                                [float(t2[0]), float(t2[1]), float(t2[2])],
                            )
                        )
                        yaw_samples.append(_yaw_from_match(m2, yaw_axis_correction))

                    # Robust cap: accept detection only if a 12-sample window is consistent.
                    window_n = min(int(validation_samples), len(velocity_samples))
                    v_tail = velocity_samples[-window_n:] if window_n > 0 else []
                    y_tail = yaw_samples[-window_n:] if window_n > 0 else []
                    min_required = validation_min_samples
                    track_ok, track_reason, track_stats = _validate_track_window(
                        v_tail,
                        y_tail,
                        min_samples=min_required,
                        max_fit_residual_m=validation_max_fit_residual_m,
                        max_fit_z_residual_m=validation_max_fit_z_residual_m,
                        max_yaw_residual_deg=validation_max_yaw_residual_deg,
                    )
                    if not track_ok:
                        _log_debug(
                            "track_reject",
                            cycle=cycle,
                            reason=track_reason,
                            stats=track_stats,
                            samples=len(velocity_samples),
                        )
                        continue
                    _log_debug(
                        "track_accept",
                        cycle=cycle,
                        stats=track_stats,
                        samples=len(velocity_samples),
                    )

                    # Robust depth estimate from sampled track points.
                    if velocity_samples:
                        z_samples = [
                            float(p[2])
                            for _, p in velocity_samples
                            if _is_finite_vec(p, 3)
                        ]
                        if z_samples:
                            target_base[2] = _estimate_stable_depth_m(
                                z_samples,
                                min_samples=z_min_samples,
                                mad_scale=z_outlier_mad_scale,
                                min_tol_m=z_outlier_min_tol_m,
                            )
                    if len(velocity_samples) >= velocity_min_samples:
                        velocity_base_mps, _ = _estimate_linear_velocity(
                            velocity_samples
                        )
                        if lock_z_velocity:
                            velocity_base_mps[2] = 0.0
                        speed = _norm3(velocity_base_mps)
                        if (
                            max_object_speed_mps > 0
                            and speed > max_object_speed_mps
                            and speed > 1e-9
                        ):
                            s = max_object_speed_mps / speed
                            velocity_base_mps = [
                                velocity_base_mps[0] * s,
                                velocity_base_mps[1] * s,
                                velocity_base_mps[2] * s,
                            ]
                        pred = [
                            velocity_base_mps[0] * prediction_horizon_s,
                            velocity_base_mps[1] * prediction_horizon_s,
                            velocity_base_mps[2] * prediction_horizon_s,
                        ]
                        if lock_z_velocity:
                            pred[2] = 0.0
                        pred_n = _norm3(pred)
                        if (
                            max_prediction_displacement_m > 0
                            and pred_n > max_prediction_displacement_m
                            and pred_n > 1e-9
                        ):
                            s = max_prediction_displacement_m / pred_n
                            pred = [pred[0] * s, pred[1] * s, pred[2] * s]
                        target_base = _add_vec(target_base, pred)
                    # Optional tuning gain to compensate under-estimated object speed.
                    velocity_base_mps = [
                        velocity_base_mps[0] * velocity_scale,
                        velocity_base_mps[1] * velocity_scale,
                        velocity_base_mps[2] * velocity_scale,
                    ]
                    if lock_z_velocity:
                        velocity_base_mps[2] = 0.0

                yaw = _estimate_stable_yaw_deg(
                    yaw_samples, yaw_outlier_reject_deg, yaw_min_samples
                )
                yaw_off = float(pick_cfg.get("yaw_offset_deg", 0.0))
                yaw_sign = float(pick_cfg.get("yaw_sign", 1.0))
                yaw_frame = str(
                    (pall_cfg or {}).get("yaw_frame", pick_cfg.get("yaw_frame", "base"))
                ).lower()
                if yaw_frame not in ("base", "tool"):
                    yaw_frame = "base"
                place_yaw_frame = str(
                    (pall_cfg or {}).get("place_yaw_frame", yaw_frame)
                ).lower()
                if place_yaw_frame not in ("base", "tool"):
                    place_yaw_frame = yaw_frame
                yaw_auto_sign = bool((pall_cfg or {}).get("yaw_auto_sign", True))
                smart_yaw = bool((pall_cfg or {}).get("smart_yaw", True))
                yaw_snap_deg = max(
                    0.0, float((pall_cfg or {}).get("yaw_snap_deg", 0.0))
                )
                if yaw_snap_deg > 1e-6:
                    yaw = round(yaw / yaw_snap_deg) * yaw_snap_deg
                yaw_sign_effective = yaw_sign
                if yaw_auto_sign and yaw_frame == "base":
                    cam_z = _quat_rotate(
                        base_to_cam.get("rotation_quat_xyzw", [0.0, 0.0, 0.0, 1.0]),
                        [0.0, 0.0, 1.0],
                    )
                    if (
                        isinstance(cam_z, list)
                        and len(cam_z) >= 3
                        and float(cam_z[2]) < 0.0
                    ):
                        yaw_sign_effective *= -1.0
                yaw_effective_deg = _normalize_angle_deg(
                    yaw * yaw_sign_effective + yaw_off
                )
                pick_yaw_deg = (
                    _wrap_yaw_90(yaw_effective_deg) if smart_yaw else yaw_effective_deg
                )
                _log_debug(
                    "yaw_solution",
                    cycle=cycle,
                    yaw_samples=yaw_samples,
                    yaw_raw_deg=yaw,
                    yaw_effective_deg=yaw_effective_deg,
                    pick_yaw_deg=pick_yaw_deg,
                    yaw_axis_correction=yaw_axis_correction,
                    yaw_frame=yaw_frame,
                    place_yaw_frame=place_yaw_frame,
                )
                capture_q = _get_pose_quat(capture_pose) or [0.0, 0.0, 0.0, 1.0]
                yaw_rad = math.radians(pick_yaw_deg)
                orientation_yaw_target = _normalize_quat_xyzw(
                    _apply_yaw(capture_q, yaw_rad, yaw_frame)
                )
                _log_debug(
                    "orientation_solution",
                    cycle=cycle,
                    capture_orientation_quat=capture_q,
                    orientation_yaw_target_quat=orientation_yaw_target,
                    orientation_yaw_target_rpy_deg=_quat_to_rpy_deg(
                        orientation_yaw_target
                    ),
                    yaw_rad=yaw_rad,
                    yaw_frame=yaw_frame,
                )

                approach_offset = list(
                    pick_cfg.get("approach_offset_m", [0.0, 0.0, 0.06])
                )
                grasp_offset = list(pick_cfg.get("grasp_offset_m", [0.0, 0.0, 0.0]))
                retreat_offset = list(
                    pick_cfg.get("retreat_offset_m", [0.0, 0.0, 0.08])
                )
                # Keep parity with pick/place: allow safer pick height from hand-eye gripper offset.
                pick_z_offset = float(
                    pick_cfg.get(
                        "pick_z_offset_m", hand_eye_raw.get("gripper_z_offset_m", 0.0)
                    )
                )
                speed_cmd = _norm3(velocity_base_mps)
                lag_vec = [0.0, 0.0, 0.0]
                if lag_compensation_m > 1e-9 and speed_cmd > 1e-9:
                    lag_vec = [
                        velocity_base_mps[0] / speed_cmd * lag_compensation_m,
                        velocity_base_mps[1] / speed_cmd * lag_compensation_m,
                        velocity_base_mps[2] / speed_cmd * lag_compensation_m,
                    ]
                target_pick = [
                    float(target_base[0]) + lag_vec[0],
                    float(target_base[1]) + lag_vec[1],
                    float(target_base[2]) + lag_vec[2] + pick_z_offset + z_pick_bias_m,
                ]
                approach = _add_vec(target_pick, approach_offset)
                grasp = _add_vec(target_pick, grasp_offset)
                retreat = _add_vec(target_pick, retreat_offset)
                approach_pose_dbg = _pose_debug(
                    approach, orientation_yaw_target, "base"
                )
                grasp_pose_dbg = _pose_debug(grasp, orientation_yaw_target, "base")
                retreat_pose_dbg = _pose_debug(retreat, orientation_yaw_target, "base")
                _log_debug(
                    "target_solution",
                    cycle=cycle,
                    target_base=target_base,
                    velocity_base_mps=velocity_base_mps,
                    lock_z_velocity=lock_z_velocity,
                    z_pick_bias_m=z_pick_bias_m,
                    target_pick=target_pick,
                    approach_pose=approach_pose_dbg,
                    grasp_pose=grasp_pose_dbg,
                    retreat_pose=retreat_pose_dbg,
                )

                dynamic_active = bool(
                    dynamic_pick_enabled and hasattr(ctx.robot, "servo_tcp")
                )
                speed_mps = _norm3(velocity_base_mps)
                ctx.robot.open_gripper()
                if dynamic_active:
                    # VIM-like continuous pickup path:
                    # one servo timeline for approach -> pick -> retreat, no blocking pre-pick move.
                    start_pos = tcp_pose.get("position_m", [0.0, 0.0, 0.0])
                    if not _is_finite_vec(start_pos, 3):
                        start_pos = [0.0, 0.0, 0.0]
                    start_pos = [
                        float(start_pos[0]),
                        float(start_pos[1]),
                        float(start_pos[2]),
                    ]
                    start_quat = tcp_pose.get("quat_xyzw", [0.0, 0.0, 0.0, 1.0])
                    if not _is_finite_vec(start_quat, 4):
                        start_quat = capture_q
                    start_quat = _normalize_quat_xyzw(
                        [
                            float(start_quat[0]),
                            float(start_quat[1]),
                            float(start_quat[2]),
                            float(start_quat[3]),
                        ]
                    )
                    pick_quat = (
                        start_quat if align_yaw_after_pick else orientation_yaw_target
                    )
                    t_app = max(1e-3, float(servo_approach_window_s))
                    t_pick = max(1e-3, float(servo_pick_window_s))
                    t_ret = max(1e-3, float(servo_retreat_window_s))
                    t_total = t_app + t_pick + t_ret
                    if yaw_align_window_s_cfg is None:
                        yaw_align_window_s = t_app + t_pick
                    else:
                        yaw_align_window_s = max(1e-3, float(yaw_align_window_s_cfg))
                    close_t = t_app + t_pick * close_gripper_ratio
                    t0 = time.perf_counter()
                    closed = False

                    while True:
                        _ensure_stop(handle)
                        tk = time.perf_counter()
                        et = tk - t0
                        if et >= t_total:
                            break

                        if et < t_app:
                            # Enter interception path from current TCP to moving pre-pick target.
                            alpha = et / t_app
                            pre = [
                                target_pick[0]
                                + velocity_base_mps[0] * (pre_pick_lead_s + et)
                                + approach_offset[0],
                                target_pick[1]
                                + velocity_base_mps[1] * (pre_pick_lead_s + et)
                                + approach_offset[1],
                                target_pick[2]
                                + velocity_base_mps[2] * (pre_pick_lead_s + et)
                                + approach_offset[2],
                            ]
                            p = _lerp(start_pos, pre, alpha)
                            profile = pre_pick_profile
                        elif et < (t_app + t_pick):
                            # Keep moving while transitioning pre-pick -> grasp.
                            u = (et - t_app) / t_pick
                            o = _lerp(approach_offset, grasp_offset, u)
                            p = [
                                target_pick[0]
                                + velocity_base_mps[0] * (pick_lead_s + et)
                                + o[0],
                                target_pick[1]
                                + velocity_base_mps[1] * (pick_lead_s + et)
                                + o[1],
                                target_pick[2]
                                + velocity_base_mps[2] * (pick_lead_s + et)
                                + o[2],
                            ]
                            profile = pick_profile
                        else:
                            # Continue in motion while moving grasp -> retreat.
                            u = (et - t_app - t_pick) / t_ret
                            o = _lerp(grasp_offset, retreat_offset, u)
                            p = [
                                target_pick[0]
                                + velocity_base_mps[0] * (retreat_lead_s + et)
                                + o[0],
                                target_pick[1]
                                + velocity_base_mps[1] * (retreat_lead_s + et)
                                + o[1],
                                target_pick[2]
                                + velocity_base_mps[2] * (retreat_lead_s + et)
                                + o[2],
                            ]
                            profile = retreat_profile

                        if align_yaw_after_pick:
                            q_cmd = pick_quat
                        elif yaw_blend_during_pick:
                            yaw_alpha = min(1.0, et / yaw_align_window_s)
                            q_cmd = _quat_slerp(
                                start_quat, orientation_yaw_target, yaw_alpha
                            )
                        else:
                            q_cmd = orientation_yaw_target

                        ctx.robot.servo_tcp(
                            {"position_m": p, "quat_xyzw": q_cmd, "frame": "base"},
                            profile,
                        )
                        if not closed and et >= close_t:
                            ctx.robot.close_gripper()
                            closed = True
                        _sleep_rate(servo_rate_hz, tk)

                    if not closed:
                        ctx.robot.close_gripper()
                else:
                    if speed_mps < velocity_deadband_mps:
                        _log(
                            ctx,
                            run_id,
                            "PALLATIZING_WARN",
                            cycle=cycle,
                            reason="dynamic_pick_disabled_or_unsupported",
                            dynamic_pick_enabled=dynamic_pick_enabled,
                            has_servo_tcp=hasattr(ctx.robot, "servo_tcp"),
                            speed_mps=speed_mps,
                        )
                    static_pick_quat = (
                        capture_q if align_yaw_after_pick else orientation_yaw_target
                    )
                    ctx.robot.movel(
                        {
                            "position_m": approach,
                            "quat_xyzw": static_pick_quat,
                            "frame": "base",
                        },
                        pre_pick_profile,
                    )
                    ctx.robot.movel(
                        {
                            "position_m": grasp,
                            "quat_xyzw": static_pick_quat,
                            "frame": "base",
                        },
                        pick_profile,
                    )
                    ctx.robot.close_gripper()
                    ctx.robot.movel(
                        {
                            "position_m": retreat,
                            "quat_xyzw": static_pick_quat,
                            "frame": "base",
                        },
                        retreat_profile,
                    )

                _log(ctx, run_id, "PICK_DONE", cycle=cycle)
                place_cfg = robot_cfg.get("place", {})
                place_profile = str(place_cfg.get("profile", default_profile))
                place_target = place_pose.get("tcp") or place_pose.get("tcp_pose") or {}
                place_pos = place_target.get("position_m")
                if place_pos:
                    place_seed_quat = place_target.get("quat_xyzw") or capture_q
                    place_yaw_mode = str(
                        (pall_cfg or {}).get("place_yaw_mode", "template_absolute")
                    ).lower()
                    place_yaw_offset_deg = float(
                        (pall_cfg or {}).get("place_yaw_offset_deg", 0.0)
                    )
                    if align_yaw_after_pick:
                        if place_yaw_mode == "inverse_detected_yaw":
                            rpy = _quat_to_rpy_deg(
                                _normalize_quat_xyzw(list(place_seed_quat))
                            )
                            target_yaw_deg = _normalize_angle_deg(
                                rpy[2] - yaw + place_yaw_offset_deg
                            )
                            place_quat = _rpy_deg_to_quat_xyzw(
                                rpy[0], rpy[1], target_yaw_deg
                            )
                        elif place_yaw_mode == "template_absolute":
                            rpy = _quat_to_rpy_deg(
                                _normalize_quat_xyzw(list(place_seed_quat))
                            )
                            target_yaw_deg = _normalize_angle_deg(
                                rpy[2] + pick_yaw_deg + place_yaw_offset_deg
                            )
                            place_quat = _rpy_deg_to_quat_xyzw(
                                rpy[0], rpy[1], target_yaw_deg
                            )
                        else:
                            place_quat = _normalize_quat_xyzw(
                                _apply_yaw(place_seed_quat, yaw_rad, place_yaw_frame)
                            )
                    else:
                        place_quat = orientation_yaw_target
                    force_yaw = place_cfg.get("force_yaw_deg", None)
                    if force_yaw is not None:
                        seed_rpy = _quat_to_rpy_deg(
                            _normalize_quat_xyzw(list(place_seed_quat))
                        )
                        place_quat = _rpy_deg_to_quat_xyzw(
                            float(seed_rpy[0]), float(seed_rpy[1]), float(force_yaw)
                        )
                    app = _add_vec(
                        place_pos, place_cfg.get("approach_offset_m", [0.0, 0.0, 0.08])
                    )
                    ret = _add_vec(
                        place_pos, place_cfg.get("retreat_offset_m", [0.0, 0.0, 0.08])
                    )
                    _log_debug(
                        "place_solution",
                        cycle=cycle,
                        place_yaw_mode=place_yaw_mode,
                        place_yaw_offset_deg=place_yaw_offset_deg,
                        detected_pick_yaw_deg=pick_yaw_deg,
                        place_seed_quat=place_seed_quat,
                        place_quat=place_quat,
                        place_quat_rpy_deg=_quat_to_rpy_deg(
                            _normalize_quat_xyzw(list(place_quat))
                        ),
                        place_approach_pose=_pose_debug(
                            app, place_quat, place_target.get("frame", "base")
                        ),
                        place_pose=_pose_debug(
                            place_pos, place_quat, place_target.get("frame", "base")
                        ),
                        place_retreat_pose=_pose_debug(
                            ret, place_quat, place_target.get("frame", "base")
                        ),
                    )
                    ctx.robot.movel(
                        {
                            "position_m": app,
                            "quat_xyzw": place_quat,
                            "frame": place_target.get("frame", "base"),
                        },
                        place_profile,
                    )
                    ctx.robot.movel(
                        {
                            "position_m": place_pos,
                            "quat_xyzw": place_quat,
                            "frame": place_target.get("frame", "base"),
                        },
                        place_profile,
                    )
                    ctx.robot.open_gripper()
                    ctx.robot.movel(
                        {
                            "position_m": ret,
                            "quat_xyzw": place_quat,
                            "frame": place_target.get("frame", "base"),
                        },
                        place_profile,
                    )
                else:
                    _move_to_pose(ctx, place_pose, default_profile)
                    ctx.robot.open_gripper()
                _log(ctx, run_id, "PICK_PLACE_DONE", cycle=cycle)
                break
        finally:
            try:
                ctx.vision.stop_session(request_id)
            except Exception:
                pass
            if handle.vision_request_id == request_id:
                handle.vision_request_id = None

        cycle += 1
        if not repeat:
            break
        if max_cycles > 0 and cycle >= max_cycles:
            break
