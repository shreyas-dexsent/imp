"""Compatibility shim for the generic pick/place runtime."""

from orchestrator.tasks._pick_runtime import (
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
    _resolve_hand_eye,
    _resolve_runtime_handeye,
    _resolve_task_payload,
    _resolve_templates_dir,
    _rpy_deg_to_quat_xyzw,
    _transform_point,
    _vision_timeout_s,
    _wait_for_match,
    _workspace_ok,
    run_pick_place_core,
)


def run_pick_place_demo(ctx, state, handle) -> None:
    run_pick_place_core(ctx, state, handle)
