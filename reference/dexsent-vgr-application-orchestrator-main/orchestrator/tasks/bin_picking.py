"""Implementation for `orchestrator.tasks.bin_picking`."""

from copy import deepcopy
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import threading
import time
from typing import Any, Callable, Dict, List, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np

from orchestrator.core.context import StationContext
from orchestrator.core.runs import RunState
from orchestrator.robot_engine_bridge import evaluate_scene as evaluate_robot_engine_scene
from orchestrator.tasks._pick_runtime import (
    PickRuntimeHooks,
    _add_vec,
    _apply_tcp_calibration_to_base,
    _compose_transform,
    _is_finite_vec,
    _normalize_quat_xyzw,
    _normalize_vec,
    _quat_mul_xyzw,
    _quat_rotate,
    _resolve_hand_eye,
    _resolve_runtime_handeye,
    _resolve_runtime_tcp_calibration,
    _transform_point,
    run_pick_place_core,
)

_DEFAULT_BIN_PICKING_POSE_MODULE = "megapose_bin_picking"
_BIN_PICKING_POSE_MODULES = {
    "megapose_bin_picking",
    "ppf_icp_bin_picking",
}
_BIN_PICKING_NO_CANDIDATE_ERRORS = {
    "megapose_no_candidate",
    "parallel_jaw_no_grasp_candidate",
    "ppf_icp_no_candidate",
}


def _clean_path_string(raw_path: Optional[str]) -> str:
    value = str(raw_path or "").strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        value = value[1:-1].strip()
    return value


def _coerce_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        num = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(num):
        return default
    return num


def _coerce_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return bool(value)
    raw = str(value).strip().lower()
    if raw in {"1", "true", "yes", "y", "on"}:
        return True
    if raw in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _vec3_or_none(value: Any) -> Optional[List[float]]:
    if not isinstance(value, (list, tuple)) or len(value) < 3:
        return None
    out = []
    for idx in range(3):
        num = _coerce_float(value[idx], None)
        if num is None:
            return None
        out.append(num)
    return out


def _distance_between(a: List[float], b: List[float]) -> float:
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    dz = a[2] - b[2]
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def _subtract_vec(a: List[float], b: List[float]) -> List[float]:
    return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]


def _dot_vec(a: List[float], b: List[float]) -> float:
    return (a[0] * b[0]) + (a[1] * b[1]) + (a[2] * b[2])


def _cross_vec(a: List[float], b: List[float]) -> List[float]:
    return [
        (a[1] * b[2]) - (a[2] * b[1]),
        (a[2] * b[0]) - (a[0] * b[2]),
        (a[0] * b[1]) - (a[1] * b[0]),
    ]


def _distance_point_to_segment(
    point: List[float],
    seg_a: List[float],
    seg_b: List[float],
) -> float:
    """Shortest distance from point to line segment AB."""
    ab = _subtract_vec(seg_b, seg_a)
    ap = _subtract_vec(point, seg_a)
    ab_len_sq = _dot_vec(ab, ab)
    if ab_len_sq <= 1e-12:
        return _distance_between(point, seg_a)
    t = max(0.0, min(1.0, _dot_vec(ap, ab) / ab_len_sq))
    closest = [seg_a[0] + ab[0] * t, seg_a[1] + ab[1] * t, seg_a[2] + ab[2] * t]
    return _distance_between(point, closest)


def _rotate_vec_about_axis(vec: List[float], axis: List[float], angle_rad: float) -> List[float]:
    axis_n = _normalize_vec(axis)
    if axis_n is None:
        return list(vec)
    cos_t = math.cos(float(angle_rad))
    sin_t = math.sin(float(angle_rad))
    dot = _dot_vec(axis_n, vec)
    cross = _cross_vec(axis_n, vec)
    return [
        (vec[0] * cos_t) + (cross[0] * sin_t) + (axis_n[0] * dot * (1.0 - cos_t)),
        (vec[1] * cos_t) + (cross[1] * sin_t) + (axis_n[1] * dot * (1.0 - cos_t)),
        (vec[2] * cos_t) + (cross[2] * sin_t) + (axis_n[2] * dot * (1.0 - cos_t)),
    ]


def _canonical_gripper_type(value: Any) -> str:
    raw = str(value or "").strip().lower()
    if raw in {"vacuum", "suction", "vacume"}:
        return "vacuum"
    return "parallel_jaw"


def _normalize_grasp_family_label(value: Any) -> Optional[str]:
    raw = str(value or "").strip().lower()
    if raw in {"internal", "inside", "inner"}:
        return "internal"
    if raw in {"external", "outside", "outer"}:
        return "external"
    return None


def _parse_group_index(value: Any) -> Optional[int]:
    group_index = int(_coerce_float(value, 0.0) or 0)
    return group_index if group_index > 0 else None


def _parse_grasp_family_selection_policy(
    payload: Dict[str, Any],
    label_map: dict[int, str],
) -> Dict[str, Any]:
    raw_policy = payload.get("selection_policy")
    policy = raw_policy if isinstance(raw_policy, dict) else {}
    raw_groups = policy.get("groups")
    if raw_groups is None:
        raw_groups = policy.get("priority_groups")
    if raw_groups is None:
        raw_groups = payload.get("priority_groups")
    if not isinstance(raw_groups, list):
        return {}

    disabled_groups: set[int] = set()
    for value in (
        policy.get("disabled_groups")
        or payload.get("disabled_groups")
        or []
    ):
        group_index = _parse_group_index(value)
        if group_index is not None:
            disabled_groups.add(group_index)

    groups: List[Dict[str, Any]] = []
    all_groups: List[Dict[str, Any]] = []
    for order, raw_group in enumerate(raw_groups):
        if isinstance(raw_group, dict):
            group_index = _parse_group_index(
                raw_group.get("group_index")
                or raw_group.get("index")
                or raw_group.get("group")
            )
            enabled = bool(raw_group.get("enabled", True))
            priority = int(_coerce_float(raw_group.get("priority"), order + 1) or (order + 1))
            raw_label = (
                raw_group.get("label")
                or raw_group.get("family_label")
                or label_map.get(group_index or 0)
            )
            note = str(raw_group.get("note") or raw_group.get("disabled_reason") or "").strip()
        else:
            group_index = _parse_group_index(raw_group)
            enabled = True
            priority = order + 1
            raw_label = label_map.get(group_index or 0)
            note = ""
        if group_index is None:
            continue
        if group_index in disabled_groups:
            enabled = False
        label = _normalize_grasp_family_label(raw_label)
        entry: Dict[str, Any] = {
            "group_index": group_index,
            "priority": priority,
            "order": order,
            "enabled": bool(enabled),
        }
        if label is not None:
            entry["label"] = label
        if note:
            entry["note"] = note
        all_groups.append(entry)
        if enabled:
            groups.append(entry)

    groups.sort(
        key=lambda item: (
            int(_coerce_float(item.get("priority"), 0.0) or 0),
            int(_coerce_float(item.get("order"), 0.0) or 0),
            int(_coerce_float(item.get("group_index"), 0.0) or 0),
        )
    )
    all_groups.sort(
        key=lambda item: (
            int(_coerce_float(item.get("priority"), 0.0) or 0),
            int(_coerce_float(item.get("order"), 0.0) or 0),
            int(_coerce_float(item.get("group_index"), 0.0) or 0),
        )
    )
    if not groups and not all_groups:
        return {}
    return {
        "enabled": bool(policy.get("enabled", payload.get("priority_selection_enabled", True))),
        "mode": str(policy.get("mode") or "priority_groups").strip().lower(),
        "fallback": str(policy.get("fallback") or "next_segmentation").strip().lower(),
        "groups": groups,
        "all_groups": all_groups,
    }


def _grasp_family_policy_enabled(policy: Any) -> bool:
    return (
        isinstance(policy, dict)
        and bool(policy.get("enabled", True))
        and bool((policy.get("groups") or []) or (policy.get("all_groups") or []))
    )


def _grasp_family_policy_group_key(
    policy: Any,
    group_index: Any,
) -> Optional[tuple[int, int]]:
    if not _grasp_family_policy_enabled(policy):
        return (0, 0)
    parsed_group_index = _parse_group_index(group_index)
    if parsed_group_index is None:
        return None
    for order, group in enumerate(policy.get("groups") or []):
        if _parse_group_index((group or {}).get("group_index")) != parsed_group_index:
            continue
        return (
            int(_coerce_float((group or {}).get("priority"), order + 1) or (order + 1)),
            int(_coerce_float((group or {}).get("order"), order) or order),
        )
    return None


def _grasp_allowed_by_family_policy(policy: Any, grasp: Dict[str, Any]) -> bool:
    if not _grasp_family_policy_enabled(policy):
        return True
    return _grasp_family_policy_group_key(
        policy,
        grasp.get("generator_group_index"),
    ) is not None


def _apply_grasp_family_mode(policy: Any, mode: str) -> Any:
    """Return a filtered copy of policy restricted to the given grasp family mode.

    "internal_only" keeps only groups labelled internal.
    "external_only" keeps only groups labelled external.
    "default" (or anything else) returns the policy unchanged.
    """
    mode = (mode or "default").strip().lower()
    if mode not in {"internal_only", "external_only"} or not _grasp_family_policy_enabled(policy):
        return policy
    required = "internal" if mode == "internal_only" else "external"
    filtered = dict(policy)
    filtered["groups"] = [
        g for g in (policy.get("groups") or [])
        if _normalize_grasp_family_label(g.get("label")) == required
    ]
    if not filtered["groups"]:
        return {}
    return filtered


def _load_grasp_family_labels_model(
    contacts_path: Path,
) -> tuple[dict[int, str], Optional[str], Dict[str, Any]]:
    labels_path = contacts_path.with_name("grasp_family_labels.json")
    if not labels_path.exists() or not labels_path.is_file():
        return {}, None, {}
    try:
        payload = json.loads(labels_path.read_text(encoding="utf-8"))
    except Exception:
        return {}, None, {}
    if not isinstance(payload, dict):
        return {}, None, {}
    raw_labels = payload.get("labels")
    result: dict[int, str] = {}
    if isinstance(raw_labels, dict):
        for raw_key, raw_value in raw_labels.items():
            label = _normalize_grasp_family_label(raw_value)
            if label is None:
                continue
            group_index = _parse_group_index(raw_key)
            if group_index is None:
                continue
            result[group_index] = label
    selection_policy = _parse_grasp_family_selection_policy(payload, result)
    for group in selection_policy.get("all_groups") or []:
        group_index = _parse_group_index((group or {}).get("group_index"))
        label = _normalize_grasp_family_label((group or {}).get("label"))
        if group_index is not None and label is not None and group_index not in result:
            result[group_index] = label
    return result, str(labels_path), selection_policy


def _parse_signed_axis(value: Any) -> Optional[Dict[str, Any]]:
    raw = str(value or "").strip().lower()
    if not raw:
        return None
    sign = 1.0
    if raw.startswith("-"):
        sign = -1.0
        raw = raw[1:]
    if raw not in {"x", "y", "z"}:
        return None
    return {"axis": raw, "sign": sign}


def _quat_from_axes(
    x_axis: List[float],
    y_axis: List[float],
    z_axis: List[float],
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


def _quat_angle_distance_rad(a_xyzw: List[float], b_xyzw: List[float]) -> float:
    a_n = _normalize_quat_xyzw(a_xyzw)
    b_n = _normalize_quat_xyzw(b_xyzw)
    if a_n is None or b_n is None:
        return math.pi
    dot = (
        (a_n[0] * b_n[0])
        + (a_n[1] * b_n[1])
        + (a_n[2] * b_n[2])
        + (a_n[3] * b_n[3])
    )
    dot = max(-1.0, min(1.0, abs(float(dot))))
    return 2.0 * math.acos(dot)


def _orthogonalize(
    primary: List[float],
    secondary: List[float],
) -> Optional[List[float]]:
    primary_n = _normalize_vec(primary)
    secondary_n = _normalize_vec(secondary)
    if primary_n is None or secondary_n is None:
        return None
    projected = [
        secondary_n[0] - (_dot_vec(secondary_n, primary_n) * primary_n[0]),
        secondary_n[1] - (_dot_vec(secondary_n, primary_n) * primary_n[1]),
        secondary_n[2] - (_dot_vec(secondary_n, primary_n) * primary_n[2]),
    ]
    return _normalize_vec(projected)


def _repair_legacy_symmetric_ring_approach_axis(
    *,
    payload: Dict[str, Any],
    raw_grasp: Dict[str, Any],
    raw_grasp_index: int,
    jaw_axis_local: List[float],
    stored_approach_axis_local: Optional[List[float]],
) -> Optional[List[float]]:
    if str(payload.get("approach_axis_convention") or "").strip().lower() == "pregrasp_to_center":
        return stored_approach_axis_local
    generator = payload.get("generator") if isinstance(payload.get("generator"), dict) else {}
    if str(generator.get("mode") or "").strip().lower() != "symmetric_ring":
        return stored_approach_axis_local
    if str(generator.get("symmetric_approach_mode") or "").strip().lower() != "axis_roll":
        return stored_approach_axis_local
    group_index = int(_coerce_float(raw_grasp.get("generator_group_index"), 0.0) or 0)
    if group_index <= 0:
        grasp_count_per_ring = int(
            _coerce_float(generator.get("grasp_count_per_ring"), 0.0) or 0
        )
        if grasp_count_per_ring > 0:
            group_index = int(raw_grasp_index // grasp_count_per_ring) + 1
    if group_index <= 0:
        return stored_approach_axis_local
    axis_records = generator.get("pair_axis_records") or []
    axis_record = None
    for record in axis_records:
        if int(_coerce_float((record or {}).get("group_index"), 0.0) or 0) == group_index:
            axis_record = record
            break
    axis_direction = _normalize_vec(
        _vec3_or_none((axis_record or {}).get("axis_direction")) or []
    )
    if axis_direction is None:
        return stored_approach_axis_local
    # Legacy symmetric-ring files stored the symmetry/finger axis itself as
    # approach_axis_local. Detect that case and reconstruct the true approach
    # axis from the finger axis and jaw axis instead.
    if stored_approach_axis_local is not None:
        stored_n = _normalize_vec(stored_approach_axis_local)
        if stored_n is not None and abs(_dot_vec(stored_n, axis_direction)) < 0.9:
            return stored_approach_axis_local
    finger_axis_local = _orthogonalize(jaw_axis_local, axis_direction)
    if finger_axis_local is None:
        return stored_approach_axis_local
    roll_values = generator.get("approach_roll_deg_per_group")
    roll_deg = None
    if isinstance(roll_values, list) and len(roll_values) >= group_index:
        roll_deg = _coerce_float(roll_values[group_index - 1], None)
    if roll_deg is None:
        roll_deg = _coerce_float(generator.get("approach_roll_deg"), 0.0) or 0.0
    if abs(float(roll_deg)) > 1e-9:
        finger_axis_local = _normalize_vec(
            _rotate_vec_about_axis(
                finger_axis_local,
                jaw_axis_local,
                math.radians(float(roll_deg)),
            )
        )
    if finger_axis_local is None:
        return stored_approach_axis_local
    repaired = _normalize_vec(_cross_vec(finger_axis_local, jaw_axis_local))
    return repaired or stored_approach_axis_local


def _build_parallel_jaw_orientation(
    jaw_axis_base: List[float],
    approach_axis_base: List[float],
    jaw_tool_axis: str,
    approach_tool_axis: str,
) -> Optional[List[float]]:
    jaw_axis = _normalize_vec(jaw_axis_base)
    approach_axis = _normalize_vec(approach_axis_base)
    jaw_tool = _parse_signed_axis(jaw_tool_axis)
    approach_tool = _parse_signed_axis(approach_tool_axis)
    if jaw_axis is None or approach_axis is None or jaw_tool is None or approach_tool is None:
        return None
    if jaw_tool["axis"] == approach_tool["axis"]:
        return None

    world_axes: Dict[str, List[float]] = {
        jaw_tool["axis"]: [
            jaw_axis[0] * jaw_tool["sign"],
            jaw_axis[1] * jaw_tool["sign"],
            jaw_axis[2] * jaw_tool["sign"],
        ],
        approach_tool["axis"]: [
            approach_axis[0] * approach_tool["sign"],
            approach_axis[1] * approach_tool["sign"],
            approach_axis[2] * approach_tool["sign"],
        ],
    }

    if "x" in world_axes and "y" in world_axes:
        x_axis = _normalize_vec(world_axes["x"])
        y_axis = _orthogonalize(x_axis or [], world_axes["y"])
        if x_axis is None or y_axis is None:
            return None
        z_axis = _normalize_vec(_cross_vec(x_axis, y_axis))
        if z_axis is None:
            return None
        y_axis = _normalize_vec(_cross_vec(z_axis, x_axis)) or y_axis
        return _quat_from_axes(x_axis, y_axis, z_axis)
    if "y" in world_axes and "z" in world_axes:
        y_axis = _normalize_vec(world_axes["y"])
        z_axis = _orthogonalize(y_axis or [], world_axes["z"])
        if y_axis is None or z_axis is None:
            return None
        x_axis = _normalize_vec(_cross_vec(y_axis, z_axis))
        if x_axis is None:
            return None
        z_axis = _normalize_vec(_cross_vec(x_axis, y_axis)) or z_axis
        return _quat_from_axes(x_axis, y_axis, z_axis)
    if "x" in world_axes and "z" in world_axes:
        x_axis = _normalize_vec(world_axes["x"])
        z_axis = _orthogonalize(x_axis or [], world_axes["z"])
        if x_axis is None or z_axis is None:
            return None
        y_axis = _normalize_vec(_cross_vec(z_axis, x_axis))
        if y_axis is None:
            return None
        z_axis = _normalize_vec(_cross_vec(x_axis, y_axis)) or z_axis
        return _quat_from_axes(x_axis, y_axis, z_axis)
    return None


def _parallel_jaw_approach_tool_axis(pick_cfg: Optional[Dict[str, Any]]) -> str:
    """Return the flange-frame tool approach axis for parallel-jaw planning."""
    cfg = pick_cfg if isinstance(pick_cfg, dict) else {}
    return str(
        cfg.get("parallel_jaw_approach_tool_axis")
        or cfg.get("pick_contacts_tool_axis")
        or "z"
    ).strip().lower()


def _parallel_jaw_jaw_tool_axis(pick_cfg: Optional[Dict[str, Any]]) -> str:
    """Return the flange-frame jaw/contact axis for parallel-jaw planning."""
    cfg = pick_cfg if isinstance(pick_cfg, dict) else {}
    return str(
        cfg.get("parallel_jaw_jaw_tool_axis")
        or cfg.get("pick_contacts_jaw_axis")
        or cfg.get("_tcp_jaw_tool_axis")
        or "y"
    ).strip().lower()


def _resolve_parallel_jaw_equivalent(
    *,
    contact_a_local_m: List[float],
    contact_b_local_m: List[float],
    normal_a_local: List[float],
    normal_b_local: List[float],
    contact_a_base_m: List[float],
    contact_b_base_m: List[float],
    jaw_axis_local: List[float],
    jaw_axis_base: List[float],
    approach_axis_base: List[float],
    jaw_tool_axis: str,
    approach_tool_axis: str,
    reference_quat_xyzw: Optional[List[float]],
    allow_contact_axis_flip: bool = False,
) -> Dict[str, Any]:
    primary_orientation = _build_parallel_jaw_orientation(
        jaw_axis_base=jaw_axis_base,
        approach_axis_base=approach_axis_base,
        jaw_tool_axis=jaw_tool_axis,
        approach_tool_axis=approach_tool_axis,
    )
    primary_distance = (
        _quat_angle_distance_rad(primary_orientation, reference_quat_xyzw)
        if primary_orientation is not None and reference_quat_xyzw is not None
        else math.pi
    )

    if not allow_contact_axis_flip:
        return {
            "contact_a_local_m": contact_a_local_m,
            "contact_b_local_m": contact_b_local_m,
            "normal_a_local": normal_a_local,
            "normal_b_local": normal_b_local,
            "contact_a_base_m": contact_a_base_m,
            "contact_b_base_m": contact_b_base_m,
            "jaw_axis_local": jaw_axis_local,
            "jaw_axis_base": jaw_axis_base,
            "explicit_orientation": primary_orientation,
            "orientation_distance_rad": primary_distance,
            "equivalent_flipped": False,
        }

    flipped_jaw_axis_local = [-jaw_axis_local[0], -jaw_axis_local[1], -jaw_axis_local[2]]
    flipped_jaw_axis_base = [-jaw_axis_base[0], -jaw_axis_base[1], -jaw_axis_base[2]]
    flipped_orientation = _build_parallel_jaw_orientation(
        jaw_axis_base=flipped_jaw_axis_base,
        approach_axis_base=approach_axis_base,
        jaw_tool_axis=jaw_tool_axis,
        approach_tool_axis=approach_tool_axis,
    )
    flipped_distance = (
        _quat_angle_distance_rad(flipped_orientation, reference_quat_xyzw)
        if flipped_orientation is not None and reference_quat_xyzw is not None
        else math.pi
    )

    use_flipped = flipped_distance < primary_distance
    if not use_flipped:
        return {
            "contact_a_local_m": contact_a_local_m,
            "contact_b_local_m": contact_b_local_m,
            "normal_a_local": normal_a_local,
            "normal_b_local": normal_b_local,
            "contact_a_base_m": contact_a_base_m,
            "contact_b_base_m": contact_b_base_m,
            "jaw_axis_local": jaw_axis_local,
            "jaw_axis_base": jaw_axis_base,
            "explicit_orientation": primary_orientation,
            "orientation_distance_rad": primary_distance,
            "equivalent_flipped": False,
        }
    return {
        "contact_a_local_m": contact_b_local_m,
        "contact_b_local_m": contact_a_local_m,
        "normal_a_local": normal_b_local,
        "normal_b_local": normal_a_local,
        "contact_a_base_m": contact_b_base_m,
        "contact_b_base_m": contact_a_base_m,
        "jaw_axis_local": flipped_jaw_axis_local,
        "jaw_axis_base": flipped_jaw_axis_base,
        "explicit_orientation": flipped_orientation,
        "orientation_distance_rad": flipped_distance,
        "equivalent_flipped": True,
    }


def _project_point_uv(
    point_xyz_m: Optional[List[float]],
    camera_intrinsics: Any,
) -> Optional[List[float]]:
    if not _is_finite_vec(point_xyz_m, 3):
        return None
    intr = camera_intrinsics if isinstance(camera_intrinsics, dict) else {}
    fx = _coerce_float(intr.get("fx"), None)
    fy = _coerce_float(intr.get("fy"), None)
    cx = _coerce_float(intr.get("cx"), None)
    cy = _coerce_float(intr.get("cy"), None)
    if None in (fx, fy, cx, cy):
        return None
    x = float(point_xyz_m[0])
    y = float(point_xyz_m[1])
    z = float(point_xyz_m[2])
    if z <= 1e-6:
        return None
    return [
        float(fx * (x / z) + cx),
        float(fy * (y / z) + cy),
    ]


def _resolve_optional_path(
    raw_path: Optional[str],
    search_roots: List[Path],
) -> Optional[Path]:
    value = _clean_path_string(raw_path)
    if not value:
        return None
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate
    for root in search_roots:
        resolved = (root / candidate).resolve()
        if resolved.exists():
            return resolved
    if search_roots:
        return (search_roots[0] / candidate).resolve()
    return candidate.resolve()


def _resolve_object_folder_path(
    ctx: StationContext,
    process_id: Optional[str],
    object_id: Optional[str],
    raw_path: Optional[str],
) -> str:
    cleaned = _clean_path_string(raw_path)
    candidates = []
    if cleaned:
        path = Path(cleaned)
        if path.is_absolute():
            candidates.append(path.resolve())
        else:
            candidates.append((Path.cwd() / path).resolve())
            candidates.append((ctx.data_root / path).resolve())

    if process_id and object_id:
        process = ctx.processes.get(process_id) or {}
        station_id = process.get("station_id")
        asset_id = process.get("asset_id") or process.get("process_id") or process_id
        if station_id and asset_id:
            candidates.append(
                (ctx.data_paths.process_objects_dir(str(station_id), str(asset_id)) / str(object_id)).resolve()
            )

    metadata = (
        ctx.objects.get_metadata(process_id, object_id) or {}
        if process_id and object_id
        else {}
    )
    megapose_meta = metadata.get("megapose") or {}
    ppf_icp_meta = metadata.get("ppf_icp") or metadata.get("ppf_icp_bin_picking") or {}
    meta_path = _clean_path_string(
        megapose_meta.get("object_folder")
        or ppf_icp_meta.get("object_folder")
        or metadata.get("object_folder")
    )
    if meta_path:
        path = Path(meta_path)
        if path.is_absolute():
            candidates.append(path.resolve())
        else:
            candidates.append((Path.cwd() / path).resolve())
            candidates.append((ctx.data_root / path).resolve())

    seen = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if candidate.exists() and candidate.is_dir():
            return str(candidate)

    if candidates:
        return str(candidates[0])
    raise RuntimeError("missing_object_folder")


def _is_websocket_vision(ctx: StationContext) -> bool:
    cfg = ctx.config.get("vision_engine", {}) if ctx and getattr(ctx, "config", None) else {}
    transport = getattr(getattr(ctx, "vision", None), "transport", None) or cfg.get("transport")
    return str(transport or "").strip().lower() == "websocket"


def _looks_like_posix_absolute_path(raw_path: Optional[str]) -> bool:
    cleaned = _clean_path_string(raw_path)
    return cleaned.startswith("/") and not cleaned.startswith("//")


def _vision_object_folder_path(
    ctx: StationContext,
    raw_path: Optional[str],
    local_path: str,
) -> str:
    cleaned = _clean_path_string(raw_path)
    if cleaned and _is_websocket_vision(ctx) and _looks_like_posix_absolute_path(cleaned):
        return cleaned
    return local_path


def _normalize_pose_reference(raw_value: Any) -> Optional[Dict[str, Any]]:
    if isinstance(raw_value, str) and str(raw_value).strip():
        return {"pose_name": str(raw_value).strip()}
    if not isinstance(raw_value, dict):
        return None
    pose_name = str(raw_value.get("pose_name") or raw_value.get("name") or "").strip()
    pose_inline = raw_value.get("pose")
    if pose_inline is None and any(
        key in raw_value for key in ("tcp_pose", "tcp", "joints", "position_m")
    ):
        pose_inline = dict(raw_value)
    if not pose_name and not isinstance(pose_inline, dict):
        return None
    out: Dict[str, Any] = {}
    if pose_name:
        out["pose_name"] = pose_name
    if isinstance(pose_inline, dict):
        out["pose"] = dict(pose_inline)
    return out


def _resolve_pose_reference(
    pose_index: Dict[str, Dict[str, Any]],
    raw_value: Any,
    error_code: str,
) -> Optional[Dict[str, Any]]:
    reference = _normalize_pose_reference(raw_value)
    if not reference:
        return None
    pose_name = str(reference.get("pose_name") or "").strip()
    pose_inline = reference.get("pose") if isinstance(reference.get("pose"), dict) else None
    if pose_name:
        pose_named = pose_index.get(pose_name)
        if isinstance(pose_named, dict):
            return {"pose": pose_named, "pose_name": pose_name}
    if pose_inline is not None:
        pose = dict(pose_inline)
        if pose_name and not pose.get("name"):
            pose["name"] = pose_name
        return {"pose": pose, "pose_name": pose_name or pose.get("name")}
    raise RuntimeError(error_code)


def _load_place_policy_model(
    module_params: Dict[str, Any],
    place_cfg: Dict[str, Any],
    search_roots: List[Path],
) -> Optional[Dict[str, Any]]:
    raw_path = place_cfg.get("place_rules_file") or module_params.get("place_rules_file")
    policy_path = _resolve_optional_path(raw_path, search_roots) if raw_path else None
    if policy_path is None:
        policy_path = _resolve_optional_path("place.json", search_roots)
    if policy_path is None or not policy_path.exists() or not policy_path.is_file():
        return None

    payload = json.loads(policy_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return None

    placement_goal = payload.get("placement_goal")
    if not isinstance(placement_goal, dict):
        placement_goal = {}

    raw_strategies = (
        payload.get("grasp_strategies")
        or payload.get("strategies")
        or payload.get("grasp_plans")
        or []
    )
    strategies: List[Dict[str, Any]] = []
    for order, raw_strategy in enumerate(raw_strategies):
        if not isinstance(raw_strategy, dict):
            continue
        if raw_strategy.get("enabled", True) is False:
            continue
        match_cfg = raw_strategy.get("match") if isinstance(raw_strategy.get("match"), dict) else raw_strategy
        grasp_ids = [
            str(value).strip()
            for value in (match_cfg.get("grasp_ids") or match_cfg.get("ids") or [])
            if str(value).strip()
        ]
        grasp_labels = [
            str(value).strip()
            for value in (
                match_cfg.get("grasp_labels")
                or match_cfg.get("labels")
                or []
            )
            if str(value).strip()
        ]
        group_indices: List[int] = []
        for value in (
            match_cfg.get("generator_group_indices")
            or match_cfg.get("grasp_group_indices")
            or match_cfg.get("group_indices")
            or []
        ):
            group_index = int(_coerce_float(value, 0.0) or 0)
            if group_index > 0:
                group_indices.append(group_index)
        mode_raw = str(raw_strategy.get("mode") or "direct").strip().lower()
        if mode_raw in {
            "intermediate",
            "intermediate_pose",
            "regrasp",
            "intermediate_regrasp",
        }:
            mode = "intermediate_regrasp"
        else:
            mode = "direct"
        intermediate_cfg = raw_strategy.get("intermediate_regrasp")
        if not isinstance(intermediate_cfg, dict):
            intermediate_cfg = raw_strategy.get("intermediate")
        if not isinstance(intermediate_cfg, dict):
            intermediate_cfg = {}
        final_place_cfg = raw_strategy.get("final_place")
        if not isinstance(final_place_cfg, dict):
            final_place_cfg = {}
        strategies.append(
            {
                "name": str(raw_strategy.get("name") or f"strategy_{order + 1}"),
                "mode": mode,
                "priority": int(_coerce_float(raw_strategy.get("priority"), 0.0) or 0),
                "order": order,
                "match": {
                    "grasp_ids": grasp_ids,
                    "grasp_labels": grasp_labels,
                    "generator_group_indices": group_indices,
                },
                "intermediate_regrasp": dict(intermediate_cfg),
                "final_place": dict(final_place_cfg),
            }
        )

    return {
        "path": str(policy_path),
        "format": str(payload.get("format") or "vgr_place_policy/v1"),
        "enabled": bool(payload.get("enabled", True)),
        "placement_goal": placement_goal,
        "strategies": strategies,
    }


def _place_strategy_has_selector(strategy: Dict[str, Any]) -> bool:
    match_cfg = strategy.get("match") if isinstance(strategy.get("match"), dict) else {}
    return bool(
        (match_cfg.get("grasp_ids") or [])
        or (match_cfg.get("grasp_labels") or [])
        or (match_cfg.get("generator_group_indices") or [])
    )


def _place_strategy_matches(
    strategy: Dict[str, Any],
    selected_pick_contact: Dict[str, Any],
) -> bool:
    match_cfg = strategy.get("match") if isinstance(strategy.get("match"), dict) else {}
    grasp_id = str(selected_pick_contact.get("id") or "").strip()
    grasp_label = str(selected_pick_contact.get("label") or "").strip()
    group_index = int(_coerce_float(selected_pick_contact.get("generator_group_index"), 0.0) or 0)
    has_selector = False

    grasp_ids = match_cfg.get("grasp_ids") or []
    if grasp_ids:
        has_selector = True
        if grasp_id not in grasp_ids:
            return False

    grasp_labels = match_cfg.get("grasp_labels") or []
    if grasp_labels:
        has_selector = True
        if grasp_label not in grasp_labels:
            return False

    group_indices = match_cfg.get("generator_group_indices") or []
    if group_indices:
        has_selector = True
        if group_index <= 0 or group_index not in group_indices:
            return False

    return True if has_selector else True


def _select_place_strategy(
    place_policy_model: Optional[Dict[str, Any]],
    selected_pick_contact: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    if (
        not place_policy_model
        or place_policy_model.get("enabled", True) is False
        or not isinstance(selected_pick_contact, dict)
    ):
        return None
    strategies = place_policy_model.get("strategies") or []
    specific_matches: List[Dict[str, Any]] = []
    default_matches: List[Dict[str, Any]] = []
    for strategy in strategies:
        if not isinstance(strategy, dict):
            continue
        if not _place_strategy_matches(strategy, selected_pick_contact):
            continue
        if _place_strategy_has_selector(strategy):
            specific_matches.append(strategy)
        else:
            default_matches.append(strategy)
    matches = specific_matches or default_matches
    if not matches:
        return None
    matches.sort(
        key=lambda item: (
            int(_coerce_float(item.get("priority"), 0.0) or 0),
            int(_coerce_float(item.get("order"), 0.0) or 0),
        )
    )
    return dict(matches[0])


def _vision_http_url(ctx: StationContext) -> str:
    cfg = ctx.config.get("vision_engine", {}) if ctx and ctx.config else {}
    return str(cfg.get("http_url", "http://127.0.0.1:8000")).rstrip("/")


def _post_json(url: str, payload: Dict[str, Any], timeout_s: float) -> Dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(req, timeout=float(timeout_s)) as resp:
            raw = resp.read().decode("utf-8")
            return json.loads(raw) if raw else {}
    except HTTPError as exc:
        try:
            detail = exc.read().decode("utf-8")
        except Exception:
            detail = str(exc)
        raise RuntimeError(f"bin_picking_prewarm_http_{exc.code}: {detail}") from exc
    except URLError as exc:
        raise RuntimeError(f"bin_picking_prewarm_unreachable: {exc}") from exc


def _start_bin_picking_prewarm(
    ctx: StationContext,
    state: RunState,
    task_payload: Dict[str, Any],
) -> None:
    recipe = task_payload
    if isinstance(task_payload.get("recipe"), dict):
        recipe = task_payload["recipe"]
    if not isinstance(recipe, dict):
        return

    vision_cfg = recipe.get("vision") or {}
    if not isinstance(vision_cfg, dict):
        return
    module_name = str(vision_cfg.get("module") or "").strip().lower()
    prewarm_path_by_module = {
        "megapose_bin_picking": "/modules/megapose_bin_picking/prewarm",
        "ppf_icp_bin_picking": "/modules/ppf_icp_bin_picking/prewarm",
    }
    prewarm_path = prewarm_path_by_module.get(module_name)
    if prewarm_path is None:
        return

    module_params = dict(vision_cfg.get("params") or {})
    if not bool(module_params.get("prewarm_before_run", False)):
        return

    timeout_s = _coerce_float(module_params.get("prewarm_timeout_s"), 180.0) or 180.0
    url = f"{_vision_http_url(ctx)}{prewarm_path}"
    payload = {"params": module_params}
    run_id = str(getattr(state, "run_id", "") or "").strip()

    def _worker() -> None:
        event_base = {
            "event": "BIN_PICKING_PREWARM",
            "timestamp_ns": time.time_ns(),
            "url": url,
            "module": module_name,
            "object_folder": module_params.get("object_folder"),
        }
        try:
            if run_id:
                ctx.runs.append_event(run_id, {**event_base, "status": "started"})
            result = _post_json(url, payload, timeout_s=timeout_s)
            if run_id:
                ctx.runs.append_event(
                    run_id,
                    {
                        **event_base,
                        "status": "ok",
                        "result": result,
                        "timestamp_ns": time.time_ns(),
                    },
                )
        except Exception as exc:
            if run_id:
                ctx.runs.append_event(
                    run_id,
                    {
                        **event_base,
                        "status": "error",
                        "error": str(exc),
                        "timestamp_ns": time.time_ns(),
                    },
                )

    threading.Thread(target=_worker, daemon=True).start()


def _quat_rotate(quat_xyzw: List[float], vec: List[float]) -> List[float]:
    """Rotate a vector by a quaternion."""
    x, y, z, w = [float(v) for v in quat_xyzw[:4]]
    vx, vy, vz = [float(v) for v in vec[:3]]
    
    # Quaternion multiplication: q * v * q^-1
    q_vec = [x, y, z]
    uv = [q_vec[1]*vz - q_vec[2]*vy, q_vec[2]*vx - q_vec[0]*vz, q_vec[0]*vy - q_vec[1]*vx]
    uuv = [q_vec[1]*uv[2] - q_vec[2]*uv[1], q_vec[2]*uv[0] - q_vec[0]*uv[2], q_vec[0]*uv[1] - q_vec[1]*uv[0]]
    
    return [
        vx + 2.0*(w*uv[0] + uuv[0]),
        vy + 2.0*(w*uv[1] + uuv[1]),
        vz + 2.0*(w*uv[2] + uuv[2]),
    ]


def _create_parallel_jaw_grasp(
    grasp_id: str, 
    position_m: List[float], 
    rotation_rpy_deg: List[float], 
    stroke_mm: float, 
    grasp_family: str, 
    invariance: Dict[str, Any]
) -> Dict[str, Any]:
    """Create a parallel jaw grasp dictionary from authored grasp parameters."""
    # Convert mm to m
    opening_width_m = stroke_mm / 1000.0

    # Create transformation matrix for grasp pose
    grasp_quat = _rpy_to_quat_deg(rotation_rpy_deg)
    grasp_matrix = [
        [1.0, 0.0, 0.0, position_m[0]],
        [0.0, 1.0, 0.0, position_m[1]],
        [0.0, 0.0, 1.0, position_m[2]],
        [0.0, 0.0, 0.0, 1.0],
    ]
    # Apply rotation
    rot_matrix = _quat_xyzw_to_matrix(grasp_quat)
    grasp_matrix = [
        [rot_matrix[0][0], rot_matrix[0][1], rot_matrix[0][2], position_m[0]],
        [rot_matrix[1][0], rot_matrix[1][1], rot_matrix[1][2], position_m[1]],
        [rot_matrix[2][0], rot_matrix[2][1], rot_matrix[2][2], position_m[2]],
        [0.0, 0.0, 0.0, 1.0],
    ]

    # Contact points relative to grasp center along Y axis (gripper jaw opening direction)
    half_width = opening_width_m / 2.0
    contact_a_local = [0.0, -half_width, 0.0, 1.0]
    contact_b_local = [0.0, half_width, 0.0, 1.0]

    # Transform contacts to object frame
    contact_a_obj = [0.0, 0.0, 0.0]
    contact_b_obj = [0.0, 0.0, 0.0]
    for i in range(3):
        contact_a_obj[i] = sum(grasp_matrix[i][j] * contact_a_local[j] for j in range(4))
        contact_b_obj[i] = sum(grasp_matrix[i][j] * contact_b_local[j] for j in range(4))

    # Center is the grasp position
    center_local_m = position_m.copy()

    # Jaw axis = gripper +Y in object frame (perpendicular to invariance axis when axis != Y)
    jaw_axis_local = _quat_rotate(grasp_quat, [0.0, 1.0, 0.0])

    # Approach axis = gripper +Z in object frame (matches grasp studio: +Z = approach direction,
    # i.e. the direction the gripper travels from pregrasp toward the grasp center, with the
    # gripper body sitting at TCP -Z and the object at TCP +Z).
    approach_axis_local = _quat_rotate(grasp_quat, [0.0, 0.0, 1.0])

    # Normals: contact_a (at -Y) normal points +Y toward center; contact_b (at +Y) points -Y
    normal_a_local = list(jaw_axis_local)
    normal_b_local = [-v for v in jaw_axis_local]

    return {
        "id": grasp_id,
        "label": grasp_id,
        "grasp_type": "parallel_jaw_pair",
        "contact_a_local_m": contact_a_obj,
        "contact_b_local_m": contact_b_obj,
        "normal_a_local": normal_a_local,
        "normal_b_local": normal_b_local,
        "center_local_m": center_local_m,
        "jaw_axis_local": jaw_axis_local,
        "approach_axis_local": approach_axis_local,
        "opening_width_m": opening_width_m,
        "generator_group_index": 1 if grasp_family == "external" else 2,
        "grasp_family_label": grasp_family,
        "invariance": invariance,
    }


def _expand_grasp_invariance(
    base_grasp_id: str, 
    base_grasp: Dict[str, Any], 
    invariance: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """Expand a base grasp into multiple grasps based on invariance settings."""
    if not invariance or not bool(invariance.get("enabled")):
        return []
    
    import math
    steps = max(1, min(36, int(float(invariance.get("steps") or 8))))
    enabled = invariance.get("enabledSteps")
    enabled_steps = set(int(v) for v in enabled) if isinstance(enabled, list) else set(range(steps + 1))
    lower = math.radians(float(invariance.get("lowerLimit") or 0.0))
    upper = math.radians(float(invariance.get("upperLimit") or 360.0))
    axis_name = str(invariance.get("axis") or "y").lower()
    axis = {"x": [1.0, 0.0, 0.0], "y": [0.0, 1.0, 0.0], "z": [0.0, 0.0, 1.0]}.get(axis_name, [0.0, 1.0, 0.0])
    axis_pos = [float(v or 0.0) for v in (invariance.get("axisPos") or [0.0, 0.0, 0.0])[:3]]
    
    expanded_grasps = []
    for step in range(steps + 1):
        if step == 0 or step not in enabled_steps:
            continue
        
        angle = lower + (upper - lower) * (step / steps)
        
        # Create rotation matrix around axis
        axis_norm = math.sqrt(sum(a*a for a in axis))
        if axis_norm < 1e-12:
            continue
        ax, ay, az = [a / axis_norm for a in axis]
        c, s = math.cos(angle), math.sin(angle)
        r = [
            [c + ax*ax*(1-c), ax*ay*(1-c) - az*s, ax*az*(1-c) + ay*s],
            [ay*ax*(1-c) + az*s, c + ay*ay*(1-c), ay*az*(1-c) - ax*s],
            [az*ax*(1-c) - ay*s, az*ay*(1-c) + ax*s, c + az*az*(1-c)],
        ]
        
        # Apply rotation to grasp pose
        center = base_grasp["center_local_m"]
        contact_a = base_grasp["contact_a_local_m"] 
        contact_b = base_grasp["contact_b_local_m"]
        jaw_axis = base_grasp["jaw_axis_local"]
        approach_axis = base_grasp["approach_axis_local"]
        normal_a = base_grasp["normal_a_local"]
        normal_b = base_grasp["normal_b_local"]
        
        # Rotate points around axis_pos
        def rotate_point(pt):
            # Translate to axis origin
            pt_rel = [pt[i] - axis_pos[i] for i in range(3)]
            # Rotate
            pt_rot = [
                sum(r[i][j] * pt_rel[j] for j in range(3))
                for i in range(3)
            ]
            # Translate back
            return [pt_rot[i] + axis_pos[i] for i in range(3)]
        
        rotated_center = rotate_point(center)
        rotated_contact_a = rotate_point(contact_a)
        rotated_contact_b = rotate_point(contact_b)
        rotated_jaw_axis = rotate_point([center[i] + jaw_axis[i] for i in range(3)])
        rotated_jaw_axis = [rotated_jaw_axis[i] - rotated_center[i] for i in range(3)]
        rotated_approach_axis = rotate_point([center[i] + approach_axis[i] for i in range(3)])
        rotated_approach_axis = [rotated_approach_axis[i] - rotated_center[i] for i in range(3)]
        rotated_normal_a = rotate_point([contact_a[i] + normal_a[i] for i in range(3)])
        rotated_normal_a = [rotated_normal_a[i] - rotated_contact_a[i] for i in range(3)]
        rotated_normal_b = rotate_point([contact_b[i] + normal_b[i] for i in range(3)])
        rotated_normal_b = [rotated_normal_b[i] - rotated_contact_b[i] for i in range(3)]
        
        grasp_id = f"{base_grasp_id}_inv_{step}"
        expanded_grasp = base_grasp.copy()
        expanded_grasp.update({
            "id": grasp_id,
            "label": grasp_id,
            "center_local_m": rotated_center,
            "contact_a_local_m": rotated_contact_a,
            "contact_b_local_m": rotated_contact_b,
            "jaw_axis_local": rotated_jaw_axis,
            "approach_axis_local": rotated_approach_axis,
            "normal_a_local": rotated_normal_a,
            "normal_b_local": rotated_normal_b,
            "invariance_step": step,
            "source_grasp_id": base_grasp_id,
        })
        expanded_grasps.append(expanded_grasp)
    
    return expanded_grasps


def _translation_matrix(xyz: List[float]) -> List[List[float]]:
    """Create a 4x4 translation matrix."""
    return [
        [1.0, 0.0, 0.0, float(xyz[0])],
        [0.0, 1.0, 0.0, float(xyz[1])],
        [0.0, 0.0, 1.0, float(xyz[2])],
        [0.0, 0.0, 0.0, 1.0],
    ]


def _rpy_matrix_deg(position_m: List[float], rpy_deg: List[float]) -> List[List[float]]:
    """Create a 4x4 transformation matrix from position and RPY angles in degrees."""
    from orchestrator.tasks._pick_runtime import _rpy_deg_to_quat_xyzw
    quat = _rpy_deg_to_quat_xyzw(float(rpy_deg[0]), float(rpy_deg[1]), float(rpy_deg[2]))
    R = _quat_xyzw_to_matrix(quat)
    return [
        [R[0][0], R[0][1], R[0][2], float(position_m[0])],
        [R[1][0], R[1][1], R[1][2], float(position_m[1])],
        [R[2][0], R[2][1], R[2][2], float(position_m[2])],
        [0.0, 0.0, 0.0, 1.0],
    ]


def _matrix_to_rpy_deg(matrix: List[List[float]]) -> List[float]:
    """Extract RPY angles in degrees from a 4x4 transformation matrix."""
    # Extract rotation matrix
    R = [
        [matrix[0][0], matrix[0][1], matrix[0][2]],
        [matrix[1][0], matrix[1][1], matrix[1][2]],
        [matrix[2][0], matrix[2][1], matrix[2][2]],
    ]
    # Convert to RPY using scipy or manual calculation
    # For simplicity, use atan2 approach
    if abs(R[2][0]) < 1.0:
        pitch = math.asin(-R[2][0])
        roll = math.atan2(R[2][1] / math.cos(pitch), R[2][2] / math.cos(pitch))
        yaw = math.atan2(R[1][0] / math.cos(pitch), R[0][0] / math.cos(pitch))
    else:
        # Gimbal lock
        yaw = math.atan2(-R[0][1], R[1][1])
        roll = 0.0
        pitch = math.copysign(math.pi / 2.0, -R[2][0])
    return [math.degrees(roll), math.degrees(pitch), math.degrees(yaw)]


def _rpy_to_quat_deg(rpy_deg: List[float]) -> List[float]:
    """Convert grasp-studio Euler angles (deg, XYZ intrinsic order) to quaternion (xyzw).

    The grasp studio composes orientations with three.js' default Euler order "XYZ":
    R = Rx(rx) · Ry(ry) · Rz(rz). Earlier this delegated to _rpy_deg_to_quat_xyzw,
    which uses ZYX (Rz·Ry·Rx); for non-trivial combos like [-90, 0, -90] those
    produce different orientations and the planner's jaw/approach axes ended up
    pointing along the wrong object axes.
    """
    rx = math.radians(float(rpy_deg[0]))
    ry = math.radians(float(rpy_deg[1]))
    rz = math.radians(float(rpy_deg[2]))
    cx, sx = math.cos(rx * 0.5), math.sin(rx * 0.5)
    cy, sy = math.cos(ry * 0.5), math.sin(ry * 0.5)
    cz, sz = math.cos(rz * 0.5), math.sin(rz * 0.5)
    # qx of Rx(rx) * Ry(ry) * Rz(rz) (matches THREE.Quaternion.setFromEuler order 'XYZ')
    qx = sx * cy * cz + cx * sy * sz
    qy = cx * sy * cz - sx * cy * sz
    qz = cx * cy * sz + sx * sy * cz
    qw = cx * cy * cz - sx * sy * sz
    n = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw) or 1.0
    return [qx / n, qy / n, qz / n, qw / n]


def _quat_xyzw_to_matrix(quat_xyzw: List[float]) -> List[List[float]]:
    """Convert quaternion to 3x3 rotation matrix."""
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


def _matrix_multiply(a: List[List[float]], b: List[List[float]]) -> List[List[float]]:
    """Multiply two 4x4 matrices."""
    result = [[0.0 for _ in range(4)] for _ in range(4)]
    for i in range(4):
        for j in range(4):
            for k in range(4):
                result[i][j] += a[i][k] * b[k][j]
    return result


def _load_authored_grasps_model(
    ctx: StationContext,
    module_params: Dict[str, Any],
    pick_cfg: Dict[str, Any],
    search_roots: List[Path],
) -> Optional[Dict[str, Any]]:
    """Load grasps from grasp_authoring.json and apply metadata.json offsets."""
    object_folder = _clean_path_string(module_params.get("object_folder"))
    if not object_folder:
        return None

    object_folder_path = Path(object_folder)
    grasp_authoring_path = object_folder_path / "grasp_authoring.json"
    if not grasp_authoring_path.exists() or not grasp_authoring_path.is_file():
        return None

    try:
        authoring_payload = json.loads(grasp_authoring_path.read_text(encoding="utf-8"))
    except Exception:
        return None

    if not isinstance(authoring_payload, dict):
        return None

    # Get grasp family mode from configuration
    grasp_family_mode = str(pick_cfg.get("grasp_family_mode", "default")).strip().lower()

    # Load metadata.json: bin_picking_frame is the studio's authoring origin (CAD-frame offset).
    # Megapose, however, reports the object's pose at the AABB centroid of the original CAD mesh
    # (because the runtime mesh is recentered to its AABB centre before pose estimation). The
    # studio places its `objectFrameGroup` at `bin_picking_frame`, so grasp positions in
    # grasp_authoring.json are expressed in the bin_picking_frame. To make the planner agree
    # with the studio's visualization we shift each grasp by `(bin_picking_frame - aabb_centroid)`
    # so it ends up in the same frame the vision system reports — the AABB centroid frame.
    metadata_path = object_folder_path / "metadata.json"
    bin_picking_frame_offset = {"position_m": [0.0, 0.0, 0.0], "rotation_rpy_deg": [0.0, 0.0, 0.0]}
    if metadata_path.exists() and metadata_path.is_file():
        try:
            metadata_payload = json.loads(metadata_path.read_text(encoding="utf-8"))
            if isinstance(metadata_payload, dict):
                bin_picking_meta = metadata_payload.get("bin_picking_frame")
                if isinstance(bin_picking_meta, dict):
                    bin_picking_frame_offset.update(bin_picking_meta)
        except Exception:
            pass  # Use defaults if metadata loading fails

    # Compute AABB centroid of the original CAD mesh (in metres, in CAD frame). The mesh in
    # the object folder is what the studio loads and what megapose uses (after recentering),
    # so reading it here gives us the exact same centroid megapose used.
    aabb_centroid_cad_m = [0.0, 0.0, 0.0]
    cad_units_scale = 0.001  # default mm
    for ext in ("obj", "stl", "ply"):
        mesh_candidate = object_folder_path / f"{object_folder_path.name}.{ext}"
        if not mesh_candidate.exists():
            matches = sorted(object_folder_path.glob(f"*.{ext}"))
            mesh_candidate = matches[0] if matches else None
        if mesh_candidate is None or not mesh_candidate.exists():
            continue
        try:
            mn = [float("inf")] * 3
            mx = [float("-inf")] * 3
            with mesh_candidate.open("r", encoding="utf-8", errors="ignore") as fh:
                for raw in fh:
                    if not raw.startswith("v "):
                        continue
                    parts = raw.split()
                    if len(parts) < 4:
                        continue
                    try:
                        v = [float(parts[1]), float(parts[2]), float(parts[3])]
                    except ValueError:
                        continue
                    for i in range(3):
                        if v[i] < mn[i]:
                            mn[i] = v[i]
                        if v[i] > mx[i]:
                            mx[i] = v[i]
            if all(math.isfinite(mn[i]) and math.isfinite(mx[i]) for i in range(3)):
                aabb_centroid_cad_m = [
                    0.5 * (mn[i] + mx[i]) * cad_units_scale for i in range(3)
                ]
        except Exception:
            pass
        break

    bp_pos_m = list(bin_picking_frame_offset.get("position_m") or [0.0, 0.0, 0.0])
    # Shift to apply to each authored grasp position (in bin_picking_frame, CAD-axis-aligned)
    # so the result is expressed in the AABB-centroid frame megapose reports against.
    bin_to_aabb_shift_m = [
        float(bp_pos_m[i]) - float(aabb_centroid_cad_m[i]) for i in range(3)
    ]

    raw_grasps = authoring_payload.get("grasps") or []
    grasps: List[Dict[str, Any]] = []

    for idx, raw_grasp in enumerate(raw_grasps):
        if not isinstance(raw_grasp, dict):
            continue
        if raw_grasp.get("enabled", True) is False:
            continue

        # Extract authored grasp data
        base_grasp_id = str(raw_grasp.get("id") or f"grasp_{idx + 1:03d}")
        mesh_units = str(authoring_payload.get("mesh_units") or "mm").strip().lower()
        unit_scale = 1.0 if mesh_units == "m" else 0.001
        raw_position = _vec3_or_none(raw_grasp.get("position")) or [0.0, 0.0, 0.0]
        position_m = [
            raw_position[0] * unit_scale + bin_to_aabb_shift_m[0],
            raw_position[1] * unit_scale + bin_to_aabb_shift_m[1],
            raw_position[2] * unit_scale + bin_to_aabb_shift_m[2],
        ]
        rotation_rpy_deg = _vec3_or_none(raw_grasp.get("rotation")) or [0.0, 0.0, 0.0]
        stroke_mm = _coerce_float(raw_grasp.get("stroke"), 41.0) or 41.0
        grasp_type = str(raw_grasp.get("type") or "parallel_jaw").strip().lower()
        grasp_family = str(raw_grasp.get("grasp_family") or "external").strip().lower()
        invariance = raw_grasp.get("invariance") if isinstance(raw_grasp.get("invariance"), dict) else {}
        # Shift the rotation-invariance pivot too so the rotation axis stays anchored to the
        # same physical point on the object after the bin_picking→AABB-centroid frame shift.
        if isinstance(invariance, dict) and invariance.get("axisPos") is not None:
            try:
                ax_pos = list(invariance.get("axisPos") or [0.0, 0.0, 0.0])
                invariance = dict(invariance)
                invariance["axisPos"] = [
                    float(ax_pos[0]) + bin_to_aabb_shift_m[0] / unit_scale,
                    float(ax_pos[1]) + bin_to_aabb_shift_m[1] / unit_scale,
                    float(ax_pos[2]) + bin_to_aabb_shift_m[2] / unit_scale,
                ]
            except Exception:
                pass

        # Filter by grasp family mode
        if grasp_family_mode == "internal_only" and grasp_family != "internal":
            continue
        if grasp_family_mode == "external_only" and grasp_family != "external":
            continue

        # Generate base grasp
        base_grasp = _create_parallel_jaw_grasp(
            base_grasp_id, position_m, rotation_rpy_deg, stroke_mm, grasp_family, invariance
        )
        
        # Add base grasp
        grasps.append(base_grasp)
        
        # Expand invariance if enabled
        if invariance and bool(invariance.get("enabled")):
            invariance_grasps = _expand_grasp_invariance(base_grasp_id, base_grasp, invariance)
            grasps.extend(invariance_grasps)

    if not grasps:
        return None

    # Create grasp family labels and policy based on authored grasps
    grasp_family_labels_map = {}
    for grasp in grasps:
        family = grasp.get("grasp_family_label")
        group_idx = grasp.get("generator_group_index")
        if family and group_idx:
            grasp_family_labels_map[group_idx] = family

    grasp_family_selection_policy = _parse_grasp_family_selection_policy(
        {"selection_policy": {"enabled": True, "groups": []}},  # Default policy
        grasp_family_labels_map
    )

    return {
        "path": str(grasp_authoring_path),
        "source_mesh_path": None,
        "format": "vgr_grasp_authoring/v1",
        "default_gripper_type": "parallel_jaw",
        "selection_mode": "closest_to_tcp",
        "tool_axis": "-z",
        "jaw_axis": "x",
        "force_yaw_deg": 0.0,
        "mesh_units": "mm",
        "mesh_scale": 1.0,
        "contacts": [],  # No vacuum contacts from authored grasps
        "grasps": grasps,
        "grasp_family_labels": {str(k): v for k, v in grasp_family_labels_map.items()},
        "grasp_family_labels_path": None,
        "grasp_family_selection_policy": grasp_family_selection_policy,
    }


def _load_pick_contacts_model(
    module_params: Dict[str, Any],
    pick_cfg: Dict[str, Any],
    search_roots: List[Path],
) -> Optional[Dict[str, Any]]:
    raw_path = (
        pick_cfg.get("pick_contacts_file")
        or module_params.get("pick_contacts_file")
    )
    contacts_path = _resolve_optional_path(raw_path, search_roots)
    if contacts_path is None or not contacts_path.exists() or not contacts_path.is_file():
        return None

    payload = json.loads(contacts_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return None

    mesh_units = str(
        payload.get("mesh_units")
        or module_params.get("mesh_units")
        or "mm"
    ).strip().lower()
    mesh_scale = _coerce_float(
        payload.get("mesh_scale")
        if payload.get("mesh_scale") is not None
        else module_params.get("mesh_scale", 1.0),
        1.0,
    )
    raw_source_mesh_path = _clean_path_string(payload.get("source_mesh_path"))
    source_mesh_path = _resolve_optional_path(
        raw_source_mesh_path,
        [contacts_path.parent, *search_roots],
    )
    if source_mesh_path is not None and not source_mesh_path.exists():
        source_mesh_path = None
    if source_mesh_path is None and raw_source_mesh_path:
        fallback_name = Path(raw_source_mesh_path.replace("\\", "/")).name
        fallback_candidate = contacts_path.parent / fallback_name
        if fallback_name and fallback_candidate.exists() and fallback_candidate.is_file():
            source_mesh_path = fallback_candidate
    unit_scale = 1.0 if mesh_units == "m" else 0.001
    (
        grasp_family_labels_map,
        grasp_family_labels_path,
        grasp_family_selection_policy,
    ) = _load_grasp_family_labels_model(
        contacts_path
    )

    contacts: List[Dict[str, Any]] = []
    for idx, raw_contact in enumerate(payload.get("contacts") or []):
        if not isinstance(raw_contact, dict):
            continue
        if raw_contact.get("enabled", True) is False:
            continue
        point_local_m = _vec3_or_none(
            raw_contact.get("point_local_m")
            or raw_contact.get("position_local_m")
            or raw_contact.get("point_m")
        )
        if point_local_m is None:
            point_mesh_units = _vec3_or_none(
                raw_contact.get("point_local_mesh_units")
                or raw_contact.get("position_local_mesh_units")
                or raw_contact.get("point_mesh_units")
            )
            if point_mesh_units is not None:
                point_local_m = [
                    point_mesh_units[0] * unit_scale * float(mesh_scale or 1.0),
                    point_mesh_units[1] * unit_scale * float(mesh_scale or 1.0),
                    point_mesh_units[2] * unit_scale * float(mesh_scale or 1.0),
                ]
        normal_local = _normalize_vec(
            _vec3_or_none(
                raw_contact.get("normal_local")
                or raw_contact.get("surface_normal_local")
                or raw_contact.get("normal")
            )
            or []
        )
        if point_local_m is None or normal_local is None:
            continue
        contacts.append(
            {
                "id": str(raw_contact.get("id") or f"contact_{idx + 1:03d}"),
                "label": str(
                    raw_contact.get("label")
                    or raw_contact.get("name")
                    or raw_contact.get("id")
                    or f"contact_{idx + 1:03d}"
                ),
                "point_local_m": point_local_m,
                "normal_local": normal_local,
                "face_index": raw_contact.get("face_index"),
                "sample_index": raw_contact.get("sample_index"),
                "grasp_type": "vacuum_contact",
            }
        )

    grasps: List[Dict[str, Any]] = []
    for idx, raw_grasp in enumerate(
        payload.get("grasps")
        or payload.get("grasp_candidates")
        or []
    ):
        if not isinstance(raw_grasp, dict):
            continue
        if raw_grasp.get("enabled", True) is False:
            continue
        contact_a_local_m = _vec3_or_none(
            raw_grasp.get("contact_a_local_m")
            or raw_grasp.get("point_a_local_m")
            or raw_grasp.get("left_contact_local_m")
        )
        contact_b_local_m = _vec3_or_none(
            raw_grasp.get("contact_b_local_m")
            or raw_grasp.get("point_b_local_m")
            or raw_grasp.get("right_contact_local_m")
        )
        normal_a_local = _normalize_vec(
            _vec3_or_none(
                raw_grasp.get("normal_a_local")
                or raw_grasp.get("contact_a_normal_local")
                or raw_grasp.get("left_normal_local")
            )
            or []
        )
        normal_b_local = _normalize_vec(
            _vec3_or_none(
                raw_grasp.get("normal_b_local")
                or raw_grasp.get("contact_b_normal_local")
                or raw_grasp.get("right_normal_local")
            )
            or []
        )
        if contact_a_local_m is None or contact_b_local_m is None:
            continue
        center_local_m = _vec3_or_none(raw_grasp.get("center_local_m"))
        if center_local_m is None:
            center_local_m = [
                0.5 * (contact_a_local_m[0] + contact_b_local_m[0]),
                0.5 * (contact_a_local_m[1] + contact_b_local_m[1]),
                0.5 * (contact_a_local_m[2] + contact_b_local_m[2]),
            ]
        jaw_axis_local = _normalize_vec(
            _vec3_or_none(raw_grasp.get("jaw_axis_local"))
            or [
                contact_b_local_m[0] - contact_a_local_m[0],
                contact_b_local_m[1] - contact_a_local_m[1],
                contact_b_local_m[2] - contact_a_local_m[2],
            ]
        )
        approach_axis_local = _normalize_vec(
            _vec3_or_none(raw_grasp.get("approach_axis_local"))
            or _vec3_or_none(raw_grasp.get("normal_local"))
            or []
        )
        if jaw_axis_local is None or approach_axis_local is None:
            continue
        approach_axis_local = _repair_legacy_symmetric_ring_approach_axis(
            payload=payload,
            raw_grasp=raw_grasp,
            raw_grasp_index=idx,
            jaw_axis_local=jaw_axis_local,
            stored_approach_axis_local=approach_axis_local,
        )
        if approach_axis_local is None:
            continue
        opening_width_m = _coerce_float(raw_grasp.get("opening_width_m"), None)
        if opening_width_m is None:
            opening_width_m = _distance_between(contact_a_local_m, contact_b_local_m)
        generator_group_index = int(
            _coerce_float(raw_grasp.get("generator_group_index"), 0.0) or 0
        ) or None
        raw_family_label = _normalize_grasp_family_label(
            raw_grasp.get("grasp_family_label")
            or raw_grasp.get("grasp_family")
            or raw_grasp.get("family_label")
        )
        mapped_family_label = (
            grasp_family_labels_map.get(int(generator_group_index))
            if generator_group_index is not None
            else None
        )
        grasps.append(
            {
                "id": str(raw_grasp.get("id") or f"grasp_{idx + 1:03d}"),
                "label": str(
                    raw_grasp.get("label")
                    or raw_grasp.get("name")
                    or raw_grasp.get("id")
                    or f"grasp_{idx + 1:03d}"
                ),
                "grasp_type": "parallel_jaw_pair",
                "contact_a_local_m": contact_a_local_m,
                "contact_b_local_m": contact_b_local_m,
                "normal_a_local": normal_a_local,
                "normal_b_local": normal_b_local,
                "center_local_m": center_local_m,
                "jaw_axis_local": jaw_axis_local,
                "approach_axis_local": approach_axis_local,
                "opening_width_m": float(opening_width_m),
                "face_indices": raw_grasp.get("face_indices"),
                "sample_indices": raw_grasp.get("sample_indices"),
                "generator_group_index": generator_group_index,
                "grasp_family_label": mapped_family_label or raw_family_label,
            }
        )

    if not contacts and not grasps:
        return None

    tool_axis = str(
        pick_cfg.get("pick_contacts_tool_axis")
        or payload.get("tool_axis")
        or payload.get("surface_align_axis")
        or "-z"
    ).strip().lower()
    force_yaw_raw = pick_cfg.get("pick_contacts_force_yaw_deg")
    if force_yaw_raw is None:
        force_yaw_raw = payload.get("force_yaw_deg")
    jaw_axis = str(
        pick_cfg.get("pick_contacts_jaw_axis")
        or payload.get("jaw_axis")
        or "x"
    ).strip().lower()

    default_gripper_type = _canonical_gripper_type(
        payload.get("default_gripper_type")
        or payload.get("gripper_type")
        or ("parallel_jaw" if grasps and not contacts else "vacuum")
    )
    selection_mode = str(
        payload.get("selection_mode") or "closest_to_tcp"
    ).strip().lower()
    if default_gripper_type == "parallel_jaw" and selection_mode in {
        "",
        "closest_contact_to_camera",
        "closest_to_camera",
    }:
        selection_mode = "camera_normal"

    return {
        "path": str(contacts_path),
        "source_mesh_path": None if source_mesh_path is None else str(source_mesh_path),
        "format": str(payload.get("format") or "vgr_pick_contacts/v1"),
        "default_gripper_type": default_gripper_type,
        "selection_mode": selection_mode,
        "tool_axis": tool_axis or "-z",
        "jaw_axis": jaw_axis or "x",
        "force_yaw_deg": _coerce_float(force_yaw_raw, 0.0),
        "mesh_units": mesh_units,
        "mesh_scale": float(mesh_scale or 1.0),
        "contacts": contacts,
        "grasps": grasps,
        "grasp_family_labels": {
            str(group_index): label
            for group_index, label in grasp_family_labels_map.items()
        },
        "grasp_family_labels_path": grasp_family_labels_path,
        "grasp_family_selection_policy": grasp_family_selection_policy,
    }


def _transform_points_by_pose(
    points_local_m: Any,
    pose_quat_xyzw: List[float],
    translation_m: List[float],
) -> List[List[float]]:
    """Apply (quat, trans) to Nx3 points. Returns list of [x,y,z] in metres."""
    try:
        import numpy as _np
    except Exception:
        return []
    if points_local_m is None:
        return []
    try:
        pts = _np.asarray(points_local_m, dtype=_np.float32).reshape(-1, 3)
    except Exception:
        return []
    if pts.size == 0:
        return []
    qx, qy, qz, qw = (float(v) for v in pose_quat_xyzw)
    tx, ty, tz = (float(v) for v in translation_m)
    # Quaternion rotation via cross products: v' = v + 2*q_vec × (q_vec × v + q_w*v)
    q_vec = _np.array([qx, qy, qz], dtype=_np.float32)
    w = _np.float32(qw)
    t = _np.cross(_np.broadcast_to(q_vec, pts.shape), pts) * _np.float32(2.0)
    rotated = pts + w * t + _np.cross(_np.broadcast_to(q_vec, pts.shape), t)
    rotated += _np.array([tx, ty, tz], dtype=_np.float32)
    return rotated.tolist()


def _extract_safety_points_base_m(
    match: Dict[str, Any],
    base_to_cam: Dict[str, Any],
) -> Dict[str, List[List[float]]]:
    """Pull target/neighbor point clouds from a vision match and transform to base frame.

    Returns {"target": [...], "neighbor": [...]}; empty lists if absent (fallback).
    Points in camera frame are transformed via base_to_cam (same chain used for grasp poses).
    """
    empty = {"target": [], "neighbor": []}
    safety_pcd = match.get("safety_pcd") if isinstance(match, dict) else None
    if not isinstance(safety_pcd, dict):
        return empty
    frame = str(safety_pcd.get("frame") or "camera").lower()
    target_raw = safety_pcd.get("target_points_camera_m") or []
    neighbor_raw = safety_pcd.get("neighbor_points_camera_m") or []
    if frame != "camera":
        # If already in base, return as-is.
        return {
            "target": [list(p) for p in target_raw if _is_finite_vec(p, 3)],
            "neighbor": [list(p) for p in neighbor_raw if _is_finite_vec(p, 3)],
        }
    cam_quat = base_to_cam.get("rotation_quat_xyzw") or [0.0, 0.0, 0.0, 1.0]
    cam_trans = base_to_cam.get("translation_m") or [0.0, 0.0, 0.0]
    if not _is_finite_vec(cam_quat, 4) or not _is_finite_vec(cam_trans, 3):
        return empty

    def _xform(pts: List[Any]) -> List[List[float]]:
        out: List[List[float]] = []
        for p in pts:
            if not _is_finite_vec(p, 3):
                continue
            rotated = _quat_rotate(cam_quat, [float(p[0]), float(p[1]), float(p[2])])
            out.append(_add_vec(rotated, cam_trans))
        return out

    return {"target": _xform(target_raw), "neighbor": _xform(neighbor_raw)}


def _scene_point_cloud_path(
    match: Dict[str, Any],
    *,
    runs_root: Optional[Path] = None,
    run_id: str = "",
) -> Optional[Path]:
    for container_key in ("visualization_3d", "debug_paths"):
        container = match.get(container_key)
        if not isinstance(container, dict):
            continue
        for key in ("scene_point_cloud_ply_path", "scene_point_cloud"):
            raw = str(container.get(key) or "").strip()
            if raw:
                path = Path(raw)
                if path.exists() and path.is_file():
                    return path
                if runs_root is not None and run_id:
                    parts = path.parts
                    if len(parts) >= 3:
                        candidate = Path(runs_root) / run_id / "vision" / parts[-3] / parts[-2] / parts[-1]
                        if candidate.exists() and candidate.is_file():
                            return candidate
    return None


def _load_ply_xyz_points(path: Path) -> np.ndarray:
    with path.open("rb") as fh:
        fmt = ""
        vertex_count = 0
        properties: List[tuple[str, str]] = []
        in_vertex = False
        while True:
            raw = fh.readline()
            if not raw:
                break
            line = raw.decode("utf-8", errors="ignore").strip()
            if line.startswith("format "):
                fmt = line.split()[1]
            elif line.startswith("element "):
                parts = line.split()
                in_vertex = len(parts) >= 3 and parts[1] == "vertex"
                if in_vertex:
                    vertex_count = int(parts[2])
            elif in_vertex and line.startswith("property "):
                parts = line.split()
                if len(parts) >= 3 and parts[1] != "list":
                    properties.append((parts[1], parts[2]))
            elif line == "end_header":
                break

        if vertex_count <= 0 or not properties:
            return np.empty((0, 3), dtype=np.float32)

        x_idx = next((i for i, (_, name) in enumerate(properties) if name == "x"), -1)
        y_idx = next((i for i, (_, name) in enumerate(properties) if name == "y"), -1)
        z_idx = next((i for i, (_, name) in enumerate(properties) if name == "z"), -1)
        if min(x_idx, y_idx, z_idx) < 0:
            return np.empty((0, 3), dtype=np.float32)

        if fmt == "ascii":
            pts: List[List[float]] = []
            for _ in range(vertex_count):
                parts = fh.readline().decode("utf-8", errors="ignore").split()
                if len(parts) > max(x_idx, y_idx, z_idx):
                    pts.append([float(parts[x_idx]), float(parts[y_idx]), float(parts[z_idx])])
            return np.asarray(pts, dtype=np.float32).reshape(-1, 3)

        if fmt != "binary_little_endian":
            return np.empty((0, 3), dtype=np.float32)

        dtype_map = {
            "char": "i1",
            "uchar": "u1",
            "int8": "i1",
            "uint8": "u1",
            "short": "<i2",
            "ushort": "<u2",
            "int16": "<i2",
            "uint16": "<u2",
            "int": "<i4",
            "uint": "<u4",
            "int32": "<i4",
            "uint32": "<u4",
            "float": "<f4",
            "float32": "<f4",
            "double": "<f8",
            "float64": "<f8",
        }
        dtype_fields = []
        for idx, (typ, name) in enumerate(properties):
            np_type = dtype_map.get(typ)
            if np_type is None:
                return np.empty((0, 3), dtype=np.float32)
            dtype_fields.append((name or f"field_{idx}", np_type))
        data = np.frombuffer(fh.read(), dtype=np.dtype(dtype_fields), count=vertex_count)
        return np.column_stack([data["x"], data["y"], data["z"]]).astype(np.float32)


def _extract_scene_point_cloud_base_m(
    match: Dict[str, Any],
    base_to_cam: Dict[str, Any],
    *,
    runs_root: Optional[Path] = None,
    run_id: str = "",
) -> Optional[List[List[float]]]:
    path = _scene_point_cloud_path(match, runs_root=runs_root, run_id=run_id)
    if path is None:
        return None
    try:
        pts = _load_ply_xyz_points(path)
        pts = pts[np.all(np.isfinite(pts), axis=1)]
        if len(pts) == 0:
            return None
        rot = np.asarray(
            _quat_xyzw_to_matrix(
                base_to_cam.get("rotation_quat_xyzw") or [0.0, 0.0, 0.0, 1.0]
            ),
            dtype=np.float64,
        )
        trans = np.asarray(base_to_cam.get("translation_m") or [0.0, 0.0, 0.0], dtype=np.float64)
        pts_base = (pts.astype(np.float64) @ rot.T) + trans[None, :]
        return pts_base.astype(np.float32).tolist()
    except Exception:
        return None


_GRASP_COLLISION_PIPELINE_CACHE: Dict[str, tuple[Any, Dict[str, Any]]] = {}
_POINTCLOUD_COLLISION_MESH_CACHE: Dict[str, Dict[str, Any]] = {}


def _default_grasp_collision_scene_yaml_path() -> str:
    return str(
        (
            Path(__file__).resolve().parents[3]
            / "data/stations/station-1/assets/asset-1/bin_picking/scene.yaml"
        ).resolve()
    )


def _task_obstacles_scene_yaml(ctx: StationContext, state: RunState) -> Optional[str]:
    """The per-task scene.yaml (tasks/<task_id>/scene.yaml) holding the obstacle set
    the operator edits, if it exists. None otherwise."""
    try:
        station_id = str(getattr(state, "station_id", "") or "").strip()
        process_id = str(getattr(state, "process_id", "") or "").strip()
        task_id = str(getattr(state, "task_id", "") or "").strip()
        if station_id and process_id and task_id:
            p = ctx.data_paths.process_task_scene_dir(station_id, process_id, task_id) / "scene.yaml"
            if p.exists():
                return str(p.resolve())
    except Exception:
        pass
    return None


def _gripper_collision_dir_for_scene(scene_path: Path) -> Path:
    """Find <asset>/gripper/collision walking up from a scene.yaml path.
    Works whether the scene is at <asset>/bin_picking/scene.yaml or
    <asset>/tasks/<task_id>/scene.yaml."""
    for parent in [scene_path.parent, *scene_path.parents]:
        cand = parent / "gripper" / "collision"
        if cand.is_dir():
            return cand
    # Last-resort fallback (original behaviour): two levels up from the scene.
    return scene_path.parent.parent / "gripper" / "collision"


def _load_scene_obstacles(scene_path: Path) -> List[Dict[str, Any]]:
    """Return the environment.obstacles list from a scene.yaml (mesh entries only),
    each as {name, abs_mesh_path, transform_4x4, scale}."""
    out: List[Dict[str, Any]] = []
    try:
        import yaml as _yaml
        data = _yaml.safe_load(scene_path.read_text(encoding="utf-8")) or {}
        obstacles = (data.get("environment") or {}).get("obstacles") or []
        scene_dir = scene_path.parent
        for obs in obstacles:
            if not isinstance(obs, dict) or obs.get("type") != "mesh":
                continue
            rel = str(obs.get("path") or "").strip()
            if not rel:
                continue
            mesh_path = Path(rel) if Path(rel).is_absolute() else (scene_dir / rel)
            if not mesh_path.exists():
                continue
            tf = obs.get("transform") or {}
            T = np.eye(4, dtype=np.float64)
            if isinstance(tf, dict):
                trans = tf.get("translation") or [0.0, 0.0, 0.0]
                rot = tf.get("rotation")
                if rot is not None:
                    try:
                        T[:3, :3] = np.asarray(rot, dtype=np.float64).reshape(3, 3)
                    except Exception:
                        pass
                T[:3, 3] = np.asarray([float(v) for v in trans[:3]], dtype=np.float64)
            else:
                try:
                    T = np.asarray(tf, dtype=np.float64).reshape(4, 4)
                except Exception:
                    T = np.eye(4, dtype=np.float64)
            out.append({
                "name": str(obs.get("name") or f"obstacle_{len(out) + 1}"),
                "abs_mesh_path": str(mesh_path.resolve()),
                "transform_4x4": T,
                "scale": float(obs.get("scale") or 1.0),
            })
    except Exception:
        pass
    return out


def _station_gripper_collision_dir() -> Path:
    return (
        Path(__file__).resolve().parents[3]
        / "data/stations/station-1/assets/asset-1/gripper/collision"
    ).resolve()


def _filter_obstacle_points_outside_target_region(
    obstacle_points_base_m: Any,
    target_points_base_m: Any,
    *,
    exclusion_radius_m: float = 0.012,
) -> List[List[float]]:
    """Remove obstacle-cloud points that overlap the segmented target/object region."""
    try:
        obs_raw = [] if obstacle_points_base_m is None else obstacle_points_base_m
        target_raw = [] if target_points_base_m is None else target_points_base_m
        obs = np.asarray(obs_raw, dtype=np.float32).reshape(-1, 3)
        target = np.asarray(target_raw, dtype=np.float32).reshape(-1, 3)
    except Exception:
        return []
    if len(obs) == 0 or len(target) == 0:
        return obs.astype(float).tolist()
    radius_sq = float(exclusion_radius_m) * float(exclusion_radius_m)
    keep = np.ones(len(obs), dtype=bool)
    chunk = 4096
    target64 = target.astype(np.float64)
    for start in range(0, len(obs), chunk):
        block = obs[start:start + chunk].astype(np.float64)
        diff = block[:, None, :] - target64[None, :, :]
        min_d2 = np.min(np.sum(diff * diff, axis=2), axis=1)
        keep[start:start + chunk] = min_d2 > radius_sq
    return obs[keep].astype(float).tolist()


def _parse_mesh_vertices_for_aabb(mesh_path: Path, scale: float = 1.0) -> List[List[float]]:
    if not mesh_path.exists() or not mesh_path.is_file():
        return []
    suffix = mesh_path.suffix.lower()
    vertices: List[List[float]] = []
    try:
        if suffix == ".obj":
            for raw in mesh_path.read_text(encoding="utf-8", errors="ignore").splitlines():
                line = raw.strip()
                if not line.startswith("v "):
                    continue
                parts = line.split()
                if len(parts) >= 4:
                    vertices.append(
                        [
                            float(parts[1]) * float(scale),
                            float(parts[2]) * float(scale),
                            float(parts[3]) * float(scale),
                        ]
                    )
            return vertices
        if suffix == ".stl":
            data = mesh_path.read_bytes()
            if len(data) >= 84:
                tri_count = int(np.frombuffer(data[80:84], dtype="<u4", count=1)[0])
                expected = 84 + tri_count * 50
                if expected == len(data):
                    off = 84
                    for _ in range(tri_count):
                        off += 12
                        for _v in range(3):
                            x, y, z = np.frombuffer(data[off:off + 12], dtype="<f4", count=3)
                            vertices.append(
                                [
                                    float(x) * float(scale),
                                    float(y) * float(scale),
                                    float(z) * float(scale),
                                ]
                            )
                            off += 12
                        off += 2
                    return vertices
            text = data.decode("utf-8", errors="ignore")
            for raw in text.splitlines():
                parts = raw.strip().split()
                if len(parts) == 4 and parts[0].lower() == "vertex":
                    vertices.append(
                        [
                            float(parts[1]) * float(scale),
                            float(parts[2]) * float(scale),
                            float(parts[3]) * float(scale),
                        ]
                    )
    except Exception:
        return []
    return vertices


def _scene_obstacle_aabb(
    scene_yaml_path: Optional[str],
    *,
    margin_m: float = 0.03,
) -> Optional[tuple[np.ndarray, np.ndarray]]:
    if not scene_yaml_path:
        return None
    pts_all: List[np.ndarray] = []
    try:
        for obs in _load_scene_obstacles(Path(scene_yaml_path)):
            mesh_path = Path(str(obs.get("abs_mesh_path") or ""))
            scale = float(obs.get("scale") or 1.0)
            vertices = _parse_mesh_vertices_for_aabb(mesh_path, scale=scale)
            if not vertices:
                continue
            pts = np.asarray(vertices, dtype=np.float64).reshape(-1, 3)
            T = np.asarray(obs.get("transform_4x4"), dtype=np.float64).reshape(4, 4)
            pts_h = np.concatenate([pts, np.ones((len(pts), 1), dtype=np.float64)], axis=1)
            pts_w = (T @ pts_h.T).T[:, :3]
            pts_all.append(pts_w)
    except Exception:
        return None
    if not pts_all:
        return None
    all_pts = np.concatenate(pts_all, axis=0)
    margin = float(max(0.0, margin_m))
    return all_pts.min(axis=0) - margin, all_pts.max(axis=0) + margin


def _filter_points_to_aabb(
    points: np.ndarray,
    aabb: Optional[tuple[np.ndarray, np.ndarray]],
) -> np.ndarray:
    if aabb is None or len(points) == 0:
        return points
    lo, hi = aabb
    mask = np.all((points >= lo[None, :]) & (points <= hi[None, :]), axis=1)
    return points[mask]


def _voxel_surface_mesh_from_points(
    points: np.ndarray,
    *,
    voxel_size_m: float,
    min_cluster_points: int,
    min_cluster_voxels: int,
    max_clusters: int,
    max_voxels: int,
) -> Dict[str, Any]:
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    pts = pts[np.all(np.isfinite(pts), axis=1)]
    voxel = max(0.001, float(voxel_size_m or 0.001))
    empty = {
        "points": pts.astype(np.float32),
        "vertices": [],
        "indices": [],
        "stats": {
            "input_points": int(len(pts)),
            "filtered_points": 0,
            "voxel_size_m": voxel,
            "component_count": 0,
            "kept_component_count": 0,
            "kept_voxel_count": 0,
        },
    }
    if len(pts) == 0:
        return empty
    cells = np.floor(pts / voxel).astype(np.int64)
    unique, inverse, counts = np.unique(
        cells, axis=0, return_inverse=True, return_counts=True
    )
    point_counts = np.asarray(counts, dtype=np.int64)
    occupied = {tuple(int(v) for v in row): idx for idx, row in enumerate(unique)}
    visited = np.zeros(len(unique), dtype=bool)
    components: List[Dict[str, Any]] = []
    neighbors = [
        (1, 0, 0), (-1, 0, 0),
        (0, 1, 0), (0, -1, 0),
        (0, 0, 1), (0, 0, -1),
    ]
    for start_idx, row in enumerate(unique):
        if visited[start_idx]:
            continue
        stack = [tuple(int(v) for v in row)]
        visited[start_idx] = True
        comp_indices: List[int] = []
        comp_point_count = 0
        while stack:
            cell = stack.pop()
            idx = occupied[cell]
            comp_indices.append(idx)
            comp_point_count += int(point_counts[idx])
            cx, cy, cz = cell
            for dx, dy, dz in neighbors:
                ncell = (cx + dx, cy + dy, cz + dz)
                nidx = occupied.get(ncell)
                if nidx is None or visited[nidx]:
                    continue
                visited[nidx] = True
                stack.append(ncell)
        components.append(
            {
                "indices": comp_indices,
                "point_count": comp_point_count,
                "voxel_count": len(comp_indices),
            }
        )
    components.sort(
        key=lambda item: (int(item["point_count"]), int(item["voxel_count"])),
        reverse=True,
    )
    kept_components: List[Dict[str, Any]] = []
    for comp in components:
        if int(comp["point_count"]) < int(min_cluster_points):
            continue
        if int(comp["voxel_count"]) < int(min_cluster_voxels):
            continue
        kept_components.append(comp)
        if max_clusters > 0 and len(kept_components) >= int(max_clusters):
            break
    kept_cell_indices: List[int] = [
        idx for comp in kept_components for idx in (comp.get("indices") or [])
    ]
    if max_voxels > 0 and len(kept_cell_indices) > int(max_voxels):
        kept_cell_indices = kept_cell_indices[: int(max_voxels)]
    if not kept_cell_indices:
        empty["stats"].update(
            {
                "component_count": len(components),
                "largest_component_points": int(components[0]["point_count"]) if components else 0,
                "largest_component_voxels": int(components[0]["voxel_count"]) if components else 0,
            }
        )
        return empty
    kept_set = {tuple(int(v) for v in unique[idx]) for idx in kept_cell_indices}
    keep_point_mask = np.isin(inverse, np.asarray(kept_cell_indices, dtype=np.int64))
    filtered_points = pts[keep_point_mask]
    vertices: List[List[float]] = []
    indices: List[int] = []
    vertex_index: Dict[tuple[float, float, float], int] = {}

    def _v(coord: tuple[float, float, float]) -> int:
        key = (round(coord[0], 7), round(coord[1], 7), round(coord[2], 7))
        existing = vertex_index.get(key)
        if existing is not None:
            return existing
        vertex_index[key] = len(vertices)
        vertices.append([float(coord[0]), float(coord[1]), float(coord[2])])
        return len(vertices) - 1

    face_defs = [
        ((-1, 0, 0), [(0, 0, 0), (0, 0, 1), (0, 1, 1), (0, 1, 0)]),
        ((1, 0, 0), [(1, 0, 0), (1, 1, 0), (1, 1, 1), (1, 0, 1)]),
        ((0, -1, 0), [(0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1)]),
        ((0, 1, 0), [(0, 1, 0), (0, 1, 1), (1, 1, 1), (1, 1, 0)]),
        ((0, 0, -1), [(0, 0, 0), (0, 1, 0), (1, 1, 0), (1, 0, 0)]),
        ((0, 0, 1), [(0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)]),
    ]
    for cell in kept_set:
        cx, cy, cz = cell
        origin = np.asarray([cx, cy, cz], dtype=np.float64) * voxel
        for normal, corners in face_defs:
            ncell = (cx + normal[0], cy + normal[1], cz + normal[2])
            if ncell in kept_set:
                continue
            ids = [
                _v(
                    (
                        origin[0] + corner[0] * voxel,
                        origin[1] + corner[1] * voxel,
                        origin[2] + corner[2] * voxel,
                    )
                )
                for corner in corners
            ]
            indices.extend([ids[0], ids[1], ids[2], ids[0], ids[2], ids[3]])
    return {
        "points": filtered_points.astype(np.float32),
        "vertices": vertices,
        "indices": indices,
        "stats": {
            "input_points": int(len(pts)),
            "filtered_points": int(len(filtered_points)),
            "voxel_size_m": voxel,
            "component_count": int(len(components)),
            "kept_component_count": int(len(kept_components)),
            "kept_voxel_count": int(len(kept_cell_indices)),
            "largest_component_points": int(components[0]["point_count"]) if components else 0,
            "largest_component_voxels": int(components[0]["voxel_count"]) if components else 0,
            "vertex_count": len(vertices),
            "triangle_count": len(indices) // 3,
        },
    }


def _write_ascii_stl_mesh(path: Path, vertices: List[List[float]], indices: List[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        fh.write("solid pointcloud_collision_mesh\n")
        for i in range(0, len(indices), 3):
            try:
                a = vertices[int(indices[i])]
                b = vertices[int(indices[i + 1])]
                c = vertices[int(indices[i + 2])]
            except Exception:
                continue
            fh.write("  facet normal 0 0 0\n")
            fh.write("    outer loop\n")
            fh.write(f"      vertex {a[0]} {a[1]} {a[2]}\n")
            fh.write(f"      vertex {b[0]} {b[1]} {b[2]}\n")
            fh.write(f"      vertex {c[0]} {c[1]} {c[2]}\n")
            fh.write("    endloop\n")
            fh.write("  endfacet\n")
        fh.write("endsolid pointcloud_collision_mesh\n")


def _pointcloud_collision_mesh(
    points: Any,
    *,
    scene_yaml_path: Optional[str],
    voxel_size_m: float,
    min_cluster_points: int,
    min_cluster_voxels: int,
    max_clusters: int,
    max_voxels: int,
    bin_margin_m: float,
) -> Dict[str, Any]:
    raw = np.asarray([] if points is None else points, dtype=np.float64).reshape(-1, 3)
    raw = raw[np.all(np.isfinite(raw), axis=1)]
    aabb = _scene_obstacle_aabb(scene_yaml_path, margin_m=bin_margin_m)
    clipped = _filter_points_to_aabb(raw, aabb)
    mesh = _voxel_surface_mesh_from_points(
        clipped,
        voxel_size_m=voxel_size_m,
        min_cluster_points=min_cluster_points,
        min_cluster_voxels=min_cluster_voxels,
        max_clusters=max_clusters,
        max_voxels=max_voxels,
    )
    mesh["stats"]["aabb_filter_enabled"] = aabb is not None
    mesh["stats"]["aabb_input_points"] = int(len(raw))
    mesh["stats"]["aabb_kept_points"] = int(len(clipped))
    if mesh.get("vertices") and mesh.get("indices"):
        verts = np.asarray(mesh["vertices"], dtype=np.float32)
        idx = np.asarray(mesh["indices"], dtype=np.int64)
        digest = hashlib.sha1()
        digest.update(np.asarray([float(voxel_size_m)], dtype=np.float32).tobytes())
        digest.update(verts.tobytes())
        digest.update(idx.tobytes())
        key = digest.hexdigest()
        cached = _POINTCLOUD_COLLISION_MESH_CACHE.get(key)
        if cached is None:
            mesh_dir = Path("/tmp/dexsent_vgr_collision_meshes")
            mesh_path = mesh_dir / f"pointcloud_mesh_{key}.stl"
            _write_ascii_stl_mesh(mesh_path, mesh["vertices"], mesh["indices"])
            cached = {"mesh_path": str(mesh_path), "cache_key": key}
            _POINTCLOUD_COLLISION_MESH_CACHE[key] = cached
        mesh.update(cached)
    return mesh


def _per_finger_m(stroke_mm: float) -> float:
    return max(0.0, min(0.04, float(stroke_mm or 0.0) / 2000.0))


def _parallel_jaw_pregrasp_collision_width_m(
    pick_cfg: Optional[Dict[str, Any]],
    grasp: Dict[str, Any],
    family_label: Optional[str],
) -> Dict[str, Any]:
    cfg = pick_cfg if isinstance(pick_cfg, dict) else {}
    nominal_width_m = _coerce_float(grasp.get("opening_width_m"), None)
    if nominal_width_m is None:
        nominal_width_m = _coerce_float(grasp.get("stroke_mm"), 0.0)
        nominal_width_m = (nominal_width_m or 0.0) / 1000.0
    family_mode = "internal" if family_label == "internal" else "external"
    legacy_open_offset = _coerce_float(
        cfg.get("parallel_jaw_pregrasp_open_offset_m"),
        _coerce_float(cfg.get("parallel_jaw_open_offset_m"), 0.02),
    ) or 0.0
    open_offset = _coerce_float(
        cfg.get(f"parallel_jaw_{family_mode}_pregrasp_open_offset_m"),
        legacy_open_offset,
    ) or 0.0
    if family_mode == "internal":
        pregrasp_width_m = max(0.0, float(nominal_width_m or 0.0) - open_offset)
        action = "close"
    else:
        pregrasp_width_m = max(0.0, float(nominal_width_m or 0.0) + open_offset)
        action = "open"
    return {
        "nominal_width_m": float(nominal_width_m or 0.0),
        "pregrasp_width_m": pregrasp_width_m,
        "collision_stroke_mm": pregrasp_width_m * 1000.0,
        "pregrasp_open_offset_m": open_offset,
        "pregrasp_action": action,
        "grasp_family_mode": family_mode,
    }


def _grasp_collision_pipeline(scene_yaml_path: str, obstacles_scene_yaml_path: Optional[str] = None):
    """Lazy-build and cache robot_engine pointcloud + gripper mesh collision state.

    `scene_yaml_path` provides the robot/URDF/collision semantics (must be a full
    scene.yaml). `obstacles_scene_yaml_path`, if given, supplies the mesh obstacle
    set (e.g. the per-task tasks/<task_id>/scene.yaml the operator edits) — those
    replace whatever environment.obstacles the semantics file shipped with.
    """
    sem_path = str(Path(scene_yaml_path).expanduser().resolve())
    obs_path = (
        str(Path(obstacles_scene_yaml_path).expanduser().resolve())
        if obstacles_scene_yaml_path
        else ""
    )
    cache_key = f"{sem_path}|{obs_path}"
    cached = _GRASP_COLLISION_PIPELINE_CACHE.get(cache_key)
    if cached is not None:
        return cached
    from robot_engine.environment.mesh_obstacle import MeshObstacle
    from robot_engine.planning_core import MotionPlanningPipeline

    pipeline = MotionPlanningPipeline.from_config(sem_path)

    # If a separate obstacles scene was supplied (and differs from the semantics
    # file), replace the mesh obstacles with that one's — keeping non-mesh ones.
    if obs_path and obs_path != sem_path:
        try:
            existing = list(pipeline.list_obstacles())
            # Drop existing mesh obstacles (we'll re-add from the override). We
            # can't introspect type cheaply, so drop everything except the live
            # pointcloud slot name, then re-add the override meshes.
            for name in existing:
                if name == "bin_obstacles":
                    continue
                try:
                    pipeline.remove_obstacle(name)
                except Exception:
                    pass
            for obs in _load_scene_obstacles(Path(obs_path)):
                try:
                    pipeline.add_mesh_obstacle(
                        obs["name"],
                        obs["abs_mesh_path"],
                        obs["transform_4x4"],
                        scale=float(obs.get("scale") or 1.0),
                    )
                except Exception:
                    pass
        except Exception:
            pass

    collision_root = _gripper_collision_dir_for_scene(Path(obs_path or sem_path))
    gripper_meshes = {
        "fr3_leftfinger": MeshObstacle(
            "fr3_leftfinger",
            collision_root / "finger.stl",
            np.eye(4),
            scale=0.001,
        ),
        "fr3_rightfinger": MeshObstacle(
            "fr3_rightfinger",
            collision_root / "finger.stl",
            np.eye(4),
            scale=0.001,
        ),
    }
    state = {"gripper_meshes": gripper_meshes}
    _GRASP_COLLISION_PIPELINE_CACHE[cache_key] = (pipeline, state)
    return pipeline, state


def _translation_4x4(xyz: List[float] | tuple[float, float, float]) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, 3] = np.asarray(xyz, dtype=np.float64)
    return T


def _rpy_4x4(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    Ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    Rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = Rz @ Ry @ Rx
    return T


def _tcp_pose_to_gripper_root_pose(
    T_world_tcp_4x4: Any,
    tcp_offset_m: tuple[float, float, float] = (0.0, 0.0, 0.204),
) -> np.ndarray:
    T_link8_tcp = np.eye(4, dtype=np.float64)
    T_link8_tcp[:3, 3] = np.asarray(tcp_offset_m, dtype=np.float64)
    return np.asarray(T_world_tcp_4x4, dtype=np.float64) @ np.linalg.inv(T_link8_tcp)


def _parallel_jaw_tcp_matrix_from_axes(
    center_base_m: List[float],
    jaw_axis_base: List[float],
    approach_axis_base: List[float],
) -> np.ndarray:
    """Build the same TCP convention used by authored grasps: X=jaw, -Z=approach."""
    x_axis = np.asarray(_normalize_vec(list(jaw_axis_base)) or [1.0, 0.0, 0.0], dtype=np.float64)
    z_axis = -np.asarray(
        _normalize_vec(list(approach_axis_base)) or [0.0, 0.0, -1.0],
        dtype=np.float64,
    )
    y_axis = np.cross(z_axis, x_axis)
    norm = float(np.linalg.norm(y_axis))
    if norm < 1e-9:
        y_axis = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    else:
        y_axis = y_axis / norm
    x_axis = np.cross(y_axis, z_axis)
    x_axis = x_axis / max(float(np.linalg.norm(x_axis)), 1e-9)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = np.column_stack([x_axis, y_axis, z_axis])
    T[:3, 3] = np.asarray(center_base_m, dtype=np.float64)
    return T


def _set_gripper_mesh_poses(gripper_meshes: Dict[str, Any], T_world_root: np.ndarray, stroke_mm: float) -> None:
    per = _per_finger_m(stroke_mm)
    # Finger collision-mesh origin inside each finger link — must match the URDF
    # <collision><origin> for fr3_*finger (franka_hand.urdf): rpy="3.14159 0 -1.5708"
    # xyz="0 0.0063 0". The 0.0063 puts the inner pad face at the link frame so the
    # fingers touch at stroke 0.
    collision_origin = _translation_4x4((0.0, 0.0063, 0.0)) @ _rpy_4x4(3.14159, 0.0, -1.5708)
    left = (
        T_world_root
        @ _translation_4x4((0.0, 0.0, 0.0584))
        @ _translation_4x4((0.0, per, 0.0))
        @ collision_origin
    )
    right = (
        T_world_root
        @ _translation_4x4((0.0, 0.0, 0.0584))
        @ _rpy_4x4(0.0, 0.0, math.pi)
        @ _translation_4x4((0.0, per, 0.0))
        @ collision_origin
    )
    gripper_meshes["fr3_leftfinger"].set_transform(left)
    gripper_meshes["fr3_rightfinger"].set_transform(right)


def _mesh_aabb_contains_world_points(mesh_obstacle: Any, points_world: np.ndarray) -> bool:
    if points_world.size == 0:
        return False
    try:
        vertices = np.asarray(mesh_obstacle.mesh.vertices, dtype=np.float64).reshape(-1, 3)
        T = np.asarray(mesh_obstacle._T_world_mesh, dtype=np.float64).reshape(4, 4)
    except Exception:
        return False
    if len(vertices) == 0:
        return False
    lo = vertices.min(axis=0)
    hi = vertices.max(axis=0)
    rot = T[:3, :3]
    trans = T[:3, 3]
    chunk = 200000
    for start in range(0, len(points_world), chunk):
        local = (points_world[start:start + chunk].astype(np.float64) - trans[None, :]) @ rot
        inside = np.all((local >= lo[None, :]) & (local <= hi[None, :]), axis=1)
        if bool(np.any(inside)):
            return True
    return False


def _density_aware_downsample(
    points: np.ndarray,
    *,
    grid_m: float = 0.01,
    keep_all_below: int = 3,
    dense_keep_ratio: float = 0.34,
    seed: int = 0,
) -> np.ndarray:
    """Downsample a point cloud preserving spatial coverage.

    Points are binned into a `grid_m` grid. Cells with few points (<= keep_all_below)
    are kept in full so sparsely-sampled regions (object edges, thin features) keep
    every point. Dense cells are thinned to `dense_keep_ratio` of their points (at
    least keep_all_below). Net effect: thin where redundant, full where it's already
    sparse — instead of a flat voxel collapse that erases sparse regions too.
    """
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    n = len(pts)
    if n == 0 or grid_m <= 0.0:
        return pts.astype(np.float32)
    rng = np.random.default_rng(seed)
    cells = np.floor(pts / float(grid_m)).astype(np.int64)
    # Map each row to a cell key.
    _, inverse, counts = np.unique(cells, axis=0, return_inverse=True, return_counts=True)
    inverse = np.asarray(inverse).reshape(-1)
    order = np.argsort(inverse, kind="stable")
    keep_mask = np.zeros(n, dtype=bool)
    start = 0
    for cell_idx in range(len(counts)):
        cnt = int(counts[cell_idx])
        idxs = order[start:start + cnt]
        start += cnt
        if cnt <= int(keep_all_below):
            keep_mask[idxs] = True
            continue
        k = max(int(keep_all_below), int(math.ceil(cnt * float(dense_keep_ratio))))
        if k >= cnt:
            keep_mask[idxs] = True
        else:
            chosen = rng.choice(idxs, size=k, replace=False)
            keep_mask[chosen] = True
    return pts[keep_mask].astype(np.float32)


def _filter_grasps_by_pointcloud_collision(
    survivors: List[tuple[str, Any, float]],
    obstacle_points_base_m: Any,
    *,
    scene_yaml_path: str,
    obstacles_scene_yaml_path: Optional[str] = None,
    consider_pointcloud: bool = True,
    voxel_size_m: float = 0.01,
    inflation_m: float = 0.005,
    max_points: int = 200_000,
    density_grid_m: float = 0.0,
    mesh_min_cluster_points: int = 30,
    mesh_min_cluster_voxels: int = 4,
    mesh_max_clusters: int = 8,
    mesh_max_voxels: int = 20000,
    mesh_bin_margin_m: float = 0.03,
) -> Dict[str, Dict[str, Any]]:
    """Check candidate finger meshes against raw scene points and scene meshes."""
    pipeline, state = _grasp_collision_pipeline(scene_yaml_path, obstacles_scene_yaml_path)
    raw_points = [] if obstacle_points_base_m is None else obstacle_points_base_m
    pts = (
        np.asarray(raw_points, dtype=np.float32).reshape(-1, 3)
        if consider_pointcloud
        else np.empty((0, 3), dtype=np.float32)
    )
    if pts.size:
        pts = pts[np.all(np.isfinite(pts), axis=1)]
    mesh_obstacle_name = "pointcloud_object_mesh"
    try:
        if mesh_obstacle_name in set(pipeline.list_obstacles()):
            pipeline.remove_obstacle(mesh_obstacle_name)
    except Exception:
        pass

    # Mesh world obstacles (for example, the bin). The live scene point cloud is
    # checked directly as raw points below, without octree or mesh conversion.
    world_obstacles: List[tuple[str, Any]] = []
    try:
        for name in pipeline.list_obstacles():
            obs = pipeline._world.get_obstacle(name)
            if obs is not None and obs.coal_geometry() is not None:
                world_obstacles.append((str(name), obs))
    except Exception:
        world_obstacles = []

    out: Dict[str, Dict[str, Any]] = {}
    for grasp_id, T_world_tcp, stroke_mm in survivors:
        t0 = time.perf_counter()
        T_world_root = _tcp_pose_to_gripper_root_pose(T_world_tcp)
        pairs: List[tuple[str, str]] = []
        if world_obstacles or pts.size:
            from robot_engine.environment.planning_collision_world import _coal_collides

            gripper_meshes = state["gripper_meshes"]
            _set_gripper_mesh_poses(gripper_meshes, T_world_root, stroke_mm)
            for link_name in ("fr3_leftfinger", "fr3_rightfinger"):
                link = gripper_meshes[link_name]
                l_geom = link.coal_geometry()
                l_T = link.coal_transform()
                if pts.size and _mesh_aabb_contains_world_points(link, pts):
                    pairs.append((link_name, "raw_scene_point_cloud"))
                    break
                for obs_name, obs in world_obstacles:
                    if _coal_collides(l_geom, l_T, obs.coal_geometry(), obs.coal_transform()):
                        pairs.append((link_name, obs_name))
        passed = not pairs
        out[str(grasp_id)] = {
            "passed": passed,
            "reason": "" if passed else "pointcloud_collision",
            "pairs": pairs,
            "ms": (time.perf_counter() - t0) * 1000.0,
            "stroke_mm": float(stroke_mm or 0.0),
            "T_world_root": T_world_root.tolist(),
            "pointcloud_raw_stats": {
                "input_points": int(len(raw_points) if hasattr(raw_points, "__len__") else 0),
                "used_points": int(len(pts)),
            },
        }
    return out


def _infer_inner_grasp_from_normals(
    normal_a_local: Optional[List[float]],
    normal_b_local: Optional[List[float]],
    jaw_axis_local: Optional[List[float]],
) -> Optional[bool]:
    """Fallback inner/outer inference from contact normals when grasp_family_label is missing.

    Outer grasp: fingers close IN, object normals point AWAY from grasp center
        => dot(n_a, jaw_axis) and dot(n_b, jaw_axis) have OPPOSITE signs.
    Inner grasp: fingers expand OUT into cavity walls, normals point TOWARD center
        => the two dot products have the SAME sign (both inward relative to jaw axis).
    """
    if (
        not _is_finite_vec(normal_a_local, 3)
        or not _is_finite_vec(normal_b_local, 3)
        or not _is_finite_vec(jaw_axis_local, 3)
    ):
        return None
    jaw_n = _normalize_vec(list(jaw_axis_local))
    na_n = _normalize_vec(list(normal_a_local))
    nb_n = _normalize_vec(list(normal_b_local))
    if jaw_n is None or na_n is None or nb_n is None:
        return None
    da = _dot_vec(na_n, jaw_n)
    db = _dot_vec(nb_n, jaw_n)
    if abs(da) < 1e-3 or abs(db) < 1e-3:
        return None
    # Outer: opposite signs (canonical antipodal grasp).
    # Inner: same signs (both contact normals lean the same way along jaw axis).
    return (da * db) > 0.0


def _select_pick_contact(
    pick_contacts_model: Optional[Dict[str, Any]],
    object_center_base_m: List[float],
    object_quat_base_xyzw: List[float],
    reference_position_m: List[float],
    camera_position_m: Optional[List[float]] = None,
    reference_quat_base_xyzw: Optional[List[float]] = None,
    tcp_offset_rpy_deg: Optional[List[float]] = None,
    neighbor_points_base_m: Optional[List[List[float]]] = None,
    pick_cfg: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    if not pick_contacts_model:
        return None
    contacts = pick_contacts_model.get("contacts") or []
    if not contacts or not _is_finite_vec(object_center_base_m, 3):
        return None
    selection_mode = str(
        pick_contacts_model.get("selection_mode") or "closest_to_tcp"
    ).strip().lower()
    reference = (
        list(reference_position_m)
        if _is_finite_vec(reference_position_m, 3)
        else [0.0, 0.0, 0.0]
    )
    camera_reference = (
        list(camera_position_m)
        if _is_finite_vec(camera_position_m, 3)
        else [0.0, 0.0, 0.0]
    )
    cfg = pick_cfg if isinstance(pick_cfg, dict) else {}
    neighbors = neighbor_points_base_m or []
    cup_radius = float(cfg.get("vacuum_cup_radius_m") or 0.015)
    vacuum_margin = float(cfg.get("vacuum_safety_margin_m") or 0.003)
    vacuum_descent = float(
        cfg.get("vacuum_pregrasp_descent_m")
        or cfg.get("pick_z_offset_m")
        or 0.050
    )
    best = None
    best_key = None
    for contact in contacts:
        point_local_m = contact.get("point_local_m")
        normal_local = contact.get("normal_local")
        if not _is_finite_vec(point_local_m, 3):
            continue
        normal_base = _normalize_vec(_quat_rotate(object_quat_base_xyzw, normal_local))
        if normal_base is None:
            continue
        point_base_m = _add_vec(
            object_center_base_m,
            _quat_rotate(object_quat_base_xyzw, point_local_m),
        )
        distance_to_reference_m = _distance_between(point_base_m, reference)
        distance_to_camera_m = _distance_between(point_base_m, camera_reference)
        upward_score = float(normal_base[2])
        to_camera_dir = _normalize_vec(_subtract_vec(camera_reference, point_base_m))
        camera_alignment_score = (
            float(_dot_vec(normal_base, to_camera_dir))
            if to_camera_dir is not None
            else -1.0
        )

        # TCP-aware approach alignment: dot(current_tool_approach, required_approach).
        # Prefer contacts whose surface normal is closest to the TCP's natural pointing
        # direction, minimizing wrist/joint rotation to reach the contact.
        tcp_approach_alignment_score = float(normal_base[2])  # fallback = upward
        if reference_quat_base_xyzw is not None:
            from orchestrator.tasks._pick_runtime import _rpy_deg_to_quat_xyzw
            actual_tcp_q = _normalize_quat_xyzw(
                list(reference_quat_base_xyzw)
            )
            if tcp_offset_rpy_deg is not None:
                rpy = list(tcp_offset_rpy_deg)
                tcp_offset_q = _rpy_deg_to_quat_xyzw(
                    float(rpy[0]) if len(rpy) > 0 else 0.0,
                    float(rpy[1]) if len(rpy) > 1 else 0.0,
                    float(rpy[2]) if len(rpy) > 2 else 0.0,
                )
                actual_tcp_q = _quat_mul_xyzw(actual_tcp_q, tcp_offset_q)
            tool_approach = _normalize_vec(_quat_rotate(actual_tcp_q, [0.0, 0.0, -1.0]))
            if tool_approach is not None:
                required = [-normal_base[0], -normal_base[1], -normal_base[2]]
                tcp_approach_alignment_score = float(_dot_vec(tool_approach, required))

        # Vacuum safety: suction-cup path = cylinder from pregrasp to cup tip,
        # axis = -normal_base (approach), radius = cup_radius.
        safety_min_clearance = float("inf")
        safety_infeasible = False
        if neighbors:
            approach_n = [-normal_base[0], -normal_base[1], -normal_base[2]]
            pre_point = [
                point_base_m[i] - approach_n[i] * vacuum_descent
                for i in range(3)
            ]
            for pt in neighbors:
                d = _distance_point_to_segment(pt, point_base_m, pre_point) - cup_radius
                if d < safety_min_clearance:
                    safety_min_clearance = d
                    if d < -vacuum_margin:
                        break
            safety_infeasible = safety_min_clearance < vacuum_margin

        if selection_mode in {
            "upward_then_camera",
            "upward",
            "closest_upward",
            "upward_closest_camera",
        }:
            score_key = (
                safety_infeasible,
                -safety_min_clearance if math.isfinite(safety_min_clearance) else 0.0,
                -upward_score,
                distance_to_camera_m,
                distance_to_reference_m,
            )
        elif selection_mode in {
            "toward_camera_normal",
            "camera_normal",
            "normal_toward_camera",
            "camera_normal_then_point",
        }:
            score_key = (
                safety_infeasible,
                -safety_min_clearance if math.isfinite(safety_min_clearance) else 0.0,
                -camera_alignment_score,
                distance_to_camera_m,
                distance_to_reference_m,
            )
        elif selection_mode in {
            "min_rotation",
            "min_approach_rotation",
            "tcp_approach_aligned",
        }:
            # Primary: contact whose approach direction is most aligned with the
            # current TCP natural pointing direction, so the wrist needs less rotation.
            # Secondary: proximity to TCP tip (translation cost).
            score_key = (
                safety_infeasible,
                -safety_min_clearance if math.isfinite(safety_min_clearance) else 0.0,
                -tcp_approach_alignment_score,
                distance_to_reference_m,
            )
        else:
            score_key = (
                safety_infeasible,
                -safety_min_clearance if math.isfinite(safety_min_clearance) else 0.0,
                distance_to_reference_m,
            )
        if best_key is not None and score_key >= best_key:
            continue
        best_key = score_key
        best = {
            "id": contact.get("id"),
            "label": contact.get("label"),
            "point_local_m": list(point_local_m),
            "normal_local": list(normal_local),
            "point_base_m": point_base_m,
            "normal_base": normal_base,
            "distance_to_reference_m": distance_to_reference_m,
            "distance_to_camera_m": distance_to_camera_m,
            "upward_score": upward_score,
            "camera_alignment_score": camera_alignment_score,
            "face_index": contact.get("face_index"),
            "sample_index": contact.get("sample_index"),
            "safety_min_clearance_m": (
                None
                if not math.isfinite(safety_min_clearance)
                else float(safety_min_clearance)
            ),
            "safety_infeasible": bool(safety_infeasible),
            "safety_neighbor_count": len(neighbors),
        }
    return best


def _select_parallel_jaw_grasp(
    pick_contacts_model: Optional[Dict[str, Any]],
    object_center_base_m: List[float],
    object_quat_base_xyzw: List[float],
    reference_position_m: List[float],
    camera_position_m: Optional[List[float]] = None,
    camera_z_axis_base: Optional[List[float]] = None,
    reference_quat_base_xyzw: Optional[List[float]] = None,
    jaw_tool_axis: str = "x",
    approach_tool_axis: str = "-z",
    max_approach_angle_deg: Optional[float] = None,
    neighbor_points_base_m: Optional[List[List[float]]] = None,
    pick_cfg: Optional[Dict[str, Any]] = None,
    approach_reference_quat_base_xyzw: Optional[List[float]] = None,
) -> Optional[Dict[str, Any]]:
    if not pick_contacts_model:
        return None
    grasps = pick_contacts_model.get("grasps") or []
    if not grasps or not _is_finite_vec(object_center_base_m, 3):
        return None
    selection_mode = str(
        pick_contacts_model.get("selection_mode") or "closest_contact_to_camera"
    ).strip().lower()
    reference = (
        list(reference_position_m)
        if _is_finite_vec(reference_position_m, 3)
        else [0.0, 0.0, 0.0]
    )
    camera_reference = (
        list(camera_position_m)
        if _is_finite_vec(camera_position_m, 3)
        else [0.0, 0.0, 0.0]
    )
    camera_z_reference = (
        _normalize_vec(list(camera_z_axis_base))
        if _is_finite_vec(camera_z_axis_base, 3)
        else None
    )
    reference_quat = (
        _normalize_quat_xyzw(list(reference_quat_base_xyzw))
        if _is_finite_vec(reference_quat_base_xyzw, 4)
        else None
    )
    approach_reference_quat = (
        _normalize_quat_xyzw(list(approach_reference_quat_base_xyzw))
        if _is_finite_vec(approach_reference_quat_base_xyzw, 4)
        else reference_quat
    )
    approach_tool = _parse_signed_axis(approach_tool_axis)
    tool_approach_axis_base = None
    if approach_reference_quat is not None and approach_tool is not None:
        tool_axis_local = {
            "x": [1.0, 0.0, 0.0],
            "y": [0.0, 1.0, 0.0],
            "z": [0.0, 0.0, 1.0],
        }[approach_tool["axis"]]
        tool_axis_local = [
            tool_axis_local[0] * approach_tool["sign"],
            tool_axis_local[1] * approach_tool["sign"],
            tool_axis_local[2] * approach_tool["sign"],
        ]
        tool_approach_axis_base = _normalize_vec(
            _quat_rotate(approach_reference_quat, tool_axis_local)
        )
    jaw_tool = _parse_signed_axis(jaw_tool_axis)
    tool_jaw_axis_base = None
    if reference_quat is not None and jaw_tool is not None:
        jaw_axis_local = {
            "x": [1.0, 0.0, 0.0],
            "y": [0.0, 1.0, 0.0],
            "z": [0.0, 0.0, 1.0],
        }[jaw_tool["axis"]]
        jaw_axis_local = [
            jaw_axis_local[0] * jaw_tool["sign"],
            jaw_axis_local[1] * jaw_tool["sign"],
            jaw_axis_local[2] * jaw_tool["sign"],
        ]
        tool_jaw_axis_base = _normalize_vec(
            _quat_rotate(reference_quat, jaw_axis_local)
        )
    tool_camera_axis_alignment = (
        float(_dot_vec(tool_approach_axis_base, camera_z_reference))
        if tool_approach_axis_base is not None and camera_z_reference is not None
        else None
    )
    max_approach_angle_rad = _coerce_float(max_approach_angle_deg, None)
    min_alignment_cos = None
    if max_approach_angle_rad is not None:
        max_approach_angle_rad = math.radians(max(0.0, float(max_approach_angle_rad)))
        min_alignment_cos = math.cos(max_approach_angle_rad)
    grasp_family_mode = str((pick_cfg or {}).get("grasp_family_mode", "default"))
    family_selection_policy = _apply_grasp_family_mode(
        pick_contacts_model.get("grasp_family_selection_policy") or {},
        grasp_family_mode,
    )
    priority_groups_enabled = _grasp_family_policy_enabled(family_selection_policy)
    cfg_for_scoring = pick_cfg if isinstance(pick_cfg, dict) else {}
    max_jaw_axis_abs_z = _coerce_float(
        cfg_for_scoring.get("parallel_jaw_max_jaw_axis_z_abs"),
        None,
    )
    max_jaw_axis_tilt_deg = _coerce_float(
        cfg_for_scoring.get("parallel_jaw_max_jaw_axis_tilt_from_xy_deg"),
        20.0,
    )
    max_jaw_axis_tilt_deg = max(
        0.0,
        min(90.0, float(max_jaw_axis_tilt_deg or 20.0)),
    )
    if max_jaw_axis_abs_z is None:
        max_jaw_axis_abs_z = math.sin(
            math.radians(max_jaw_axis_tilt_deg)
        )
    max_jaw_axis_abs_z = max(0.0, min(1.0, float(max_jaw_axis_abs_z)))
    min_jaw_axis_alignment_score = _coerce_float(
        cfg_for_scoring.get("parallel_jaw_min_jaw_axis_alignment_score"),
        None,
    )
    if min_jaw_axis_alignment_score is None:
        min_jaw_axis_alignment_score = 0.0
    min_jaw_axis_alignment_score = max(
        -1.0,
        min(1.0, float(min_jaw_axis_alignment_score or 0.0)),
    )
    grasp_up_axis_raw = str(
        cfg_for_scoring.get("parallel_jaw_grasp_up_axis")
        or cfg_for_scoring.get("grasp_up_axis")
        or cfg_for_scoring.get("_tcp_grasp_up_axis")
        or ""
    ).strip().lower()
    grasp_up_axis = _parse_signed_axis(grasp_up_axis_raw)
    grasp_up_min_dot = _coerce_float(
        cfg_for_scoring.get("parallel_jaw_grasp_up_min_dot")
        if cfg_for_scoring.get("parallel_jaw_grasp_up_min_dot") is not None
        else (
            cfg_for_scoring.get("grasp_up_min_dot")
            if cfg_for_scoring.get("grasp_up_min_dot") is not None
            else cfg_for_scoring.get("_tcp_grasp_up_min_dot")
        ),
        0.0,
    )
    grasp_up_min_dot = max(-1.0, min(1.0, float(grasp_up_min_dot or 0.0)))
    grasp_up_local_axis = None
    if grasp_up_axis is not None:
        grasp_up_local_axis = {
            "x": [1.0, 0.0, 0.0],
            "y": [0.0, 1.0, 0.0],
            "z": [0.0, 0.0, 1.0],
        }[grasp_up_axis["axis"]]
        grasp_up_local_axis = [
            grasp_up_local_axis[0] * grasp_up_axis["sign"],
            grasp_up_local_axis[1] * grasp_up_axis["sign"],
            grasp_up_local_axis[2] * grasp_up_axis["sign"],
        ]
    consider_pointcloud_collision = _coerce_bool(
        cfg_for_scoring.get("pointcloud_collision_consider_pointcloud"),
        True,
    )
    collision_check_mode = str(
        cfg_for_scoring.get("pointcloud_collision_mode")
        or cfg_for_scoring.get("parallel_jaw_collision_mode")
        or "full"
    ).strip().lower()
    fast_collision_mode = collision_check_mode in {"fast", "fastmode", "fast_mode"}
    pending_collision_candidates: List[Dict[str, Any]] = []
    best = None
    best_key = None
    audit: Optional[Dict[str, Any]] = None
    if isinstance(cfg_for_scoring.get("_grasp_planning_audit"), dict):
        audit = cfg_for_scoring.get("_grasp_planning_audit")
        audit.clear()
        audit.update(
            {
                "selection_mode": selection_mode,
                "grasp_family_mode": grasp_family_mode,
                "object_center_base_m": list(object_center_base_m),
                "object_quat_base_xyzw": list(object_quat_base_xyzw),
                "approach_tool_axis": approach_tool_axis,
                "jaw_tool_axis": jaw_tool_axis,
                "max_approach_angle_deg": max_approach_angle_deg,
                "min_alignment_cos": min_alignment_cos,
                "max_jaw_axis_tilt_from_xy_deg": max_jaw_axis_tilt_deg,
                "max_jaw_axis_abs_z": max_jaw_axis_abs_z,
                "min_jaw_axis_alignment_score": min_jaw_axis_alignment_score,
                "tool_approach_axis_base": tool_approach_axis_base,
                "tool_jaw_axis_base": tool_jaw_axis_base,
                "approach_reference_quat_base_xyzw": approach_reference_quat,
                "jaw_reference_quat_base_xyzw": reference_quat,
                "grasp_up_axis": grasp_up_axis_raw,
                "grasp_up_min_dot": grasp_up_min_dot,
                "camera_z_axis_base": camera_z_reference,
                "pointcloud_collision_consider_pointcloud": consider_pointcloud_collision,
                "pointcloud_collision_mode": (
                    "fast" if fast_collision_mode else collision_check_mode
                ),
                "candidates": [],
            }
        )
    for grasp in grasps:
        candidate_audit: Dict[str, Any] = {
            "id": grasp.get("id"),
            "label": grasp.get("label"),
            "generator_group_index": grasp.get("generator_group_index"),
            "grasp_family_label": grasp.get("grasp_family_label"),
            "status": "rejected",
        }
        family_label = _normalize_grasp_family_label(
            grasp.get("grasp_family_label")
            or grasp.get("grasp_family")
            or grasp.get("family_label")
        )
        candidate_audit["normalized_family_label"] = family_label
        if grasp_family_mode.strip().lower() == "internal_only" and family_label != "internal":
            candidate_audit["reject_reason"] = "family_not_internal"
            if audit is not None:
                audit["candidates"].append(candidate_audit)
            continue
        if grasp_family_mode.strip().lower() == "external_only" and family_label != "external":
            candidate_audit["reject_reason"] = "family_not_external"
            if audit is not None:
                audit["candidates"].append(candidate_audit)
            continue
        family_priority_key = _grasp_family_policy_group_key(
            family_selection_policy,
            grasp.get("generator_group_index"),
        )
        if priority_groups_enabled and family_priority_key is None:
            candidate_audit["reject_reason"] = "family_priority_disabled"
            if audit is not None:
                audit["candidates"].append(candidate_audit)
            continue
        if family_priority_key is None:
            family_priority_key = (0, 0)
        contact_a_local_m = grasp.get("contact_a_local_m")
        contact_b_local_m = grasp.get("contact_b_local_m")
        center_local_m = grasp.get("center_local_m")
        jaw_axis_local = grasp.get("jaw_axis_local")
        approach_axis_local = grasp.get("approach_axis_local")
        if not (
            _is_finite_vec(contact_a_local_m, 3)
            and _is_finite_vec(contact_b_local_m, 3)
            and _is_finite_vec(center_local_m, 3)
            and _is_finite_vec(jaw_axis_local, 3)
            and _is_finite_vec(approach_axis_local, 3)
        ):
            candidate_audit["reject_reason"] = "invalid_grasp_geometry"
            if audit is not None:
                audit["candidates"].append(candidate_audit)
            continue
        contact_a_base_m = _add_vec(
            object_center_base_m,
            _quat_rotate(object_quat_base_xyzw, contact_a_local_m),
        )
        contact_b_base_m = _add_vec(
            object_center_base_m,
            _quat_rotate(object_quat_base_xyzw, contact_b_local_m),
        )
        center_base_m = _add_vec(
            object_center_base_m,
            _quat_rotate(object_quat_base_xyzw, center_local_m),
        )
        jaw_axis_base = _normalize_vec(_quat_rotate(object_quat_base_xyzw, jaw_axis_local))
        approach_axis_base = _normalize_vec(
            _quat_rotate(object_quat_base_xyzw, approach_axis_local)
        )
        if jaw_axis_base is None or approach_axis_base is None:
            candidate_audit["reject_reason"] = "invalid_transformed_axes"
            if audit is not None:
                audit["candidates"].append(candidate_audit)
            continue
        candidate_audit.update(
            {
                "center_base_m": center_base_m,
                "contact_a_base_m": contact_a_base_m,
                "contact_b_base_m": contact_b_base_m,
                "jaw_axis_base_initial": jaw_axis_base,
                "approach_axis_base": approach_axis_base,
            }
        )
        equivalent = _resolve_parallel_jaw_equivalent(
            contact_a_local_m=list(contact_a_local_m),
            contact_b_local_m=list(contact_b_local_m),
            normal_a_local=list(grasp.get("normal_a_local") or []),
            normal_b_local=list(grasp.get("normal_b_local") or []),
            contact_a_base_m=contact_a_base_m,
            contact_b_base_m=contact_b_base_m,
            jaw_axis_local=list(jaw_axis_local),
            jaw_axis_base=jaw_axis_base,
            approach_axis_base=approach_axis_base,
            jaw_tool_axis=jaw_tool_axis,
            approach_tool_axis=approach_tool_axis,
            reference_quat_xyzw=reference_quat,
            allow_contact_axis_flip=bool(
                cfg_for_scoring.get("parallel_jaw_allow_contact_axis_flip", False)
            ),
        )
        contact_a_local_m = equivalent["contact_a_local_m"]
        contact_b_local_m = equivalent["contact_b_local_m"]
        contact_a_base_m = equivalent["contact_a_base_m"]
        contact_b_base_m = equivalent["contact_b_base_m"]
        jaw_axis_local = equivalent["jaw_axis_local"]
        jaw_axis_base = equivalent["jaw_axis_base"]
        normal_a_local = equivalent["normal_a_local"]
        normal_b_local = equivalent["normal_b_local"]
        jaw_axis_alignment_score = (
            float(_dot_vec(jaw_axis_base, tool_jaw_axis_base))
            if tool_jaw_axis_base is not None
            else 1.0
        )
        explicit_orientation = equivalent.get("explicit_orientation")
        grasp_up_axis_base = None
        grasp_up_dot_base_z = None
        if grasp_up_local_axis is not None and _is_finite_vec(explicit_orientation, 4):
            grasp_up_axis_base = _normalize_vec(
                _quat_rotate(explicit_orientation, grasp_up_local_axis)
            )
            if grasp_up_axis_base is not None:
                grasp_up_dot_base_z = float(grasp_up_axis_base[2])
        candidate_audit.update(
            {
                "contact_a_base_m": contact_a_base_m,
                "contact_b_base_m": contact_b_base_m,
                "jaw_axis_base": jaw_axis_base,
                "jaw_axis_alignment_score": jaw_axis_alignment_score,
                "explicit_orientation_quat_xyzw": explicit_orientation,
                "grasp_up_axis": grasp_up_axis_raw,
                "grasp_up_axis_base": grasp_up_axis_base,
                "grasp_up_dot_base_z": grasp_up_dot_base_z,
                "grasp_up_min_dot": grasp_up_min_dot,
                "equivalent_flipped": bool(equivalent.get("equivalent_flipped")),
                "orientation_distance_rad": float(
                    equivalent.get("orientation_distance_rad") or math.pi
                ),
            }
        )
        to_camera_dir_preview = _normalize_vec(_subtract_vec(camera_reference, center_base_m))
        approach_alignment_score_preview = (
            float(_dot_vec(approach_axis_base, to_camera_dir_preview))
            if to_camera_dir_preview is not None
            else -1.0
        )
        tool_approach_alignment_score_preview = (
            float(_dot_vec(approach_axis_base, tool_approach_axis_base))
            if tool_approach_axis_base is not None
            else -1.0
        )
        threshold_alignment_score_preview = (
            tool_approach_alignment_score_preview
            if tool_approach_axis_base is not None
            else approach_alignment_score_preview
        )
        jaw_axis_abs_z_preview = abs(float(jaw_axis_base[2]))
        failed_filters_preview: List[str] = []
        if jaw_axis_alignment_score < min_jaw_axis_alignment_score:
            failed_filters_preview.append("jaw_axis_direction_mismatch")
        if jaw_axis_abs_z_preview > max_jaw_axis_abs_z:
            failed_filters_preview.append("jaw_axis_tilt_exceeds_limit")
        if (
            min_alignment_cos is not None
            and threshold_alignment_score_preview < min_alignment_cos
        ):
            failed_filters_preview.append("approach_cone_exceeded")
        if grasp_up_local_axis is not None and (
            grasp_up_dot_base_z is None or grasp_up_dot_base_z < grasp_up_min_dot
        ):
            failed_filters_preview.append("grasp_up_axis_not_facing_base_z")
        candidate_audit.update(
            {
                "approach_to_camera_score": approach_alignment_score_preview,
                "jaw_axis_abs_z": jaw_axis_abs_z_preview,
                "jaw_axis_tilt_from_xy_deg": math.degrees(
                    math.asin(max(0.0, min(1.0, jaw_axis_abs_z_preview)))
                ),
                "tool_approach_alignment_score": tool_approach_alignment_score_preview,
                "robot_z_alignment_score": tool_approach_alignment_score_preview,
                "tool_approach_angle_deg": math.degrees(
                    math.acos(
                        max(-1.0, min(1.0, tool_approach_alignment_score_preview))
                    )
                ),
                "threshold_alignment_score": threshold_alignment_score_preview,
                "threshold_alignment_angle_deg": math.degrees(
                    math.acos(
                        max(-1.0, min(1.0, threshold_alignment_score_preview))
                    )
                ),
                "failed_filters": failed_filters_preview,
            }
        )
        if jaw_axis_alignment_score < min_jaw_axis_alignment_score:
            candidate_audit["reject_reason"] = "jaw_axis_direction_mismatch"
            if audit is not None:
                audit["candidates"].append(candidate_audit)
            continue
        if grasp_up_local_axis is not None and (
            grasp_up_dot_base_z is None or grasp_up_dot_base_z < grasp_up_min_dot
        ):
            candidate_audit["reject_reason"] = "grasp_up_axis_not_facing_base_z"
            if audit is not None:
                audit["candidates"].append(candidate_audit)
            continue
        closest_contact_distance_to_camera_m = min(
            _distance_between(contact_a_base_m, camera_reference),
            _distance_between(contact_b_base_m, camera_reference),
        )
        center_distance_to_camera_m = _distance_between(center_base_m, camera_reference)
        distance_to_reference_m = _distance_between(center_base_m, reference)
        to_camera_dir = _normalize_vec(_subtract_vec(camera_reference, center_base_m))
        approach_axis_camera_z_score = (
            float(_dot_vec(approach_axis_base, camera_z_reference))
            if camera_z_reference is not None
            else -1.0
        )
        approach_alignment_score = (
            float(_dot_vec(approach_axis_base, to_camera_dir))
            if to_camera_dir is not None
            else -1.0
        )
        jaw_axis_abs_z = abs(float(jaw_axis_base[2]))
        jaw_axis_xy_score = max(0.0, 1.0 - jaw_axis_abs_z)
        jaw_axis_horizontal_ok = jaw_axis_abs_z <= max_jaw_axis_abs_z
        candidate_audit.update(
            {
                "closest_contact_distance_to_camera_m": closest_contact_distance_to_camera_m,
                "center_distance_to_camera_m": center_distance_to_camera_m,
                "distance_to_reference_m": distance_to_reference_m,
                "approach_to_camera_score": approach_alignment_score,
                "camera_z_alignment_score": approach_axis_camera_z_score,
                "jaw_axis_abs_z": jaw_axis_abs_z,
                "jaw_axis_tilt_from_xy_deg": math.degrees(
                    math.asin(max(0.0, min(1.0, jaw_axis_abs_z)))
                ),
                "jaw_axis_horizontal_ok": jaw_axis_horizontal_ok,
            }
        )
        if not jaw_axis_horizontal_ok:
            candidate_audit["reject_reason"] = "jaw_axis_tilt_exceeds_limit"
            if audit is not None:
                audit["candidates"].append(candidate_audit)
            continue
        tool_approach_alignment_score = (
            float(_dot_vec(approach_axis_base, tool_approach_axis_base))
            if tool_approach_axis_base is not None
            else -1.0
        )
        tool_approach_angle_deg = math.degrees(
            math.acos(max(-1.0, min(1.0, tool_approach_alignment_score)))
        )
        effective_camera_alignment_score = (
            approach_axis_camera_z_score
            if camera_z_reference is not None
            else approach_alignment_score
        )
        effective_tool_alignment_score = (
            tool_approach_alignment_score
            if tool_approach_axis_base is not None
            else approach_alignment_score
        )
        # The cone still rejects grasps that are not reachable from the camera
        # side, but final ranking should minimize rotation from the current
        # robot approach axis.
        approach_to_camera_score = approach_alignment_score
        threshold_alignment_score = (
            effective_tool_alignment_score
            if tool_approach_axis_base is not None
            else approach_to_camera_score
        )
        candidate_audit.update(
            {
                "tool_approach_alignment_score": tool_approach_alignment_score,
                "robot_z_alignment_score": tool_approach_alignment_score,
                "tool_approach_angle_deg": tool_approach_angle_deg,
                "threshold_alignment_score": threshold_alignment_score,
                "threshold_alignment_angle_deg": math.degrees(
                    math.acos(max(-1.0, min(1.0, threshold_alignment_score)))
                ),
            }
        )
        if min_alignment_cos is not None and threshold_alignment_score < min_alignment_cos:
            candidate_audit["reject_reason"] = "approach_cone_exceeded"
            if audit is not None:
                audit["candidates"].append(candidate_audit)
            continue
        orientation_distance_rad = float(
            equivalent.get("orientation_distance_rad") or math.pi
        )

        cfg_for_safety = pick_cfg if isinstance(pick_cfg, dict) else {}
        prefer_inner = bool(cfg_for_safety.get("parallel_jaw_prefer_inner_grasp", True))
        safety_is_inner = family_label == "internal"
        stroke_mm = float(
            _coerce_float(
                grasp.get("stroke_mm"),
                (float(grasp.get("opening_width_m") or 0.0) * 1000.0),
            )
            or 0.0
        )
        collision_widths = _parallel_jaw_pregrasp_collision_width_m(
            cfg_for_safety,
            grasp,
            family_label,
        )
        collision_stroke_mm = float(collision_widths.get("collision_stroke_mm") or 0.0)
        T_world_tcp = np.eye(4, dtype=np.float64)
        if _is_finite_vec(explicit_orientation, 4):
            T_world_tcp[:3, :3] = np.asarray(
                _quat_xyzw_to_matrix(list(explicit_orientation)),
                dtype=np.float64,
            )
        T_world_tcp[:3, 3] = np.asarray(center_base_m, dtype=np.float64)
        scene_yaml_path = str(
            cfg_for_safety.get("pointcloud_collision_scene_yaml")
            or cfg_for_safety.get("grasp_collision_scene_yaml")
            or _default_grasp_collision_scene_yaml_path()
        )
        obstacles_scene_yaml_path = (
            cfg_for_safety.get("pointcloud_collision_obstacles_scene_yaml") or None
        )
        if fast_collision_mode:
            inner_bonus = (-1 if (prefer_inner and safety_is_inner) else 0)
            if priority_groups_enabled:
                score_key = (
                    False,
                    -effective_tool_alignment_score,
                    -jaw_axis_alignment_score,
                    jaw_axis_abs_z,
                    family_priority_key[0],
                    family_priority_key[1],
                    -threshold_alignment_score,
                    orientation_distance_rad,
                    -approach_to_camera_score,
                    closest_contact_distance_to_camera_m,
                    center_distance_to_camera_m,
                    distance_to_reference_m,
                )
            elif selection_mode in {
                "camera_normal_then_point",
                "toward_camera_normal",
                "camera_normal",
                "normal_toward_camera",
            }:
                score_key = (
                    False,
                    -effective_tool_alignment_score,
                    -jaw_axis_alignment_score,
                    jaw_axis_abs_z,
                    inner_bonus,
                    -approach_to_camera_score,
                    -effective_camera_alignment_score,
                    orientation_distance_rad,
                    closest_contact_distance_to_camera_m,
                    center_distance_to_camera_m,
                    distance_to_reference_m,
                )
            elif selection_mode in {
                "closest_center_to_camera",
                "center_to_camera",
                "camera_center",
            }:
                score_key = (
                    False,
                    -effective_tool_alignment_score,
                    -jaw_axis_alignment_score,
                    jaw_axis_abs_z,
                    inner_bonus,
                    -approach_to_camera_score,
                    center_distance_to_camera_m,
                    closest_contact_distance_to_camera_m,
                    distance_to_reference_m,
                )
            else:
                score_key = (
                    False,
                    -effective_tool_alignment_score,
                    -jaw_axis_alignment_score,
                    jaw_axis_abs_z,
                    inner_bonus,
                    -approach_to_camera_score,
                    -effective_camera_alignment_score,
                    orientation_distance_rad,
                    closest_contact_distance_to_camera_m,
                    center_distance_to_camera_m,
                    distance_to_reference_m,
                )
            candidate_audit.update(
                {
                    "pointcloud_collision_consider_pointcloud": consider_pointcloud_collision,
                    "pointcloud_collision_mode": "fast",
                    "stroke_mm": collision_stroke_mm,
                    "authored_stroke_mm": stroke_mm,
                    "pregrasp_open_width_m": collision_widths.get("pregrasp_width_m"),
                    "pregrasp_open_offset_m": collision_widths.get("pregrasp_open_offset_m"),
                    "pregrasp_action": collision_widths.get("pregrasp_action"),
                    "safety_is_inner_grasp": safety_is_inner,
                    "safety_neighbor_count": len(neighbor_points_base_m or []),
                    "collision_sort_key": [
                        float(candidate_audit.get("threshold_alignment_angle_deg") or 180.0),
                        -float(threshold_alignment_score),
                        jaw_axis_abs_z,
                        orientation_distance_rad,
                    ],
                }
            )
            pending_collision_candidates.append(
                {
                    "audit": candidate_audit,
                    "grasp_id": str(grasp.get("id")),
                    "grasp": grasp,
                    "T_world_tcp": T_world_tcp,
                    "stroke_mm": collision_stroke_mm,
                    "authored_stroke_mm": stroke_mm,
                    "collision_widths": collision_widths,
                    "scene_yaml_path": scene_yaml_path,
                    "obstacles_scene_yaml_path": obstacles_scene_yaml_path,
                    "score_key": score_key,
                    "sort_key": (
                        float(candidate_audit.get("threshold_alignment_angle_deg") or 180.0),
                        -float(threshold_alignment_score),
                        jaw_axis_abs_z,
                        orientation_distance_rad,
                        closest_contact_distance_to_camera_m,
                        center_distance_to_camera_m,
                        distance_to_reference_m,
                    ),
                    "best_payload": {
                        "id": grasp.get("id"),
                        "label": grasp.get("label"),
                        "grasp_type": "parallel_jaw_pair",
                        "contact_a_local_m": list(contact_a_local_m),
                        "contact_b_local_m": list(contact_b_local_m),
                        "normal_a_local": list(normal_a_local),
                        "normal_b_local": list(normal_b_local),
                        "center_local_m": list(center_local_m),
                        "jaw_axis_local": list(jaw_axis_local),
                        "approach_axis_local": list(approach_axis_local),
                        "opening_width_m": float(grasp.get("opening_width_m") or 0.0),
                        "contact_a_base_m": contact_a_base_m,
                        "contact_b_base_m": contact_b_base_m,
                        "center_base_m": center_base_m,
                        "jaw_axis_base": jaw_axis_base,
                        "approach_axis_base": approach_axis_base,
                        "closest_contact_distance_to_camera_m": closest_contact_distance_to_camera_m,
                        "center_distance_to_camera_m": center_distance_to_camera_m,
                        "distance_to_reference_m": distance_to_reference_m,
                        "approach_alignment_score": approach_alignment_score,
                        "approach_to_camera_score": approach_to_camera_score,
                        "camera_z_alignment_score": approach_axis_camera_z_score,
                        "jaw_axis_abs_z": jaw_axis_abs_z,
                        "jaw_axis_alignment_score": jaw_axis_alignment_score,
                        "min_jaw_axis_alignment_score": min_jaw_axis_alignment_score,
                        "grasp_up_axis": grasp_up_axis_raw,
                        "grasp_up_axis_base": grasp_up_axis_base,
                        "grasp_up_dot_base_z": grasp_up_dot_base_z,
                        "grasp_up_min_dot": grasp_up_min_dot,
                        "jaw_axis_xy_score": jaw_axis_xy_score,
                        "jaw_axis_horizontal_ok": jaw_axis_horizontal_ok,
                        "max_jaw_axis_abs_z": max_jaw_axis_abs_z,
                        "tool_approach_alignment_score": tool_approach_alignment_score,
                        "robot_z_alignment_score": tool_approach_alignment_score,
                        "tool_camera_axis_alignment_score": tool_camera_axis_alignment,
                        "threshold_alignment_score": threshold_alignment_score,
                        "orientation_distance_rad": orientation_distance_rad,
                        "explicit_orientation_quat_xyzw": explicit_orientation,
                        "equivalent_flipped": bool(equivalent.get("equivalent_flipped")),
                        "face_indices": grasp.get("face_indices"),
                        "sample_indices": grasp.get("sample_indices"),
                        "generator_group_index": grasp.get("generator_group_index"),
                        "grasp_family_label": grasp.get("grasp_family_label"),
                        "grasp_family_priority": (
                            family_priority_key[0] if priority_groups_enabled else None
                        ),
                        "grasp_family_priority_order": (
                            family_priority_key[1] if priority_groups_enabled else None
                        ),
                        "pointcloud_collision_passed": True,
                        "pointcloud_collision_pairs": [],
                        "pointcloud_check_ms": None,
                        "pointcloud_collision_consider_pointcloud": consider_pointcloud_collision,
                        "pointcloud_collision_mode": "fast",
                        "stroke_mm": collision_stroke_mm,
                        "authored_stroke_mm": stroke_mm,
                        "pregrasp_open_width_m": collision_widths.get("pregrasp_width_m"),
                        "pregrasp_open_offset_m": collision_widths.get("pregrasp_open_offset_m"),
                        "pregrasp_action": collision_widths.get("pregrasp_action"),
                        "T_world_gripper_root": None,
                        "safety_is_inner_grasp": safety_is_inner,
                        "safety_neighbor_count": len(neighbor_points_base_m or []),
                    },
                }
            )
            continue

        pc_results = _filter_grasps_by_pointcloud_collision(
            [(str(grasp.get("id")), T_world_tcp, collision_stroke_mm)],
            neighbor_points_base_m or [],
            scene_yaml_path=scene_yaml_path,
            obstacles_scene_yaml_path=obstacles_scene_yaml_path,
            consider_pointcloud=consider_pointcloud_collision,
            voxel_size_m=float(cfg_for_safety.get("pointcloud_collision_voxel_size_m") or 0.001),
            inflation_m=float(cfg_for_safety.get("pointcloud_collision_inflation_m") or 0.0),
            density_grid_m=float(cfg_for_safety.get("pointcloud_collision_density_grid_m") or 0.0),
            mesh_min_cluster_points=int(cfg_for_safety.get("pointcloud_collision_mesh_min_cluster_points") or 30),
            mesh_min_cluster_voxels=int(cfg_for_safety.get("pointcloud_collision_mesh_min_cluster_voxels") or 4),
            mesh_max_clusters=int(cfg_for_safety.get("pointcloud_collision_mesh_max_clusters") or 8),
            mesh_max_voxels=int(cfg_for_safety.get("pointcloud_collision_mesh_max_voxels") or 100000),
            mesh_bin_margin_m=float(cfg_for_safety.get("pointcloud_collision_mesh_bin_margin_m") or 0.03),
        )
        pc_result = pc_results.get(str(grasp.get("id"))) or {}
        safety_is_infeasible = not bool(pc_result.get("passed", True))
        candidate_audit.update(
            {
                "pointcloud_collision_passed": not safety_is_infeasible,
                "pointcloud_collision_pairs": pc_result.get("pairs", []),
                "pointcloud_check_ms": pc_result.get("ms"),
                "pointcloud_raw_stats": pc_result.get("pointcloud_raw_stats"),
                "pointcloud_collision_consider_pointcloud": consider_pointcloud_collision,
                "pointcloud_collision_mode": (
                    "fast" if fast_collision_mode else collision_check_mode
                ),
                "stroke_mm": pc_result.get("stroke_mm", collision_stroke_mm),
                "authored_stroke_mm": stroke_mm,
                "pregrasp_open_width_m": collision_widths.get("pregrasp_width_m"),
                "pregrasp_open_offset_m": collision_widths.get("pregrasp_open_offset_m"),
                "pregrasp_action": collision_widths.get("pregrasp_action"),
                "T_world_gripper_root": pc_result.get("T_world_root"),
                "safety_is_inner_grasp": safety_is_inner,
                "safety_neighbor_count": len(neighbor_points_base_m or []),
            }
        )
        if safety_is_infeasible:
            candidate_audit["reject_reason"] = str(
                pc_result.get("reason") or "pointcloud_collision"
            )
            if audit is not None:
                audit["candidates"].append(candidate_audit)
            continue
        inner_bonus = (-1 if (prefer_inner and safety_is_inner) else 0)

        if priority_groups_enabled:
            score_key = (
                safety_is_infeasible,
                -effective_tool_alignment_score,
                -jaw_axis_alignment_score,
                jaw_axis_abs_z,
                family_priority_key[0],
                family_priority_key[1],
                -threshold_alignment_score,
                orientation_distance_rad,
                -approach_to_camera_score,
                closest_contact_distance_to_camera_m,
                center_distance_to_camera_m,
                distance_to_reference_m,
            )
        elif selection_mode in {
            "camera_normal_then_point",
            "toward_camera_normal",
            "camera_normal",
            "normal_toward_camera",
        }:
            score_key = (
                safety_is_infeasible,
                -effective_tool_alignment_score,
                -jaw_axis_alignment_score,
                jaw_axis_abs_z,
                inner_bonus,
                -approach_to_camera_score,
                -effective_camera_alignment_score,
                orientation_distance_rad,
                closest_contact_distance_to_camera_m,
                center_distance_to_camera_m,
                distance_to_reference_m,
            )
        elif selection_mode in {
            "closest_center_to_camera",
            "center_to_camera",
            "camera_center",
        }:
            score_key = (
                safety_is_infeasible,
                -effective_tool_alignment_score,
                -jaw_axis_alignment_score,
                jaw_axis_abs_z,
                inner_bonus,
                -approach_to_camera_score,
                center_distance_to_camera_m,
                closest_contact_distance_to_camera_m,
                distance_to_reference_m,
            )
        else:
            score_key = (
                safety_is_infeasible,
                -effective_tool_alignment_score,
                -jaw_axis_alignment_score,
                jaw_axis_abs_z,
                inner_bonus,
                -approach_to_camera_score,
                -effective_camera_alignment_score,
                orientation_distance_rad,
                closest_contact_distance_to_camera_m,
                center_distance_to_camera_m,
                distance_to_reference_m,
            )
        candidate_audit.update(
            {
                "status": "accepted",
                "reject_reason": None,
                "score_key": list(score_key),
            }
        )
        if audit is not None:
            audit["candidates"].append(candidate_audit)
        if best_key is not None and score_key >= best_key:
            continue
        best_key = score_key
        best = {
            "id": grasp.get("id"),
            "label": grasp.get("label"),
            "grasp_type": "parallel_jaw_pair",
            "contact_a_local_m": list(contact_a_local_m),
            "contact_b_local_m": list(contact_b_local_m),
            "normal_a_local": list(normal_a_local),
            "normal_b_local": list(normal_b_local),
            "center_local_m": list(center_local_m),
            "jaw_axis_local": list(jaw_axis_local),
            "approach_axis_local": list(approach_axis_local),
            "opening_width_m": float(grasp.get("opening_width_m") or 0.0),
            "contact_a_base_m": contact_a_base_m,
            "contact_b_base_m": contact_b_base_m,
            "center_base_m": center_base_m,
            "jaw_axis_base": jaw_axis_base,
            "approach_axis_base": approach_axis_base,
            "closest_contact_distance_to_camera_m": closest_contact_distance_to_camera_m,
            "center_distance_to_camera_m": center_distance_to_camera_m,
            "distance_to_reference_m": distance_to_reference_m,
            "approach_alignment_score": approach_alignment_score,
            "approach_to_camera_score": approach_to_camera_score,
            "camera_z_alignment_score": approach_axis_camera_z_score,
            "jaw_axis_abs_z": jaw_axis_abs_z,
            "jaw_axis_alignment_score": jaw_axis_alignment_score,
            "min_jaw_axis_alignment_score": min_jaw_axis_alignment_score,
            "grasp_up_axis": grasp_up_axis_raw,
            "grasp_up_axis_base": grasp_up_axis_base,
            "grasp_up_dot_base_z": grasp_up_dot_base_z,
            "grasp_up_min_dot": grasp_up_min_dot,
            "jaw_axis_xy_score": jaw_axis_xy_score,
            "jaw_axis_horizontal_ok": jaw_axis_horizontal_ok,
            "max_jaw_axis_abs_z": max_jaw_axis_abs_z,
            "tool_approach_alignment_score": tool_approach_alignment_score,
            "robot_z_alignment_score": tool_approach_alignment_score,
            "tool_camera_axis_alignment_score": tool_camera_axis_alignment,
            "threshold_alignment_score": threshold_alignment_score,
            "orientation_distance_rad": orientation_distance_rad,
            "explicit_orientation_quat_xyzw": explicit_orientation,
            "equivalent_flipped": bool(equivalent.get("equivalent_flipped")),
            "face_indices": grasp.get("face_indices"),
            "sample_indices": grasp.get("sample_indices"),
            "generator_group_index": grasp.get("generator_group_index"),
            "grasp_family_label": grasp.get("grasp_family_label"),
            "grasp_family_priority": family_priority_key[0] if priority_groups_enabled else None,
            "grasp_family_priority_order": (
                family_priority_key[1] if priority_groups_enabled else None
            ),
            "pointcloud_collision_passed": True,
            "pointcloud_collision_pairs": [],
            "pointcloud_check_ms": pc_result.get("ms"),
            "pointcloud_raw_stats": pc_result.get("pointcloud_raw_stats"),
            "pointcloud_collision_consider_pointcloud": consider_pointcloud_collision,
            "pointcloud_collision_mode": (
                "fast" if fast_collision_mode else collision_check_mode
            ),
            "stroke_mm": pc_result.get("stroke_mm", collision_stroke_mm),
            "authored_stroke_mm": stroke_mm,
            "pregrasp_open_width_m": collision_widths.get("pregrasp_width_m"),
            "pregrasp_open_offset_m": collision_widths.get("pregrasp_open_offset_m"),
            "pregrasp_action": collision_widths.get("pregrasp_action"),
            "T_world_gripper_root": pc_result.get("T_world_root"),
            "safety_is_inner_grasp": safety_is_inner,
            "safety_neighbor_count": len(neighbor_points_base_m or []),
        }
    if fast_collision_mode and pending_collision_candidates:
        for order, record in enumerate(
            sorted(pending_collision_candidates, key=lambda item: item["sort_key"])
        ):
            candidate_audit = record["audit"]
            candidate_audit["collision_check_order"] = order
            if best is not None:
                candidate_audit.update(
                    {
                        "status": "skipped",
                        "reject_reason": "collision_fast_skipped_after_first_free",
                    }
                )
                if audit is not None:
                    audit["candidates"].append(candidate_audit)
                continue

            pc_results = _filter_grasps_by_pointcloud_collision(
                [(record["grasp_id"], record["T_world_tcp"], record["stroke_mm"])],
                neighbor_points_base_m or [],
                scene_yaml_path=record["scene_yaml_path"],
                obstacles_scene_yaml_path=record["obstacles_scene_yaml_path"],
                consider_pointcloud=consider_pointcloud_collision,
                voxel_size_m=float(
                    cfg_for_scoring.get("pointcloud_collision_voxel_size_m") or 0.001
                ),
                inflation_m=float(
                    cfg_for_scoring.get("pointcloud_collision_inflation_m") or 0.0
                ),
                density_grid_m=float(
                    cfg_for_scoring.get("pointcloud_collision_density_grid_m") or 0.0
                ),
                mesh_min_cluster_points=int(
                    cfg_for_scoring.get("pointcloud_collision_mesh_min_cluster_points") or 30
                ),
                mesh_min_cluster_voxels=int(
                    cfg_for_scoring.get("pointcloud_collision_mesh_min_cluster_voxels") or 4
                ),
                mesh_max_clusters=int(
                    cfg_for_scoring.get("pointcloud_collision_mesh_max_clusters") or 8
                ),
                mesh_max_voxels=int(
                    cfg_for_scoring.get("pointcloud_collision_mesh_max_voxels") or 100000
                ),
                mesh_bin_margin_m=float(
                    cfg_for_scoring.get("pointcloud_collision_mesh_bin_margin_m") or 0.03
                ),
            )
            pc_result = pc_results.get(record["grasp_id"]) or {}
            safety_is_infeasible = not bool(pc_result.get("passed", True))
            candidate_audit.update(
                {
                    "pointcloud_collision_passed": not safety_is_infeasible,
                    "pointcloud_collision_pairs": pc_result.get("pairs", []),
                    "pointcloud_check_ms": pc_result.get("ms"),
                    "pointcloud_raw_stats": pc_result.get("pointcloud_raw_stats"),
                    "stroke_mm": pc_result.get("stroke_mm", record["stroke_mm"]),
                    "T_world_gripper_root": pc_result.get("T_world_root"),
                }
            )
            if safety_is_infeasible:
                candidate_audit["reject_reason"] = str(
                    pc_result.get("reason") or "pointcloud_collision"
                )
                if audit is not None:
                    audit["candidates"].append(candidate_audit)
                continue

            score_key = record["score_key"]
            candidate_audit.update(
                {
                    "status": "accepted",
                    "reject_reason": None,
                    "score_key": list(score_key),
                }
            )
            if audit is not None:
                audit["candidates"].append(candidate_audit)
            best_key = score_key
            best = dict(record["best_payload"])
            best.update(
                {
                    "pointcloud_collision_passed": True,
                    "pointcloud_collision_pairs": [],
                    "pointcloud_check_ms": pc_result.get("ms"),
                    "pointcloud_raw_stats": pc_result.get("pointcloud_raw_stats"),
                    "stroke_mm": pc_result.get("stroke_mm", record["stroke_mm"]),
                    "authored_stroke_mm": record.get("authored_stroke_mm"),
                    "pregrasp_open_width_m": (
                        record.get("collision_widths") or {}
                    ).get("pregrasp_width_m"),
                    "pregrasp_open_offset_m": (
                        record.get("collision_widths") or {}
                    ).get("pregrasp_open_offset_m"),
                    "pregrasp_action": (
                        record.get("collision_widths") or {}
                    ).get("pregrasp_action"),
                    "T_world_gripper_root": pc_result.get("T_world_root"),
                }
            )
    if audit is not None:
        audit["selected_id"] = best.get("id") if isinstance(best, dict) else None
        audit["selected_score_key"] = list(best_key) if best_key is not None else None
        audit["accepted_count"] = len(
            [c for c in audit.get("candidates", []) if c.get("status") == "accepted"]
        )
        reasons: Dict[str, int] = {}
        for candidate in audit.get("candidates", []):
            reason = str(candidate.get("reject_reason") or "accepted")
            reasons[reason] = reasons.get(reason, 0) + 1
        audit["reason_counts"] = reasons
        audit["pointcloud_rejected_count"] = int(reasons.get("pointcloud_collision", 0))
        audit["pointcloud_obstacle_point_count"] = len(neighbor_points_base_m or [])
        audit["pointcloud_collision_mode"] = (
            "fast" if fast_collision_mode else collision_check_mode
        )
        audit["pointcloud_collision_representation"] = "raw_points"
        for candidate in audit.get("candidates", []):
            if isinstance(candidate, dict) and isinstance(candidate.get("pointcloud_raw_stats"), dict):
                audit["pointcloud_raw_stats"] = candidate.get("pointcloud_raw_stats")
                break
        # For the debug GLB / summary, point at the obstacle set actually used
        # (the per-task scene when present) so the bin renders where the operator
        # placed it; keep the semantics scene path as a separate field.
        audit["pointcloud_semantics_scene_yaml"] = str(
            cfg_for_scoring.get("pointcloud_collision_scene_yaml")
            or cfg_for_scoring.get("grasp_collision_scene_yaml")
            or _default_grasp_collision_scene_yaml_path()
        )
        audit["pointcloud_scene_yaml"] = str(
            cfg_for_scoring.get("pointcloud_collision_obstacles_scene_yaml")
            or cfg_for_scoring.get("pointcloud_collision_scene_yaml")
            or cfg_for_scoring.get("grasp_collision_scene_yaml")
            or _default_grasp_collision_scene_yaml_path()
        )
    return best


def _write_parallel_jaw_grasp_planning_debug(
    ctx: StationContext,
    state: RunState,
    cycle: int,
    attempt: int,
    match: Dict[str, Any],
    audit: Dict[str, Any],
    write_glb: bool = False,
    write_selected_glb: bool = True,
    write_colliding_glb: bool = True,
    colliding_glb_cap: int = 12,
) -> Optional[str]:
    if not isinstance(audit, dict):
        return None
    try:
        run_root = Path(ctx.runs.root) / state.run_id / "grasp_planning"
        run_root.mkdir(parents=True, exist_ok=True)
        rank_value = _coerce_float(match.get("rank"), -1.0)
        rank = int(rank_value if rank_value is not None else -1)
        stem = f"cycle-{cycle:03d}_attempt-{attempt:02d}_rank-{rank:03d}"
        artifact_dir = run_root / stem
        artifact_dir.mkdir(parents=True, exist_ok=True)
        selected_dir = artifact_dir / "selected"
        colliding_dir = artifact_dir / "colliding"
        selected_dir.mkdir(exist_ok=True)
        colliding_dir.mkdir(exist_ok=True)
        audit["artifact_dir"] = str(artifact_dir)
        payload = {
            "run_id": state.run_id,
            "cycle": cycle,
            "attempt": attempt,
            "detection_rank": rank,
            "object_id": match.get("object_id"),
            "label": match.get("label"),
            "score": match.get("score"),
            "center_uv": match.get("center_uv"),
            "center_xyz_m": match.get("center_xyz_m"),
            "pose_quat_xyzw": match.get("pose_quat_xyzw") or match.get("quaternion_xyzw"),
            "audit": audit,
        }
        json_path = artifact_dir / "candidates.json"
        json_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        csv_path = artifact_dir / "candidates.csv"
        columns = [
            "id",
            "status",
            "reject_reason",
            "grasp_family_label",
            "generator_group_index",
            "tool_approach_angle_deg",
            "tool_approach_alignment_score",
            "threshold_alignment_angle_deg",
            "threshold_alignment_score",
            "jaw_axis_tilt_from_xy_deg",
            "jaw_axis_abs_z",
            "jaw_axis_alignment_score",
            "grasp_up_axis",
            "grasp_up_dot_base_z",
            "grasp_up_min_dot",
            "pointcloud_collision_passed",
            "pointcloud_collision_consider_pointcloud",
            "pointcloud_collision_mode",
            "collision_check_order",
            "pointcloud_collision_pairs",
            "pointcloud_check_ms",
            "pointcloud_raw_stats",
            "stroke_mm",
            "authored_stroke_mm",
            "pregrasp_open_width_m",
            "pregrasp_open_offset_m",
            "pregrasp_action",
            "T_world_gripper_root",
            "score_key",
        ]

        def _csv_cell(value: Any) -> str:
            if isinstance(value, (dict, list, tuple)):
                text = json.dumps(value, separators=(",", ":"), ensure_ascii=False)
            elif value is None:
                text = ""
            else:
                text = str(value)
            return '"' + text.replace('"', '""') + '"'

        selected_id = audit.get("selected_id")
        rows = [",".join(columns)]
        for candidate in audit.get("candidates") or []:
            row = dict(candidate)
            if row.get("id") == selected_id and row.get("status") == "accepted":
                row["status"] = "selected"
            rows.append(",".join(_csv_cell(row.get(col)) for col in columns))
        csv_path.write_text("\n".join(rows) + "\n", encoding="utf-8")

        raw_obstacle_points = audit.get("obstacle_points_base_m") or []
        audit["obstacle_points_base_m_filtered"] = raw_obstacle_points
        audit["pointcloud_obstacle_point_count_filtered"] = len(raw_obstacle_points)
        _write_points_ply(artifact_dir / "obstacle_cloud.ply", raw_obstacle_points)
        # GLB export is ~2 s/file via trimesh. The selected-grasp scenes (3 files)
        # are written by default so the result can be visualized (bin mesh + cloud
        # + selected gripper pose); the per-candidate flood stays behind write_glb.
        if (write_selected_glb or write_glb) and audit.get("selected_id"):
            selected_scene_path = selected_dir / "selected_with_obstacles.glb"
            gripper_scene_path = selected_dir / "gripper_at_selected.glb"
            _write_grasp_planning_glb(
                selected_scene_path,
                match=match,
                audit=audit,
                include_candidates=False,
            )
            _write_grasp_planning_glb(
                gripper_scene_path,
                match=match,
                audit=audit,
                include_candidates=False,
            )
        if write_selected_glb or write_glb:
            candidates_scene_path = artifact_dir / "candidates_scene.glb"
            _write_grasp_planning_glb(
                candidates_scene_path,
                match=match,
                audit=audit,
                include_candidates=True,
            )

        selected = None
        colliding: List[Dict[str, Any]] = []
        # Per-colliding-candidate GLBs: each is the full scene (bin mesh, obstacle
        # cloud, object CAD, base/camera/object/grasp frames) with that one gripper
        # pose, colliding links red. write_glb=True writes them all; otherwise we
        # write up to colliding_glb_cap of them so a bad run can't cost ~minutes.
        colliding_written = 0
        for candidate in audit.get("candidates") or []:
            if not isinstance(candidate, dict):
                continue
            if candidate.get("id") == audit.get("selected_id"):
                selected = {
                    "id": candidate.get("id"),
                    "stroke_mm": candidate.get("stroke_mm"),
                    "T_world_gripper_root": candidate.get("T_world_gripper_root"),
                    "T_world_tcp": candidate.get("center_base_m"),
                    "pointcloud_check_ms": candidate.get("pointcloud_check_ms"),
                }
            if candidate.get("reject_reason") == "pointcloud_collision":
                glb_name = f"{candidate.get('id')}_gripper.glb"
                wrote_this = False
                if write_colliding_glb and (
                    write_glb or colliding_written < int(colliding_glb_cap)
                ):
                    _write_grasp_planning_glb(
                        colliding_dir / glb_name,
                        match=match,
                        audit={**audit, "selected_id": candidate.get("id")},
                        include_candidates=False,
                    )
                    colliding_written += 1
                    wrote_this = True
                colliding.append(
                    {
                        "id": candidate.get("id"),
                        "reason": candidate.get("reject_reason"),
                        "pairs": candidate.get("pointcloud_collision_pairs") or [],
                        "stroke_mm": candidate.get("stroke_mm"),
                        "T_world_gripper_root": candidate.get("T_world_gripper_root"),
                        "pointcloud_check_ms": candidate.get("pointcloud_check_ms"),
                        "glb_path": f"colliding/{glb_name}" if wrote_this else None,
                    }
                )

        summary_path = artifact_dir / "summary.json"
        summary_path.write_text(
            json.dumps(
                {
                    "selected_id": audit.get("selected_id"),
                    "accepted_count": audit.get("accepted_count"),
                    "rejected_counts": {
                        k: v
                        for k, v in (audit.get("reason_counts") or {}).items()
                        if k != "accepted"
                    },
                    "pointcloud_collision_consider_pointcloud": audit.get(
                        "pointcloud_collision_consider_pointcloud"
                    ),
                    "pointcloud_collision_mode": audit.get("pointcloud_collision_mode"),
                    "pointcloud_collision_representation": audit.get(
                        "pointcloud_collision_representation"
                    ),
                    "pointcloud_raw_stats": audit.get("pointcloud_raw_stats"),
                    "obstacle_point_count": audit.get("pointcloud_obstacle_point_count"),
                    "obstacle_point_count_filtered": audit.get(
                        "pointcloud_obstacle_point_count_filtered"
                    ),
                    "scene_yaml": audit.get("pointcloud_scene_yaml"),
                    "selected": selected,
                    "colliding": colliding,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        return str(artifact_dir)
    except Exception:
        return None


def _write_points_ply(path: Path, points: Any) -> None:
    pts = [p for p in (points or []) if _is_finite_vec(p, 3)]
    with path.open("w", encoding="utf-8") as fh:
        fh.write("ply\nformat ascii 1.0\n")
        fh.write(f"element vertex {len(pts)}\n")
        fh.write("property float x\nproperty float y\nproperty float z\n")
        fh.write("end_header\n")
        for p in pts:
            fh.write(f"{float(p[0])} {float(p[1])} {float(p[2])}\n")


def _write_grasp_planning_glb(
    path: Path,
    *,
    match: Dict[str, Any],
    audit: Dict[str, Any],
    include_candidates: bool,
) -> None:
    """Write a small dependency-free GLB scene for grasp-planning inspection."""
    import struct

    buffer = bytearray()
    buffer_views: List[Dict[str, Any]] = []
    accessors: List[Dict[str, Any]] = []
    meshes: List[Dict[str, Any]] = []
    nodes: List[Dict[str, Any]] = []
    materials = [
        {"name": "object_gray", "pbrMetallicRoughness": {"baseColorFactor": [0.55, 0.55, 0.55, 0.65], "metallicFactor": 0.0, "roughnessFactor": 0.7}, "alphaMode": "BLEND"},
        {"name": "x_red", "pbrMetallicRoughness": {"baseColorFactor": [1.0, 0.05, 0.05, 1.0]}},
        {"name": "y_green", "pbrMetallicRoughness": {"baseColorFactor": [0.05, 0.8, 0.15, 1.0]}},
        {"name": "z_blue", "pbrMetallicRoughness": {"baseColorFactor": [0.05, 0.25, 1.0, 1.0]}},
        {"name": "grasp_orange", "pbrMetallicRoughness": {"baseColorFactor": [1.0, 0.45, 0.05, 1.0]}},
        {"name": "selected_black", "pbrMetallicRoughness": {"baseColorFactor": [0.02, 0.02, 0.02, 1.0]}},
        {"name": "gripper_green", "pbrMetallicRoughness": {"baseColorFactor": [0.0, 0.75, 0.25, 1.0]}},
        {"name": "gripper_gray", "pbrMetallicRoughness": {"baseColorFactor": [0.55, 0.57, 0.58, 1.0]}},
        {"name": "colliding_red", "pbrMetallicRoughness": {"baseColorFactor": [1.0, 0.02, 0.02, 1.0]}},
        {"name": "obstacle_points", "pbrMetallicRoughness": {"baseColorFactor": [1.0, 0.18, 0.05, 0.85]}, "alphaMode": "BLEND"},
        {"name": "target_points", "pbrMetallicRoughness": {"baseColorFactor": [0.05, 0.55, 1.0, 0.8]}, "alphaMode": "BLEND"},
        {"name": "scene_obstacle_mesh", "pbrMetallicRoughness": {"baseColorFactor": [0.6, 0.75, 0.9, 0.35], "metallicFactor": 0.0, "roughnessFactor": 0.8}, "alphaMode": "BLEND"},
    ]

    def _align(n: int = 4) -> None:
        while len(buffer) % n:
            buffer.append(0)

    def _add_view(data: bytes, target: Optional[int] = None) -> int:
        _align(4)
        offset = len(buffer)
        buffer.extend(data)
        view: Dict[str, Any] = {"buffer": 0, "byteOffset": offset, "byteLength": len(data)}
        if target is not None:
            view["target"] = target
        buffer_views.append(view)
        return len(buffer_views) - 1

    def _add_vec3_accessor(points: List[List[float]], target: Optional[int] = None) -> int:
        flat = [float(v) for point in points for v in point[:3]]
        data = struct.pack("<" + "f" * len(flat), *flat) if flat else b""
        view = _add_view(data, target)
        mins = [min(point[i] for point in points) for i in range(3)] if points else [0, 0, 0]
        maxs = [max(point[i] for point in points) for i in range(3)] if points else [0, 0, 0]
        accessors.append(
            {
                "bufferView": view,
                "componentType": 5126,
                "count": len(points),
                "type": "VEC3",
                "min": mins,
                "max": maxs,
            }
        )
        return len(accessors) - 1

    def _add_indices_accessor(indices: List[int]) -> int:
        data = struct.pack("<" + "I" * len(indices), *[int(i) for i in indices]) if indices else b""
        view = _add_view(data, 34963)
        accessors.append(
            {
                "bufferView": view,
                "componentType": 5125,
                "count": len(indices),
                "type": "SCALAR",
            }
        )
        return len(accessors) - 1

    def _add_mesh_node(
        name: str,
        points: List[List[float]],
        indices: Optional[List[int]],
        material: int,
        mode: int,
    ) -> None:
        if not points:
            return
        pos_accessor = _add_vec3_accessor(points, 34962)
        primitive: Dict[str, Any] = {
            "attributes": {"POSITION": pos_accessor},
            "mode": mode,
            "material": material,
        }
        if indices:
            primitive["indices"] = _add_indices_accessor(indices)
        meshes.append({"name": name, "primitives": [primitive]})
        nodes.append({"name": name, "mesh": len(meshes) - 1})

    def _add_voxel_points(
        name: str,
        points: List[List[float]],
        material: int,
        *,
        radius: float = 0.003,
        max_points: int = 2500,
    ) -> None:
        """Render point clouds as tiny cubes; GL_POINTS are too easy to miss in viewers."""
        if not points:
            return
        pts = points
        if max_points > 0 and len(pts) > max_points:
            step = max(1, int(math.ceil(len(pts) / float(max_points))))
            pts = pts[::step][:max_points]
        vertices: List[List[float]] = []
        indices: List[int] = []
        offsets = [
            (-radius, -radius, -radius),
            (radius, -radius, -radius),
            (radius, radius, -radius),
            (-radius, radius, -radius),
            (-radius, -radius, radius),
            (radius, -radius, radius),
            (radius, radius, radius),
            (-radius, radius, radius),
        ]
        cube_faces = [
            (0, 1, 2), (0, 2, 3),
            (4, 6, 5), (4, 7, 6),
            (0, 4, 5), (0, 5, 1),
            (1, 5, 6), (1, 6, 2),
            (2, 6, 7), (2, 7, 3),
            (3, 7, 4), (3, 4, 0),
        ]
        for p in pts:
            if not _is_finite_vec(p, 3):
                continue
            base = len(vertices)
            vertices.extend(
                [
                    [float(p[0]) + dx, float(p[1]) + dy, float(p[2]) + dz]
                    for dx, dy, dz in offsets
                ]
            )
            for a, b, c in cube_faces:
                indices.extend([base + a, base + b, base + c])
        _add_mesh_node(name, vertices, indices, material, 4)

    def _add_raw_points(
        name: str,
        points: List[List[float]],
        material: int,
    ) -> None:
        if not points:
            return
        # GL_POINTS are invisible in several GLB viewers. Use tiny cubes so the
        # raw cloud is visible, while keeping the marker size below the old voxel blocks.
        _add_voxel_points(
            name,
            points,
            material,
            radius=0.0006,
            max_points=60000,
        )

    def _add_cylinder_between(
        name: str,
        a: List[float],
        b: List[float],
        material: int,
        *,
        radius: float = 0.003,
        segments: int = 12,
    ) -> None:
        if not (_is_finite_vec(a, 3) and _is_finite_vec(b, 3)):
            return
        axis = _subtract_vec(list(b), list(a))
        axis_n = _normalize_vec(axis)
        if axis_n is None:
            return
        ref = [0.0, 0.0, 1.0] if abs(axis_n[2]) < 0.9 else [0.0, 1.0, 0.0]
        u = _normalize_vec(_cross_vec(axis_n, ref))
        if u is None:
            return
        v = _normalize_vec(_cross_vec(axis_n, u))
        if v is None:
            return
        points: List[List[float]] = []
        for base in (a, b):
            for idx in range(segments):
                theta = (2.0 * math.pi * idx) / segments
                radial = [
                    (u[i] * math.cos(theta) * radius) + (v[i] * math.sin(theta) * radius)
                    for i in range(3)
                ]
                points.append([float(base[i]) + radial[i] for i in range(3)])
        center_a = len(points)
        points.append(list(a))
        center_b = len(points)
        points.append(list(b))
        indices: List[int] = []
        for idx in range(segments):
            j = (idx + 1) % segments
            a0 = idx
            a1 = j
            b0 = segments + idx
            b1 = segments + j
            indices.extend([a0, b0, b1, a0, b1, a1])
            indices.extend([center_a, a1, a0])
            indices.extend([center_b, b0, b1])
        _add_mesh_node(name, points, indices, material, 4)

    def _add_frame(
        name: str,
        origin: List[float],
        quat: List[float],
        *,
        length: float = 0.05,
        material_offset: int = 1,
    ) -> None:
        if not (_is_finite_vec(origin, 3) and _is_finite_vec(quat, 4)):
            return
        axes = [
            ([1.0, 0.0, 0.0], material_offset, "x"),
            ([0.0, 1.0, 0.0], material_offset + 1, "y"),
            ([0.0, 0.0, 1.0], material_offset + 2, "z"),
        ]
        for local, material, suffix in axes:
            vec = _quat_rotate(quat, local)
            end = [float(origin[i]) + (float(vec[i]) * length) for i in range(3)]
            _add_cylinder_between(
                f"{name}_{suffix}",
                list(origin),
                end,
                material,
                radius=max(0.0015, length * 0.035),
            )

    def _parse_obj_mesh(mesh_path: Any) -> tuple[List[List[float]], List[int]]:
        obj_path = Path(str(mesh_path or ""))
        if not obj_path.exists() or not obj_path.is_file():
            return [], []
        vertices: List[List[float]] = []
        indices: List[int] = []
        try:
            for raw in obj_path.read_text(encoding="utf-8", errors="ignore").splitlines():
                line = raw.strip()
                if line.startswith("v "):
                    parts = line.split()
                    if len(parts) >= 4:
                        vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
                elif line.startswith("f "):
                    face: List[int] = []
                    for token in line.split()[1:]:
                        idx = token.split("/")[0]
                        if not idx:
                            continue
                        value = int(idx)
                        face.append((len(vertices) + value) if value < 0 else (value - 1))
                    for i in range(1, len(face) - 1):
                        indices.extend([face[0], face[i], face[i + 1]])
        except Exception:
            return [], []
        return vertices, indices

    def _parse_stl_mesh(mesh_path: Any, scale: float = 1.0) -> tuple[List[List[float]], List[int]]:
        stl_path = Path(str(mesh_path or ""))
        if not stl_path.exists() or not stl_path.is_file():
            return [], []
        try:
            data = stl_path.read_bytes()
        except Exception:
            return [], []
        vertices: List[List[float]] = []
        indices: List[int] = []
        try:
            if len(data) >= 84:
                tri_count = struct.unpack_from("<I", data, 80)[0]
                expected = 84 + (int(tri_count) * 50)
                if expected == len(data):
                    offset = 84
                    for _ in range(int(tri_count)):
                        offset += 12  # normal
                        base = len(vertices)
                        for _vertex_idx in range(3):
                            x, y, z = struct.unpack_from("<fff", data, offset)
                            vertices.append(
                                [
                                    float(x) * float(scale),
                                    float(y) * float(scale),
                                    float(z) * float(scale),
                                ]
                            )
                            offset += 12
                        indices.extend([base, base + 1, base + 2])
                        offset += 2
                    return vertices, indices
        except Exception:
            vertices = []
            indices = []

        # ASCII STL fallback.
        try:
            text = data.decode("utf-8", errors="ignore")
            face: List[int] = []
            for raw in text.splitlines():
                parts = raw.strip().split()
                if len(parts) == 4 and parts[0].lower() == "vertex":
                    base = len(vertices)
                    vertices.append(
                        [
                            float(parts[1]) * float(scale),
                            float(parts[2]) * float(scale),
                            float(parts[3]) * float(scale),
                        ]
                    )
                    face.append(base)
                    if len(face) == 3:
                        indices.extend(face)
                        face = []
        except Exception:
            return [], []
        return vertices, indices

    def _load_tri_mesh(mesh_path: Any, scale: float = 1.0) -> tuple[List[List[float]], List[int]]:
        mesh_path_obj = Path(str(mesh_path or ""))
        try:
            import trimesh  # type: ignore
            loaded = trimesh.load(str(mesh_path_obj), force=None)
            if isinstance(loaded, trimesh.Scene):
                meshes_loaded = [
                    g for g in loaded.geometry.values()
                    if isinstance(g, trimesh.Trimesh)
                ]
                if not meshes_loaded:
                    return [], []
                mesh = trimesh.util.concatenate(meshes_loaded)
            else:
                mesh = loaded
            if mesh is None or not hasattr(mesh, "vertices"):
                return [], []
            vertices = (np.asarray(mesh.vertices, dtype=np.float64) * float(scale)).tolist()
            indices = np.asarray(mesh.faces, dtype=np.int64).reshape(-1).tolist()
            return vertices, [int(i) for i in indices]
        except Exception:
            if mesh_path_obj.suffix.lower() == ".stl":
                return _parse_stl_mesh(mesh_path_obj, scale)
            return _parse_obj_mesh(mesh_path)

    def _transform_points(
        points: List[List[float]],
        quat: List[float],
        trans: List[float],
        scale: float,
    ) -> List[List[float]]:
        out: List[List[float]] = []
        for point in points:
            scaled = [float(point[i]) * scale for i in range(3)]
            rot = _quat_rotate(quat, scaled)
            out.append([float(trans[i]) + float(rot[i]) for i in range(3)])
        return out

    def _transform_points_matrix(
        points: List[List[float]],
        matrix: Any,
    ) -> List[List[float]]:
        try:
            T = np.asarray(matrix, dtype=np.float64).reshape(4, 4)
        except Exception:
            return []
        out: List[List[float]] = []
        for point in points:
            p = T @ np.array([float(point[0]), float(point[1]), float(point[2]), 1.0])
            out.append([float(p[0]), float(p[1]), float(p[2])])
        return out

    def _add_gripper_meshes(candidate: Dict[str, Any], selected: bool) -> None:
        root = candidate.get("T_world_gripper_root")
        if root is None:
            return
        stroke = float(candidate.get("stroke_mm") or 0.0)
        per = _per_finger_m(stroke)
        T_root = np.asarray(root, dtype=np.float64)
        collision_root = _station_gripper_collision_dir()
        finger_vertices, finger_indices = _load_tri_mesh(collision_root / "finger.stl", 0.001)
        # Keep in sync with _set_gripper_mesh_poses / the URDF <collision><origin>.
        collision_origin = _translation_4x4((0.0, 0.0063, 0.0)) @ _rpy_4x4(3.14159, 0.0, -1.5708)
        link_poses = {
            "fr3_leftfinger": T_root
            @ _translation_4x4((0.0, 0.0, 0.0584))
            @ _translation_4x4((0.0, per, 0.0))
            @ collision_origin,
            "fr3_rightfinger": T_root
            @ _translation_4x4((0.0, 0.0, 0.0584))
            @ _rpy_4x4(0.0, 0.0, math.pi)
            @ _translation_4x4((0.0, per, 0.0))
            @ collision_origin,
        }
        colliding_links = {
            str(link)
            for pair in (candidate.get("pointcloud_collision_pairs") or [])
            for link in pair
        }
        reason = str(candidate.get("reject_reason") or "").strip()
        is_rejected = bool(reason) and reason != "accepted"
        # selected -> green(6); accepted -> gray(7); rejected -> red(8). Colliding
        # links always red(8) regardless.
        base_material = 6 if selected else (8 if is_rejected else 7)
        for link_name, pose in link_poses.items():
            material = 8 if link_name in colliding_links else base_material
            _add_mesh_node(
                f"{candidate.get('id')}_{link_name}",
                _transform_points_matrix(finger_vertices, pose),
                finger_indices,
                material,
                4,
            )

    object_center = audit.get("object_center_base_m")
    object_quat = audit.get("object_quat_base_xyzw")
    _add_frame("robot_base_frame", [0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0], length=0.12)
    camera_position = audit.get("camera_position_base_m")
    camera_quat = audit.get("camera_quat_base_xyzw")
    if _is_finite_vec(camera_position, 3) and _is_finite_vec(camera_quat, 4):
        _add_frame("camera_frame", camera_position, camera_quat, length=0.07)
    if _is_finite_vec(object_center, 3) and _is_finite_vec(object_quat, 4):
        raw_vertices, raw_indices = _parse_obj_mesh(match.get("mesh_path"))
        scale = 1.0
        raw_min = raw_max = None
        if raw_vertices:
            raw_min = [min(v[i] for v in raw_vertices) for i in range(3)]
            raw_max = [max(v[i] for v in raw_vertices) for i in range(3)]
        extents = match.get("object_extents_m")
        if raw_min is not None and _is_finite_vec(extents, 3):
            raw_ext = [max(1e-9, raw_max[i] - raw_min[i]) for i in range(3)]
            ratios = [
                float(extents[i]) / raw_ext[i]
                for i in range(3)
                if float(extents[i]) > 0 and raw_ext[i] > 1e-9
            ]
            if ratios:
                scale = sorted(ratios)[len(ratios) // 2]
        # Megapose reports the pose at the CAD mesh's AABB centroid (the runtime mesh
        # is recentered before pose estimation). The raw CAD vertices are in the CAD
        # frame, whose origin is generally NOT the AABB centroid — so recenter the
        # vertices to their own AABB centroid before applying the reported pose, or the
        # displayed mesh lands offset from where the object actually is.
        if raw_min is not None:
            centroid = [0.5 * (raw_min[i] + raw_max[i]) for i in range(3)]
            recentered = [[v[0] - centroid[0], v[1] - centroid[1], v[2] - centroid[2]] for v in raw_vertices]
        else:
            recentered = raw_vertices
        object_vertices = _transform_points(recentered, object_quat, object_center, scale)
        _add_mesh_node("object_mesh_base_pose", object_vertices, raw_indices, 0, 4)
        _add_frame("object_pose_frame", object_center, object_quat, length=0.06)

    # Render exactly the raw scene points used by collision. No octree, voxel
    # blocks, target exclusion, AABB clip, or connected-component filtering.
    obstacle_points = audit.get("obstacle_points_base_m") or []
    _add_raw_points("collision_raw_scene_point_cloud", obstacle_points, 9)

    scene_yaml_path = audit.get("pointcloud_scene_yaml")
    if scene_yaml_path:
        try:
            import yaml
            scene_yaml_file = Path(str(scene_yaml_path))
            if scene_yaml_file.exists():
                scene_cfg = yaml.safe_load(scene_yaml_file.read_text(encoding="utf-8"))
                scene_obstacles = (scene_cfg.get("environment") or {}).get("obstacles") or []
                scene_dir = scene_yaml_file.parent
                for obs in scene_obstacles:
                    if not isinstance(obs, dict) or obs.get("type") != "mesh":
                        continue
                    obs_path = obs.get("path")
                    if not obs_path:
                        continue
                    obs_mesh_path = Path(obs_path) if Path(obs_path).is_absolute() else scene_dir / obs_path
                    obs_vertices, obs_indices = _load_tri_mesh(obs_mesh_path)
                    if not obs_vertices:
                        continue
                    transform = obs.get("transform") or {}
                    trans = transform.get("translation") or [0.0, 0.0, 0.0]
                    rot_matrix = transform.get("rotation")
                    if rot_matrix is not None:
                        try:
                            R = np.asarray(rot_matrix, dtype=np.float64).reshape(3, 3)
                            T = np.eye(4)
                            T[:3, :3] = R
                            T[:3, 3] = [float(v) for v in trans]
                            obs_vertices = _transform_points_matrix(obs_vertices, T)
                        except Exception:
                            obs_vertices = [[float(v[0]) + float(trans[0]), float(v[1]) + float(trans[1]), float(v[2]) + float(trans[2])] for v in obs_vertices]
                    else:
                        obs_vertices = [[float(v[0]) + float(trans[0]), float(v[1]) + float(trans[1]), float(v[2]) + float(trans[2])] for v in obs_vertices]
                    _add_mesh_node(f"scene_obstacle_{obs.get('name', 'mesh')}", obs_vertices, obs_indices, 11, 4)
        except Exception:
            pass

    candidates = [c for c in audit.get("candidates") or [] if isinstance(c, dict)]
    selected_id = audit.get("selected_id")

    def _has_gripper_pose(c: Dict[str, Any]) -> bool:
        return c.get("T_world_gripper_root") is not None and _is_finite_vec(
            c.get("center_base_m"), 3
        ) and _is_finite_vec(c.get("explicit_orientation_quat_xyzw"), 4)

    if include_candidates:
        # All candidates we have a gripper pose for (accepted AND rejected — e.g.
        # pointcloud_collision rejects, so you can see why a grasp failed even when
        # nothing is accepted), plus the selected one.
        frame_candidates = [
            c for c in candidates
            if c.get("id") == selected_id or _has_gripper_pose(c)
        ]
    else:
        frame_candidates = [c for c in candidates if c.get("id") == selected_id]
    for candidate in frame_candidates:
        center = candidate.get("center_base_m")
        quat = candidate.get("explicit_orientation_quat_xyzw")
        if _is_finite_vec(center, 3) and _is_finite_vec(quat, 4):
            is_selected = candidate.get("id") == selected_id
            name = "selected_grasp_tcp_frame" if is_selected else f"candidate_{candidate.get('id')}_tcp_frame"
            _add_frame(name, center, quat, length=0.045)
            _add_gripper_meshes(candidate, is_selected)
            contact_a = candidate.get("contact_a_base_m")
            contact_b = candidate.get("contact_b_base_m")
            if _is_finite_vec(contact_a, 3) and _is_finite_vec(contact_b, 3):
                _add_cylinder_between(
                    f"{name}_contact_line",
                    contact_a,
                    contact_b,
                    5,
                    radius=0.002,
                )

    if not meshes:
        _add_frame("robot_base_frame", [0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0], length=0.12)

    while len(buffer) % 4:
        buffer.append(0)
    gltf = {
        "asset": {"version": "2.0", "generator": "bin_picking_grasp_planning_debug"},
        "scene": 0,
        "scenes": [{"nodes": list(range(len(nodes)))}],
        "nodes": nodes,
        "meshes": meshes,
        "materials": materials,
        "buffers": [{"byteLength": len(buffer)}],
        "bufferViews": buffer_views,
        "accessors": accessors,
    }
    json_chunk = json.dumps(gltf, separators=(",", ":")).encode("utf-8")
    while len(json_chunk) % 4:
        json_chunk += b" "
    total_len = 12 + 8 + len(json_chunk) + 8 + len(buffer)
    with path.open("wb") as fh:
        fh.write(struct.pack("<III", 0x46546C67, 2, total_len))
        fh.write(struct.pack("<I4s", len(json_chunk), b"JSON"))
        fh.write(json_chunk)
        fh.write(struct.pack("<I4s", len(buffer), b"BIN\x00"))
        fh.write(buffer)


def _match_pose_in_base(
    match: Dict[str, Any],
    base_to_cam: Dict[str, Any],
) -> Optional[tuple[List[float], List[float]]]:
    """Transform a match's camera-frame pose to base frame. Returns (translation, quat_xyzw)."""
    center_cam = match.get("center_xyz_m") or match.get("pose_origin_xyz_m")
    quat_cam = match.get("pose_quat_xyzw") or match.get("quaternion_xyzw")
    if not _is_finite_vec(center_cam, 3) or not _is_finite_vec(quat_cam, 4):
        return None
    cam_quat = base_to_cam.get("rotation_quat_xyzw") or [0.0, 0.0, 0.0, 1.0]
    cam_trans = base_to_cam.get("translation_m") or [0.0, 0.0, 0.0]
    if not _is_finite_vec(cam_quat, 4) or not _is_finite_vec(cam_trans, 3):
        return None
    trans_base = _add_vec(
        _quat_rotate(cam_quat, [float(v) for v in center_cam]),
        [float(v) for v in cam_trans],
    )
    quat_base = _normalize_quat_xyzw(
        _quat_mul_xyzw(cam_quat, [float(v) for v in quat_cam])
    ) or [0.0, 0.0, 0.0, 1.0]
    return trans_base, quat_base


def _inner_grasp_within_approach_cone(
    pick_contacts_model: Dict[str, Any],
    object_quat_base_xyzw: List[float],
    reference_quat_base_xyzw: List[float],
    approach_tool_axis: str,
    max_approach_angle_deg: Optional[float],
    pick_cfg: Optional[Dict[str, Any]] = None,
) -> bool:
    """Return True if at least one inner grasp exists whose approach axis (after
    applying the object pose) is within the approach-angle cone around the tool
    approach axis."""
    grasp_family_mode = str((pick_cfg or {}).get("grasp_family_mode", "default")).strip().lower()
    if grasp_family_mode == "external_only":
        return False
    grasps = (pick_contacts_model or {}).get("grasps") or []
    if not grasps:
        return False
    family_selection_policy = _apply_grasp_family_mode(
        (pick_contacts_model or {}).get("grasp_family_selection_policy") or {},
        grasp_family_mode,
    )
    max_deg = _coerce_float(max_approach_angle_deg, 45.0) or 45.0
    min_cos = math.cos(math.radians(max(0.0, float(max_deg))))
    # Compute tool approach axis in base frame.
    approach_tool = _parse_signed_axis(approach_tool_axis or "-z")
    if approach_tool is None or not _is_finite_vec(reference_quat_base_xyzw, 4):
        return False
    tool_local = {"x": [1.0, 0.0, 0.0], "y": [0.0, 1.0, 0.0], "z": [0.0, 0.0, 1.0]}[
        approach_tool["axis"]
    ]
    tool_local = [v * approach_tool["sign"] for v in tool_local]
    tool_base = _normalize_vec(_quat_rotate(reference_quat_base_xyzw, tool_local))
    if tool_base is None:
        return False
    for g in grasps:
        if not _grasp_allowed_by_family_policy(family_selection_policy, g):
            continue
        label = (g.get("grasp_family_label") or "").strip().lower()
        is_inner = label in {"internal", "inside", "inner"}
        if not is_inner:
            # Try normal-based inference as fallback.
            inferred = _infer_inner_grasp_from_normals(
                g.get("normal_a_local"),
                g.get("normal_b_local"),
                g.get("jaw_axis_local"),
            )
            is_inner = bool(inferred)
        if not is_inner:
            continue
        approach_local = g.get("approach_axis_local")
        if not _is_finite_vec(approach_local, 3):
            continue
        approach_base = _normalize_vec(
            _quat_rotate(object_quat_base_xyzw, [float(v) for v in approach_local])
        )
        if approach_base is None:
            continue
        if _dot_vec(approach_base, tool_base) >= min_cos:
            return True
    return False


def _any_outer_grasp_collision_free(
    pick_contacts_model: Dict[str, Any],
    object_center_base_m: List[float],
    object_quat_base_xyzw: List[float],
    reference_quat_base_xyzw: List[float],
    approach_tool_axis: str,
    max_approach_angle_deg: Optional[float],
    neighbor_points_base_m: List[List[float]],
    pick_cfg: Dict[str, Any],
) -> bool:
    """Return True if at least one outer cone-passing grasp clears the pointcloud."""
    grasp_family_mode = str((pick_cfg or {}).get("grasp_family_mode", "default")).strip().lower()
    if grasp_family_mode == "internal_only":
        return False
    grasps = (pick_contacts_model or {}).get("grasps") or []
    if not grasps:
        return False
    family_selection_policy = _apply_grasp_family_mode(
        (pick_contacts_model or {}).get("grasp_family_selection_policy") or {},
        grasp_family_mode,
    )
    max_deg = _coerce_float(max_approach_angle_deg, 45.0) or 45.0
    min_cos = math.cos(math.radians(max(0.0, float(max_deg))))
    approach_tool = _parse_signed_axis(approach_tool_axis or "-z")
    tool_local = {"x": [1.0, 0.0, 0.0], "y": [0.0, 1.0, 0.0], "z": [0.0, 0.0, 1.0]}[
        approach_tool["axis"]
    ]
    tool_local = [v * approach_tool["sign"] for v in tool_local]
    tool_base = _normalize_vec(
        _quat_rotate(reference_quat_base_xyzw, tool_local)
    ) or [0.0, 0.0, -1.0]
    for g in grasps:
        if not _grasp_allowed_by_family_policy(family_selection_policy, g):
            continue
        label = (g.get("grasp_family_label") or "").strip().lower()
        if label in {"internal", "inside", "inner"}:
            continue
        # Infer if unlabeled.
        if not label:
            inferred = _infer_inner_grasp_from_normals(
                g.get("normal_a_local"),
                g.get("normal_b_local"),
                g.get("jaw_axis_local"),
            )
            if inferred:
                continue
        approach_local = g.get("approach_axis_local")
        jaw_local = g.get("jaw_axis_local")
        center_local = g.get("center_local_m")
        ca_local = g.get("contact_a_local_m")
        cb_local = g.get("contact_b_local_m")
        if not _is_finite_vec(approach_local, 3) or not _is_finite_vec(jaw_local, 3):
            continue
        if not _is_finite_vec(ca_local, 3) or not _is_finite_vec(cb_local, 3):
            continue
        if not _is_finite_vec(center_local, 3):
            continue
        approach_base = _normalize_vec(
            _quat_rotate(object_quat_base_xyzw, [float(v) for v in approach_local])
        )
        if approach_base is None:
            continue
        if _dot_vec(approach_base, tool_base) < min_cos:
            continue  # Outside approach cone.
        center_base = _add_vec(
            object_center_base_m,
            _quat_rotate(object_quat_base_xyzw, [float(v) for v in center_local]),
        )
        jaw_base = _normalize_vec(
            _quat_rotate(object_quat_base_xyzw, [float(v) for v in jaw_local])
        )
        if jaw_base is None:
            continue
        T_world_tcp = _parallel_jaw_tcp_matrix_from_axes(center_base, jaw_base, approach_base)
        stroke_mm = float(
            _coerce_float(
                g.get("stroke_mm"),
                float(g.get("opening_width_m") or 0.0) * 1000.0,
            )
            or 0.0
        )
        collision_widths = _parallel_jaw_pregrasp_collision_width_m(
            pick_cfg,
            g,
            _normalize_grasp_family_label(g.get("grasp_family_label")),
        )
        result = _filter_grasps_by_pointcloud_collision(
            [
                (
                    str(g.get("id")),
                    T_world_tcp,
                    float(collision_widths.get("collision_stroke_mm") or stroke_mm),
                )
            ],
            neighbor_points_base_m or [],
            scene_yaml_path=str(
                (pick_cfg or {}).get("pointcloud_collision_scene_yaml")
                or (pick_cfg or {}).get("grasp_collision_scene_yaml")
                or _default_grasp_collision_scene_yaml_path()
            ),
            obstacles_scene_yaml_path=(pick_cfg or {}).get("pointcloud_collision_obstacles_scene_yaml") or None,
            consider_pointcloud=_coerce_bool(
                (pick_cfg or {}).get("pointcloud_collision_consider_pointcloud"),
                True,
            ),
            voxel_size_m=float((pick_cfg or {}).get("pointcloud_collision_voxel_size_m") or 0.001),
            inflation_m=float((pick_cfg or {}).get("pointcloud_collision_inflation_m") or 0.0),
            density_grid_m=float((pick_cfg or {}).get("pointcloud_collision_density_grid_m") or 0.0),
            mesh_min_cluster_points=int((pick_cfg or {}).get("pointcloud_collision_mesh_min_cluster_points") or 30),
            mesh_min_cluster_voxels=int((pick_cfg or {}).get("pointcloud_collision_mesh_min_cluster_voxels") or 4),
            mesh_max_clusters=int((pick_cfg or {}).get("pointcloud_collision_mesh_max_clusters") or 8),
            mesh_max_voxels=int((pick_cfg or {}).get("pointcloud_collision_mesh_max_voxels") or 100000),
            mesh_bin_margin_m=float((pick_cfg or {}).get("pointcloud_collision_mesh_bin_margin_m") or 0.03),
        ).get(str(g.get("id"))) or {}
        if bool(result.get("passed", True)):
            return True
    return False


def _build_all_poses_ranker(
    *,
    ctx: StationContext,
    state: RunState,
    pick_contacts_model: Dict[str, Any],
    base_to_cam: Dict[str, Any],
    base_to_ee: Dict[str, Any],
    approach_tool_axis: str,
    max_approach_angle_deg: Optional[float],
    mesh_path: Optional[str],
    mesh_units: str,
    mesh_scale: float,
) -> Callable[[List[Dict[str, Any]], Dict[str, Any]], List[Dict[str, Any]]]:
    """Build a closure that reorders matches using:
      1) Pickable-filter: inner-grasp-within-cone OR outer-grasp-collision-free
         against union of OTHER posed-object mesh sample points.
      2) Centrality: ties broken by distance to XY centroid of all pose translations.
    """

    def ranker(matches: List[Dict[str, Any]], pick_cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
        if not matches:
            return []
        posed: List[Dict[str, Any]] = []
        for m in matches:
            pb = _match_pose_in_base(m, base_to_cam)
            if pb is None:
                continue
            trans_base, quat_base = pb
            safety_points = _extract_safety_points_base_m(m, base_to_cam)
            scene_points = (
                _extract_scene_point_cloud_base_m(
                    m,
                    base_to_cam,
                    runs_root=Path(ctx.runs.root),
                    run_id=state.run_id,
                )
                or safety_points.get("neighbor")
                or []
            )
            posed.append(
                {
                    "match": m,
                    "trans_base": trans_base,
                    "quat_base": quat_base,
                    "neighbor_points_base_m": scene_points,
                }
            )
        if not posed:
            return list(matches)

        # Centroid XY of all pose translations (bin-center proxy).
        cx = sum(p["trans_base"][0] for p in posed) / len(posed)
        cy = sum(p["trans_base"][1] for p in posed) / len(posed)

        ref_quat = base_to_ee.get("rotation_quat_xyzw") or [0.0, 0.0, 0.0, 1.0]
        ref_quat = _normalize_quat_xyzw(list(ref_quat)) or [0.0, 0.0, 0.0, 1.0]

        # Determine pickable for each.
        for p in posed:
            inner_ok = _inner_grasp_within_approach_cone(
                pick_contacts_model,
                p["quat_base"],
                ref_quat,
                approach_tool_axis,
                max_approach_angle_deg,
                pick_cfg=pick_cfg,
            )
            if inner_ok:
                p["pickable"] = True
                p["pickable_reason"] = "inner_within_cone"
                continue
            outer_ok = _any_outer_grasp_collision_free(
                pick_contacts_model,
                list(p["trans_base"]),
                p["quat_base"],
                ref_quat,
                approach_tool_axis,
                max_approach_angle_deg,
                p["neighbor_points_base_m"],
                pick_cfg,
            )
            p["pickable"] = bool(outer_ok)
            p["pickable_reason"] = (
                "outer_collision_free" if outer_ok else "all_grasps_pointcloud_collide"
            )

        # Annotate matches for downstream visibility.
        for p in posed:
            m = p["match"]
            dx = p["trans_base"][0] - cx
            dy = p["trans_base"][1] - cy
            dist_xy = math.sqrt(dx * dx + dy * dy)
            m["all_poses_pickable"] = bool(p["pickable"])
            m["all_poses_pickable_reason"] = p["pickable_reason"]
            m["all_poses_centroid_dist_xy_m"] = dist_xy
            m["all_poses_pose_base_m"] = list(p["trans_base"])

        def _key(p: Dict[str, Any]) -> tuple:
            return (
                not bool(p["pickable"]),        # pickable first
                float(p["match"].get("all_poses_centroid_dist_xy_m") or 0.0),
                -float(p["match"].get("selection_mask_pixels") or 0.0),
            )

        posed.sort(key=_key)
        return [p["match"] for p in posed]

    return ranker


def _build_segmentation_pickable_ranker(
    *,
    pick_contacts_model: Dict[str, Any],
    base_to_cam: Dict[str, Any],
    base_to_ee: Dict[str, Any],
    base_to_tcp: Optional[Dict[str, Any]] = None,
    tcp_offset_rpy_deg: Optional[List[float]] = None,
    gripper_type: str,
    approach_tool_axis: str,
    jaw_tool_axis: str,
) -> Callable[[List[Dict[str, Any]], Dict[str, Any]], List[Dict[str, Any]]]:
    """Filter already-ranked segmentations to those with a usable pick candidate.

    The incoming match list is already ordered by the generic segmentation ranker
    (nearer/topmost/more isolated). This pass preserves that order, but drops a
    segmentation when the configured grasp-family priority groups cannot provide
    any candidate inside the current approach cone.
    """

    _tcp = base_to_tcp if isinstance(base_to_tcp, dict) else base_to_ee
    camera_position = base_to_cam.get("translation_m", [0.0, 0.0, 0.0])
    camera_z_axis = _quat_rotate(
        base_to_cam.get("rotation_quat_xyzw", [0.0, 0.0, 0.0, 1.0]),
        [0.0, 0.0, 1.0],
    )
    reference_position = _tcp.get("translation_m", [0.0, 0.0, 0.0])
    reference_quat = _tcp.get("rotation_quat_xyzw", [0.0, 0.0, 0.0, 1.0])

    def _pose_for_match(match: Dict[str, Any]) -> Optional[tuple[List[float], List[float]]]:
        pose = _match_pose_in_base(match, base_to_cam)
        if pose is not None:
            return pose
        center_cam = match.get("center_xyz_m") or match.get("pose_origin_xyz_m")
        if not _is_finite_vec(center_cam, 3):
            return None
        trans_base = _transform_point([float(v) for v in center_cam], base_to_cam)
        quat_base = _normalize_quat_xyzw(
            base_to_cam.get("rotation_quat_xyzw") or [0.0, 0.0, 0.0, 1.0]
        ) or [0.0, 0.0, 0.0, 1.0]
        return trans_base, quat_base

    def ranker(matches: List[Dict[str, Any]], pick_cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
        if not matches:
            return []
        pickable: List[Dict[str, Any]] = []
        for idx, match in enumerate(matches):
            if not isinstance(match, dict):
                continue
            pose = _pose_for_match(match)
            selected_pick_contact = None
            if pose is not None:
                object_center_base, object_quat_base = pose
                safety_points = _extract_safety_points_base_m(match, base_to_cam)
                neighbor_points_base_m = (
                _extract_scene_point_cloud_base_m(
                    match,
                    base_to_cam,
                    runs_root=Path(ctx.runs.root),
                    run_id=state.run_id,
                )
                    or safety_points.get("neighbor")
                    or []
                )
                if gripper_type == "parallel_jaw" and (pick_contacts_model.get("grasps") or []):
                    selected_pick_contact = _select_parallel_jaw_grasp(
                        pick_contacts_model,
                        list(object_center_base),
                        object_quat_base,
                        reference_position,
                        camera_position,
                        camera_z_axis,
                        reference_quat,
                        jaw_tool_axis,
                        approach_tool_axis,
                        pick_cfg.get("parallel_jaw_max_approach_angle_deg"),
                        neighbor_points_base_m=neighbor_points_base_m,
                        pick_cfg=pick_cfg,
                        approach_reference_quat_base_xyzw=base_to_ee.get(
                            "rotation_quat_xyzw",
                            [0.0, 0.0, 0.0, 1.0],
                        ),
                    )
                else:
                    selected_pick_contact = _select_pick_contact(
                        pick_contacts_model,
                        list(object_center_base),
                        object_quat_base,
                        reference_position,
                        camera_position,
                        reference_quat_base_xyzw=list(reference_quat),
                        tcp_offset_rpy_deg=None,
                        neighbor_points_base_m=neighbor_points_base_m,
                        pick_cfg=pick_cfg,
                    )
            match["pickable_candidate"] = bool(selected_pick_contact)
            match["pickable_rank_index"] = idx
            if selected_pick_contact:
                match["pickable_candidate_id"] = selected_pick_contact.get("id")
                match["pickable_grasp_group_index"] = selected_pick_contact.get(
                    "generator_group_index"
                )
                match["pickable_grasp_family_priority"] = selected_pick_contact.get(
                    "grasp_family_priority"
                )
                pickable.append(match)
        return pickable

    return ranker


def _prepare_bin_picking_task(
    ctx: StationContext,
    state: RunState,
    task_payload: Dict[str, Any],
) -> Dict[str, Any]:
    payload = deepcopy(task_payload or {})
    if not isinstance(payload, dict):
        return {}

    recipe = payload
    if isinstance(payload.get("recipe"), dict):
        recipe = payload["recipe"]

    vision_cfg = recipe.setdefault("vision", {})
    if not isinstance(vision_cfg, dict):
        vision_cfg = {}
        recipe["vision"] = vision_cfg
    filters = vision_cfg.setdefault("filters", {})
    if not isinstance(filters, dict):
        filters = {}
        vision_cfg["filters"] = filters
    module_params = dict(vision_cfg.get("params") or {})
    object_id = (
        module_params.get("object_id")
        or vision_cfg.get("object_id")
        or recipe.get("object_id")
    )
    if object_id:
        vision_cfg["object_id"] = object_id
        module_params["object_id"] = object_id
    metadata = (
        ctx.objects.get_metadata(state.process_id, object_id) or {}
        if state.process_id and object_id
        else {}
    )
    megapose_meta = metadata.get("megapose") or {}
    ppf_icp_meta = metadata.get("ppf_icp") or metadata.get("ppf_icp_bin_picking") or {}
    requested_module = str(
        vision_cfg.get("module")
        or module_params.get("module")
        or _DEFAULT_BIN_PICKING_POSE_MODULE
    ).strip().lower()
    if requested_module not in _BIN_PICKING_POSE_MODULES:
        requested_module = _DEFAULT_BIN_PICKING_POSE_MODULE
    module_meta = ppf_icp_meta if requested_module == "ppf_icp_bin_picking" else megapose_meta
    raw_object_folder = (
        module_params.get("object_folder")
        or vision_cfg.get("object_folder")
        or vision_cfg.get("object_folder_path")
    )
    object_folder = _resolve_object_folder_path(
        ctx,
        state.process_id,
        object_id,
        raw_object_folder,
    )
    vision_object_folder = _vision_object_folder_path(ctx, raw_object_folder, object_folder)
    object_folder_path = Path(object_folder)

    # Load optional bin ROI (2D OBB) for this asset/process; vision engine filters
    # segmentation masks outside this region at runtime.
    bin_roi_payload = None
    try:
        if state.process_id:
            process_record = ctx.processes.get(state.process_id) or {}
            station_id = process_record.get("station_id") or ""
            if station_id:
                bin_path = ctx.data_paths.process_bin_path(
                    station_id, state.process_id
                )
                if bin_path.exists():
                    bin_roi_payload = json.loads(
                        bin_path.read_text(encoding="utf-8")
                    )
    except Exception:
        bin_roi_payload = None
    if isinstance(bin_roi_payload, dict):
        module_params["bin_roi"] = bin_roi_payload

    vision_cfg["module"] = requested_module
    vision_cfg["process_mode"] = "trigger_only"
    vision_cfg.setdefault("shm_max_result_size", 2 * 1024 * 1024)
    module_params["object_folder"] = vision_object_folder
    module_params.pop("module", None)
    module_params.setdefault("input_color_order", "bgr")
    module_params.setdefault("selection_top_k", int(module_meta.get("selection_top_k", 5)))
    module_params.setdefault(
        "selection_min_confidence",
        float(module_meta.get("selection_min_confidence", 0.2)),
    )
    module_params.setdefault("yolo_conf", float(module_meta.get("yolo_conf", 0.2)))
    module_params.setdefault(
        "selection_min_area_px",
        float(module_meta.get("selection_min_area_px", 0.0)),
    )
    module_params.setdefault(
        "selection_max_area_px",
        float(module_meta.get("selection_max_area_px", 0.0)),
    )
    module_params.setdefault(
        "selection_min_area_ratio",
        float(module_meta.get("selection_min_area_ratio", 0.01)),
    )
    module_params.setdefault(
        "selection_max_area_ratio",
        float(module_meta.get("selection_max_area_ratio", 0.20)),
    )
    module_params.setdefault("yolo_device", str(module_meta.get("yolo_device", "cpu")))
    module_params.setdefault("mesh_units", str(module_meta.get("mesh_units", "mm")))
    module_params.setdefault("mesh_scale", float(module_meta.get("mesh_scale", 1.0)))
    module_params.setdefault("axis", list(module_meta.get("axis", [0.0, 0.0, 1.0])))
    module_params.setdefault("yolo_imgsz", int(module_meta.get("yolo_imgsz", 640)))
    module_params.setdefault(
        "all_poses_mode",
        bool(module_meta.get("all_poses_mode", False)),
    )
    module_params.setdefault(
        "all_poses_max",
        max(1, int(module_meta.get("all_poses_max", 20) or 20)),
    )
    module_params.setdefault(
        "render_all_poses_3d",
        bool(module_meta.get("render_all_poses_3d", True)),
    )
    module_params.setdefault(
        "processing_max_width",
        int(module_meta.get("processing_max_width", 1280)),
    )
    module_params.setdefault(
        "processing_max_height",
        int(module_meta.get("processing_max_height", 720)),
    )
    module_params.setdefault(
        "prewarm_before_run",
        bool(
            module_meta.get(
                "prewarm_before_run",
                requested_module == "megapose_bin_picking",
            )
        ),
    )
    module_params.setdefault(
        "prewarm_timeout_s",
        float(module_meta.get("prewarm_timeout_s", 180.0)),
    )
    module_params.setdefault(
        "startup_warmup_frames",
        int(module_meta.get("startup_warmup_frames", 0)),
    )
    module_params.setdefault(
        "max_no_candidate_frames",
        int(module_meta.get("max_no_candidate_frames", 1)),
    )
    module_params.setdefault(
        "first_frame_timeout_s",
        float(module_meta.get("first_frame_timeout_s", 5.0)),
    )
    if requested_module == "megapose_bin_picking":
        for ppf_key in (
            "ppf_relative_sampling_step",
            "ppf_relative_distance_step",
            "ppf_num_angles",
            "ppf_scene_sample_step",
            "ppf_scene_distance",
            "ppf_search_position_threshold",
            "ppf_search_rotation_threshold",
            "ppf_weighted_clustering",
            "ppf_model_sample_points",
            "ppf_scene_max_points",
            "ppf_normal_neighbors",
            "ppf_min_votes",
            "ppf_min_vote_ratio",
            "icp_enabled",
            "icp_iterations",
            "icp_tolerance",
            "icp_rejection_scale",
            "icp_num_levels",
            "icp_sample_type",
            "icp_num_max_corr",
            "icp_max_residual_m",
        ):
            module_params.pop(ppf_key, None)
        module_params.setdefault("model", str(megapose_meta.get("model", "rgbd")))
        module_params.setdefault("device", str(megapose_meta.get("device", "cpu")))
        module_params.setdefault("refiner_iterations", int(megapose_meta.get("refiner_iterations", 1)))
        module_params.setdefault("coarse_grid_size", int(megapose_meta.get("coarse_grid_size", 72)))
        module_params.setdefault(
            "depth_ray_refinement",
            bool(megapose_meta.get("depth_ray_refinement", True)),
        )
        module_params.setdefault(
            "depth_ray_refinement_iterations",
            int(megapose_meta.get("depth_ray_refinement_iterations", 2)),
        )
        module_params.setdefault(
            "depth_ray_refinement_mesh_samples",
            int(megapose_meta.get("depth_ray_refinement_mesh_samples", 2500)),
        )
        module_params.setdefault(
            "depth_ray_refinement_observed_samples",
            int(megapose_meta.get("depth_ray_refinement_observed_samples", 4000)),
        )
        module_params.setdefault(
            "depth_ray_refinement_max_correspondence_m",
            float(megapose_meta.get("depth_ray_refinement_max_correspondence_m", 0.02)),
        )
        module_params.setdefault(
            "depth_ray_refinement_max_shift_m",
            float(megapose_meta.get("depth_ray_refinement_max_shift_m", 0.05)),
        )
        module_params.setdefault(
            "depth_ray_front_layer_thickness_m",
            float(megapose_meta.get("depth_ray_front_layer_thickness_m", 0.01)),
        )
        module_params.setdefault(
            "depth_ray_center_keep_ratio",
            float(megapose_meta.get("depth_ray_center_keep_ratio", 0.7)),
        )
        module_params.setdefault("renderer_workers", 0)
        module_params.setdefault("cpu_threads", 8)
        module_params.setdefault("cpu_interop_threads", 1)
        module_params.setdefault(
            "prewarm_estimator_inference",
            bool(megapose_meta.get("prewarm_estimator_inference", True)),
        )
    else:
        for megapose_key in (
            "model",
            "device",
            "refiner_iterations",
            "coarse_grid_size",
            "depth_ray_refinement",
            "depth_ray_refinement_iterations",
            "depth_ray_refinement_mesh_samples",
            "depth_ray_refinement_observed_samples",
            "depth_ray_refinement_max_correspondence_m",
            "depth_ray_refinement_max_shift_m",
            "depth_ray_front_layer_thickness_m",
            "depth_ray_center_keep_ratio",
            "prewarm_estimator_inference",
            "renderer_workers",
            "cpu_threads",
            "cpu_interop_threads",
        ):
            module_params.pop(megapose_key, None)
        module_params.setdefault(
            "ppf_relative_sampling_step",
            float(ppf_icp_meta.get("ppf_relative_sampling_step", 0.025)),
        )
        module_params.setdefault(
            "ppf_relative_distance_step",
            float(ppf_icp_meta.get("ppf_relative_distance_step", 0.025)),
        )
        module_params.setdefault("ppf_num_angles", int(ppf_icp_meta.get("ppf_num_angles", 60)))
        module_params.setdefault(
            "ppf_scene_sample_step",
            float(ppf_icp_meta.get("ppf_scene_sample_step", 0.05)),
        )
        module_params.setdefault(
            "ppf_scene_distance",
            float(ppf_icp_meta.get("ppf_scene_distance", 0.015)),
        )
        module_params.setdefault(
            "ppf_search_position_threshold",
            float(ppf_icp_meta.get("ppf_search_position_threshold", -1.0)),
        )
        module_params.setdefault(
            "ppf_search_rotation_threshold",
            float(ppf_icp_meta.get("ppf_search_rotation_threshold", -1.0)),
        )
        module_params.setdefault(
            "ppf_weighted_clustering",
            bool(ppf_icp_meta.get("ppf_weighted_clustering", False)),
        )
        module_params.setdefault(
            "ppf_model_sample_points",
            int(ppf_icp_meta.get("ppf_model_sample_points", 1500)),
        )
        module_params.setdefault(
            "ppf_scene_max_points",
            int(ppf_icp_meta.get("ppf_scene_max_points", 12000)),
        )
        module_params.setdefault(
            "ppf_normal_neighbors",
            int(ppf_icp_meta.get("ppf_normal_neighbors", 24)),
        )
        module_params.setdefault(
            "ppf_min_votes",
            float(ppf_icp_meta.get("ppf_min_votes", 800.0)),
        )
        module_params.setdefault(
            "ppf_min_vote_ratio",
            float(ppf_icp_meta.get("ppf_min_vote_ratio", 0.2)),
        )
        module_params.setdefault("icp_enabled", bool(ppf_icp_meta.get("icp_enabled", True)))
        module_params.setdefault("icp_iterations", int(ppf_icp_meta.get("icp_iterations", 80)))
        module_params.setdefault("icp_tolerance", float(ppf_icp_meta.get("icp_tolerance", 0.05)))
        module_params.setdefault(
            "icp_rejection_scale",
            float(ppf_icp_meta.get("icp_rejection_scale", 4.0)),
        )
        module_params.setdefault("icp_num_levels", int(ppf_icp_meta.get("icp_num_levels", 6)))
        module_params.setdefault("icp_sample_type", str(ppf_icp_meta.get("icp_sample_type", "uniform")))
        module_params.setdefault("icp_num_max_corr", int(ppf_icp_meta.get("icp_num_max_corr", 1)))
        module_params.setdefault(
            "icp_max_residual_m",
            float(ppf_icp_meta.get("icp_max_residual_m", 0.03)),
        )
    module_params.setdefault("save_debug_image", True)
    module_params.setdefault("save_segmented_region", True)
    module_params.setdefault("save_pose_annotated_image", True)
    module_params.setdefault("save_pose_visualizations", True)
    module_params.setdefault("save_pose_3d_assets", True)
    module_params.setdefault("safety_pcd_max_points", 50000)
    fusion_cfg = vision_cfg.get("fusion") if isinstance(vision_cfg.get("fusion"), dict) else {}
    fusion_pipeline = (
        fusion_cfg.get("pipeline")
        if isinstance(fusion_cfg.get("pipeline"), dict)
        else {}
    )
    if _coerce_float(module_params.get("depth_max_m"), 0.0) <= 0.0:
        module_params["depth_max_m"] = float(
            module_meta.get(
                "depth_max_m",
                fusion_pipeline.get(
                    "view_max_distance_m",
                    fusion_pipeline.get("depth_max_m", 1.0),
                ),
            )
            or 1.0
        )
    if getattr(state, "run_id", None) and (
        bool(module_params.get("save_debug_image", True))
        or bool(module_params.get("save_segmented_region", True))
        or bool(module_params.get("save_pose_annotated_image", True))
    ):
        module_params["save_outputs"] = True
        module_params.setdefault(
            "output_root",
            str(ctx.data_paths.runs / str(state.run_id) / "vision"),
        )
    else:
        module_params.setdefault("save_outputs", False)
    if "label_override" not in module_params and object_id:
        module_params["label_override"] = object_id
    vision_cfg["params"] = module_params
    # Bin picking consumes a single 6DoF vision result. Do not inherit the
    # generic template-matching acceptance thresholds, or valid MegaPose results
    # with small ROIs will be silently ignored while the task keeps waiting.
    filters.setdefault("min_inliers", 0)
    filters.setdefault("min_score", 0.0)
    filters.setdefault("min_area_px", 0)
    filters.setdefault("min_bbox_px", 1)
    filters.setdefault("min_scale", 0.0)
    filters.setdefault("max_scale", 1000.0)

    robot_cfg = recipe.setdefault("robot", {})
    if not isinstance(robot_cfg, dict):
        robot_cfg = {}
        recipe["robot"] = robot_cfg
    pick_cfg = robot_cfg.setdefault("pick", {})
    if not isinstance(pick_cfg, dict):
        pick_cfg = {}
        robot_cfg["pick"] = pick_cfg
    # `0` means keep waiting indefinitely for a new object to appear.
    robot_cfg.setdefault("max_pick_attempts", 0)
    robot_cfg.setdefault("no_candidate_retry_delay_s", 0.5)
    robot_cfg.setdefault("stay_at_capture_on_no_candidate", True)
    robot_cfg.setdefault("use_intermediate_grasp_pose", True)
    robot_cfg.setdefault("intermediate_pose_name", "")
    robot_cfg.setdefault("intermediate_pose", {})
    robot_cfg.setdefault("intermediate_move_strategy", "cartesian")
    pick_cfg.setdefault("orientation_mode", "vision_pose")
    pick_cfg.setdefault("vision_pose_quat_offset_xyzw", [0.0, 0.0, 0.0, 1.0])
    pick_cfg.setdefault("parallel_jaw_max_approach_angle_deg", 45.0)
    pick_cfg.setdefault("parallel_jaw_max_jaw_axis_tilt_from_xy_deg", 20.0)
    pick_cfg.setdefault("parallel_jaw_internal_pregrasp_open_offset_m", 0.020)
    pick_cfg.setdefault("parallel_jaw_internal_grasp_close_offset_m", -0.005)
    pick_cfg.setdefault("parallel_jaw_external_pregrasp_open_offset_m", 0.020)
    pick_cfg.setdefault("parallel_jaw_external_grasp_close_offset_m", -0.005)
    pick_cfg.setdefault("grasp_family_mode", "default")
    pick_cfg.setdefault("parallel_jaw_pregrasp_descent_m", 0.050)
    pick_cfg.setdefault("parallel_jaw_prefer_inner_grasp", True)
    pick_cfg.setdefault(
        "pointcloud_collision_scene_yaml",
        _default_grasp_collision_scene_yaml_path(),
    )
    # The mesh obstacle set (e.g. the bin) comes from the selected task's
    # scene.yaml when it exists, so the grasp collision check + debug GLB use the
    # bin where the operator placed it. Falls back to the semantics scene.
    _task_obstacles_scene = _task_obstacles_scene_yaml(ctx, state)
    if _task_obstacles_scene:
        pick_cfg.setdefault(
            "pointcloud_collision_obstacles_scene_yaml",
            _task_obstacles_scene,
        )
    pick_cfg.setdefault("pointcloud_collision_mode", "full")
    pick_cfg.setdefault("pointcloud_collision_consider_pointcloud", True)
    # GLB debug exports (each ~2 s via trimesh; all include the full scene — bin
    # mesh, obstacle point cloud, object CAD, base/camera/object/grasp frames):
    #   selected_glb           -> selected/*.glb (selected grasp scene)         [on]
    #   colliding_glb          -> colliding/<id>_gripper.glb per colliding cand [on, capped]
    #   colliding_glb_max      -> cap on how many colliding GLBs to write       [12]
    #   grasp_debug_glb        -> write ALL colliding GLBs uncapped + everything [off]
    pick_cfg.setdefault("parallel_jaw_grasp_debug_selected_glb", True)
    pick_cfg.setdefault("parallel_jaw_grasp_debug_colliding_glb", True)
    pick_cfg.setdefault("parallel_jaw_grasp_debug_colliding_glb_max", 12)
    pick_cfg.setdefault("parallel_jaw_grasp_debug_glb", False)
    pick_cfg.setdefault("pointcloud_collision_representation", "raw_points")
    # Raw scene-point collision uses the exact perception scene_point_cloud.ply.
    # These legacy knobs are kept for JSON compatibility but are not used to
    # voxelize, downsample, crop, or filter the live cloud.
    pick_cfg.setdefault("pointcloud_collision_voxel_size_m", 0.001)
    pick_cfg.setdefault("pointcloud_collision_inflation_m", 0.0)
    pick_cfg.setdefault("pointcloud_collision_density_grid_m", 0.0)
    pick_cfg.setdefault("pointcloud_collision_mesh_min_cluster_points", 30)
    pick_cfg.setdefault("pointcloud_collision_mesh_min_cluster_voxels", 4)
    pick_cfg.setdefault("pointcloud_collision_mesh_max_clusters", 8)
    pick_cfg.setdefault("pointcloud_collision_mesh_max_voxels", 100000)
    pick_cfg.setdefault("pointcloud_collision_mesh_bin_margin_m", 0.03)
    pick_cfg.setdefault("pointcloud_target_exclusion_radius_m", 0.012)
    pick_cfg.setdefault("vacuum_cup_radius_m", 0.015)
    pick_cfg.setdefault("vacuum_safety_margin_m", 0.003)
    pick_cfg.setdefault("vacuum_pregrasp_descent_m", 0.050)
    pick_cfg.setdefault("segmentation_safety_ranking_enabled", True)
    pick_cfg.setdefault("segmentation_grasp_candidate_ranking_enabled", False)
    pick_cfg.setdefault(
        "all_poses_mode",
        bool(module_params.get("all_poses_mode", False)),
    )

    if bool(module_params.get("all_poses_mode", False)):
        try:
            configured_timeout_s = float(vision_cfg.get("timeout_s", 0.0) or 0.0)
        except (TypeError, ValueError):
            configured_timeout_s = 0.0
        if configured_timeout_s <= 0.0:
            vision_cfg["timeout_s"] = 120.0

    if not pick_cfg.get("pick_contacts_file"):
        for _default_name in ("grasp_authoring.json", "pick_contacts.json"):
            default_contacts = object_folder_path / _default_name
            if default_contacts.exists() and default_contacts.is_file():
                pick_cfg["pick_contacts_file"] = str(default_contacts)
                break

    retry_errors = [str(v).strip() for v in robot_cfg.get("retry_errors") or [] if str(v).strip()]
    raw_max_attempts = int(robot_cfg.get("max_pick_attempts", 0) or 0)
    allow_no_candidate_retry = raw_max_attempts <= 0 or raw_max_attempts > 1
    if allow_no_candidate_retry:
        for no_candidate_error in sorted(_BIN_PICKING_NO_CANDIDATE_ERRORS):
            if no_candidate_error not in retry_errors:
                retry_errors.append(no_candidate_error)
    else:
        retry_errors = [
            code
            for code in retry_errors
            if code not in _BIN_PICKING_NO_CANDIDATE_ERRORS
        ]
    robot_cfg["retry_errors"] = retry_errors

    return payload


def _load_bin_picking_runtime_state(
    ctx: StationContext,
    state: RunState,
    recipe: Dict[str, Any],
    vision_cfg: Dict[str, Any],
    robot_cfg: Dict[str, Any],
    pick_cfg: Dict[str, Any],
    module_params: Dict[str, Any],
    cycle: int,
    log_debug,
) -> Dict[str, Any]:
    search_roots = [Path.cwd(), ctx.data_root]
    object_folder = _clean_path_string(module_params.get("object_folder"))
    if object_folder:
        search_roots.insert(0, Path(object_folder))
    place_cfg = robot_cfg.get("place") if isinstance(robot_cfg.get("place"), dict) else {}
    pick_contacts_model = None
    # Load grasp model
    explicit_contacts_file = pick_cfg.get("pick_contacts_file") or module_params.get("pick_contacts_file")
    if explicit_contacts_file:
        if str(explicit_contacts_file).endswith("grasp_authoring.json"):
            pick_contacts_model = _load_authored_grasps_model(
                ctx,
                module_params,
                pick_cfg,
                search_roots,
            )
        else:
            pick_contacts_model = _load_pick_contacts_model(
                module_params,
                pick_cfg,
                search_roots,
            )
    else:
        # No explicit file — try authored grasps then fall back to pick_contacts
        pick_contacts_model = _load_authored_grasps_model(
            ctx,
            module_params,
            pick_cfg,
            search_roots,
        )
        if pick_contacts_model is None:
            pick_contacts_model = _load_pick_contacts_model(
                module_params,
                pick_cfg,
                search_roots,
            )
    if pick_contacts_model:
        log_debug(
            "pick_contacts_model",
            cycle=cycle,
            path=pick_contacts_model.get("path"),
                contact_count=len(pick_contacts_model.get("contacts") or []),
                grasp_count=len(pick_contacts_model.get("grasps") or []),
                default_gripper_type=pick_contacts_model.get("default_gripper_type"),
                tool_axis=pick_contacts_model.get("tool_axis"),
                jaw_axis=pick_contacts_model.get("jaw_axis"),
                force_yaw_deg=pick_contacts_model.get("force_yaw_deg"),
                grasp_family_selection_policy=pick_contacts_model.get(
                    "grasp_family_selection_policy"
                ),
            )
    place_policy_model = _load_place_policy_model(
        module_params,
        place_cfg,
        search_roots,
    )
    if place_policy_model:
        log_debug(
            "place_policy_model",
            cycle=cycle,
            path=place_policy_model.get("path"),
            format=place_policy_model.get("format"),
            enabled=place_policy_model.get("enabled"),
            placement_goal=place_policy_model.get("placement_goal"),
            strategy_count=len(place_policy_model.get("strategies") or []),
        )
    if isinstance(pick_cfg, dict):
        pick_cfg.pop("_all_poses_ranker", None)
        pick_cfg.pop("_segmentation_pickable_ranker", None)
    if (
        isinstance(pick_cfg, dict)
        and isinstance(pick_contacts_model, dict)
    ):
        hand_eye_raw, _pref, _used = _resolve_runtime_handeye(
            ctx,
            state.process_id,
            robot_cfg,
        )
        hand_eye = _resolve_hand_eye(hand_eye_raw)
        hand_eye_frame = (
            str(hand_eye.get("hand_eye_frame") or "gripper_to_camera").strip().lower()
        )
        robot_state_now = ctx.robot.get_state() or {}
        tcp_pose = (
            robot_state_now.get("flange_pose")
            if isinstance(robot_state_now.get("flange_pose"), dict)
            else robot_state_now.get("tcp_pose")
        ) or {}
        base_to_ee = {
            "translation_m": tcp_pose.get("position_m", [0.0, 0.0, 0.0]),
            "rotation_quat_xyzw": tcp_pose.get("quat_xyzw", [0.0, 0.0, 0.0, 1.0]),
        }
        _tcp_calib, _tcp_calib_source = _resolve_runtime_tcp_calibration(
            ctx,
            state.process_id,
            hand_eye_raw,
        )
        pick_cfg["_tcp_grasp_up_axis"] = str(_tcp_calib.get("grasp_up_axis") or "").strip().lower()
        pick_cfg["_tcp_grasp_up_min_dot"] = _coerce_float(
            _tcp_calib.get("grasp_up_min_dot"),
            0.0,
        )
        pick_cfg["_tcp_jaw_tool_axis"] = str(_tcp_calib.get("jaw_tool_axis") or _tcp_calib.get("parallel_jaw_jaw_tool_axis") or "").strip().lower()
        base_to_tcp = _apply_tcp_calibration_to_base(
            base_to_ee,
            _tcp_calib,
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
        if bool(pick_cfg.get("all_poses_mode", False)):
            pick_cfg["_all_poses_ranker"] = _build_all_poses_ranker(
                ctx=ctx,
                state=state,
                pick_contacts_model=pick_contacts_model,
                base_to_cam=base_to_cam,
                base_to_ee=base_to_ee,
                approach_tool_axis=_parallel_jaw_approach_tool_axis(pick_cfg),
                max_approach_angle_deg=_coerce_float(
                    pick_cfg.get("parallel_jaw_max_approach_angle_deg"),
                    45.0,
                ),
                mesh_path=pick_contacts_model.get("source_mesh_path"),
                mesh_units=str(
                    pick_contacts_model.get("mesh_units")
                    or module_params.get("mesh_units")
                    or "mm"
                ),
                mesh_scale=_coerce_float(
                    pick_contacts_model.get("mesh_scale"),
                    _coerce_float(module_params.get("mesh_scale"), 1.0) or 1.0,
                )
                or 1.0,
            )
        elif bool(pick_cfg.get("segmentation_grasp_candidate_ranking_enabled", False)):
            gripper_type = _canonical_gripper_type(
                pick_cfg.get("gripper_type")
                or robot_cfg.get("gripper_type")
                or pick_contacts_model.get("default_gripper_type")
            )
            pick_cfg["_segmentation_pickable_ranker"] = _build_segmentation_pickable_ranker(
                pick_contacts_model=pick_contacts_model,
                base_to_cam=base_to_cam,
                base_to_ee=base_to_ee,
                base_to_tcp=base_to_tcp,
                tcp_offset_rpy_deg=None,
                gripper_type=gripper_type,
                approach_tool_axis=_parallel_jaw_approach_tool_axis(pick_cfg),
                jaw_tool_axis=_parallel_jaw_jaw_tool_axis(pick_cfg),
            )
    return {
        "pick_contacts_model": pick_contacts_model,
        "place_policy_model": place_policy_model,
        "selected_pick_contact": None,
        "selected_place_strategy": None,
    }


def _resolve_bin_picking_target_override(
    ctx: StationContext,
    state: RunState,
    recipe: Dict[str, Any],
    vision_cfg: Dict[str, Any],
    robot_cfg: Dict[str, Any],
    pick_cfg: Dict[str, Any],
    module_params: Dict[str, Any],
    runtime_state: Any,
    cycle: int,
    attempt: int,
    match: Dict[str, Any],
    base_to_cam: Dict[str, Any],
    base_to_ee: Dict[str, Any],
    object_center_base: List[float],
    target_base: List[float],
    log_debug,
) -> Optional[Dict[str, Any]]:
    runtime = runtime_state if isinstance(runtime_state, dict) else {}
    pick_contacts_model = runtime.get("pick_contacts_model")
    place_policy_model = runtime.get("place_policy_model")
    if not pick_contacts_model or not _is_finite_vec(object_center_base, 3):
        if isinstance(runtime_state, dict):
            runtime_state["selected_pick_contact"] = None
            runtime_state["selected_place_strategy"] = None
        return None

    # Compute base_to_tcp using the station TCP calibration so that reference
    # position/orientation reflects the actual tool tip, not the raw flange.
    try:
        _hand_eye_for_tcp, _pref_for_tcp, _used_for_tcp = _resolve_runtime_handeye(
            ctx,
            state.process_id,
            robot_cfg,
        )
    except Exception:
        _hand_eye_for_tcp = {}
    _tcp_calib, _tcp_calib_source = _resolve_runtime_tcp_calibration(
        ctx,
        state.process_id,
        _hand_eye_for_tcp,
    )
    pick_cfg["_tcp_grasp_up_axis"] = str(_tcp_calib.get("grasp_up_axis") or "").strip().lower()
    pick_cfg["_tcp_grasp_up_min_dot"] = _coerce_float(
        _tcp_calib.get("grasp_up_min_dot"),
        0.0,
    )
    base_to_tcp = _apply_tcp_calibration_to_base(
        base_to_ee,
        _tcp_calib,
    )

    vision_pose_quat_cam = _normalize_quat_xyzw(
        match.get("pose_quat_xyzw")
        or match.get("quaternion_xyzw")
        or [0.0, 0.0, 0.0, 1.0]
    )
    object_quat_base = _quat_mul_xyzw(
        base_to_cam.get("rotation_quat_xyzw", [0.0, 0.0, 0.0, 1.0]),
        vision_pose_quat_cam,
    )
    # Prefer the exact scene_point_cloud.ply from perception for collision/debug.
    # No target exclusion, AABB crop, downsample, octree, or mesh conversion.
    safety_points = _extract_safety_points_base_m(match, base_to_cam)
    scene_points_base_m = (
        _extract_scene_point_cloud_base_m(
            match,
            base_to_cam,
            runs_root=Path(ctx.runs.root),
            run_id=state.run_id,
        )
        or safety_points.get("neighbor")
        or []
    )
    neighbor_points_base_m = scene_points_base_m

    gripper_type = _canonical_gripper_type(
        pick_cfg.get("gripper_type")
        or robot_cfg.get("gripper_type")
        or pick_contacts_model.get("default_gripper_type")
    )
    if gripper_type == "parallel_jaw":
        selected_pick_contact = None
        if pick_contacts_model.get("grasps") or []:
            grasp_audit: Dict[str, Any] = {}
            pick_cfg["_grasp_planning_audit"] = grasp_audit
            try:
                selected_pick_contact = _select_parallel_jaw_grasp(
                    pick_contacts_model,
                    list(object_center_base),
                    object_quat_base,
                    base_to_tcp.get("translation_m", [0.0, 0.0, 0.0]),
                    base_to_cam.get("translation_m", [0.0, 0.0, 0.0]),
                    _quat_rotate(
                        base_to_cam.get("rotation_quat_xyzw", [0.0, 0.0, 0.0, 1.0]),
                        [0.0, 0.0, 1.0],
                    ),
                    base_to_tcp.get("rotation_quat_xyzw", [0.0, 0.0, 0.0, 1.0]),
                    _parallel_jaw_jaw_tool_axis(pick_cfg),
                    _parallel_jaw_approach_tool_axis(pick_cfg),
                    pick_cfg.get("parallel_jaw_max_approach_angle_deg"),
                    neighbor_points_base_m=neighbor_points_base_m,
                    pick_cfg=pick_cfg,
                    approach_reference_quat_base_xyzw=base_to_ee.get(
                        "rotation_quat_xyzw",
                        [0.0, 0.0, 0.0, 1.0],
                    ),
                )
            finally:
                pick_cfg.pop("_grasp_planning_audit", None)
            grasp_audit["camera_position_base_m"] = list(
                base_to_cam.get("translation_m") or [0.0, 0.0, 0.0]
            )
            grasp_audit["camera_quat_base_xyzw"] = list(
                base_to_cam.get("rotation_quat_xyzw") or [0.0, 0.0, 0.0, 1.0]
            )
            grasp_audit["obstacle_points_base_m_raw_count"] = len(
                scene_points_base_m
            )
            grasp_audit["obstacle_points_base_m"] = neighbor_points_base_m
            grasp_audit["target_points_base_m"] = safety_points.get("target") or []
            debug_dir = _write_parallel_jaw_grasp_planning_debug(
                ctx,
                state,
                cycle,
                attempt,
                match,
                grasp_audit,
                write_glb=bool(pick_cfg.get("parallel_jaw_grasp_debug_glb", False)),
                write_selected_glb=bool(pick_cfg.get("parallel_jaw_grasp_debug_selected_glb", True)),
                write_colliding_glb=bool(pick_cfg.get("parallel_jaw_grasp_debug_colliding_glb", True)),
                colliding_glb_cap=int(pick_cfg.get("parallel_jaw_grasp_debug_colliding_glb_max", 12) or 12),
            )
            if debug_dir:
                log_debug(
                    "grasp_planning_debug",
                    cycle=cycle,
                    attempt=attempt,
                    rank=match.get("rank"),
                    selected_id=grasp_audit.get("selected_id"),
                    accepted_count=grasp_audit.get("accepted_count"),
                    reason_counts=grasp_audit.get("reason_counts"),
                    debug_dir=debug_dir,
                )
    else:
        selected_pick_contact = _select_pick_contact(
            pick_contacts_model,
            list(object_center_base),
            object_quat_base,
            base_to_tcp.get("translation_m", [0.0, 0.0, 0.0]),
            base_to_cam.get("translation_m", [0.0, 0.0, 0.0]),
            reference_quat_base_xyzw=list(base_to_tcp.get("rotation_quat_xyzw") or [0,0,0,1]),
            tcp_offset_rpy_deg=None,
            neighbor_points_base_m=neighbor_points_base_m,
            pick_cfg=pick_cfg,
        )
    if not selected_pick_contact:
        if isinstance(runtime_state, dict):
            runtime_state["selected_pick_contact"] = None
            runtime_state["selected_place_strategy"] = None
            rejected_rank = int(_coerce_float(match.get("rank"), -1.0) or -1)
            if rejected_rank >= 0:
                rejected = runtime_state.setdefault("rejected_detection_ranks", [])
                if rejected_rank not in rejected:
                    rejected.append(rejected_rank)
                log_debug(
                    "pick_contact_reject",
                    cycle=cycle,
                    attempt=attempt,
                    rank=rejected_rank,
                    rejected_detection_ranks=list(rejected),
                    reason="parallel_jaw_no_grasp_candidate",
                )
        if gripper_type == "parallel_jaw" and (pick_contacts_model.get("grasps") or []):
            return {"_no_grasp_candidate": True}
        return None
    if isinstance(runtime_state, dict):
        runtime_state["selected_pick_contact"] = selected_pick_contact
        runtime_state["selected_place_strategy"] = _select_place_strategy(
            place_policy_model,
            selected_pick_contact,
        )
        _audit_robot_engine_selected_grasp(
            ctx,
            state,
            pick_cfg,
            selected_pick_contact,
            match,
            cycle,
            attempt,
            log_debug,
        )
    selected_place_strategy = runtime.get("selected_place_strategy")

    object_center_camera = _vec3_or_none(match.get("center_xyz_m")) or [0.0, 0.0, 0.0]
    if selected_pick_contact.get("grasp_type") == "parallel_jaw_pair":
        center_camera_m = _add_vec(
            object_center_camera,
            _quat_rotate(vision_pose_quat_cam, selected_pick_contact.get("center_local_m")),
        )
        contact_a_camera_m = _add_vec(
            object_center_camera,
            _quat_rotate(vision_pose_quat_cam, selected_pick_contact.get("contact_a_local_m")),
        )
        contact_b_camera_m = _add_vec(
            object_center_camera,
            _quat_rotate(vision_pose_quat_cam, selected_pick_contact.get("contact_b_local_m")),
        )
        jaw_axis_camera = _normalize_vec(
            _quat_rotate(vision_pose_quat_cam, selected_pick_contact.get("jaw_axis_local"))
        )
        approach_axis_camera = _normalize_vec(
            _quat_rotate(
                vision_pose_quat_cam,
                selected_pick_contact.get("approach_axis_local"),
            )
        )
        approach_tip_camera_m = (
            _add_vec(
                center_camera_m,
                [
                    -approach_axis_camera[0] * 0.03,
                    -approach_axis_camera[1] * 0.03,
                    -approach_axis_camera[2] * 0.03,
                ],
            )
            if approach_axis_camera is not None
            else None
        )
        center_uv = _project_point_uv(center_camera_m, match.get("camera_intrinsics"))
        contact_a_uv = _project_point_uv(contact_a_camera_m, match.get("camera_intrinsics"))
        contact_b_uv = _project_point_uv(contact_b_camera_m, match.get("camera_intrinsics"))
        approach_tip_uv = _project_point_uv(
            approach_tip_camera_m, match.get("camera_intrinsics")
        )
        match_out = dict(match)
        match_out["pick_contact"] = {
            "id": selected_pick_contact.get("id"),
            "label": selected_pick_contact.get("label"),
            "grasp_type": "parallel_jaw_pair",
            "point_local_m": selected_pick_contact.get("center_local_m"),
            "point_base_m": selected_pick_contact.get("center_base_m"),
            "point_camera_m": center_camera_m,
            "point_uv": center_uv,
            "center_local_m": selected_pick_contact.get("center_local_m"),
            "center_base_m": selected_pick_contact.get("center_base_m"),
            "center_camera_m": center_camera_m,
            "center_uv": center_uv,
            "contact_a_local_m": selected_pick_contact.get("contact_a_local_m"),
            "contact_b_local_m": selected_pick_contact.get("contact_b_local_m"),
            "contact_a_base_m": selected_pick_contact.get("contact_a_base_m"),
            "contact_b_base_m": selected_pick_contact.get("contact_b_base_m"),
            "contact_a_camera_m": contact_a_camera_m,
            "contact_b_camera_m": contact_b_camera_m,
            "contact_a_uv": contact_a_uv,
            "contact_b_uv": contact_b_uv,
            "jaw_axis_local": selected_pick_contact.get("jaw_axis_local"),
            "jaw_axis_base": selected_pick_contact.get("jaw_axis_base"),
            "jaw_axis_camera": jaw_axis_camera,
            "approach_axis_local": selected_pick_contact.get("approach_axis_local"),
            "approach_axis_base": selected_pick_contact.get("approach_axis_base"),
            "approach_axis_camera": approach_axis_camera,
            "normal_local": selected_pick_contact.get("approach_axis_local"),
            "normal_base": selected_pick_contact.get("approach_axis_base"),
            "normal_camera": approach_axis_camera,
            "normal_tip_camera_m": approach_tip_camera_m,
            "normal_tip_uv": approach_tip_uv,
            "opening_width_m": selected_pick_contact.get("opening_width_m"),
            "distance_to_reference_m": selected_pick_contact.get("distance_to_reference_m"),
            "distance_to_camera_m": selected_pick_contact.get("center_distance_to_camera_m"),
            "closest_contact_distance_to_camera_m": selected_pick_contact.get(
                "closest_contact_distance_to_camera_m"
            ),
            "camera_alignment_score": selected_pick_contact.get("approach_alignment_score"),
            "approach_to_camera_score": selected_pick_contact.get("approach_to_camera_score"),
            "camera_z_alignment_score": selected_pick_contact.get("camera_z_alignment_score"),
            "jaw_axis_abs_z": selected_pick_contact.get("jaw_axis_abs_z"),
            "jaw_axis_alignment_score": selected_pick_contact.get(
                "jaw_axis_alignment_score"
            ),
            "min_jaw_axis_alignment_score": selected_pick_contact.get(
                "min_jaw_axis_alignment_score"
            ),
            "grasp_up_axis": selected_pick_contact.get("grasp_up_axis"),
            "grasp_up_axis_base": selected_pick_contact.get("grasp_up_axis_base"),
            "grasp_up_dot_base_z": selected_pick_contact.get("grasp_up_dot_base_z"),
            "grasp_up_min_dot": selected_pick_contact.get("grasp_up_min_dot"),
            "jaw_axis_xy_score": selected_pick_contact.get("jaw_axis_xy_score"),
            "jaw_axis_horizontal_ok": selected_pick_contact.get("jaw_axis_horizontal_ok"),
            "max_jaw_axis_abs_z": selected_pick_contact.get("max_jaw_axis_abs_z"),
            "tool_approach_alignment_score": selected_pick_contact.get(
                "tool_approach_alignment_score"
            ),
            "robot_z_alignment_score": selected_pick_contact.get(
                "robot_z_alignment_score"
            ),
            "threshold_alignment_score": selected_pick_contact.get(
                "threshold_alignment_score"
            ),
            "orientation_distance_rad": selected_pick_contact.get(
                "orientation_distance_rad"
            ),
            "pointcloud_collision_passed": selected_pick_contact.get(
                "pointcloud_collision_passed"
            ),
            "pointcloud_collision_pairs": selected_pick_contact.get(
                "pointcloud_collision_pairs"
            ),
            "pointcloud_check_ms": selected_pick_contact.get("pointcloud_check_ms"),
            "stroke_mm": selected_pick_contact.get("stroke_mm"),
            "authored_stroke_mm": selected_pick_contact.get("authored_stroke_mm"),
            "pregrasp_open_width_m": selected_pick_contact.get("pregrasp_open_width_m"),
            "pregrasp_open_offset_m": selected_pick_contact.get("pregrasp_open_offset_m"),
            "pregrasp_action": selected_pick_contact.get("pregrasp_action"),
            "T_world_gripper_root": selected_pick_contact.get("T_world_gripper_root"),
            "generator_group_index": selected_pick_contact.get("generator_group_index"),
            "grasp_family_label": selected_pick_contact.get("grasp_family_label"),
            "grasp_family_priority": selected_pick_contact.get("grasp_family_priority"),
        }
        runtime_target_base = selected_pick_contact.get("center_base_m")
    else:
        point_camera_m = _add_vec(
            object_center_camera,
            _quat_rotate(vision_pose_quat_cam, selected_pick_contact.get("point_local_m")),
        )
        normal_camera = _normalize_vec(
            _quat_rotate(vision_pose_quat_cam, selected_pick_contact.get("normal_local"))
        )
        normal_tip_camera_m = (
            _add_vec(point_camera_m, [normal_camera[0] * 0.03, normal_camera[1] * 0.03, normal_camera[2] * 0.03])
            if normal_camera is not None
            else None
        )
        point_uv = _project_point_uv(point_camera_m, match.get("camera_intrinsics"))
        normal_tip_uv = _project_point_uv(normal_tip_camera_m, match.get("camera_intrinsics"))

        match_out = dict(match)
        match_out["pick_contact"] = {
            "id": selected_pick_contact.get("id"),
            "label": selected_pick_contact.get("label"),
            "grasp_type": "vacuum_contact",
            "point_local_m": selected_pick_contact.get("point_local_m"),
            "point_base_m": selected_pick_contact.get("point_base_m"),
            "point_camera_m": point_camera_m,
            "point_uv": point_uv,
            "normal_local": selected_pick_contact.get("normal_local"),
            "normal_base": selected_pick_contact.get("normal_base"),
            "normal_camera": normal_camera,
            "normal_tip_camera_m": normal_tip_camera_m,
            "normal_tip_uv": normal_tip_uv,
            "distance_to_reference_m": selected_pick_contact.get("distance_to_reference_m"),
            "distance_to_camera_m": selected_pick_contact.get("distance_to_camera_m"),
            "upward_score": selected_pick_contact.get("upward_score"),
            "camera_alignment_score": selected_pick_contact.get("camera_alignment_score"),
        }
        runtime_target_base = selected_pick_contact.get("point_base_m")
    if selected_place_strategy:
        match_out["place_strategy"] = {
            "name": selected_place_strategy.get("name"),
            "mode": selected_place_strategy.get("mode"),
        }
    log_debug(
        "pick_contact",
        cycle=cycle,
        attempt=attempt,
        object_center_base=object_center_base,
        target_base=runtime_target_base,
        reference_position_m=base_to_tcp.get("translation_m", [0.0, 0.0, 0.0]),
        reference_quat_base_xyzw=base_to_ee.get("rotation_quat_xyzw", [0.0, 0.0, 0.0, 1.0]),
        tcp_reference_quat_base_xyzw=base_to_tcp.get("rotation_quat_xyzw", [0.0, 0.0, 0.0, 1.0]),
        tcp_calibration=_tcp_calib,
        tcp_calibration_source=_tcp_calib_source,
        pick_contact=match_out["pick_contact"],
        contacts_path=pick_contacts_model.get("path"),
        place_strategy=match_out.get("place_strategy"),
    )
    return {
        "target_base": runtime_target_base,
        "match": match_out,
        "runtime_target_data": match_out.get("pick_contact"),
    }


def _audit_robot_engine_selected_grasp(
    ctx: StationContext,
    state: RunState,
    pick_cfg: Dict[str, Any],
    selected_pick_contact: Dict[str, Any],
    match: Dict[str, Any],
    cycle: int,
    attempt: int,
    log_debug,
) -> None:
    """Log robot_engine feasibility for the selected Orch grasp."""
    if not bool(pick_cfg.get("robot_engine_validation_enabled", True)):
        return
    try:
        object_id = (
            match.get("object_id")
            or match.get("label")
            or match.get("class_name")
            or selected_pick_contact.get("object_id")
        )
        result = evaluate_robot_engine_scene(
            ctx,
            state.process_id,
            str(object_id) if object_id else None,
            {
                "distance": True,
                "collision": True,
                "target_grasp_id": selected_pick_contact.get("id"),
            },
        )
        evaluation = result.get("evaluation") or {}
        motion = result.get("motion") if isinstance(result.get("motion"), dict) else {}
        errors = evaluation.get("errors") or []
        log_debug(
            "robot_engine_grasp_validation",
            cycle=cycle,
            attempt=attempt,
            rank=match.get("rank"),
            selected_id=selected_pick_contact.get("id"),
            scene_ok=bool(evaluation.get("ok")),
            collision=bool((evaluation.get("collision") or {}).get("collision")),
            colliding_pairs=(evaluation.get("collision") or {}).get("colliding_pairs") or [],
            motion_success=bool(motion.get("success")),
            motion_rejection_reason=motion.get("rejection_reason"),
            reason_codes=[err.get("code") for err in errors if isinstance(err, dict)],
        )
    except Exception as exc:
        log_debug(
            "robot_engine_grasp_validation_error",
            cycle=cycle,
            attempt=attempt,
            rank=match.get("rank"),
            selected_id=selected_pick_contact.get("id"),
            error=str(exc),
        )


def _enrich_bin_picking_vision_only_match(
    ctx: StationContext,
    state: RunState,
    recipe: Dict[str, Any],
    vision_cfg: Dict[str, Any],
    robot_cfg: Dict[str, Any],
    pick_cfg: Dict[str, Any],
    module_params: Dict[str, Any],
    runtime_state: Any,
    match: Dict[str, Any],
    log_debug,
) -> Optional[Dict[str, Any]]:
    runtime = runtime_state if isinstance(runtime_state, dict) else {}
    pick_contacts_model = runtime.get("pick_contacts_model")
    if not isinstance(pick_contacts_model, dict):
        return None
    target_cam = _vec3_or_none(match.get("center_xyz_m"))
    if not target_cam:
        return None
    hand_eye_raw, _pref, _used = _resolve_runtime_handeye(
        ctx,
        state.process_id,
        robot_cfg,
    )
    hand_eye = _resolve_hand_eye(hand_eye_raw)
    hand_eye_frame = (
        str(hand_eye.get("hand_eye_frame") or "gripper_to_camera").strip().lower()
    )
    robot_state_now = ctx.robot.get_state() or {}
    tcp_pose = (
        robot_state_now.get("flange_pose")
        if isinstance(robot_state_now.get("flange_pose"), dict)
        else robot_state_now.get("tcp_pose")
    ) or {}
    base_to_ee = {
        "translation_m": tcp_pose.get("position_m", [0.0, 0.0, 0.0]),
        "rotation_quat_xyzw": tcp_pose.get("quat_xyzw", [0.0, 0.0, 0.0, 1.0]),
    }
    _tcp_calib, _tcp_calib_source = _resolve_runtime_tcp_calibration(
        ctx,
        state.process_id,
        hand_eye_raw,
    )
    base_to_tcp = _apply_tcp_calibration_to_base(
        base_to_ee,
        _tcp_calib,
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
    target_override = _resolve_bin_picking_target_override(
        ctx,
        state,
        recipe,
        vision_cfg,
        robot_cfg,
        pick_cfg,
        module_params,
        runtime_state,
        0,
        0,
        match,
        base_to_cam,
        base_to_ee,
        object_center_base,
        list(object_center_base),
        log_debug,
    )
    if not isinstance(target_override, dict):
        return None
    match_out = target_override.get("match")
    return match_out if isinstance(match_out, dict) else None


def _resolve_bin_picking_orientation_overrides(
    ctx: StationContext,
    state: RunState,
    recipe: Dict[str, Any],
    vision_cfg: Dict[str, Any],
    robot_cfg: Dict[str, Any],
    pick_cfg: Dict[str, Any],
    module_params: Dict[str, Any],
    runtime_state: Any,
    cycle: int,
    attempt: int,
    match: Dict[str, Any],
    base_to_cam: Dict[str, Any],
    base_to_ee: Dict[str, Any],
    capture_pose: Dict[str, Any],
    capture_tcp_pose: Dict[str, Any],
) -> Dict[str, Any]:
    runtime = runtime_state if isinstance(runtime_state, dict) else {}
    pick_contacts_model = runtime.get("pick_contacts_model")
    selected_pick_contact = runtime.get("selected_pick_contact")
    overrides: Dict[str, Any] = {}
    if match.get("pose_quat_xyzw") or match.get("quaternion_xyzw"):
        overrides["orientation_mode_default"] = "vision_pose"
    if not selected_pick_contact:
        return overrides

    vision_pose_quat_cam = _normalize_quat_xyzw(
        match.get("pose_quat_xyzw")
        or match.get("quaternion_xyzw")
        or [0.0, 0.0, 0.0, 1.0]
    )

    if selected_pick_contact.get("grasp_type") == "parallel_jaw_pair":
        jaw_axis_base = _normalize_vec(selected_pick_contact.get("jaw_axis_base") or [])
        approach_axis_base = _normalize_vec(
            selected_pick_contact.get("approach_axis_base") or []
        )
        jaw_axis_camera = _normalize_vec(
            _quat_rotate(
                vision_pose_quat_cam,
                selected_pick_contact.get("jaw_axis_local", [1.0, 0.0, 0.0]),
            )
        )
        approach_axis_camera = _normalize_vec(
            _quat_rotate(
                vision_pose_quat_cam,
                selected_pick_contact.get("approach_axis_local", [0.0, 0.0, 1.0]),
            )
        )
        explicit_orientation = None
        if jaw_axis_base is not None and approach_axis_base is not None:
            explicit_orientation = _build_parallel_jaw_orientation(
                jaw_axis_base=jaw_axis_base,
                approach_axis_base=approach_axis_base,
                jaw_tool_axis=_parallel_jaw_jaw_tool_axis(pick_cfg),
                approach_tool_axis=_parallel_jaw_approach_tool_axis(pick_cfg),
            )
        overrides.update(
            {
                "force_align_with_surface": True,
                "smart_yaw_default": False,
                "yaw_decouple_surface_default": True,
                "yaw_frame_default": "base",
                "surface_align_axis_default": _parallel_jaw_approach_tool_axis(pick_cfg),
                "normal_cam": approach_axis_camera,
                "normal_base": approach_axis_base,
            }
        )
        if explicit_orientation is not None:
            overrides["orientation_quat_xyzw"] = explicit_orientation
        return overrides

    normal_cam = _normalize_vec(
        _quat_rotate(
            vision_pose_quat_cam,
            selected_pick_contact.get("normal_local", [0.0, 0.0, 1.0]),
        )
    )
    overrides.update(
        {
            "force_align_with_surface": True,
            "yaw_deg": _coerce_float(
                (pick_contacts_model or {}).get("force_yaw_deg"),
                0.0,
            ),
            "smart_yaw_default": False,
            "yaw_decouple_surface_default": True,
            "yaw_frame_default": "base",
            "surface_align_axis_default": str(
                (pick_contacts_model or {}).get("tool_axis", "-z")
            ).lower(),
            "normal_cam": normal_cam,
            "normal_base": selected_pick_contact.get("normal_base"),
        }
    )
    return overrides


def _resolve_bin_picking_place_plan(
    ctx: StationContext,
    state: RunState,
    recipe: Dict[str, Any],
    vision_cfg: Dict[str, Any],
    robot_cfg: Dict[str, Any],
    pick_cfg: Dict[str, Any],
    module_params: Dict[str, Any],
    runtime_state: Any,
    execution: Dict[str, Any],
    log_debug,
) -> Optional[Dict[str, Any]]:
    runtime = runtime_state if isinstance(runtime_state, dict) else {}
    selected_pick_contact = runtime.get("selected_pick_contact")
    selected_place_strategy = runtime.get("selected_place_strategy")
    place_policy_model = runtime.get("place_policy_model")
    if not selected_place_strategy and isinstance(selected_pick_contact, dict):
        selected_place_strategy = _select_place_strategy(
            place_policy_model,
            selected_pick_contact,
        )
    if not isinstance(selected_place_strategy, dict):
        return None

    pose_index = execution.get("pose_index") if isinstance(execution.get("pose_index"), dict) else {}
    plan: Dict[str, Any] = {
        "strategy_name": selected_place_strategy.get("name"),
        "mode": selected_place_strategy.get("mode"),
    }
    final_place_cfg = (
        selected_place_strategy.get("final_place")
        if isinstance(selected_place_strategy.get("final_place"), dict)
        else {}
    )
    final_place_ref = (
        final_place_cfg.get("pose")
        or final_place_cfg.get("pose_name")
        or final_place_cfg.get("name")
    )
    if final_place_ref is None and any(
        key in final_place_cfg for key in ("tcp_pose", "tcp", "joints", "position_m")
    ):
        final_place_ref = final_place_cfg
    if final_place_ref is not None:
        resolved_final_place = _resolve_pose_reference(
            pose_index,
            final_place_ref,
            "place_policy_final_place_pose_missing",
        )
        if resolved_final_place:
            plan["place_pose"] = resolved_final_place.get("pose")
    if final_place_cfg.get("profile"):
        plan["place_profile"] = str(final_place_cfg.get("profile")).strip()
    if final_place_cfg.get("parallel_jaw_open_width_m") is not None:
        plan["place_open_width_m"] = _coerce_float(
            final_place_cfg.get("parallel_jaw_open_width_m"),
            None,
        )
    if final_place_cfg.get("approach_offset_m") is not None:
        plan["place_approach_offset_m"] = final_place_cfg.get("approach_offset_m")
    if final_place_cfg.get("retreat_offset_m") is not None:
        plan["place_retreat_offset_m"] = final_place_cfg.get("retreat_offset_m")
    if final_place_cfg.get("quat_xyzw") is not None:
        plan["place_quat_xyzw"] = final_place_cfg.get("quat_xyzw")
    if final_place_cfg.get("force_yaw_deg") is not None:
        plan["force_place_yaw_deg"] = _coerce_float(
            final_place_cfg.get("force_yaw_deg"),
            None,
        )

    if str(selected_place_strategy.get("mode") or "").strip().lower() != "intermediate_regrasp":
        log_debug(
            "place_strategy_selected",
            cycle=execution.get("cycle"),
            attempt=execution.get("attempt"),
            strategy=selected_place_strategy,
        )
        return plan

    intermediate_cfg = (
        selected_place_strategy.get("intermediate_regrasp")
        if isinstance(selected_place_strategy.get("intermediate_regrasp"), dict)
        else {}
    )
    resolved_place = _resolve_pose_reference(
        pose_index,
        intermediate_cfg.get("place_pose") or intermediate_cfg.get("place_pose_name"),
        "intermediate_regrasp_place_pose_missing",
    )
    resolved_pick = _resolve_pose_reference(
        pose_index,
        intermediate_cfg.get("pick_pose") or intermediate_cfg.get("pick_pose_name"),
        "intermediate_regrasp_pick_pose_missing",
    )
    if not resolved_place or not resolved_pick:
        raise RuntimeError("intermediate_regrasp_pose_missing")

    plan["intermediate_regrasp"] = {
        "place_pose": resolved_place.get("pose"),
        "place_pose_name": resolved_place.get("pose_name"),
        "pick_pose": resolved_pick.get("pose"),
        "pick_pose_name": resolved_pick.get("pose_name"),
        "place_profile": str(
            intermediate_cfg.get("place_profile") or execution.get("default_profile") or ""
        ).strip(),
        "pick_profile": str(
            intermediate_cfg.get("pick_profile") or execution.get("default_profile") or ""
        ).strip(),
        "place_approach_offset_m": intermediate_cfg.get(
            "place_approach_offset_m",
            [0.0, 0.0, 0.08],
        ),
        "place_retreat_offset_m": intermediate_cfg.get(
            "place_retreat_offset_m",
            intermediate_cfg.get("place_approach_offset_m", [0.0, 0.0, 0.08]),
        ),
        "pick_approach_offset_m": intermediate_cfg.get(
            "pick_approach_offset_m",
            [0.0, 0.0, 0.08],
        ),
        "pick_retreat_offset_m": intermediate_cfg.get(
            "pick_retreat_offset_m",
            intermediate_cfg.get("pick_approach_offset_m", [0.0, 0.0, 0.08]),
        ),
        "place_open_width_m": _coerce_float(
            intermediate_cfg.get("place_open_width_m"),
            None,
        ),
        "pick_open_width_m": _coerce_float(
            intermediate_cfg.get("pick_open_width_m"),
            None,
        ),
        "pick_close_width_m": _coerce_float(
            intermediate_cfg.get("pick_close_width_m"),
            None,
        ),
        "settle_time_s": max(
            0.0,
            float(intermediate_cfg.get("settle_time_s", 0.0) or 0.0),
        ),
    }
    log_debug(
        "place_strategy_selected",
        cycle=execution.get("cycle"),
        attempt=execution.get("attempt"),
        strategy=selected_place_strategy,
        place_plan=plan,
    )
    return plan


def run_bin_picking(ctx: StationContext, state: RunState, handle: Any) -> None:
    prepared_task = _prepare_bin_picking_task(ctx, state, state.task)
    _start_bin_picking_prewarm(ctx, state, prepared_task)
    state_bin = replace(
        state,
        task_type="bin_picking",
        task=prepared_task,
    )
    hooks = PickRuntimeHooks(
        load_runtime_state=_load_bin_picking_runtime_state,
        enrich_vision_only_match=_enrich_bin_picking_vision_only_match,
        resolve_target_override=_resolve_bin_picking_target_override,
        resolve_orientation_overrides=_resolve_bin_picking_orientation_overrides,
        resolve_place_plan=_resolve_bin_picking_place_plan,
    )
    run_pick_place_core(ctx, state_bin, handle, hooks=hooks)
