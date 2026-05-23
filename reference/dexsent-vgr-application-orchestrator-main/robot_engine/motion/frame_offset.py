from __future__ import annotations

import numpy as np

from robot_engine.math_utils import as_matrix, to_transform
from robot_engine.motion.motion_request import Axis, AxisDirection, AxisFrame, FrameOffsetRequest, MotionType
from robot_engine.motion.motion_result import MotionRejectionReason, MotionResult


def compute_offset_frame(request: FrameOffsetRequest) -> MotionResult:
    if request.distance <= 0 or not np.isfinite(request.distance):
        return MotionResult(
            success=False,
            motion_type=MotionType.LINEAR,
            start_frame=request.frame,
            target_frame=request.frame,
            failed_stage="frame_offset",
            rejection_reason=MotionRejectionReason.INVALID_DISTANCE,
            debug_info={"distance": request.distance},
        )
    try:
        target = offset_transform(request)
        return MotionResult(
            success=True,
            motion_type=MotionType.LINEAR,
            start_frame=request.frame,
            target_frame=target,
            generated_frames=[target],
            cartesian_waypoints=[request.frame, target],
            debug_info={"axis": request.axis.value, "direction": request.direction.value, "reference_frame": request.reference_frame.value},
        )
    except ValueError as exc:
        return MotionResult(
            success=False,
            motion_type=MotionType.LINEAR,
            start_frame=request.frame,
            target_frame=request.frame,
            failed_stage="frame_offset",
            rejection_reason=MotionRejectionReason.INVALID_AXIS,
            debug_info={"error": str(exc)},
        )


def offset_transform(request: FrameOffsetRequest):
    frame = as_matrix(request.frame)
    ref = _reference_matrix(request)
    signed_distance = request.distance * (1.0 if request.direction == AxisDirection.POSITIVE else -1.0)
    axis_vec = _axis_vector(request.axis)
    world_delta = ref[:3, :3] @ (axis_vec * signed_distance)
    out = frame.copy()
    out[:3, 3] = frame[:3, 3] + world_delta
    return to_transform(request.frame.parent_frame, request.output_child_frame or request.frame.child_frame, out)


def _reference_matrix(request: FrameOffsetRequest) -> np.ndarray:
    if request.reference_frame == AxisFrame.BASE:
        return np.eye(4)
    if request.reference_frame == AxisFrame.CUSTOM:
        if request.reference_transform is None:
            raise ValueError("CUSTOM axis frame requires reference_transform")
        return as_matrix(request.reference_transform)
    # TCP/object/grasp/bin offsets use the supplied frame orientation unless a
    # specific reference transform is provided.
    return as_matrix(request.reference_transform) if request.reference_transform is not None else as_matrix(request.frame)


def _axis_vector(axis: Axis) -> np.ndarray:
    if axis == Axis.X:
        return np.array([1.0, 0.0, 0.0])
    if axis == Axis.Y:
        return np.array([0.0, 1.0, 0.0])
    if axis == Axis.Z:
        return np.array([0.0, 0.0, 1.0])
    raise ValueError(f"Unsupported axis: {axis}")
