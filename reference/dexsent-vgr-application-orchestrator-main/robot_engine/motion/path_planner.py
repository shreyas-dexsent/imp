from __future__ import annotations

from robot_engine.motion.frame_offset import compute_offset_frame
from robot_engine.motion.joint_motion import plan_joint_move_to_frame
from robot_engine.motion.linear_motion import plan_linear_move_to_frame
from robot_engine.motion.motion_request import ExportTrajectoryRequest, MotionRequest, MotionType
from robot_engine.motion.motion_result import MotionRejectionReason, MotionSequence
from robot_engine.motion.trajectory_validator import validate_trajectory


def plan_motion(request: MotionRequest):
    if request.motion_type == MotionType.JOINT:
        return plan_joint_move_to_frame(request)
    if request.motion_type == MotionType.LINEAR:
        return plan_linear_move_to_frame(request)
    from robot_engine.motion.motion_primitive import failed_result

    return failed_result(request.motion_type, request, "dispatch", MotionRejectionReason.UNSUPPORTED_MOTION_TYPE)


def plan_motion_sequence(request: MotionSequence):
    from robot_engine.motion.motion_sequence import plan_motion_sequence as _plan_motion_sequence

    return _plan_motion_sequence(request)


def export_robot_trajectory(request: ExportTrajectoryRequest):
    trajectory = request.trajectory
    if hasattr(trajectory, "model_dump"):
        return trajectory.model_dump()
    return trajectory


__all__ = [
    "plan_motion",
    "plan_joint_move_to_frame",
    "plan_linear_move_to_frame",
    "compute_offset_frame",
    "plan_motion_sequence",
    "validate_trajectory",
    "export_robot_trajectory",
]
