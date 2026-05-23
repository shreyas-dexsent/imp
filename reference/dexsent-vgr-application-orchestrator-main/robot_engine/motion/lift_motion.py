from __future__ import annotations

from robot_engine.motion.frame_offset import offset_transform
from robot_engine.motion.motion_request import FrameOffsetRequest, LiftOptions, MotionRequest
from robot_engine.motion.motion_result import MotionRejectionReason
from robot_engine.motion.path_planner import plan_motion


def plan_lift_motion(request: MotionRequest):
    options = request.lift or LiftOptions()
    if options.distance <= 0:
        from robot_engine.motion.motion_primitive import failed_result

        return failed_result(options.motion_type, request, "lift_offset", MotionRejectionReason.INVALID_DISTANCE)
    start = request.start_frame
    if start is None:
        from robot_engine.motion.motion_primitive import frame_from_fk

        start = frame_from_fk(request, request.current_joint_state)
    if start is None:
        from robot_engine.motion.motion_primitive import failed_result

        return failed_result(options.motion_type, request, "lift_start", MotionRejectionReason.FRAME_NOT_FOUND)
    lift_target = offset_transform(
        FrameOffsetRequest(
            frame=start,
            axis=options.axis,
            direction=options.direction,
            distance=options.distance,
            reference_frame=options.reference_frame,
            output_child_frame=f"{start.child_frame}_lift",
        )
    )
    return plan_motion(request.model_copy(update={"motion_type": options.motion_type, "start_frame": start, "target_frame": lift_target, "label": "lift"}))
