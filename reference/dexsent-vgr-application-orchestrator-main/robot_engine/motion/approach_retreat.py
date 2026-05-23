from __future__ import annotations

from robot_engine.motion.frame_offset import offset_transform
from robot_engine.motion.motion_request import ApproachOptions, FrameOffsetRequest, MotionRequest, RetreatOptions
from robot_engine.motion.motion_result import MotionRejectionReason, MotionSequenceResult
from robot_engine.motion.motion_result import MotionSequence as MotionSequenceRequest
from robot_engine.motion.motion_sequence import plan_motion_sequence


def plan_approach_to_frame(request: MotionRequest) -> MotionSequenceResult:
    options = request.approach or ApproachOptions(enabled=True)
    if not options.enabled:
        return plan_motion_sequence(MotionSequenceRequest(name="target_only", segments=[request]))
    if options.distance <= 0:
        return MotionSequenceResult(success=False, failed_stage="approach_offset", rejection_reason=MotionRejectionReason.INVALID_DISTANCE)
    approach_frame = offset_transform(
        FrameOffsetRequest(
            frame=request.target_frame,
            axis=options.axis,
            direction=options.direction,
            distance=options.distance,
            reference_frame=options.reference_frame,
            output_child_frame=f"{request.target_frame.child_frame}_approach",
        )
    )
    approach_request = request.model_copy(update={"motion_type": options.motion_type, "target_frame": approach_frame, "label": "approach"})
    target_request = request.model_copy(update={"label": "target"})
    return plan_motion_sequence(MotionSequenceRequest(name="approach_to_target", segments=[approach_request, target_request]))


def plan_retreat_from_frame(request: MotionRequest) -> MotionSequenceResult:
    options = request.retreat or RetreatOptions(enabled=True)
    if not options.enabled:
        return plan_motion_sequence(MotionSequenceRequest(name="target_only", segments=[request]))
    if options.distance <= 0:
        return MotionSequenceResult(success=False, failed_stage="retreat_offset", rejection_reason=MotionRejectionReason.INVALID_DISTANCE)
    retreat_frame = offset_transform(
        FrameOffsetRequest(
            frame=request.target_frame,
            axis=options.axis,
            direction=options.direction,
            distance=options.distance,
            reference_frame=options.reference_frame,
            output_child_frame=f"{request.target_frame.child_frame}_retreat",
        )
    )
    target_request = request.model_copy(update={"label": "target"})
    retreat_request = request.model_copy(update={"motion_type": options.motion_type, "target_frame": retreat_frame, "label": "retreat"})
    return plan_motion_sequence(MotionSequenceRequest(name="target_to_retreat", segments=[target_request, retreat_request]))
