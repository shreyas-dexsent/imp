from __future__ import annotations

from typing import List

from robot_engine.kinematics.kinematic_chain import KinematicChain
from robot_engine.motion.motion_request import MotionRequest
from robot_engine.motion.motion_result import JointTrajectory, MotionSegment, MotionSequence, MotionSequenceResult


def plan_motion_sequence(sequence: MotionSequence) -> MotionSequenceResult:
    from robot_engine.motion.path_planner import plan_motion

    segments: List[MotionSegment] = []
    trajectories = []
    current_state = None
    for index, request in enumerate(sequence.segments):
        request = MotionRequest.model_validate(request)
        if current_state is not None:
            request.current_joint_state = current_state
        result = plan_motion(request)
        segments.append(MotionSegment(name=request.label or f"segment_{index}", motion_type=request.motion_type, result=result))
        if not result.success:
            return MotionSequenceResult(
                success=False,
                segments=segments,
                failed_stage=result.failed_stage,
                failed_segment_index=index,
                failed_waypoint_index=result.failed_waypoint_index,
                rejection_reason=result.rejection_reason,
                debug_info=result.debug_info,
            )
        if result.trajectory is not None:
            trajectories.append(result.trajectory)
        if result.joint_waypoints:
            current_state = result.joint_waypoints[-1]
    trajectory = concatenate_trajectories(trajectories)
    clearances = [segment.result.minimum_clearance for segment in segments if segment.result and segment.result.minimum_clearance is not None]
    return MotionSequenceResult(
        success=True,
        segments=segments,
        trajectory=trajectory,
        estimated_duration=trajectory.times[-1] if trajectory and trajectory.times else sum((s.wait_seconds for s in segments), 0.0),
        minimum_clearance=min(clearances) if clearances else None,
    )


def concatenate_trajectories(trajectories: List[JointTrajectory]) -> JointTrajectory | None:
    if not trajectories:
        return None
    joint_names = trajectories[0].joint_names
    positions = []
    times = []
    offset = 0.0
    for traj in trajectories:
        start_index = 1 if positions and traj.positions else 0
        positions.extend(traj.positions[start_index:])
        times.extend([offset + t for t in traj.times[start_index:]])
        if times:
            offset = times[-1]
    return JointTrajectory(joint_names=joint_names, positions=positions, times=times)
