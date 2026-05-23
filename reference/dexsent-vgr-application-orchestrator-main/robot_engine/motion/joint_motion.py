from __future__ import annotations

from robot_engine.kinematics.kinematic_chain import KinematicChain
from robot_engine.motion.motion_primitive import (
    best_ik_solution,
    failed_result,
    frames_from_joint_waypoints,
    interpolate_joint_waypoints,
    max_joint_delta,
)
from robot_engine.motion.motion_request import MotionRequest, MotionType, TrajectoryValidationRequest
from robot_engine.motion.motion_result import MotionRejectionReason, MotionResult
from robot_engine.motion.path_smoothing import remove_duplicate_joint_waypoints
from robot_engine.motion.trajectory_generator import generate_joint_trajectory
from robot_engine.motion.trajectory_validator import validate_trajectory


def plan_joint_move_to_frame(request: MotionRequest) -> MotionResult:
    if request.motion_type != MotionType.JOINT:
        return failed_result(MotionType.JOINT, request, "dispatch", MotionRejectionReason.UNSUPPORTED_MOTION_TYPE)
    chain = KinematicChain(request.chain)
    current = chain.clamp(request.current_joint_state)
    ik_result, effort, failures = best_ik_solution(request, request.target_frame)
    if ik_result is None:
        return failed_result(MotionType.JOINT, request, "ik", MotionRejectionReason.IK_FAILED, debug={"ik_failures": failures})

    waypoints = interpolate_joint_waypoints(chain, current, ik_result.joint_positions, request.trajectory_options.max_joint_step, request.trajectory_options.joint_waypoint_count)
    waypoints = remove_duplicate_joint_waypoints(chain, waypoints)
    trajectory = generate_joint_trajectory(chain, waypoints, request.trajectory_options)
    cartesian = frames_from_joint_waypoints(request, waypoints)

    validation = validate_trajectory(
        TrajectoryValidationRequest(
            chain=request.chain,
            trajectory=trajectory,
            collision_options=request.collision_options,
            trajectory_options=request.trajectory_options,
            ik_options=request.ik_options,
        )
    )
    if request.planning_options.validate_trajectory and not validation.success:
        return MotionResult(
            success=False,
            motion_type=MotionType.JOINT,
            start_frame=cartesian[0] if cartesian else request.start_frame,
            target_frame=request.target_frame,
            generated_frames=[request.target_frame],
            cartesian_waypoints=cartesian,
            joint_waypoints=waypoints,
            trajectory=trajectory,
            estimated_duration=trajectory.times[-1] if trajectory.times else 0.0,
            minimum_clearance=validation.minimum_clearance,
            max_joint_motion=validation.max_joint_motion,
            failed_stage=validation.failed_stage,
            failed_waypoint_index=validation.failed_waypoint_index,
            rejection_reason=validation.rejection_reason or MotionRejectionReason.TRAJECTORY_VALIDATION_FAILED,
            debug_info={"ik_iterations": ik_result.iterations, "joint_effort": effort, **validation.debug_info},
        )

    return MotionResult(
        success=True,
        motion_type=MotionType.JOINT,
        start_frame=cartesian[0] if cartesian else request.start_frame,
        target_frame=request.target_frame,
        generated_frames=[request.target_frame],
        cartesian_waypoints=cartesian,
        joint_waypoints=waypoints,
        trajectory=trajectory,
        estimated_duration=trajectory.times[-1] if trajectory.times else 0.0,
        minimum_clearance=validation.minimum_clearance,
        max_joint_motion=max_joint_delta(chain, waypoints),
        debug_info={"ik_iterations": ik_result.iterations, "joint_effort": effort},
    )
