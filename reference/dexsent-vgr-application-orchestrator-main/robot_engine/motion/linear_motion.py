from __future__ import annotations

import numpy as np

from robot_engine.kinematics.kinematic_chain import KinematicChain
from robot_engine.motion.motion_primitive import (
    failed_result,
    frame_from_fk,
    interpolate_cartesian_frames,
    max_joint_delta,
    solve_target_ik,
)
from robot_engine.motion.motion_request import MotionRequest, MotionType, TrajectoryValidationRequest
from robot_engine.motion.motion_result import MotionRejectionReason, MotionResult
from robot_engine.motion.path_smoothing import remove_duplicate_joint_waypoints
from robot_engine.motion.trajectory_generator import generate_joint_trajectory
from robot_engine.motion.trajectory_validator import validate_trajectory


def plan_linear_move_to_frame(request: MotionRequest) -> MotionResult:
    if request.motion_type != MotionType.LINEAR:
        return failed_result(MotionType.LINEAR, request, "dispatch", MotionRejectionReason.UNSUPPORTED_MOTION_TYPE)
    chain = KinematicChain(request.chain)
    current = chain.clamp(request.current_joint_state)
    start_frame = request.start_frame or frame_from_fk(request, current)
    if start_frame is None:
        return failed_result(MotionType.LINEAR, request, "start_frame", MotionRejectionReason.FRAME_NOT_FOUND)

    cartesian = interpolate_cartesian_frames(start_frame, request.target_frame, request.trajectory_options.linear_waypoint_count)
    joint_waypoints = [current]
    seed = request.ik_options.seed or current
    for index, waypoint in enumerate(cartesian[1:], start=1):
        ik = solve_target_ik(request, waypoint, seed=seed)
        if not ik.ok:
            return failed_result(MotionType.LINEAR, request, "linear_ik", MotionRejectionReason.LINEAR_PATH_FAILED, index, {"ik_reason": ik.reason})
        if joint_waypoints:
            step = float(np.max(np.abs(np.asarray([ik.joint_positions[n] for n in chain.joint_names]) - np.asarray([seed[n] for n in chain.joint_names]))))
            if step > request.ik_options.continuity_joint_step:
                return failed_result(MotionType.LINEAR, request, "ik_continuity", MotionRejectionReason.IK_DISCONTINUITY, index, {"step": step})
        joint_waypoints.append(ik.joint_positions)
        seed = ik.joint_positions

    joint_waypoints = remove_duplicate_joint_waypoints(chain, joint_waypoints)
    trajectory = generate_joint_trajectory(chain, joint_waypoints, request.trajectory_options)
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
            motion_type=MotionType.LINEAR,
            start_frame=start_frame,
            target_frame=request.target_frame,
            generated_frames=[request.target_frame],
            cartesian_waypoints=cartesian,
            joint_waypoints=joint_waypoints,
            trajectory=trajectory,
            estimated_duration=trajectory.times[-1] if trajectory.times else 0.0,
            minimum_clearance=validation.minimum_clearance,
            max_joint_motion=validation.max_joint_motion,
            failed_stage=validation.failed_stage,
            failed_waypoint_index=validation.failed_waypoint_index,
            rejection_reason=validation.rejection_reason or MotionRejectionReason.TRAJECTORY_VALIDATION_FAILED,
            debug_info=validation.debug_info,
        )

    return MotionResult(
        success=True,
        motion_type=MotionType.LINEAR,
        start_frame=start_frame,
        target_frame=request.target_frame,
        generated_frames=[request.target_frame],
        cartesian_waypoints=cartesian,
        joint_waypoints=joint_waypoints,
        trajectory=trajectory,
        estimated_duration=trajectory.times[-1] if trajectory.times else 0.0,
        minimum_clearance=validation.minimum_clearance,
        max_joint_motion=max_joint_delta(chain, joint_waypoints),
    )
