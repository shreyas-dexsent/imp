from __future__ import annotations

import numpy as np

from robot_engine.collision.distance_queries import aabb_distance
from robot_engine.interfaces.schemas import JacobianRequest
from robot_engine.kinematics.jacobian_solver import compute_jacobian
from robot_engine.kinematics.kinematic_chain import KinematicChain
from robot_engine.math_utils import as_matrix
from robot_engine.motion.motion_primitive import q_from_values
from robot_engine.motion.motion_request import TrajectoryValidationRequest
from robot_engine.motion.motion_result import JointTrajectory, MotionRejectionReason, TrajectoryValidationResult


def validate_trajectory(request: TrajectoryValidationRequest) -> TrajectoryValidationResult:
    trajectory = request.trajectory
    if not isinstance(trajectory, JointTrajectory):
        trajectory = JointTrajectory.model_validate(trajectory)
    chain = KinematicChain(request.chain)
    min_clearance = None
    max_motion = 0.0
    previous = None

    for index, position in enumerate(trajectory.positions):
        q = q_from_values(chain, position)
        bad = chain.violates_limits(q)
        if bad:
            return _fail("joint_limits", MotionRejectionReason.JOINT_LIMIT_VIOLATION, index, min_clearance, max_motion, {"joints": bad})
        if previous is not None:
            step = float(np.max(np.abs(np.asarray(position, dtype=float) - np.asarray(previous, dtype=float))))
            max_motion = max(max_motion, step)
        previous = position

        singular = _check_singularity(request, q, index, min_clearance, max_motion)
        if singular is not None:
            return singular

        clearance_result = _check_clearance(request, chain, q, index, min_clearance, max_motion)
        if isinstance(clearance_result, TrajectoryValidationResult):
            return clearance_result
        if clearance_result is not None:
            min_clearance = clearance_result if min_clearance is None else min(min_clearance, clearance_result)

    limit_result = _check_velocity_acceleration(request, trajectory, min_clearance, max_motion)
    if limit_result is not None:
        return limit_result

    return TrajectoryValidationResult(success=True, minimum_clearance=min_clearance, max_joint_motion=max_motion)


def _check_singularity(request, q, index, min_clearance, max_motion):
    jac = compute_jacobian(JacobianRequest(chain=request.chain, joint_positions=q))
    if jac.ok and jac.condition_number is not None and jac.condition_number > request.ik_options.singularity_threshold:
        return _fail(
            "singularity",
            MotionRejectionReason.SINGULARITY_RISK,
            index,
            min_clearance,
            max_motion,
            {"condition_number": jac.condition_number},
        )
    return None


def _check_clearance(request, chain: KinematicChain, q, index, min_clearance, max_motion):
    options = request.collision_options
    if not options.enabled or options.world is None:
        return None
    tcp = chain.forward_matrices(q).transforms[chain.tcp_frame]
    point_bounds = np.vstack([tcp[:3, 3], tcp[:3, 3]])
    clearances = []
    for object_id, obj in options.world.objects.items():
        distance, _, _, colliding = aabb_distance(point_bounds, obj.world_aabb())
        clearance = float(distance) - float(options.tcp_clearance_radius)
        clearances.append(clearance)
        if colliding or clearance <= 0.0:
            return _fail("collision", MotionRejectionReason.COLLISION_DETECTED, index, min_clearance, max_motion, {"object_id": object_id})
        if clearance < options.minimum_clearance:
            return _fail(
                "clearance",
                MotionRejectionReason.CLEARANCE_TOO_LOW,
                index,
                min_clearance,
                max_motion,
                {"object_id": object_id, "clearance": clearance, "minimum_clearance": options.minimum_clearance},
            )
    return min(clearances) if clearances else None


def _check_velocity_acceleration(request, trajectory: JointTrajectory, min_clearance, max_motion):
    max_velocity = float(request.trajectory_options.max_joint_velocity)
    max_acceleration = float(request.trajectory_options.max_joint_acceleration)
    for index, values in enumerate(trajectory.velocities):
        if values and float(np.max(np.abs(values))) > max_velocity + 1e-9:
            return _fail("velocity", MotionRejectionReason.VELOCITY_LIMIT_VIOLATION, index, min_clearance, max_motion, {"max_velocity": max_velocity})
    for index, values in enumerate(trajectory.accelerations):
        if values and float(np.max(np.abs(values))) > max_acceleration + 1e-9:
            return _fail("acceleration", MotionRejectionReason.ACCELERATION_LIMIT_VIOLATION, index, min_clearance, max_motion, {"max_acceleration": max_acceleration})
    return None


def _fail(stage, reason, index, min_clearance, max_motion, debug):
    return TrajectoryValidationResult(
        success=False,
        minimum_clearance=min_clearance,
        max_joint_motion=max_motion,
        failed_stage=stage,
        failed_waypoint_index=index,
        rejection_reason=reason,
        debug_info=debug,
    )
