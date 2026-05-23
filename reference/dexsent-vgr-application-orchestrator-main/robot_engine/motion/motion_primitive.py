from __future__ import annotations

import math
from typing import Dict, Iterable, List

import numpy as np

from robot_engine.interfaces.schemas import AlgorithmError, FKRequest, IKRequest, IKResult, JacobianRequest, Transform3D
from robot_engine.kinematics.fk_solver import compute_fk
from robot_engine.kinematics.ik_solver import solve_ik
from robot_engine.kinematics.jacobian_solver import compute_jacobian
from robot_engine.kinematics.kinematic_chain import KinematicChain
from robot_engine.math_utils import as_matrix, to_transform
from robot_engine.motion.motion_request import IKOptions, MotionRequest, MotionType
from robot_engine.motion.motion_result import MotionRejectionReason


def ordered_q(chain: KinematicChain, q: Dict[str, float]) -> List[float]:
    return [float(q.get(name, 0.0)) for name in chain.joint_names]


def q_from_values(chain: KinematicChain, values: Iterable[float]) -> Dict[str, float]:
    return {name: float(value) for name, value in zip(chain.joint_names, values)}


def joint_effort(chain: KinematicChain, start: Dict[str, float], goal: Dict[str, float]) -> float:
    return float(np.linalg.norm(np.asarray(ordered_q(chain, goal)) - np.asarray(ordered_q(chain, start))))


def max_joint_delta(chain: KinematicChain, waypoints: List[Dict[str, float]]) -> float:
    if len(waypoints) < 2:
        return 0.0
    values = np.asarray([ordered_q(chain, q) for q in waypoints], dtype=float)
    return float(np.max(np.abs(np.diff(values, axis=0))))


def interpolate_joint_waypoints(chain: KinematicChain, start: Dict[str, float], goal: Dict[str, float], max_step: float, min_count: int) -> List[Dict[str, float]]:
    start_v = np.asarray(ordered_q(chain, start), dtype=float)
    goal_v = np.asarray(ordered_q(chain, goal), dtype=float)
    max_delta = float(np.max(np.abs(goal_v - start_v))) if start_v.size else 0.0
    count = max(2, int(min_count), int(math.ceil(max_delta / max(max_step, 1e-9))) + 1)
    return [q_from_values(chain, start_v + (goal_v - start_v) * t) for t in np.linspace(0.0, 1.0, count)]


def frame_from_fk(request: MotionRequest, q: Dict[str, float]) -> Transform3D | None:
    chain = KinematicChain(request.chain)
    fk = compute_fk(FKRequest(chain=request.chain, joint_positions=q, target_frame=chain.tcp_frame))
    if not fk.ok:
        return None
    return fk.transforms[chain.tcp_frame]


def frames_from_joint_waypoints(request: MotionRequest, waypoints: List[Dict[str, float]]) -> List[Transform3D]:
    frames = []
    for waypoint in waypoints:
        frame = frame_from_fk(request, waypoint)
        if frame is not None:
            frames.append(frame)
    return frames


def solve_target_ik(request: MotionRequest, target: Transform3D, seed: Dict[str, float] | None = None):
    options = request.ik_options
    result = solve_ik(
        IKRequest(
            chain=request.chain,
            target=target,
            seed=seed or options.seed or request.current_joint_state,
            max_iterations=options.max_iterations,
            position_tolerance=options.position_tolerance,
            orientation_tolerance=options.orientation_tolerance,
            damping=options.damping,
            singularity_threshold=options.singularity_threshold,
        )
    )
    if result.ok:
        return result
    return solve_position_only_ik(request, target, seed=seed)


def solve_position_only_ik(request: MotionRequest, target: Transform3D, seed: Dict[str, float] | None = None) -> IKResult:
    chain = KinematicChain(request.chain)
    target_pos = as_matrix(target)[:3, 3]
    q = chain.clamp({name: float((seed or request.ik_options.seed or request.current_joint_state).get(name, 0.0)) for name in chain.joint_names})
    pos_err = None
    for iteration in range(1, request.ik_options.max_iterations + 1):
        current = chain.forward_matrices(q).transforms[chain.tcp_frame]
        err = target_pos - current[:3, 3]
        pos_err = float(np.linalg.norm(err))
        if pos_err <= request.ik_options.position_tolerance:
            return IKResult(ok=True, joint_positions=q, iterations=iteration, position_error=pos_err, orientation_error=None, reason="IK_CONVERGED")
        jac_result = compute_jacobian(JacobianRequest(chain=request.chain, joint_positions=q, frame_id=chain.tcp_frame))
        if not jac_result.ok:
            return IKResult(ok=False, joint_positions=q, iterations=iteration, reason=jac_result.error.code, error=jac_result.error)
        jac = np.asarray(jac_result.jacobian, dtype=float)[:3, :]
        lhs = jac @ jac.T + (request.ik_options.damping ** 2) * np.eye(3)
        try:
            step = jac.T @ np.linalg.solve(lhs, err)
        except np.linalg.LinAlgError:
            step = jac.T @ np.linalg.pinv(lhs) @ err
        if float(np.linalg.norm(step)) < 1e-12:
            break
        for i, name in enumerate(chain.joint_names):
            q[name] += float(step[i])
        bad = chain.violates_limits(q)
        if bad:
            q = chain.clamp(q)
            if pos_err > request.ik_options.position_tolerance:
                continue
    return IKResult(
        ok=False,
        joint_positions=q,
        iterations=request.ik_options.max_iterations,
        position_error=pos_err,
        reason="IK_UNREACHABLE",
        error=AlgorithmError(code="IK_UNREACHABLE", message="Position-only IK did not converge.", details={"position_error": pos_err}),
    )


def best_ik_solution(request: MotionRequest, target: Transform3D):
    chain = KinematicChain(request.chain)
    seeds = [request.current_joint_state, request.ik_options.seed, *request.ik_options.seed_states]
    best = None
    best_effort = float("inf")
    failures = []
    for seed in seeds:
        result = solve_target_ik(request, target, seed=seed)
        if result.ok:
            effort = joint_effort(chain, request.current_joint_state, result.joint_positions)
            if effort < best_effort:
                best = result
                best_effort = effort
        else:
            failures.append(result.reason)
    return best, best_effort, failures


def interpolate_cartesian_frames(start: Transform3D, target: Transform3D, count: int) -> List[Transform3D]:
    start_m = as_matrix(start)
    target_m = as_matrix(target)
    frames = []
    for i, t in enumerate(np.linspace(0.0, 1.0, max(2, count))):
        mat = np.eye(4)
        mat[:3, 3] = start_m[:3, 3] + (target_m[:3, 3] - start_m[:3, 3]) * t
        # Keep orientation interpolation conservative for now: snap to target
        # only at the final waypoint, otherwise keep start orientation.
        mat[:3, :3] = target_m[:3, :3] if i == max(2, count) - 1 else start_m[:3, :3]
        frames.append(to_transform(start.parent_frame, target.child_frame if i == max(2, count) - 1 else start.child_frame, mat))
    return frames


def failed_result(motion_type: MotionType, request: MotionRequest, stage: str, reason: MotionRejectionReason, waypoint_index=None, debug=None):
    from robot_engine.motion.motion_result import MotionResult

    return MotionResult(
        success=False,
        motion_type=motion_type,
        start_frame=request.start_frame,
        target_frame=request.target_frame,
        failed_stage=stage,
        failed_waypoint_index=waypoint_index,
        rejection_reason=reason,
        debug_info=debug or {},
    )
