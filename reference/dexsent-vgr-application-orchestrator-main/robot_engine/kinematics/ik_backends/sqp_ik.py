from __future__ import annotations

import time

import numpy as np

from robot_engine.interfaces.schemas import AlgorithmError, IKRequest, IKResult
from robot_engine.kinematics.ik_backends.base import IKBackend
from robot_engine.kinematics.jacobian_solver import compute_jacobian
from robot_engine.kinematics.kinematic_chain import KinematicChain
from robot_engine.kinematics.singularity import condition_number
from robot_engine.interfaces.schemas import JacobianRequest
from robot_engine.math_utils import as_matrix, pose_error


class SQPIKBackend(IKBackend):
    backend_name = "SQP"

    def solve(self, request: IKRequest) -> IKResult:
        try:
            from scipy.optimize import minimize
        except Exception as exc:
            return IKResult(ok=False, reason="IK_BACKEND_UNAVAILABLE", backend_used=self.backend_name, error=AlgorithmError(code="IK_BACKEND_UNAVAILABLE", message=str(exc)))

        started = time.monotonic()
        try:
            chain = KinematicChain(request.chain)
            names = chain.joint_names
            target = as_matrix(request.target)
            lower = np.asarray([joint.lower for joint in chain.movable_joints], dtype=float)
            upper = np.asarray([joint.upper for joint in chain.movable_joints], dtype=float)
            q_seed = np.asarray([request.seed.get(name, 0.0) for name in names], dtype=float)
            q_seed = np.clip(q_seed, lower, upper)
            preferred = np.asarray([request.preferred_posture.get(name, q_seed[i]) for i, name in enumerate(names)], dtype=float)

            def residual_parts(qv):
                q = {name: float(v) for name, v in zip(names, qv)}
                current = chain.forward_matrices(q).transforms[chain.tcp_frame]
                err = pose_error(current, target)
                pos = err[:3]
                rot = err[3:]
                if request.mode == "position":
                    rot = np.zeros(3)
                elif request.mode == "orientation":
                    pos = np.zeros(3)
                return pos, rot

            def objective(qv):
                pos, rot = residual_parts(qv)
                cost = request.position_weight * float(pos @ pos)
                cost += request.orientation_weight * float(rot @ rot)
                if request.minimum_joint_motion_weight:
                    cost += request.minimum_joint_motion_weight * float(np.sum((qv - q_seed) ** 2))
                if request.preferred_posture_weight:
                    cost += request.preferred_posture_weight * float(np.sum((qv - preferred) ** 2))
                if request.joint_limit_avoidance_weight:
                    margin = np.minimum(qv - lower, upper - qv)
                    cost += request.joint_limit_avoidance_weight * float(np.sum(1.0 / np.maximum(margin, 1e-6) ** 2))
                if _callback_rejects(request.collision_callback, qv, names):
                    cost += 1e6
                return cost

            constraints = []
            if request.collision_callback is not None:
                constraints.append({"type": "ineq", "fun": lambda qv: 1.0 if not _callback_rejects(request.collision_callback, qv, names) else -1.0})

            options = {"maxiter": request.max_iterations, "ftol": 1e-12, "disp": False}
            if request.timeout is not None:
                options["maxiter"] = max(1, request.max_iterations)
            result = minimize(objective, q_seed, method="SLSQP", bounds=list(zip(lower, upper)), constraints=constraints, options=options)
            elapsed = time.monotonic() - started
            qv = np.clip(result.x, lower, upper)
            q = {name: float(v) for name, v in zip(names, qv)}
            pos, rot = residual_parts(qv)
            pos_norm = float(np.linalg.norm(pos))
            rot_norm = float(np.linalg.norm(rot))
            total = float(np.sqrt(request.position_weight * pos_norm**2 + request.orientation_weight * rot_norm**2))

            jac = compute_jacobian(JacobianRequest(chain=request.chain, joint_positions=q, frame_id=chain.tcp_frame))
            cond = jac.condition_number if jac.ok else None
            if cond is not None and np.isfinite(cond) and cond > request.singularity_threshold:
                return _failure(request, q, result, pos_norm, rot_norm, total, "SINGULARITY_RISK", "SQP IK solution is near a singularity.", cond, elapsed)
            if _callback_rejects(request.collision_callback, qv, names):
                return _failure(request, q, result, pos_norm, rot_norm, total, "COLLISION_DETECTED", "SQP IK solution rejected by collision callback.", cond, elapsed, collision_status=True)

            pos_ok = request.mode == "orientation" or pos_norm <= request.position_tolerance
            rot_ok = request.mode == "position" or rot_norm <= request.orientation_tolerance
            if result.success and pos_ok and rot_ok:
                return IKResult(
                    ok=True,
                    joint_positions=q,
                    all_candidate_solutions=[q] if request.return_all_solutions else [],
                    best_solution=q,
                    iterations=int(result.nit),
                    position_error=pos_norm,
                    orientation_error=rot_norm,
                    residual_total=total,
                    backend_used=self.backend_name,
                    singularity_metric=cond,
                    collision_status=False if request.collision_callback is not None else None,
                    reason="IK_CONVERGED",
                    debug_info={"objective": float(result.fun), "optimizer_success": bool(result.success), "elapsed": elapsed},
                )

            code = "COLLISION_DETECTED" if request.collision_callback is not None else ("TOLERANCE_NOT_MET" if result.success else "IK_FAILED")
            return _failure(request, q, result, pos_norm, rot_norm, total, code, "SQP IK did not satisfy requested tolerances.", cond, elapsed)
        except Exception as exc:
            return IKResult(ok=False, reason="IK_FAILED", backend_used=self.backend_name, error=AlgorithmError(code="IK_FAILED", message=str(exc)))


def _callback_rejects(callback, qv, names) -> bool:
    if callback is None:
        return False
    q_dict = {name: float(v) for name, v in zip(names, qv)}
    value = callback(q_dict)
    if hasattr(value, "collision"):
        return bool(value.collision)
    if isinstance(value, dict):
        if "collision" in value:
            return bool(value["collision"])
        if "valid" in value:
            return not bool(value["valid"])
    return bool(value)


def _failure(request, q, result, pos_norm, rot_norm, total, code, message, cond, elapsed, collision_status=None):
    return IKResult(
        ok=False,
        joint_positions=q,
        iterations=int(getattr(result, "nit", 0)),
        position_error=pos_norm,
        orientation_error=rot_norm,
        residual_total=total,
        backend_used="SQP",
        singularity_metric=cond,
        collision_status=collision_status,
        reason=code,
        error=AlgorithmError(code=code, message=message, details={"optimizer_message": str(getattr(result, "message", "")), "objective": float(getattr(result, "fun", np.nan)), "elapsed": elapsed}),
    )
