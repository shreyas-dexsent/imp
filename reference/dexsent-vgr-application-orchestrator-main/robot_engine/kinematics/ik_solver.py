from __future__ import annotations

import numpy as np

from robot_engine.interfaces.schemas import AlgorithmError, IKRequest, IKResult, JacobianRequest
from robot_engine.kinematics.jacobian_solver import compute_jacobian
from robot_engine.kinematics.kinematic_chain import KinematicChain
from robot_engine.math_utils import as_matrix, pose_error


class IKBackendRegistry:
    def __init__(self) -> None:
        self._backends = {}

    def register_backend(self, name: str, backend) -> None:
        self._backends[name.upper()] = backend

    def get(self, name: str):
        return self._backends.get(name.upper())

    def names(self) -> list[str]:
        return sorted(self._backends)


_registry = IKBackendRegistry()


def default_registry() -> IKBackendRegistry:
    if not _registry.names():
        from robot_engine.kinematics.ik_backends.analytical_ik import AnalyticalIKBackend
        from robot_engine.kinematics.ik_backends.dls_ik import DLSIKBackend
        from robot_engine.kinematics.ik_backends.eaik_adapter import EAIKAdapterBackend
        from robot_engine.kinematics.ik_backends.lm_ik import LMIKBackend
        from robot_engine.kinematics.ik_backends.optimization_ik import OptimizationIKBackend
        from robot_engine.kinematics.ik_backends.pinocchio_ik import PinocchioIKBackend
        from robot_engine.kinematics.ik_backends.sqp_ik import SQPIKBackend

        for backend in [DLSIKBackend(), LMIKBackend(), OptimizationIKBackend(), SQPIKBackend(), AnalyticalIKBackend(), EAIKAdapterBackend(), PinocchioIKBackend()]:
            _registry.register_backend(backend.backend_name, backend)
    return _registry


def solve_ik_with_backend(request: IKRequest, backend: str = "auto") -> IKResult:
    if backend.lower() in ("auto", "dls"):
        return solve_ik(request)
    candidate = default_registry().get(backend)
    if candidate is None:
        return IKResult(ok=False, reason="UNSUPPORTED_IK_BACKEND", error=AlgorithmError(code="UNSUPPORTED_IK_BACKEND", message=f"Unknown IK backend: {backend}"))
    return candidate.solve(request)


def solve_ik_multi_seed(request: IKRequest, seeds, backend: str = "auto") -> IKResult:
    best = None
    best_norm = float("inf")
    for seed in seeds:
        trial = request.model_copy(update={"seed": dict(seed)})
        result = solve_ik_with_backend(trial, backend)
        if result.ok:
            q = np.asarray(list(result.joint_positions.values()), dtype=float)
            q0 = np.asarray([seed.get(name, 0.0) for name in result.joint_positions], dtype=float)
            norm = float(np.linalg.norm(q - q0))
            if norm < best_norm:
                best = result
                best_norm = norm
    if best is not None:
        return best
    return IKResult(ok=False, reason="IK_FAILED", error=AlgorithmError(code="IK_FAILED", message="No IK seed converged.", details={"seed_count": len(list(seeds))}))


def rank_solutions(solutions, q_current, ranking_options=None):
    q0 = np.asarray(q_current, dtype=float)
    return sorted(solutions, key=lambda q: float(np.linalg.norm(np.asarray(q, dtype=float) - q0)))


def solve_ik(request: IKRequest) -> IKResult:
    try:
        chain = KinematicChain(request.chain)
        target = as_matrix(request.target)
        q = chain.clamp({name: float(request.seed.get(name, 0.0)) for name in chain.joint_names})
        reason = "MAX_ITERATIONS"
        pos_err = None
        rot_err = None

        for iteration in range(1, request.max_iterations + 1):
            current = chain.forward_matrices(q).transforms[chain.tcp_frame]
            err = pose_error(current, target)
            pos_err = float(np.linalg.norm(err[:3]))
            rot_err = float(np.linalg.norm(err[3:]))
            if pos_err <= request.position_tolerance and rot_err <= request.orientation_tolerance:
                return IKResult(ok=True, joint_positions=q, iterations=iteration, position_error=pos_err, orientation_error=rot_err, reason="IK_CONVERGED")

            jac_result = compute_jacobian(JacobianRequest(chain=request.chain, joint_positions=q, frame_id=chain.tcp_frame))
            if not jac_result.ok:
                return IKResult(ok=False, joint_positions=q, iterations=iteration, reason=jac_result.error.code, error=jac_result.error)
            jac = np.asarray(jac_result.jacobian, dtype=float)
            if jac_result.condition_number and jac_result.condition_number > request.singularity_threshold:
                reason = "SINGULAR_JACOBIAN"
                break
            lhs = jac @ jac.T + (request.damping ** 2) * np.eye(6)
            step = jac.T @ np.linalg.solve(lhs, err)
            for i, name in enumerate(chain.joint_names):
                q[name] += float(step[i])
            bad = chain.violates_limits(q)
            if bad:
                q = chain.clamp(q)
                reason = "JOINT_LIMIT_VIOLATION"
                if np.linalg.norm(step) < 1e-10:
                    break
            if np.linalg.norm(step) < 1e-12:
                reason = "IK_UNREACHABLE"
                break

        return IKResult(
            ok=False,
            joint_positions=q,
            iterations=request.max_iterations,
            position_error=pos_err,
            orientation_error=rot_err,
            reason=reason,
            error=AlgorithmError(code=reason, message="IK did not converge.", details={"position_error": pos_err, "orientation_error": rot_err}),
        )
    except Exception as exc:
        return IKResult(ok=False, reason="MODEL_LOAD_FAILED", error=AlgorithmError(code="MODEL_LOAD_FAILED", message=str(exc)))
