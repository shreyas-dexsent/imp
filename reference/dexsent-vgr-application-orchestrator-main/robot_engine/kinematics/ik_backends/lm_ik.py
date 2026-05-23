from __future__ import annotations

import numpy as np

from robot_engine.interfaces.schemas import AlgorithmError, IKRequest, IKResult, JacobianRequest
from robot_engine.kinematics.ik_backends.base import IKBackend
from robot_engine.kinematics.kinematic_chain import KinematicChain
from robot_engine.kinematics.jacobian_solver import compute_jacobian
from robot_engine.math_utils import as_matrix, pose_error


class LMIKBackend(IKBackend):
    backend_name = "LM"

    def solve(self, request: IKRequest) -> IKResult:
        try:
            from scipy.optimize import least_squares
        except Exception as exc:
            return IKResult(ok=False, reason="IK_BACKEND_UNAVAILABLE", error=AlgorithmError(code="IK_BACKEND_UNAVAILABLE", message=str(exc)))

        chain = KinematicChain(request.chain)
        names = chain.joint_names
        target = as_matrix(request.target)
        q0 = np.asarray([request.seed.get(name, 0.0) for name in names], dtype=float)
        lower = np.asarray([j.lower for j in chain.movable_joints], dtype=float)
        upper = np.asarray([j.upper for j in chain.movable_joints], dtype=float)

        def residual(qv):
            q = {name: float(v) for name, v in zip(names, qv)}
            current = chain.forward_matrices(q).transforms[chain.tcp_frame]
            r = pose_error(current, target)
            return np.r_[r[:3] / max(request.position_tolerance, 1e-9), r[3:] / max(request.orientation_tolerance, 1e-9)]

        result = least_squares(residual, np.clip(q0, lower, upper), bounds=(lower, upper), max_nfev=request.max_iterations)
        q = {name: float(v) for name, v in zip(names, result.x)}
        err = pose_error(chain.forward_matrices(q).transforms[chain.tcp_frame], target)
        pos = float(np.linalg.norm(err[:3]))
        rot = float(np.linalg.norm(err[3:]))
        ok = pos <= request.position_tolerance and rot <= request.orientation_tolerance
        reason = "IK_CONVERGED" if ok else "IK_FAILED"
        return IKResult(
            ok=ok,
            joint_positions=q,
            iterations=int(result.nfev),
            position_error=pos,
            orientation_error=rot,
            reason=reason,
            error=None if ok else AlgorithmError(code=reason, message="LM IK did not converge.", details={"backend_used": self.backend_name, "cost": result.cost}),
        )

