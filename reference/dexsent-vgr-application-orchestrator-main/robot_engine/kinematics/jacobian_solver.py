from __future__ import annotations

import numpy as np

from robot_engine.interfaces.schemas import AlgorithmError, JacobianRequest, JacobianResult
from robot_engine.kinematics.kinematic_chain import KinematicChain
from robot_engine.math_utils import pose_error


def compute_jacobian(request: JacobianRequest, eps: float = 1e-6) -> JacobianResult:
    try:
        chain = KinematicChain(request.chain)
        frame = request.frame_id or chain.tcp_frame
        q = {name: float(request.joint_positions.get(name, 0.0)) for name in chain.joint_names}
        bad = chain.violates_limits(q)
        if bad:
            return JacobianResult(ok=False, frame_id=frame, error=AlgorithmError(code="JOINT_LIMIT_VIOLATION", message="Joint limits violated.", details={"joints": bad}))
        base = chain.forward_matrices(q).transforms[frame]
        jac = np.zeros((6, len(chain.joint_names)))
        for col, name in enumerate(chain.joint_names):
            q2 = dict(q)
            q2[name] += eps
            moved = chain.forward_matrices(q2).transforms[frame]
            jac[:, col] = pose_error(base, moved) / eps
        cond = float(np.linalg.cond(jac)) if jac.size else None
        return JacobianResult(ok=True, frame_id=frame, jacobian=jac.tolist(), condition_number=cond)
    except Exception as exc:
        return JacobianResult(ok=False, frame_id=request.frame_id or request.chain.tip_frame, error=AlgorithmError(code="MODEL_LOAD_FAILED", message=str(exc)))
