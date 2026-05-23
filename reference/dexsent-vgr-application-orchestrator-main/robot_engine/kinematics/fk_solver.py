from __future__ import annotations

from robot_engine.interfaces.schemas import AlgorithmError, FKRequest, FKResult
from robot_engine.kinematics.kinematic_chain import KinematicChain


def compute_fk(request: FKRequest) -> FKResult:
    try:
        chain = KinematicChain(request.chain)
        bad = chain.violates_limits(request.joint_positions)
        if bad:
            return FKResult(ok=False, error=AlgorithmError(code="JOINT_LIMIT_VIOLATION", message="Joint limits violated.", details={"joints": bad}))
        transforms = chain.forward_transforms(request.joint_positions)
        if request.target_frame:
            transforms = {request.target_frame: transforms[request.target_frame]}
        return FKResult(ok=True, transforms=transforms)
    except Exception as exc:
        return FKResult(ok=False, error=AlgorithmError(code="MODEL_LOAD_FAILED", message=str(exc)))
