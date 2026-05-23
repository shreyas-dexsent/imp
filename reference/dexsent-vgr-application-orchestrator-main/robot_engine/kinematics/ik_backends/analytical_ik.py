from __future__ import annotations

from robot_engine.interfaces.schemas import AlgorithmError, IKRequest, IKResult
from robot_engine.kinematics.ik_backends.base import IKBackend


class AnalyticalIKBackend(IKBackend):
    backend_name = "ANALYTICAL"

    def solve(self, request: IKRequest) -> IKResult:
        return IKResult(ok=False, reason="IK_BACKEND_UNAVAILABLE", error=AlgorithmError(code="IK_BACKEND_UNAVAILABLE", message="No robot-specific analytical IK solver is registered for this model."))

