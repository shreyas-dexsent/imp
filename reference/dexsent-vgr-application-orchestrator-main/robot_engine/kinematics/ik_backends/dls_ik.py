from __future__ import annotations

from robot_engine.interfaces.schemas import IKRequest, IKResult
from robot_engine.kinematics.ik_backends.base import IKBackend
from robot_engine.kinematics.ik_solver import solve_ik


class DLSIKBackend(IKBackend):
    backend_name = "DLS"

    def solve(self, request: IKRequest) -> IKResult:
        result = solve_ik(request)
        if result.error:
            result.error.details["backend_used"] = self.backend_name
        return result

