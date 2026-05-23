from __future__ import annotations

import numpy as np

from robot_engine.interfaces.schemas import AlgorithmError, IKRequest, IKResult
from robot_engine.kinematics.ik_backends.base import IKBackend
from robot_engine.math_utils import as_matrix


class PinocchioIKBackend(IKBackend):
    backend_name = "PINOCCHIO_DLS"

    def solve(self, request: IKRequest) -> IKResult:
        robot = request.robot_model
        model = getattr(robot, "pin_model", robot)
        if model is None:
            return IKResult(ok=False, reason="IK_BACKEND_UNAVAILABLE", backend_used=self.backend_name, error=AlgorithmError(code="IK_BACKEND_UNAVAILABLE", message="Pinocchio IK requires a loaded Pinocchio model on IKRequest.robot_model."))
        try:
            import pinocchio as pin
        except Exception as exc:
            return IKResult(ok=False, reason="IK_BACKEND_UNAVAILABLE", backend_used=self.backend_name, error=AlgorithmError(code="IK_BACKEND_UNAVAILABLE", message=f"Pinocchio import failed: {exc}"))

        try:
            data = model.createData()
            target = as_matrix(request.target)
            target_se3 = pin.SE3(target[:3, :3], target[:3, 3])
            frame_name = request.target.child_frame
            frame_id = model.getFrameId(frame_name)
            if frame_id >= len(model.frames):
                return IKResult(ok=False, reason="FRAME_NOT_FOUND", backend_used=self.backend_name, error=AlgorithmError(code="FRAME_NOT_FOUND", message=f"Pinocchio frame not found: {frame_name}"))
            names = list(model.names[1:])
            q = np.zeros(model.nq)
            neutral = pin.neutral(model)
            q[: len(neutral)] = neutral
            for i, name in enumerate(names[: model.nq]):
                if name in request.seed:
                    q[i] = request.seed[name]
            q = np.minimum(np.maximum(q, model.lowerPositionLimit), model.upperPositionLimit)

            pos_err = None
            rot_err = None
            for iteration in range(1, request.max_iterations + 1):
                pin.forwardKinematics(model, data, q)
                pin.updateFramePlacements(model, data)
                current = data.oMf[frame_id]
                err6 = pin.log(current.inverse() * target_se3).vector
                pos_err = float(np.linalg.norm(err6[:3]))
                rot_err = float(np.linalg.norm(err6[3:]))
                if (request.mode == "orientation" or pos_err <= request.position_tolerance) and (request.mode == "position" or rot_err <= request.orientation_tolerance):
                    q_dict = {name: float(q[i]) for i, name in enumerate(names[: model.nq])}
                    return IKResult(ok=True, joint_positions=q_dict, best_solution=q_dict, iterations=iteration, position_error=pos_err, orientation_error=rot_err, residual_total=float(np.linalg.norm(err6)), backend_used=self.backend_name, reason="IK_CONVERGED")
                J = pin.computeFrameJacobian(model, data, q, frame_id, pin.ReferenceFrame.LOCAL)
                if request.mode == "position":
                    J_use = J[:3, :]
                    err_use = err6[:3]
                elif request.mode == "orientation":
                    J_use = J[3:, :]
                    err_use = err6[3:]
                else:
                    J_use = J
                    err_use = err6
                lhs = J_use @ J_use.T + request.damping**2 * np.eye(J_use.shape[0])
                dq = J_use.T @ np.linalg.solve(lhs, err_use)
                q = pin.integrate(model, q, dq)
                q = np.minimum(np.maximum(q, model.lowerPositionLimit), model.upperPositionLimit)
            q_dict = {name: float(q[i]) for i, name in enumerate(names[: model.nq])}
            return IKResult(ok=False, joint_positions=q_dict, iterations=request.max_iterations, position_error=pos_err, orientation_error=rot_err, residual_total=None if pos_err is None else float(np.hypot(pos_err, rot_err or 0.0)), backend_used=self.backend_name, reason="IK_FAILED", error=AlgorithmError(code="IK_FAILED", message="Pinocchio DLS IK did not converge."))
        except Exception as exc:
            return IKResult(ok=False, reason="IK_BACKEND_UNAVAILABLE", backend_used=self.backend_name, error=AlgorithmError(code="IK_BACKEND_UNAVAILABLE", message=f"Pinocchio IK failed safely: {exc}"))
