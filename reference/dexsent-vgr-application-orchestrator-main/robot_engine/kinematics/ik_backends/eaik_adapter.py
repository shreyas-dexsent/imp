from __future__ import annotations

import importlib

import numpy as np

from robot_engine.interfaces.schemas import AlgorithmError, IKRequest, IKResult
from robot_engine.kinematics.ik_backends.base import IKBackend
from robot_engine.kinematics.jacobian_solver import compute_jacobian
from robot_engine.kinematics.kinematic_chain import KinematicChain
from robot_engine.kinematics.singularity import condition_number
from robot_engine.interfaces.schemas import JacobianRequest
from robot_engine.math_utils import as_matrix, pose_error


class EAIKAdapterBackend(IKBackend):
    backend_name = "EAIK"

    def solve(self, request: IKRequest) -> IKResult:
        try:
            module = importlib.import_module("eaik")
        except Exception as exc:
            return IKResult(ok=False, reason="IK_BACKEND_UNAVAILABLE", backend_used=self.backend_name, error=AlgorithmError(code="IK_BACKEND_UNAVAILABLE", message=f"EAIK backend unavailable: {exc}"))

        solver = getattr(module, "solve_ik", None) or getattr(module, "solve", None)
        if solver is None:
            solver_cls = getattr(module, "EAIK", None)
            if solver_cls is not None:
                try:
                    solver_obj = solver_cls()
                    solver = getattr(solver_obj, "solve_ik", None) or getattr(solver_obj, "solve", None)
                except Exception as exc:
                    return _unavailable(f"EAIK object could not be constructed: {exc}")
        if solver is None:
            return _unavailable("EAIK package is importable, but no supported solve_ik/solve API was found.")

        try:
            raw = _call_solver(solver, request)
            candidates = _normalize_solutions(raw, request)
            return _filter_rank_candidates(candidates, request)
        except Exception as exc:
            return IKResult(ok=False, reason="IK_BACKEND_UNAVAILABLE", backend_used=self.backend_name, error=AlgorithmError(code="IK_BACKEND_UNAVAILABLE", message=f"EAIK adapter failed safely: {exc}"))


def _unavailable(message: str) -> IKResult:
    return IKResult(ok=False, reason="IK_BACKEND_UNAVAILABLE", backend_used="EAIK", error=AlgorithmError(code="IK_BACKEND_UNAVAILABLE", message=message))


def _call_solver(solver, request: IKRequest):
    try:
        return solver(request)
    except TypeError:
        try:
            return solver(target=as_matrix(request.target), seed=request.seed, chain=request.chain)
        except TypeError:
            return solver(as_matrix(request.target), request.seed)


def _normalize_solutions(raw, request: IKRequest):
    if raw is None:
        return []
    if hasattr(raw, "solutions"):
        raw = raw.solutions
    if isinstance(raw, dict) and "solutions" in raw:
        raw = raw["solutions"]
    if isinstance(raw, dict):
        raw = [raw]
    chain = KinematicChain(request.chain)
    names = chain.joint_names
    out = []
    for item in raw:
        if isinstance(item, dict):
            out.append({name: float(item.get(name, request.seed.get(name, 0.0))) for name in names})
        else:
            values = np.asarray(item, dtype=float).reshape(-1)
            if values.size >= len(names):
                out.append({name: float(values[i]) for i, name in enumerate(names)})
    return out


def _filter_rank_candidates(candidates, request: IKRequest) -> IKResult:
    chain = KinematicChain(request.chain)
    target = as_matrix(request.target)
    valid = []
    rejections = []
    for q in candidates:
        bad_limits = chain.violates_limits(q)
        if bad_limits:
            rejections.append({"reason": "JOINT_LIMIT_VIOLATION", "joints": bad_limits})
            continue
        current = chain.forward_matrices(q).transforms[chain.tcp_frame]
        err = pose_error(current, target)
        pos = float(np.linalg.norm(err[:3]))
        rot = float(np.linalg.norm(err[3:]))
        pos_ok = request.mode == "orientation" or pos <= request.position_tolerance
        rot_ok = request.mode == "position" or rot <= request.orientation_tolerance
        if not (pos_ok and rot_ok):
            rejections.append({"reason": "TOLERANCE_NOT_MET", "position_error": pos, "orientation_error": rot})
            continue
        if request.collision_callback is not None and _collision_rejects(request.collision_callback, q):
            rejections.append({"reason": "COLLISION_DETECTED"})
            continue
        jac = compute_jacobian(JacobianRequest(chain=request.chain, joint_positions=q, frame_id=chain.tcp_frame))
        cond = jac.condition_number if jac.ok else None
        if cond is not None and cond > request.singularity_threshold:
            rejections.append({"reason": "SINGULARITY_RISK", "condition_number": cond})
            continue
        valid.append((q, pos, rot, cond))
    if not valid:
        return IKResult(ok=False, reason="IK_FAILED", backend_used="EAIK", error=AlgorithmError(code="IK_FAILED", message="EAIK produced no valid solution after filtering.", details={"candidate_count": len(candidates), "rejections": rejections}))
    seed_v = np.asarray([request.seed.get(name, 0.0) for name in chain.joint_names], dtype=float)
    valid.sort(key=lambda item: float(np.linalg.norm(np.asarray([item[0][name] for name in chain.joint_names]) - seed_v)))
    best, pos, rot, cond = valid[0]
    return IKResult(
        ok=True,
        joint_positions=best,
        all_candidate_solutions=[item[0] for item in valid] if request.return_all_solutions else [],
        best_solution=best,
        position_error=pos,
        orientation_error=rot,
        residual_total=float(np.hypot(pos, rot)),
        backend_used="EAIK",
        singularity_metric=cond,
        collision_status=False if request.collision_callback is not None else None,
        reason="IK_CONVERGED",
        debug_info={"valid_candidates": len(valid), "rejections": rejections},
    )


def _collision_rejects(callback, q):
    value = callback(q)
    if hasattr(value, "collision"):
        return bool(value.collision)
    if isinstance(value, dict):
        if "collision" in value:
            return bool(value["collision"])
        if "valid" in value:
            return not bool(value["valid"])
    return bool(value)
