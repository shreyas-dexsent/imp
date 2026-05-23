from __future__ import annotations

from fastapi import APIRouter
from fastapi.concurrency import run_in_threadpool

from robot_engine.api.context_store import ContextNotFound, get_store
from robot_engine.api.errors import handle_context_not_found
from robot_engine.interfaces.schemas import FKRequest, IKRequest, JacobianRequest
from robot_engine.kinematics.fk_solver import compute_fk
from robot_engine.kinematics.ik_solver import solve_ik, solve_ik_with_backend
from robot_engine.kinematics.jacobian_solver import compute_jacobian

router = APIRouter(tags=["kinematics"])

# --- stateless endpoints ---

@router.post("/kinematics/fk")
async def fk(req: FKRequest):
    result = await run_in_threadpool(compute_fk, req)
    return result.model_dump()


@router.post("/kinematics/jacobian")
async def jacobian(req: JacobianRequest):
    result = await run_in_threadpool(compute_jacobian, req)
    return result.model_dump()


@router.post("/kinematics/ik")
async def ik(req: IKRequest):
    result = await run_in_threadpool(solve_ik, req)
    return result.model_dump()


@router.post("/kinematics/ik/{backend}")
async def ik_backend(backend: str, req: IKRequest):
    result = await run_in_threadpool(solve_ik_with_backend, req, backend)
    return result.model_dump()


# --- context-backed endpoints ---

def _get_ctx(context_id: str):
    store = get_store()
    try:
        return store.get(context_id)
    except ContextNotFound as exc:
        raise handle_context_not_found(exc)


@router.post("/contexts/{context_id}/kinematics/fk")
async def ctx_fk(context_id: str, req: FKRequest):
    _get_ctx(context_id)  # validate context exists
    result = await run_in_threadpool(compute_fk, req)
    return result.model_dump()


@router.post("/contexts/{context_id}/kinematics/jacobian")
async def ctx_jacobian(context_id: str, req: JacobianRequest):
    _get_ctx(context_id)
    result = await run_in_threadpool(compute_jacobian, req)
    return result.model_dump()


@router.post("/contexts/{context_id}/kinematics/ik")
async def ctx_ik(context_id: str, req: IKRequest):
    _get_ctx(context_id)
    result = await run_in_threadpool(solve_ik, req)
    return result.model_dump()
