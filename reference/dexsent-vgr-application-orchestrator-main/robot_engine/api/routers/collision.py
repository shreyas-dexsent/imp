from __future__ import annotations

from typing import List, Optional

from fastapi import APIRouter
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel

from robot_engine.api.context_store import ContextNotFound, get_store
from robot_engine.api.errors import handle_context_not_found
from robot_engine.collision.continuous_collision import (
    conservative_continuous_collision,
    conservative_continuous_path_collision,
)
from robot_engine.interfaces.schemas import (
    CollisionCheckRequest,
    CollisionMatrix,
    CollisionObjectConfig,
    MinimumDistanceRequest,
)

router = APIRouter(prefix="/contexts/{context_id}/collision", tags=["collision"])


def _get_ctx(context_id: str):
    store = get_store()
    try:
        return store.get(context_id)
    except ContextNotFound as exc:
        raise handle_context_not_found(exc)


class BuildWorldRequest(BaseModel):
    objects: List[CollisionObjectConfig]
    matrix: Optional[CollisionMatrix] = None


@router.post("/world")
async def build_world(context_id: str, req: BuildWorldRequest):
    ctx = _get_ctx(context_id)

    def _build():
        world, geom_statuses, errors = ctx.build_collision_world_status(req.objects, req.matrix)
        return {
            "ok": not errors,
            "object_ids": sorted(world.objects),
            "active_pairs": [[a, b] for a, b in world.active_pairs()],
            "geometry": [s.model_dump() for s in geom_statuses],
            "errors": [e.model_dump() for e in errors],
        }

    return await run_in_threadpool(_build)


@router.post("/matrix")
async def set_matrix(context_id: str, matrix: CollisionMatrix):
    ctx = _get_ctx(context_id)
    error = ctx.set_collision_matrix(matrix)
    if error:
        return {"ok": False, "error": error.model_dump()}
    return {"ok": True}


@router.post("/check")
async def check_collision(context_id: str, req: CollisionCheckRequest):
    ctx = _get_ctx(context_id)
    result = await run_in_threadpool(ctx.check_collisions, req)
    return result.model_dump()


@router.post("/distance")
async def distance(context_id: str, req: MinimumDistanceRequest):
    ctx = _get_ctx(context_id)
    results = await run_in_threadpool(ctx.query_minimum_distances, req)
    return {"results": [r.model_dump() for r in results]}


class ContinuousSegmentRequest(BaseModel):
    q0: List[float]
    q1: List[float]
    resolution: float = 0.05


@router.post("/continuous-segment")
async def continuous_segment(context_id: str, req: ContinuousSegmentRequest):
    ctx = _get_ctx(context_id)

    def _check():
        result = conservative_continuous_collision(req.q0, req.q1, world=ctx.world, resolution=req.resolution)
        return {
            "success": result.success,
            "collision": result.collision,
            "first_collision_waypoint": result.first_collision_waypoint,
            "minimum_clearance": result.minimum_clearance,
            "interpolation_samples": result.interpolation_samples,
            "rejection_reason": result.rejection_reason,
        }

    return await run_in_threadpool(_check)


class ContinuousPathRequest(BaseModel):
    q_waypoints: List[List[float]]
    resolution: float = 0.05


@router.post("/continuous-path")
async def continuous_path(context_id: str, req: ContinuousPathRequest):
    ctx = _get_ctx(context_id)

    def _check():
        result = conservative_continuous_path_collision(req.q_waypoints, world=ctx.world, resolution=req.resolution)
        return {
            "success": result.success,
            "collision": result.collision,
            "first_collision_waypoint": result.first_collision_waypoint,
            "first_collision_segment": result.first_collision_segment,
            "minimum_clearance": result.minimum_clearance,
            "interpolation_samples": result.interpolation_samples,
            "rejection_reason": result.rejection_reason,
        }

    return await run_in_threadpool(_check)
