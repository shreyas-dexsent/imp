from __future__ import annotations

from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel

from robot_engine.api.context_store import ContextNotFound, get_store
from robot_engine.api.errors import handle_context_not_found
from robot_engine.interfaces.schemas import UISceneRequest

router = APIRouter(prefix="/contexts", tags=["contexts"])


class ContextCreatedResponse(BaseModel):
    context_id: str


class ContextStatusResponse(BaseModel):
    context_id: str
    has_world: bool
    robot_ids: List[str]
    object_ids: List[str]


@router.post("", response_model=ContextCreatedResponse, status_code=201)
async def create_context():
    store = get_store()
    context_id = store.create()
    return ContextCreatedResponse(context_id=context_id)


@router.get("/{context_id}", response_model=ContextStatusResponse)
async def get_context(context_id: str):
    store = get_store()
    try:
        ctx = store.get(context_id)
    except ContextNotFound as exc:
        raise handle_context_not_found(exc)
    return ContextStatusResponse(
        context_id=context_id,
        has_world=ctx.world is not None,
        robot_ids=sorted(ctx.robot_models),
        object_ids=sorted(ctx.world.objects) if ctx.world else [],
    )


@router.delete("/{context_id}", status_code=204)
async def delete_context(context_id: str):
    store = get_store()
    try:
        store.get(context_id)
    except ContextNotFound as exc:
        raise handle_context_not_found(exc)
    store.delete(context_id)


@router.post("/{context_id}/scene")
async def load_scene(context_id: str, request: UISceneRequest):
    store = get_store()
    try:
        ctx = store.get(context_id)
    except ContextNotFound as exc:
        raise handle_context_not_found(exc)
    result = await run_in_threadpool(ctx.load_scene_from_ui, request)
    return result.model_dump()
