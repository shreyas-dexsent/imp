from __future__ import annotations

from fastapi import APIRouter
from fastapi.concurrency import run_in_threadpool

from robot_engine.api.context_store import ContextNotFound, get_store
from robot_engine.api.errors import handle_context_not_found
from robot_engine.frames.frame_graph import FrameGraph
from robot_engine.interfaces.schemas import Transform3D

router = APIRouter(prefix="/contexts/{context_id}/frames", tags=["frames"])


def _get_graph(ctx) -> FrameGraph:
    if not hasattr(ctx, "_frame_graph"):
        ctx._frame_graph = FrameGraph()
    return ctx._frame_graph


def _ok(**kw):
    return {"success": True, "error_code": "OK", "error_message": "", **kw}


def _fail(code: str, msg: str):
    return {"success": False, "error_code": code, "error_message": msg}


@router.post("")
async def add_frame(context_id: str, transform: Transform3D):
    store = get_store()
    try:
        ctx = store.get(context_id)
    except ContextNotFound as exc:
        raise handle_context_not_found(exc)

    graph = _get_graph(ctx)
    try:
        graph.add_frame(
            transform.child_frame,
            parent_frame_id=transform.parent_frame,
            transform=transform,
        )
        return _ok(child_frame=transform.child_frame, parent_frame=transform.parent_frame)
    except Exception as exc:
        return _fail("INVALID_TRANSFORM", str(exc))


@router.put("/{child_frame}")
async def update_frame(context_id: str, child_frame: str, transform: Transform3D):
    store = get_store()
    try:
        ctx = store.get(context_id)
    except ContextNotFound as exc:
        raise handle_context_not_found(exc)

    graph = _get_graph(ctx)
    try:
        graph.update_transform(transform.parent_frame, child_frame, transform)
        return _ok(child_frame=child_frame)
    except Exception as exc:
        return _fail("INVALID_TRANSFORM", str(exc))


@router.get("/{parent_frame}/{child_frame}")
async def get_transform(context_id: str, parent_frame: str, child_frame: str):
    store = get_store()
    try:
        ctx = store.get(context_id)
    except ContextNotFound as exc:
        raise handle_context_not_found(exc)

    graph = _get_graph(ctx)
    try:
        result = graph.get_transform(parent_frame, child_frame)
        return _ok(transform=result.model_dump())
    except KeyError as exc:
        return _fail("FRAME_NOT_FOUND", str(exc))
    except Exception as exc:
        return _fail("INVALID_FRAME_CHAIN", str(exc))


@router.get("")
async def list_frames(context_id: str):
    store = get_store()
    try:
        ctx = store.get(context_id)
    except ContextNotFound as exc:
        raise handle_context_not_found(exc)

    graph = _get_graph(ctx)
    return _ok(frames=graph.list_frames())
