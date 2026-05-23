from __future__ import annotations

import shutil
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool

from robot_engine.api.context_store import ContextNotFound, get_store
from robot_engine.api.errors import handle_context_not_found
from robot_engine.interfaces.schemas import GripperConfig, ObjectAssetConfig, RobotModelConfig

router = APIRouter(prefix="/contexts/{context_id}/assets", tags=["assets"])


def _get_ctx(context_id: str):
    store = get_store()
    try:
        return store.get(context_id)
    except ContextNotFound as exc:
        raise handle_context_not_found(exc)


@router.post("/robot")
async def load_robot(context_id: str, config: RobotModelConfig):
    ctx = _get_ctx(context_id)
    robot = await run_in_threadpool(ctx.load_robot_model, config)
    return {
        "ok": robot.error is None,
        "robot_id": config.robot_id,
        "joint_names": robot.get_joint_names(),
        "frame_names": robot.get_frame_names(),
        "error": robot.error.model_dump() if robot.error else None,
    }


@router.post("/gripper")
async def load_gripper(context_id: str, config: GripperConfig):
    ctx = _get_ctx(context_id)
    status = await run_in_threadpool(ctx.load_gripper_model, config)
    return status.model_dump()


@router.post("/collision")
async def load_collision_asset(context_id: str, config: ObjectAssetConfig):
    ctx = _get_ctx(context_id)
    status = await run_in_threadpool(ctx.load_collision_asset_status, config)
    return status.model_dump()


@router.post("/upload")
async def upload_asset(
    context_id: str,
    file: UploadFile = File(...),
    object_id: str = Form(...),
    frame_id: str = Form(...),
    scale: float = Form(1.0),
):
    store = get_store()
    try:
        ctx = store.get(context_id)
        tmp_dir = store.tmp_dir(context_id)
    except ContextNotFound as exc:
        raise handle_context_not_found(exc)

    dest = tmp_dir / file.filename
    with dest.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    config = ObjectAssetConfig(object_id=object_id, mesh_path=str(dest), frame_id=frame_id, scale=scale)
    status = await run_in_threadpool(ctx.load_collision_asset_status, config)
    return {**status.model_dump(), "upload_path": str(dest)}
