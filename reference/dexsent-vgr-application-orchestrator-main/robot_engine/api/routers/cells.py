from __future__ import annotations

import shutil
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

from robot_engine.api.imp_bus import BusMessage, get_bus
from robot_engine.api.orchestrator_store import (
    GeometryRecord,
    GripperSelection,
    PoseRecord,
    RobotSelection,
    TestCaseRecord,
    check_path_with_world,
    get_orchestrator_store,
    identity_transform,
    offset_tcp_pose,
    plan_joint_path,
    pose_from_tcp_dict,
)
from robot_engine.interfaces.schemas import CollisionMatrix, TCPConfig, Transform3D

router = APIRouter(prefix="/cells", tags=["cells"])


class CellCreateRequest(BaseModel):
    name: str = "Untitled cell"
    cell_type: str = "custom"


class CellUpdateRequest(BaseModel):
    name: Optional[str] = None
    cell_type: Optional[str] = None


class RobotSelectRequest(RobotSelection):
    pass


class GripperSelectRequest(GripperSelection):
    pass


class GripperAttachRequest(BaseModel):
    flange_frame: Transform3D
    tcp_frame: Transform3D


class PoseRecordRequest(BaseModel):
    name: str
    source: str = "current_tcp"
    store: List[str] = Field(default_factory=lambda: ["tcp_pose", "joint_state"])


class RobotStateUpdateRequest(BaseModel):
    joint_state: List[float] = Field(default_factory=list)
    tcp_pose: Dict[str, Any] = Field(default_factory=dict)


class JogRequest(BaseModel):
    frame: str = "tcp"
    axis: str
    direction: str
    distance_m: float
    motion_type: str = "LINEAR"


class MoveJRequest(BaseModel):
    joints: List[float]
    planner_options: Dict[str, Any] = Field(default_factory=dict)


class MoveLRequest(BaseModel):
    tcp_pose: Dict[str, Any]
    planner_options: Dict[str, Any] = Field(default_factory=dict)


class GeometryImportRequest(GeometryRecord):
    pass


class GeometryPoseRequest(BaseModel):
    pose: Transform3D


class CheckPathRequest(BaseModel):
    q_waypoints: List[List[float]]
    resolution: float = 0.05


class PlanToPoseRequest(BaseModel):
    start_pose_id: Optional[str] = None
    target_pose_id: Optional[str] = None
    start_joints: Optional[List[float]] = None
    target_joints: Optional[List[float]] = None
    planner_options: Dict[str, Any] = Field(default_factory=dict)


def _store():
    return get_orchestrator_store()


def _runtime(cell_id: str):
    try:
        return _store().get(cell_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"cell not found: {cell_id}") from exc


async def _publish(cell_id: str, type_: str, payload: Dict[str, Any], source: str = "orchestrator") -> None:
    await get_bus().publish(BusMessage(cell_id=cell_id, type=type_, source=source, payload=payload))


def _cell_payload(runtime) -> Dict[str, Any]:
    return runtime.cell.model_dump()


@router.post("", status_code=201)
async def create_cell(req: CellCreateRequest):
    runtime = _store().create_cell(req.name, req.cell_type)
    await _publish(runtime.cell.cell_id, "cell.created", {"cell": _cell_payload(runtime)})
    return {"cell": _cell_payload(runtime)}


@router.get("")
async def list_cells():
    return {"cells": [cell.model_dump() for cell in _store().list_cells()]}


@router.get("/{cell_id}")
async def get_cell(cell_id: str):
    return {"cell": _cell_payload(_runtime(cell_id))}


@router.put("/{cell_id}")
async def update_cell(cell_id: str, req: CellUpdateRequest):
    runtime = _store().update_cell(cell_id, req.model_dump(exclude_unset=True))
    await _publish(cell_id, "cell.updated", {"cell": _cell_payload(runtime)})
    return {"cell": _cell_payload(runtime)}


@router.post("/{cell_id}/robot/select")
async def select_robot(cell_id: str, req: RobotSelectRequest):
    runtime = _runtime(cell_id)
    runtime.cell.robot = RobotSelection(**req.model_dump())
    runtime.cell.latest_joint_state = runtime.cell.latest_joint_state or [0.0] * max(len(req.joint_names), len(req.lower_limits), len(req.upper_limits), 0)
    runtime.cell.touch()
    await _publish(cell_id, "scene.updated", {"robot": runtime.cell.robot.model_dump()})
    return {"ok": True, "cell": _cell_payload(runtime)}


@router.post("/{cell_id}/gripper/select")
async def select_gripper(cell_id: str, req: GripperSelectRequest):
    runtime = _runtime(cell_id)
    runtime.cell.gripper = GripperSelection(**req.model_dump())
    runtime.cell.touch()
    await _publish(cell_id, "scene.updated", {"gripper": runtime.cell.gripper.model_dump()})
    return {"ok": True, "cell": _cell_payload(runtime)}


@router.post("/{cell_id}/gripper/attach")
async def attach_gripper(cell_id: str, req: GripperAttachRequest):
    runtime = _runtime(cell_id)
    if runtime.cell.gripper is None:
        runtime.cell.gripper = GripperSelection(gripper_id="unselected")
    runtime.cell.gripper.flange_frame = req.flange_frame
    runtime.cell.gripper.tcp_frame = req.tcp_frame
    runtime.cell.tcp = TCPConfig(tcp_id="gripper_tcp", transform=req.tcp_frame)
    runtime.engine.update_tcp(runtime.cell.tcp)
    runtime.cell.touch()
    frame_graph = [
        identity_transform("world", "robot_base").model_dump(),
        identity_transform("robot_base", "robot_flange").model_dump(),
        req.flange_frame.model_dump(),
        req.tcp_frame.model_dump(),
    ]
    await _publish(cell_id, "scene.updated", {"frame_graph": frame_graph, "tcp": runtime.cell.tcp.model_dump()})
    return {"ok": True, "frame_graph": frame_graph, "cell": _cell_payload(runtime)}


@router.post("/{cell_id}/tcp")
async def update_tcp(cell_id: str, req: TCPConfig):
    runtime = _runtime(cell_id)
    runtime.cell.tcp = req
    runtime.engine.update_tcp(req)
    runtime.cell.touch()
    await _publish(cell_id, "scene.updated", {"tcp": req.model_dump()})
    return {"ok": True, "tcp": req.model_dump()}


@router.post("/{cell_id}/objects")
async def add_object(cell_id: str, req: GeometryImportRequest):
    runtime = _runtime(cell_id)
    geom = GeometryRecord(**req.model_dump())
    if geom.pose.child_frame == "geometry":
        geom.pose.child_frame = geom.geometry_id
    runtime.cell.geometries[geom.geometry_id] = geom
    world, geometry_status, errors = await run_in_threadpool(runtime.rebuild_collision_world)
    runtime.cell.touch()
    await _publish(cell_id, "scene.updated", {"geometry": geom.model_dump(), "world_object_ids": sorted(world.objects)})
    return {
        "ok": not errors,
        "geometry": geom.model_dump(),
        "collision_geometry": [s.model_dump() for s in geometry_status],
        "errors": [e.model_dump() for e in errors],
    }


@router.post("/{cell_id}/environment")
async def add_environment(cell_id: str, req: GeometryImportRequest):
    if req.type == "object":
        req.type = "fixture"
    return await add_object(cell_id, req)


@router.post("/{cell_id}/collision-matrix")
async def set_collision_matrix(cell_id: str, req: CollisionMatrix):
    runtime = _runtime(cell_id)
    runtime.cell.collision_matrix = req
    runtime.rebuild_collision_world()
    runtime.cell.touch()
    await _publish(cell_id, "collision.status", {"collision_matrix": req.model_dump()})
    return {"ok": True, "active_pairs": [[a, b] for a, b in runtime.engine.world.active_pairs()] if runtime.engine.world else []}


@router.get("/{cell_id}/robot/state")
async def robot_state(cell_id: str):
    runtime = _runtime(cell_id)
    return {
        "connected": True,
        "mode": "SIMULATED",
        "joint_state": runtime.cell.latest_joint_state,
        "q": runtime.cell.latest_joint_state,
        "tcp_pose": runtime.cell.latest_tcp_pose,
    }


@router.post("/{cell_id}/robot/state")
async def update_robot_state(cell_id: str, req: RobotStateUpdateRequest):
    runtime = _runtime(cell_id)
    if req.joint_state:
        runtime.cell.latest_joint_state = req.joint_state
    if req.tcp_pose:
        runtime.cell.latest_tcp_pose = req.tcp_pose
    runtime.cell.touch()
    await _publish(cell_id, "robot.state", {"joint_state": runtime.cell.latest_joint_state, "tcp_pose": runtime.cell.latest_tcp_pose})
    return {"ok": True, "state": (await robot_state(cell_id))}


@router.post("/{cell_id}/robot/jog")
async def jog_robot(cell_id: str, req: JogRequest):
    runtime = _runtime(cell_id)
    try:
        runtime.cell.latest_tcp_pose = offset_tcp_pose(runtime.cell.latest_tcp_pose, req.axis, req.direction, req.distance_m)
    except ValueError as exc:
        return {"ok": False, "error_code": "INVALID_AXIS" if "axis" in str(exc) else "INVALID_DISTANCE", "error_message": str(exc)}
    runtime.cell.touch()
    await _publish(cell_id, "robot.tcp_pose", {"tcp_pose": runtime.cell.latest_tcp_pose, "command": req.model_dump()})
    return {"ok": True, "tcp_pose": runtime.cell.latest_tcp_pose}


@router.post("/{cell_id}/robot/movej")
async def movej(cell_id: str, req: MoveJRequest):
    runtime = _runtime(cell_id)
    result = plan_joint_path(runtime.cell.latest_joint_state or req.joints, req.joints, req.planner_options)
    if result["success"]:
        runtime.cell.latest_joint_state = req.joints
    runtime.cell.touch()
    await _publish(cell_id, "robot.state", {"joint_state": runtime.cell.latest_joint_state, "planning": result})
    return result


@router.post("/{cell_id}/robot/movel")
async def movel(cell_id: str, req: MoveLRequest):
    runtime = _runtime(cell_id)
    runtime.cell.latest_tcp_pose = req.tcp_pose
    runtime.cell.touch()
    await _publish(cell_id, "robot.tcp_pose", {"tcp_pose": runtime.cell.latest_tcp_pose})
    return {"success": True, "tcp_pose": runtime.cell.latest_tcp_pose, "rejection_reason": None}


@router.post("/{cell_id}/robot/stop")
async def stop_robot(cell_id: str):
    _runtime(cell_id)
    await _publish(cell_id, "robot.state", {"mode": "STOPPED"})
    return {"ok": True, "status": "stopped"}


@router.post("/{cell_id}/poses")
async def record_pose(cell_id: str, req: PoseRecordRequest):
    runtime = _runtime(cell_id)
    pose_id = req.name.strip() or str(uuid.uuid4())
    record = PoseRecord(
        pose_id=pose_id,
        name=req.name,
        tcp_pose=runtime.cell.latest_tcp_pose if "tcp_pose" in req.store else None,
        joint_state=runtime.cell.latest_joint_state if "joint_state" in req.store else [],
    )
    runtime.cell.poses[pose_id] = record
    runtime.cell.touch()
    await _publish(cell_id, "scene.updated", {"pose": record.model_dump()})
    return {"status": "saved", "pose": record.model_dump()}


@router.get("/{cell_id}/poses")
async def list_poses(cell_id: str):
    runtime = _runtime(cell_id)
    return {"poses": [pose.model_dump() for pose in runtime.cell.poses.values()]}


@router.delete("/{cell_id}/poses/{pose_id}", status_code=204)
async def delete_pose(cell_id: str, pose_id: str):
    runtime = _runtime(cell_id)
    if runtime.cell.poses.pop(pose_id, None) is None:
        raise HTTPException(status_code=404, detail=f"pose not found: {pose_id}")
    runtime.cell.touch()
    await _publish(cell_id, "scene.updated", {"deleted_pose": pose_id})


@router.post("/{cell_id}/geometry/import")
async def import_geometry(cell_id: str, req: GeometryImportRequest):
    return await add_object(cell_id, req)


@router.post("/{cell_id}/geometry/upload")
async def upload_geometry(cell_id: str, file: UploadFile = File(...)):
    runtime = _runtime(cell_id)
    dest = runtime.tmp_dir / f"{uuid.uuid4()}-{Path(file.filename or 'mesh').name}"
    with dest.open("wb") as out:
        shutil.copyfileobj(file.file, out)
    return {"ok": True, "asset_path": str(dest), "mesh_url": str(dest)}


@router.get("/{cell_id}/geometry")
async def list_geometry(cell_id: str):
    runtime = _runtime(cell_id)
    return {"geometry": [geom.model_dump() for geom in runtime.cell.geometries.values()]}


@router.put("/{cell_id}/geometry/{geometry_id}/pose")
async def update_geometry_pose(cell_id: str, geometry_id: str, req: GeometryPoseRequest):
    runtime = _runtime(cell_id)
    geom = runtime.cell.geometries.get(geometry_id)
    if geom is None:
        raise HTTPException(status_code=404, detail=f"geometry not found: {geometry_id}")
    geom.pose = req.pose
    if runtime.engine.world and geometry_id in runtime.engine.world.objects:
        runtime.engine.update_object_pose(geometry_id, req.pose)
    runtime.cell.touch()
    await _publish(cell_id, "scene.updated", {"geometry": geom.model_dump()})
    return {"ok": True, "geometry": geom.model_dump()}


@router.delete("/{cell_id}/geometry/{geometry_id}", status_code=204)
async def delete_geometry(cell_id: str, geometry_id: str):
    runtime = _runtime(cell_id)
    if runtime.cell.geometries.pop(geometry_id, None) is None:
        raise HTTPException(status_code=404, detail=f"geometry not found: {geometry_id}")
    runtime.rebuild_collision_world()
    runtime.cell.touch()
    await _publish(cell_id, "scene.updated", {"deleted_geometry": geometry_id})


@router.post("/{cell_id}/planning/check-path")
async def check_path(cell_id: str, req: CheckPathRequest):
    runtime = _runtime(cell_id)
    result = await run_in_threadpool(check_path_with_world, runtime, req.q_waypoints, req.resolution)
    await _publish(cell_id, "planning.result", result)
    return result


@router.post("/{cell_id}/planning/plan-to-pose")
async def plan_to_pose(cell_id: str, req: PlanToPoseRequest):
    runtime = _runtime(cell_id)
    start = req.start_joints
    goal = req.target_joints
    if start is None and req.start_pose_id:
        start = runtime.cell.poses.get(req.start_pose_id, PoseRecord(pose_id="", name="")).joint_state
    if goal is None and req.target_pose_id:
        goal = runtime.cell.poses.get(req.target_pose_id, PoseRecord(pose_id="", name="")).joint_state
    start = start or runtime.cell.latest_joint_state
    if not start or not goal:
        result = {"success": False, "collision_free": False, "rejection_reason": "INVALID_REQUEST", "error_message": "start and target joint states are required"}
    else:
        result = await run_in_threadpool(plan_joint_path, start, goal, req.planner_options)
    await _publish(cell_id, "planning.result", result)
    return result


@router.post("/{cell_id}/planning/plan-test-case")
async def plan_test_case(cell_id: str, req: TestCaseRecord):
    runtime = _runtime(cell_id)
    runtime.cell.test_cases[req.test_case_id] = req
    result = await _run_test_case(runtime, req)
    runtime.cell.test_results[req.test_case_id] = result
    await _publish(cell_id, "test_case.status", {"test_case_id": req.test_case_id, "result": result})
    return result


@router.post("/{cell_id}/test-cases")
async def create_test_case(cell_id: str, req: TestCaseRecord):
    runtime = _runtime(cell_id)
    runtime.cell.test_cases[req.test_case_id] = req
    runtime.cell.touch()
    await _publish(cell_id, "test_case.status", {"test_case": req.model_dump(), "status": "created"})
    return {"test_case": req.model_dump()}


@router.get("/{cell_id}/test-cases")
async def list_test_cases(cell_id: str):
    runtime = _runtime(cell_id)
    return {"test_cases": [item.model_dump() for item in runtime.cell.test_cases.values()]}


@router.post("/{cell_id}/test-cases/{test_case_id}/run")
async def run_test_case(cell_id: str, test_case_id: str):
    runtime = _runtime(cell_id)
    case = runtime.cell.test_cases.get(test_case_id)
    if case is None:
        raise HTTPException(status_code=404, detail=f"test case not found: {test_case_id}")
    await _publish(cell_id, "test_case.status", {"test_case_id": test_case_id, "status": "RUNNING"})
    result = await _run_test_case(runtime, case)
    runtime.cell.test_results[test_case_id] = result
    await _publish(cell_id, "test_case.status", {"test_case_id": test_case_id, "status": "SUCCEEDED" if result.get("success") else "FAILED", "result": result})
    return result


@router.get("/{cell_id}/test-cases/{test_case_id}/result")
async def get_test_case_result(cell_id: str, test_case_id: str):
    runtime = _runtime(cell_id)
    return {"result": runtime.cell.test_results.get(test_case_id)}


@router.websocket("/{cell_id}/stream")
async def stream_cell(websocket: WebSocket, cell_id: str):
    _runtime(cell_id)
    await websocket.accept()
    queue = await get_bus().subscribe(cell_id)
    try:
        await websocket.send_json({"type": "bridge_status", "cell_id": cell_id})
        while True:
            message = await queue.get()
            await websocket.send_json(message.model_dump(by_alias=True))
    except WebSocketDisconnect:
        pass
    finally:
        await get_bus().unsubscribe(cell_id, queue)


async def _run_test_case(runtime, case: TestCaseRecord) -> Dict[str, Any]:
    start = runtime.cell.poses.get(case.start_pose_id or "", None)
    target = runtime.cell.poses.get(case.target_pose_id or "", None)
    if start is None or target is None or not start.joint_state or not target.joint_state:
        return {"success": False, "collision_free": False, "rejection_reason": "INVALID_REQUEST", "error_message": "test case requires start and target poses with joint states"}
    return await run_in_threadpool(plan_joint_path, start.joint_state, target.joint_state, case.planner_options)
