from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter, HTTPException

from robot_engine.api.imp_bus import BusMessage, get_bus
from robot_engine.api.orchestrator_store import get_orchestrator_store
from robot_engine.api.sim_runtime import SimJointState, SimRobotConfig, SimTrajectory

router = APIRouter(prefix="/cells/{cell_id}/sim", tags=["sim"])


def _runtime(cell_id: str):
    try:
        return get_orchestrator_store().get(cell_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"cell not found: {cell_id}") from exc


async def _publish(cell_id: str, type_: str, payload: Dict[str, Any]) -> None:
    await get_bus().publish(BusMessage(cell_id=cell_id, type=type_, source="sim", payload=payload))


@router.post("/load-robot")
async def load_robot(cell_id: str, config: SimRobotConfig):
    runtime = _runtime(cell_id)
    state = runtime.sim.load_robot(config)
    runtime.cell.latest_joint_state = state["q"]
    runtime.cell.latest_tcp_pose = state["tcp_pose"]
    runtime.cell.touch()
    await _publish(cell_id, "sim.state", state)
    await _publish(cell_id, "robot.joint_state", state["joint_state"])
    return {"success": True, "state": state}


@router.post("/set-joint-state")
async def set_joint_state(cell_id: str, joint_state: SimJointState):
    runtime = _runtime(cell_id)
    state = runtime.sim.set_joint_state(joint_state)
    runtime.cell.latest_joint_state = state["q"]
    runtime.cell.latest_tcp_pose = state["tcp_pose"]
    runtime.cell.touch()
    await _publish(cell_id, "robot.joint_state", state["joint_state"])
    await _publish(cell_id, "robot.tcp_pose", {"tcp_pose": state["tcp_pose"]})
    return {"success": True, "state": state}


@router.get("/state")
async def get_state(cell_id: str):
    return {"success": True, "state": _runtime(cell_id).sim.state()}


@router.post("/play-trajectory")
async def play_trajectory(cell_id: str, trajectory: SimTrajectory):
    runtime = _runtime(cell_id)

    async def publish(type_: str, payload: Dict[str, Any]) -> None:
        state = runtime.sim.state()
        runtime.cell.latest_joint_state = state["q"]
        runtime.cell.latest_tcp_pose = state["tcp_pose"]
        runtime.cell.touch()
        await _publish(cell_id, type_, payload)

    result = await runtime.sim.play(trajectory, publish)
    return result


@router.post("/pause")
async def pause(cell_id: str):
    state = _runtime(cell_id).sim.pause()
    await _publish(cell_id, "sim.state", state)
    return {"success": True, "state": state}


@router.post("/resume")
async def resume(cell_id: str):
    state = _runtime(cell_id).sim.resume()
    await _publish(cell_id, "sim.state", state)
    return {"success": True, "state": state}


@router.post("/stop")
async def stop(cell_id: str):
    state = _runtime(cell_id).sim.stop()
    await _publish(cell_id, "sim.state", state)
    return {"success": True, "state": state}
