from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import APIRouter
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel

from robot_engine.path_planning.birrt import BiRRTPlanner
from robot_engine.path_planning.cartesian_linear_planner import CartesianLinearPlanner
from robot_engine.path_planning.collision_aware_planner import CollisionAwarePlanner
from robot_engine.path_planning.joint_direct_planner import JointDirectPlanner
from robot_engine.path_planning.path_repair import repair_path
from robot_engine.path_planning.planner_base import PathRequest
from robot_engine.path_planning.prm import PRMPlanner
from robot_engine.path_planning.rrt import RRTPlanner
from robot_engine.path_planning.rrt_connect import RRTConnectPlanner
from robot_engine.path_planning.shortcut_smoothing import shortcut_smooth_path

router = APIRouter(prefix="/planning", tags=["planning"])


class PlanRequest(BaseModel):
    start: List[float]
    goal: List[float]
    lower_limits: Optional[List[float]] = None
    upper_limits: Optional[List[float]] = None
    max_joint_step: float = 0.1
    max_iterations: int = 1000
    timeout: float = 5.0
    goal_bias: float = 0.1
    debug_info: Dict[str, Any] = {}


def _limits(req: PlanRequest):
    if req.lower_limits is not None and req.upper_limits is not None:
        return (req.lower_limits, req.upper_limits)
    return None


def _path_result(result):
    return {
        "success": result.success,
        "planner_used": result.planner_used,
        "q_waypoints": [np.asarray(q, dtype=float).tolist() for q in result.q_waypoints],
        "length": result.length,
        "minimum_clearance": result.minimum_clearance,
        "planning_time": result.planning_time,
        "rejection_reason": result.rejection_reason,
        "debug_info": result.debug_info,
    }


def _make_path_request(req: PlanRequest) -> PathRequest:
    return PathRequest(
        start=req.start,
        goal=req.goal,
        joint_limits=_limits(req),
        max_joint_step=req.max_joint_step,
        max_iterations=req.max_iterations,
        timeout=req.timeout,
        goal_bias=req.goal_bias,
        debug_info=req.debug_info,
    )


@router.post("/path/direct")
async def plan_direct(req: PlanRequest):
    result = await run_in_threadpool(JointDirectPlanner().plan, _make_path_request(req))
    return _path_result(result)


@router.post("/path/rrt")
async def plan_rrt(req: PlanRequest):
    result = await run_in_threadpool(RRTPlanner().plan, _make_path_request(req))
    return _path_result(result)


@router.post("/path/rrt-connect")
async def plan_rrt_connect(req: PlanRequest):
    result = await run_in_threadpool(RRTConnectPlanner().plan, _make_path_request(req))
    return _path_result(result)


@router.post("/path/birrt")
async def plan_birrt(req: PlanRequest):
    result = await run_in_threadpool(BiRRTPlanner().plan, _make_path_request(req))
    return _path_result(result)


@router.post("/path/prm")
async def plan_prm(req: PlanRequest):
    result = await run_in_threadpool(PRMPlanner().plan, _make_path_request(req))
    return _path_result(result)


@router.post("/path/collision-aware")
async def plan_collision_aware(req: PlanRequest):
    result = await run_in_threadpool(CollisionAwarePlanner().plan, _make_path_request(req))
    return _path_result(result)


@router.post("/path/cartesian-linear")
async def plan_cartesian_linear(req: PlanRequest):
    result = await run_in_threadpool(CartesianLinearPlanner().plan, _make_path_request(req))
    return _path_result(result)


class RepairRequest(BaseModel):
    q_waypoints: List[List[float]]
    smoothing_iterations: int = 100

@router.post("/path/repair")
async def plan_repair(req: RepairRequest):
    def _repair():
        repaired, stats = repair_path(
            [np.asarray(q, dtype=float) for q in req.q_waypoints],
            smoothing_iterations=req.smoothing_iterations,
        )
        return {
            "success": stats.get("repaired", False) or stats.get("reason") == "OK",
            "q_waypoints": [q.tolist() if hasattr(q, "tolist") else q for q in repaired],
            "stats": stats,
        }
    return await run_in_threadpool(_repair)


class SmoothRequest(BaseModel):
    q_waypoints: List[List[float]]
    iterations: int = 100

@router.post("/path/smooth")
async def smooth(req: SmoothRequest):
    def _smooth():
        path, stats = shortcut_smooth_path(
            [np.asarray(q, dtype=float) for q in req.q_waypoints],
            iterations=req.iterations,
        )
        return {
            "success": True,
            "q_waypoints": [q.tolist() for q in path],
            "stats": stats,
        }
    return await run_in_threadpool(_smooth)
