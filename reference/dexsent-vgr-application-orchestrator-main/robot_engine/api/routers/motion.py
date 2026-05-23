from __future__ import annotations

from fastapi import APIRouter
from fastapi.concurrency import run_in_threadpool

from robot_engine.motion_primitives.approach import plan_approach
from robot_engine.motion_primitives.extract import plan_extract
from robot_engine.motion_primitives.lift import plan_lift
from robot_engine.motion_primitives.move_j import plan_move_j
from robot_engine.motion_primitives.move_l import plan_move_l
from robot_engine.motion_primitives.pick_sequence import plan_pick_sequence
from robot_engine.motion_primitives.place_sequence import plan_place_sequence
from robot_engine.motion_primitives.retreat import plan_retreat

router = APIRouter(prefix="/motion", tags=["motion"])

# Motion requests use the engine's own request objects, which are Pydantic models.
# We import the relevant request type from motion.motion_request and use it directly.

from robot_engine.motion.motion_request import MotionRequest
from robot_engine.motion.motion_sequence import plan_motion_sequence
from robot_engine.motion.motion_result import MotionSequence


def _seg_result(result) -> dict:
    """Serialise MotionSegmentResult or MotionSequenceResult."""
    if hasattr(result, "segments"):
        return {
            "success": result.success,
            "failed_stage": getattr(result, "failed_stage", None),
            "rejection_reason": getattr(result, "rejection_reason", "OK"),
            "segments": [_seg_result(s) for s in result.segments],
            "debug_info": getattr(result, "debug_info", {}),
        }
    if hasattr(result, "trajectory"):
        traj = result.trajectory
        traj_dict = None
        if traj is not None and hasattr(traj, "positions"):
            traj_dict = {
                "joint_names": traj.joint_names,
                "positions": traj.positions,
                "times": traj.times,
                "velocities": traj.velocities,
                "accelerations": traj.accelerations,
            }
        elif traj is not None and hasattr(traj, "points"):
            traj_dict = {
                "timestamps": [p.time for p in traj.points],
                "q": [p.q for p in traj.points],
                "q_dot": [p.q_dot for p in traj.points],
                "q_ddot": [p.q_ddot for p in traj.points],
                "duration": traj.duration,
                "generation_method": traj.generation_method,
            }
        return {
            "success": result.success,
            "label": getattr(result, "label", ""),
            "failed_stage": getattr(result, "failed_stage", None),
            "rejection_reason": getattr(result, "rejection_reason", "OK"),
            "trajectory": traj_dict,
            "debug_info": getattr(result, "debug_info", {}),
        }
    return {"success": getattr(result, "success", False), "raw": str(result)}


@router.post("/move-j")
async def move_j(req: MotionRequest):
    result = await run_in_threadpool(plan_move_j, req)
    return _seg_result(result)


@router.post("/move-l")
async def move_l(req: MotionRequest):
    result = await run_in_threadpool(plan_move_l, req)
    return _seg_result(result)


@router.post("/approach")
async def approach(req: MotionRequest):
    result = await run_in_threadpool(plan_approach, req)
    return _seg_result(result)


@router.post("/retreat")
async def retreat(req: MotionRequest):
    result = await run_in_threadpool(plan_retreat, req)
    return _seg_result(result)


@router.post("/lift")
async def lift(req: MotionRequest):
    result = await run_in_threadpool(plan_lift, req)
    return _seg_result(result)


@router.post("/extract")
async def extract(req: MotionRequest):
    result = await run_in_threadpool(plan_extract, req)
    return _seg_result(result)


@router.post("/pick-sequence")
async def pick_sequence(req: MotionSequence):
    result = await run_in_threadpool(plan_pick_sequence, req)
    return _seg_result(result)


@router.post("/place-sequence")
async def place_sequence(req: MotionSequence):
    result = await run_in_threadpool(plan_place_sequence, req)
    return _seg_result(result)
