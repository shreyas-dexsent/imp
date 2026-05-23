from __future__ import annotations

from typing import List, Optional

from fastapi import APIRouter
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel

from robot_engine.trajectory.cubic import multi_joint_cubic_trajectory
from robot_engine.trajectory.quintic import multi_joint_quintic_trajectory, quintic_segment_interpolation
from robot_engine.trajectory.retiming import retime_joint_path
from robot_engine.trajectory.s_curve import synchronized_multi_joint_s_curve
from robot_engine.trajectory.trajectory_base import JointTrajectory, JointTrajectoryPoint
from robot_engine.trajectory.trajectory_sampler import sample_trajectory
from robot_engine.trajectory.trajectory_validator import (
    validate_acceleration_limits,
    validate_jerk_limits,
    validate_joint_position_limits,
    validate_trajectory_continuity,
    validate_velocity_limits,
)
from robot_engine.trajectory.trapezoidal import synchronized_multi_joint_trapezoidal

router = APIRouter(prefix="/trajectory", tags=["trajectory"])


def _traj_to_dict(traj: JointTrajectory) -> dict:
    return {
        "success": traj.success,
        "generation_method": traj.generation_method,
        "duration": traj.duration,
        "dof": traj.dof,
        "point_count": len(traj.points),
        "timestamps": [p.time for p in traj.points],
        "q": [p.q for p in traj.points],
        "q_dot": [p.q_dot for p in traj.points],
        "q_ddot": [p.q_ddot for p in traj.points],
        "q_jerk": [p.q_jerk for p in traj.points if p.q_jerk is not None] or None,
        "rejection_reason": traj.rejection_reason,
    }


class CubicRequest(BaseModel):
    q0: List[float]
    q1: List[float]
    v0: Optional[List[float]] = None
    v1: Optional[List[float]] = None
    duration: float = 1.0
    samples: int = 101


@router.post("/cubic")
async def cubic(req: CubicRequest):
    import numpy as np
    dof = len(req.q0)
    v0 = req.v0 or [0.0] * dof
    v1 = req.v1 or [0.0] * dof

    def _gen():
        traj = multi_joint_cubic_trajectory(req.q0, req.q1, v0, v1, req.duration, req.samples)
        return _traj_to_dict(traj)

    return await run_in_threadpool(_gen)


class QuinticRequest(BaseModel):
    q0: List[float]
    q1: List[float]
    v0: Optional[List[float]] = None
    v1: Optional[List[float]] = None
    a0: Optional[List[float]] = None
    a1: Optional[List[float]] = None
    duration: float = 1.0
    samples: int = 101


@router.post("/quintic")
async def quintic(req: QuinticRequest):
    dof = len(req.q0)
    v0 = req.v0 or [0.0] * dof
    v1 = req.v1 or [0.0] * dof
    a0 = req.a0 or [0.0] * dof
    a1 = req.a1 or [0.0] * dof

    def _gen():
        traj = multi_joint_quintic_trajectory(req.q0, req.q1, v0, v1, a0, a1, req.duration, req.samples)
        return _traj_to_dict(traj)

    return await run_in_threadpool(_gen)


class QuinticSegmentsRequest(BaseModel):
    q_waypoints: List[List[float]]
    duration_per_segment: float = 1.0


@router.post("/quintic-segments")
async def quintic_segments(req: QuinticSegmentsRequest):
    def _gen():
        traj = quintic_segment_interpolation(req.q_waypoints, duration_per_segment=req.duration_per_segment)
        return _traj_to_dict(traj)
    return await run_in_threadpool(_gen)


class TrapezoidalRequest(BaseModel):
    q0: List[float]
    q1: List[float]
    v_limits: List[float]
    a_limits: List[float]
    samples: int = 101


@router.post("/trapezoidal")
async def trapezoidal(req: TrapezoidalRequest):
    import numpy as np

    def _gen():
        profiles = synchronized_multi_joint_trapezoidal(req.q0, req.q1, req.v_limits, req.a_limits)
        duration = profiles[0].duration if profiles else 0.0
        timestamps = np.linspace(0.0, duration, req.samples).tolist()
        q_list, v_list, a_list = [], [], []
        for t in timestamps:
            values = [p.evaluate(t) for p in profiles]
            q_list.append([float(v[0]) for v in values])
            v_list.append([float(v[1]) for v in values])
            a_list.append([float(v[2]) for v in values])
        return {
            "success": True,
            "generation_method": "trapezoidal",
            "duration": duration,
            "dof": len(req.q0),
            "point_count": req.samples,
            "timestamps": timestamps,
            "q": q_list,
            "q_dot": v_list,
            "q_ddot": a_list,
            "q_jerk": None,
            "rejection_reason": "OK",
        }

    return await run_in_threadpool(_gen)


class SCurveRequest(BaseModel):
    q0: List[float]
    q1: List[float]
    v_limits: List[float]
    a_limits: List[float]
    j_limits: List[float]
    samples: int = 101


@router.post("/s-curve")
async def s_curve(req: SCurveRequest):
    def _gen():
        result = synchronized_multi_joint_s_curve(req.q0, req.q1, req.v_limits, req.a_limits, req.j_limits, req.samples)
        if isinstance(result, JointTrajectory):
            return _traj_to_dict(result)
        # APIResult failure
        return {"success": False, "error_code": "NOT_IMPLEMENTED", "error_message": str(result)}

    return await run_in_threadpool(_gen)


class RetimeRequest(BaseModel):
    q_waypoints: List[List[float]]
    velocity_limits: List[float]
    acceleration_limits: List[float]
    method: str = "trapezoidal"


@router.post("/retime")
async def retime(req: RetimeRequest):
    def _gen():
        traj = retime_joint_path(req.q_waypoints, req.velocity_limits, req.acceleration_limits, req.method)
        return _traj_to_dict(traj)

    return await run_in_threadpool(_gen)


class ValidateRequest(BaseModel):
    timestamps: List[float]
    q: List[List[float]]
    q_dot: List[List[float]]
    q_ddot: List[List[float]]
    q_jerk: Optional[List[List[float]]] = None
    joint_limits: Optional[List[List[float]]] = None
    velocity_limits: Optional[List[float]] = None
    acceleration_limits: Optional[List[float]] = None
    jerk_limits: Optional[List[float]] = None


@router.post("/validate")
async def validate(req: ValidateRequest):
    def _validate():
        points = [
            JointTrajectoryPoint(t, q, qd, qdd, qj)
            for t, q, qd, qdd, qj in zip(
                req.timestamps, req.q, req.q_dot, req.q_ddot,
                req.q_jerk or [None] * len(req.timestamps)
            )
        ]
        traj = JointTrajectory(points)
        checks = []
        ok, idx, reason = validate_trajectory_continuity(traj)
        checks.append((ok, idx, reason))
        if req.joint_limits:
            lower = [x[0] for x in req.joint_limits]
            upper = [x[1] for x in req.joint_limits]
            checks.append(validate_joint_position_limits(traj, (lower, upper)))
        if req.velocity_limits:
            checks.append(validate_velocity_limits(traj, req.velocity_limits))
        if req.acceleration_limits:
            checks.append(validate_acceleration_limits(traj, req.acceleration_limits))
        if req.jerk_limits:
            checks.append(validate_jerk_limits(traj, req.jerk_limits))
        failed = next((c for c in checks if not c[0]), None)
        if failed:
            return {"success": False, "failed_waypoint_index": failed[1], "rejection_reason": failed[2]}
        return {"success": True, "rejection_reason": "OK"}

    return await run_in_threadpool(_validate)


class SampleRequest(BaseModel):
    timestamps: List[float]
    q: List[List[float]]
    q_dot: List[List[float]]
    q_ddot: List[List[float]]
    dt: float


@router.post("/sample")
async def sample(req: SampleRequest):
    def _sample():
        points = [JointTrajectoryPoint(t, q, qd, qdd) for t, q, qd, qdd in zip(req.timestamps, req.q, req.q_dot, req.q_ddot)]
        traj = JointTrajectory(points)
        sampled = sample_trajectory(traj, req.dt)
        return {
            "timestamps": [p.time for p in sampled],
            "q": [p.q for p in sampled],
            "q_dot": [p.q_dot for p in sampled],
            "q_ddot": [p.q_ddot for p in sampled],
        }

    return await run_in_threadpool(_sample)
