from __future__ import annotations

from typing import List, Optional

import numpy as np
from fastapi import APIRouter
from pydantic import BaseModel

from robot_engine.core_math.interpolation import (
    interpolate_joint,
    interpolate_pose_SE3,
    sample_cartesian_path,
    sample_joint_path,
)
from robot_engine.core_math.lie_groups import se3_exp, se3_log
from robot_engine.core_math.rotations import (
    angular_distance,
    rotation_exp,
    rotation_log,
    slerp_quaternion,
    validate_rotation_matrix,
)
from robot_engine.core_math.transforms import (
    compose_transform,
    invert_transform,
    is_valid_transform,
    pose_error,
    relative_transform,
    validate_transform,
)

router = APIRouter(prefix="/core", tags=["core_math"])

Matrix4 = List[List[float]]
Vec3 = List[float]
Vec4 = List[float]
Vec6 = List[float]


def _ok(**kw):
    return {"success": True, "error_code": "OK", "error_message": "", **kw}


def _fail(code: str, msg: str):
    return {"success": False, "error_code": code, "error_message": msg}


# --- transforms ---

class ComposeRequest(BaseModel):
    T_ab: Matrix4
    T_bc: Matrix4

@router.post("/transforms/compose")
def compose(req: ComposeRequest):
    try:
        result = compose_transform(req.T_ab, req.T_bc)
        return _ok(matrix=result.tolist())
    except Exception as exc:
        return _fail("INVALID_TRANSFORM", str(exc))


class InvertRequest(BaseModel):
    matrix: Matrix4

@router.post("/transforms/invert")
def invert(req: InvertRequest):
    try:
        result = invert_transform(req.matrix)
        return _ok(matrix=result.tolist())
    except Exception as exc:
        return _fail("INVALID_TRANSFORM", str(exc))


class RelativeRequest(BaseModel):
    T_world_a: Matrix4
    T_world_b: Matrix4

@router.post("/transforms/relative")
def relative(req: RelativeRequest):
    try:
        result = relative_transform(req.T_world_a, req.T_world_b)
        return _ok(matrix=result.tolist())
    except Exception as exc:
        return _fail("INVALID_TRANSFORM", str(exc))


class ValidateTransformRequest(BaseModel):
    matrix: Matrix4

@router.post("/transforms/validate")
def validate_tf(req: ValidateTransformRequest):
    valid = is_valid_transform(req.matrix)
    if valid:
        return _ok(valid=True)
    return _fail("INVALID_TRANSFORM", "Matrix is not a valid SE(3) transform")


# --- rotations ---

class RotExpRequest(BaseModel):
    axis_angle: Vec3  # rotation vector

@router.post("/rotations/exp")
def rot_exp(req: RotExpRequest):
    try:
        R = rotation_exp(req.axis_angle)
        return _ok(matrix=R.tolist())
    except Exception as exc:
        return _fail("INVALID_ROTATION_MATRIX", str(exc))


class RotLogRequest(BaseModel):
    matrix: List[List[float]]  # 3x3

@router.post("/rotations/log")
def rot_log(req: RotLogRequest):
    try:
        w = rotation_log(req.matrix)
        return _ok(axis_angle=w.tolist())
    except Exception as exc:
        return _fail("INVALID_ROTATION_MATRIX", str(exc))


class SlerpRequest(BaseModel):
    q0: Vec4   # [x,y,z,w]
    q1: Vec4
    alpha: float

@router.post("/rotations/slerp")
def slerp(req: SlerpRequest):
    try:
        q = slerp_quaternion(req.q0, req.q1, req.alpha)
        return _ok(quaternion=q.tolist())
    except Exception as exc:
        return _fail("INVALID_QUATERNION", str(exc))


class AngularDistanceRequest(BaseModel):
    R_a: List[List[float]]
    R_b: List[List[float]]

@router.post("/rotations/angular-distance")
def angular_dist(req: AngularDistanceRequest):
    try:
        d = angular_distance(req.R_a, req.R_b)
        return _ok(distance=d)
    except Exception as exc:
        return _fail("INVALID_ROTATION_MATRIX", str(exc))


# --- Lie groups ---

class SE3ExpRequest(BaseModel):
    twist: Vec6   # [vx,vy,vz,wx,wy,wz]

@router.post("/lie/se3-exp")
def se3_exp_endpoint(req: SE3ExpRequest):
    try:
        T = se3_exp(req.twist)
        return _ok(matrix=T.tolist())
    except Exception as exc:
        return _fail("INVALID_TRANSFORM", str(exc))


class SE3LogRequest(BaseModel):
    matrix: Matrix4

@router.post("/lie/se3-log")
def se3_log_endpoint(req: SE3LogRequest):
    try:
        xi = se3_log(req.matrix)
        return _ok(twist=xi.tolist())
    except Exception as exc:
        return _fail("INVALID_TRANSFORM", str(exc))


# --- interpolation ---

class JointInterpRequest(BaseModel):
    q0: List[float]
    q1: List[float]
    alpha: float

@router.post("/interpolation/joint")
def joint_interp(req: JointInterpRequest):
    try:
        q = interpolate_joint(req.q0, req.q1, req.alpha)
        return _ok(q=q.tolist())
    except Exception as exc:
        return _fail("INVALID_REQUEST", str(exc))


class CartesianInterpRequest(BaseModel):
    T0: Matrix4
    T1: Matrix4
    alpha: float

@router.post("/interpolation/cartesian")
def cartesian_interp(req: CartesianInterpRequest):
    try:
        T = interpolate_pose_SE3(req.T0, req.T1, req.alpha)
        return _ok(matrix=T.tolist())
    except Exception as exc:
        return _fail("INVALID_TRANSFORM", str(exc))


class SampleJointPathRequest(BaseModel):
    q0: List[float]
    q1: List[float]
    max_joint_step: float

@router.post("/interpolation/joint-path")
def sample_joint(req: SampleJointPathRequest):
    try:
        path = sample_joint_path(req.q0, req.q1, req.max_joint_step)
        return _ok(waypoints=[q.tolist() for q in path])
    except Exception as exc:
        return _fail("INVALID_REQUEST", str(exc))


class SampleCartesianPathRequest(BaseModel):
    T0: Matrix4
    T1: Matrix4
    translation_step: float
    rotation_step: float

@router.post("/interpolation/cartesian-path")
def sample_cartesian(req: SampleCartesianPathRequest):
    try:
        frames = sample_cartesian_path(req.T0, req.T1, req.translation_step, req.rotation_step)
        return _ok(frames=[f.tolist() for f in frames])
    except Exception as exc:
        return _fail("INVALID_TRANSFORM", str(exc))


class PoseErrorRequest(BaseModel):
    T_current: Matrix4
    T_target: Matrix4

@router.post("/transforms/pose-error")
def pose_err(req: PoseErrorRequest):
    try:
        err = pose_error(req.T_current, req.T_target)
        return _ok(error=err.tolist(), position_norm=float(np.linalg.norm(err[:3])), orientation_norm=float(np.linalg.norm(err[3:])))
    except Exception as exc:
        return _fail("INVALID_TRANSFORM", str(exc))
