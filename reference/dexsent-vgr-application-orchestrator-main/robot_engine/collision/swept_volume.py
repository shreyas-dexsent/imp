from __future__ import annotations

from robot_engine.interfaces.error_codes import ErrorCode
from robot_engine.interfaces.result_types import APIResult
from robot_engine.collision.continuous_collision import conservative_continuous_collision


def swept_volume(*args, **kwargs):
    return APIResult.fail(ErrorCode.NOT_IMPLEMENTED, "Exact analytic swept-volume geometry is not implemented. Use conservative_swept_validation for interpolated continuous collision validation.")


def conservative_swept_validation(q0, q1, state_checker=None, world=None, resolution: float = 0.05):
    return conservative_continuous_collision(q0, q1, state_checker=state_checker, world=world, resolution=resolution)
