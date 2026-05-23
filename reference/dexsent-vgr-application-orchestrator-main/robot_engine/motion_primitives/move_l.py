from __future__ import annotations

from robot_engine.motion.linear_motion import plan_linear_move_to_frame
from robot_engine.motion.motion_request import MotionType


def plan_move_l(request):
    request.motion_type = MotionType.LINEAR
    return plan_linear_move_to_frame(request)

