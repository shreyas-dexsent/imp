from __future__ import annotations

from robot_engine.motion.joint_motion import plan_joint_move_to_frame
from robot_engine.motion.motion_request import MotionType


def plan_move_j(request):
    request.motion_type = MotionType.JOINT
    return plan_joint_move_to_frame(request)

