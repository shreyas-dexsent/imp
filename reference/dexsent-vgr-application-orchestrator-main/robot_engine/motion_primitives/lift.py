from __future__ import annotations

from robot_engine.motion.lift_motion import plan_lift_motion


def plan_lift(request):
    return plan_lift_motion(request)

