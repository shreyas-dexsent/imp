from __future__ import annotations

from robot_engine.motion.lift_motion import plan_lift_motion


def plan_extract(request):
    result = plan_lift_motion(request)
    if not result.success:
        result.failed_stage = result.failed_stage or "extract_lift"
    return result

