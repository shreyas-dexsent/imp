from __future__ import annotations

from robot_engine.motion.motion_sequence import plan_motion_sequence


def plan_pick_sequence(sequence):
    result = plan_motion_sequence(sequence)
    result.debug_info.setdefault("command_events", ["close_gripper", "attach_object", "lift", "retreat"])
    return result

