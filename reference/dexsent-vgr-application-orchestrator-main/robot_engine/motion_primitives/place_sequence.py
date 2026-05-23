from __future__ import annotations

from robot_engine.motion.motion_sequence import plan_motion_sequence


def plan_place_sequence(sequence):
    result = plan_motion_sequence(sequence)
    result.debug_info.setdefault("command_events", ["open_gripper", "detach_object", "retreat"])
    return result

