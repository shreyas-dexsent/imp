# motion_primitives

Thin semantic wrappers over the `motion` module implementing the robot pick-and-place vocabulary: MoveJ, MoveL, Approach, Retreat, Lift, Extract, PickSequence, PlaceSequence.

Each primitive is a pure function: `plan_<primitive>(request) → MotionSequenceResult` or `MotionSegmentResult`.

---

## `move_j.py`

### `plan_move_j(request) → MotionSegmentResult`

Forces `request.motion_type = MotionType.JOINT`, then delegates to `plan_joint_move_to_frame`.

Joint motion: IK to target → joint-space path via `CollisionAwarePlanner` → time-parameterised trajectory.

---

## `move_l.py`

### `plan_move_l(request) → MotionSegmentResult`

Forces `request.motion_type = MotionType.LINEAR`, then delegates to `plan_linear_move_to_frame`.

Linear motion: samples Cartesian path along straight TCP line → per-frame IK with continuity check → time-parameterised trajectory.

---

## `approach.py`

### `plan_approach(request) → MotionSequenceResult`

Delegates to `plan_approach_to_frame`.

Two-segment sequence:
1. Move to `T_target` offset by `approach.distance` along `approach.axis`.
2. Move to `T_target` (the grasp frame).

---

## `retreat.py`

### `plan_retreat(request) → MotionSequenceResult`

Delegates to `plan_retreat_from_frame`.

Two-segment sequence:
1. From current position to `T_target`.
2. Move to `T_target` offset by `retreat.distance` along `retreat.axis`.

---

## `lift.py`

### `plan_lift(request) → MotionSequenceResult`

Delegates to `plan_lift_motion`. Moves the TCP vertically (default `+z`) by `lift.distance` in the world frame using a linear move.

---

## `extract.py`

### `plan_extract(request) → MotionSequenceResult`

Delegates to `plan_lift_motion` (same motion as lift). On failure, sets `failed_stage = "extract_lift"` for diagnostic clarity. Semantically distinct from lift: used after pick to pull the object clear of bin walls.

---

## `pick_sequence.py`

### `plan_pick_sequence(sequence) → MotionSequenceResult`

Plans the full pick motion sequence via `plan_motion_sequence`. Additionally marks `debug_info["command_events"]` with the expected gripper/IO events:
```
["close_gripper", "attach_object", "lift", "retreat"]
```

The sequence segments are defined by the caller and typically include: approach, grasp, close-gripper, lift, retreat.

---

## `place_sequence.py`

### `plan_place_sequence(sequence) → MotionSequenceResult`

Same as `plan_pick_sequence` but marks gripper events:
```
["open_gripper", "detach_object", "retreat"]
```

The caller's sequence typically includes: approach, place, open-gripper, retreat.
