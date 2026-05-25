# spatial-transform

**Kind:** Module &nbsp;|&nbsp; **Status:** phase 3 — implemented

Lift perception poses from the camera frame to the robot base frame
(spec §9). Subscribes a perception `Pose6D` plus `imp/<station>/tf`,
composes the chain through a `TfGraph`, and publishes a `PoseTarget` in
the configured base frame. In eye-in-hand mode the live `base -> tcp`
edge is injected from FK on each `RobotState`, so the same tf lookup
handles both topologies.

## What's in the box

| Symbol | Where | Role |
|---|---|---|
| `lift_pose` | `lift.py` | Pure: `(TfGraph, base_frame, source_frame, position, quat_xyzw) -> (position, quat)`. No imp_sdk, no zenoh; unit-tested standalone. |
| `pose_to_matrix` / `matrix_to_pose` | `lift.py` | SE(3) <-> position+quat_xyzw helpers (also unit-tested). |
| `TransformModule` | `transform.py` | Compute-Runtime wrapper: marshals protobuf, maintains the in-process `TfGraph` from `imp/<station>/tf`, calls `lift_pose`, publishes a `PoseTarget`. |

## Two topologies, one lookup

```text
eye-to-hand (default)
    tf:  base -> camera   (one static edge, e.g. from calibration)
    lookup(base, camera) -> 1 hop

eye-in-hand (--eye-in-hand + --robot-system)
    tf:  tcp  -> camera   (static hand-eye edge)
    fk:  base -> tcp      (injected each tick from RobotState.q)
    lookup(base, camera) -> 2 hops, composed automatically
```

## Run

```bash
# eye-to-hand: a static hand-eye edge needs to be on tf
python -m imp_module_spatial_transform \
    --station devstation \
    --pose-key imp/devstation/perc/s1/pose

# eye-in-hand: FK injection enabled
python -m imp_module_spatial_transform \
    --station devstation \
    --pose-key imp/devstation/perc/s1/pose \
    --eye-in-hand --robot fr3 \
    --robot-system .../franka_fr3_with_franka_hand.yaml
```

## Tests

```bash
cd platform/modules/spatial-transform
PYTHONPATH=.:../spatial-tf pytest tests -q
```

8 tests covering identity passthrough, eye-to-hand translation +
rotation, two-hop chain composition, orientation round-trip via the
graph, and the "tf not yet connected" fallback.

## Source

Reference equivalent: `_pick_runtime.py` hand-rolled `_resolve_hand_eye` +
`_invert_transform` + per-call quaternion math at lines 704–800. Here the
math lives in `lift.py` and the chain composition is delegated to
`TfGraph`, so hand-eye is no longer a special case.
