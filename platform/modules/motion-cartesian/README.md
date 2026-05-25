# motion-cartesian

**Kind:** Module &nbsp;|&nbsp; **Status:** phase 3 — implemented

Cartesian (straight-line) planner — Compute-Runtime wrapper around
`algorithms.planning.plan_cartesian` (spec §9). Subscribes a **world-frame**
`PoseTarget` (the TCP goal) plus a `RobotState` (seed q), samples the linear
SE(3) path, solves IK at each sample using the previous q as the next seed,
and publishes the resulting joint-space `Path`.

## Inputs / outputs

| Port | Topic | Schema |
|---|---|---|
| `goal`  (sub) | `imp/<station>/motion/<plan>/goal`  | `PoseTarget` (world frame) |
| `start` (sub) | `imp/<station>/hal/<robot>/state`    | `RobotState` |
| `path`  (pub) | `imp/<station>/motion/<plan>/path`  | `Path` (joint-space) |

If IK fails or any segment is discontinuous, an empty-waypoint `Path` is
published (caller decides what to do; same pattern as `motion-ompl`).

## Run

```bash
PYTHONPATH=sdk/py:modules/motion-core/algorithms:modules/motion-cartesian \
  python -m imp_module_motion_cartesian \
    --world modules/motion-core/algorithms/configs/worlds/franka_table_world.yaml \
    --world-robot arm
```

Wraps `robot-algorithms planning.cartesian` (straight-line backend) via motion-core.
