# motion-path-processor

**Kind:** Module &nbsp;|&nbsp; **Status:** phase 3 — implemented

Path post-processor — Compute-Runtime wrapper around
`algorithms.optimization.shortcut_smooth` + `spline_fit` (spec §9, the
`plan_path` post-step). Subscribes a joint-space `Path` (from `motion-ompl`
or `motion-cartesian`), reconstructs the internal `algorithms.planning.Path`,
applies random-shortcut smoothing against the configured world+model,
optionally fits a polynomial spline through the shortened waypoints, and
republishes on a `processed` topic.

## Inputs / outputs

| Port | Topic | Schema |
|---|---|---|
| `path` (sub)  | `imp/<station>/motion/<plan-in>/path`  | `Path` |
| `path` (pub)  | `imp/<station>/motion/<plan-out>/path` | `Path` |

Defaults: `plan-in=plan` (matches `motion-ompl`), `plan-out=processed`.

## Run

```bash
PYTHONPATH=sdk/py:modules/motion-core/algorithms:modules/motion-path-processor \
  python -m imp_module_motion_path_processor \
    --world modules/motion-core/algorithms/configs/worlds/franka_table_world.yaml \
    --shortcut-iters 200 --spline-order quintic --spline-samples 200
```

Knobs: `--shortcut-iters` (default 100), `--max-joint-step` (default 0.05 rad),
`--spline-order {cubic,quintic,none}` (default quintic), `--spline-samples`
(default 200).

Wraps `robot-algorithms optimization` via motion-core.
