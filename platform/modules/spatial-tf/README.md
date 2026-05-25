# spatial-tf

**Kind:** Module &nbsp;|&nbsp; **Status:** phase 3 — implemented

Frame graph over Zenoh (spec §9). Subscribes `imp/<station>/tf` (`TfEdge`
messages) and maintains a live `TfGraph` so any module can ask
*"where is frame X expressed in frame Y?"*. Hand-eye, the world `base_pose`,
and any dynamic calibration are **just edges** — no special-cased channels.

## What's in the box

| Symbol | Where | Role |
|---|---|---|
| `TfGraph` | `graph.py` | Pure library: `add_edge(parent, child, T)` + `lookup(parent, child) -> 4x4`. Stores forward + inverse and does BFS composition. |
| `TfModule` | `module.py` | Compute-Runtime wrapper: subscribes `imp/<station>/tf`, feeds edges into a `TfGraph`, republishes `len(frames())` on `imp/<station>/motion/tf/frames` as a heartbeat. |

`spatial-transform` (and any future tf consumer) embeds its own `TfGraph`
instance and subscribes the same topic — no shared mutable state across modules.

## Run

```bash
# library-only usage
python - <<'PY'
import numpy as np
from imp_module_spatial_tf import TfGraph
g = TfGraph()
g.add_edge("world", "base", np.eye(4))
g.add_edge("base",  "tcp",  np.diag([1,1,1,1.0]))
print(g.lookup("world", "tcp"))
PY

# bus-resident node
python -m imp_module_spatial_tf --station devstation
```

## Tests

```bash
cd platform/modules/spatial-tf
pytest tests -q
```

12 tests cover identity, inverse, chain, diamond, disconnected components,
edge replace/remove, validation, and the canonical
`world -> base -> tcp` hand-eye composition.

## Source

Originally specified in the rev-2 plan (PLAN.md §6, "Motion core + spatial");
no direct VGR reference equivalent — the orchestrator hand-rolled per-call
transform math inside `_pick_runtime.py` rather than maintaining a graph.
