# motion-coal

**Kind:** Module &nbsp;|&nbsp; **Status:** phase 4 — collision check implemented + verified

Collision-checking imp module wrapping motion-core (Coal). `CollisionModule` is a
Compute-Runtime module (spec §9): it builds the resolved `CollisionModel` + `Scene`
from a `world.yaml` at `configure`, then on each `RobotState` runs
`algorithms.collision.is_in_collision` and publishes the contact count
(0 = collision-free) as a `Scalar`.

```bash
PYTHONPATH=sdk/py:modules/motion-core/algorithms:modules/motion-coal \
  python -m imp_module_motion_coal \
    --world modules/motion-core/algorithms/configs/worlds/franka_table_world.yaml
# cross-check against motion-core's direct is_in_collision:
python modules/motion-coal/examples/verify_collision.py \
  modules/motion-core/algorithms/configs/worlds/franka_table_world.yaml   # RESULT: OK
```

Wraps `robot-algorithms collision/` (via motion-core).
