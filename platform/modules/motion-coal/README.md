# motion-coal

**Kind:** Module &nbsp;|&nbsp; **Status:** phase 3 — collision check + Scene-fill seam implemented

Collision-checking imp module wrapping motion-core (Coal). `CollisionModule`
is a Compute-Runtime module (spec §9): it builds the resolved
`CollisionModel` + `Scene` from a `world.yaml` at `configure`, then on each
`RobotState` runs `algorithms.collision.is_in_collision` and publishes the
contact count (0 = collision-free) as a `Scalar`.

## Scene-fill seam

If `--object-pose-key` + `--object-id` are configured, the module subscribes
a perception `Pose6D` on that key and routes it into
`Scene.set_object_pose(object_id, ...)` each tick **before** running the
collision query. A moving obstacle observed by perception flips the verdict
without any orchestrator code in between — this is the §9 / debt **D3**
seam being live.

```bash
# static-scene mode (no perception input):
PYTHONPATH=sdk/py:modules/motion-core/algorithms:modules/motion-coal \
  python -m imp_module_motion_coal \
    --world modules/motion-core/algorithms/configs/worlds/franka_table_world.yaml

# Scene-fill mode: perception pose updates Scene.set_object_pose("matka")
python -m imp_module_motion_coal \
    --world modules/motion-core/algorithms/configs/worlds/franka_table_world.yaml \
    --object-pose-key imp/devstation/perc/s1/world_pose --object-id matka

# cross-check static collision against the library:
python modules/motion-coal/examples/verify_collision.py \
  modules/motion-core/algorithms/configs/worlds/franka_table_world.yaml   # RESULT: OK
```

## Attach / detach

`Scene.attach` / `Scene.detach` add and revoke the dynamic ACM allowance
plus rebind the object's geometry to a robot frame. A topic-driven schema
for grasp events lands with the task layer in P5; until then the in-process
`module.scene.attach(...)` API is what the Scene-fill integration test
exercises (see `platform/tests/test_scene_fill.py`).

Wraps `robot-algorithms collision/` via motion-core.
