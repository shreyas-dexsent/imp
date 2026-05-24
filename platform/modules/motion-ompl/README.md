# motion-ompl

**Kind:** Module &nbsp;|&nbsp; **Status:** phase 5 — joint planning implemented + verified

Joint-space planning imp module wrapping motion-core (OMPL). `PlanModule` is a
Compute-Runtime module (spec §9): it builds `CollisionModel` + `Scene` +
`KinematicModel` from a `world.yaml` at `configure`, then given a start
`RobotState` and a goal `JointSolution` plans a collision-free path with
`algorithms.planning.plan_joint` (RRT-Connect) and publishes a `Path`.

```bash
PYTHONPATH=sdk/py:modules/motion-core/algorithms:modules/motion-ompl \
  python -m imp_module_motion_ompl \
    --world modules/motion-core/algorithms/configs/worlds/franka_robot_only_world.yaml
python modules/motion-ompl/examples/verify_plan.py \
  modules/motion-core/algorithms/configs/worlds/franka_robot_only_world.yaml   # RESULT: OK
```

Verified: over-the-bus path connects start↔goal (endpoint error 0, 100 waypoints).
Wraps `robot-algorithms planning/` (via motion-core).
