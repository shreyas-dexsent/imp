# motion-ruckig

**Kind:** Module &nbsp;|&nbsp; **Status:** phase 5 — trajectory implemented + verified

Time-parameterization imp module wrapping motion-core (Ruckig). `TrajectoryModule`
is a Compute-Runtime module (spec §9): it builds the `KinematicModel` (for the
velocity/acceleration/jerk limits) at `configure`, then turns each incoming `Path`
into a time-stamped `Trajectory` via `algorithms.trajectory.time_parameterize`.

```bash
PYTHONPATH=sdk/py:modules/motion-core/algorithms:modules/motion-ruckig \
  python -m imp_module_motion_ruckig \
    --robot-system modules/motion-core/algorithms/configs/robots/franka_fr3_robot_only.yaml
python modules/motion-ruckig/examples/verify_traj.py \
  modules/motion-core/algorithms/configs/robots/franka_fr3_robot_only.yaml   # RESULT: OK
```

Verified: trajectory starts/ends at the path waypoints, time is monotone, and the
joint velocities respect the model's limits. Wraps `robot-algorithms trajectory/`.
