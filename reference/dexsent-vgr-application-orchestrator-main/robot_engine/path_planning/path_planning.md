# Path Planning

Path planning returns geometric waypoints only. It does not assign timestamps.

- `JointDirectPlanner`: straight line in joint space with joint-limit and state-validity checks.
- `CartesianLinearPlanner`: true TCP Cartesian interpolation; an IK callback must solve each waypoint.
- `RRTPlanner` / `RRTConnectPlanner`: sampled joint-space planning helpers.
- `CollisionAwarePlanner`: tries direct path first, then RRT-Connect, then shortcut smoothing.
