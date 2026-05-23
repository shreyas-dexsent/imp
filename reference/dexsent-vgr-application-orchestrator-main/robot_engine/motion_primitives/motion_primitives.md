# Motion Primitives

Motion primitives compose kinematics, path planning, trajectory generation, and validation.

- MoveJ solves IK for a target frame, plans in joint space, retimes, and validates.
- MoveL samples a Cartesian TCP line and solves IK waypoint by waypoint.
- Approach, retreat, and lift use frame-relative offsets.
- Extract and pick/place sequence produce planned segments plus placeholder command events. They do not control hardware.

