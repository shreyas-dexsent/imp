# Collision

The collision layer owns collision objects, worlds, filtering, distance checks, and path validation.

- `CollisionObject`: geometry + pose + semantic type such as robot link, gripper, object, bin, fixture, or attached object.
- `CollisionWorld`: mutable scene container for objects, matrix rules, and attached-object bookkeeping.
- `CollisionMatrix`: pair rules: `CHECK`, `ALLOW`, and `IGNORE`.
- `narrowphase.py`: exact pair checks through Coal; AABB checks are broadphase helpers.
- `path_collision_checker.py`: validates waypoints and subdivided segments and reports the first failed waypoint/segment.

Continuous collision is implemented as conservative interpolation over active collision checks. Exact analytic swept-volume geometry generation returns `NOT_IMPLEMENTED`.
