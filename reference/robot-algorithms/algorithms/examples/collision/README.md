# Collision

## 1. Description

The collision layer answers four geometric questions about a robot configuration:

```text
Is this configuration touching anything?     is_in_collision
Which pair is closest, and how far?          min_distance
Which pairs are inside a clearance band?     clearance
Does this sampled joint-space edge hit?      check_edge_collision
```

It does **not** check joint limits, singularity, velocity, acceleration, jerk, torque, controller compatibility, or human safety. Those checks belong to the planner / trajectory / validator layers.

The library exposes ten geometry types, each mapping to one native Coal type:

| Spec | Coal type | Solid / surface | Best for |
|---|---|---|---|
| `BoxGeometrySpec` | `coal.Box` | solid | exact boxes |
| `SphereGeometrySpec` | `coal.Sphere` | solid | balls, safety bubbles |
| `CylinderGeometrySpec` | `coal.Cylinder` | solid | rods, posts |
| `CapsuleGeometrySpec` | `coal.Capsule` | solid | arm-like elongated obstacles |
| `MeshGeometrySpec` (file) | `coal.BVHModelOBBRSS` | **surface** | CAD assets (OBJ / STL / PLY / DAE / GLTF / 3MF / OFF) |
| `MeshDataGeometrySpec` (memory) | `coal.BVHModelOBBRSS` | **surface** | perception meshes already in memory |
| `ConvexHullGeometrySpec` | `coal.Convex` | solid | graspable workpieces, convex CAD |
| `OctreeGeometrySpec` | `coal.OcTree` | sparse occupancy | live point clouds, dynamic clutter |
| `HeightFieldGeometrySpec` | `coal.HeightFieldOBBRSS` | terrain solid | depth scans, table surfaces |
| `ConvexDecompositionSpec` (mesh processing) | multiple `coal.Convex` | solid | concave assets via V-HACD |

**Surface vs solid** is the most common source of bugs. A `BVHModelOBBRSS` from a triangle mesh treats the mesh as a hollow shell — a robot link fully inside a closed mesh reports positive distance to the nearest triangle. Convex hulls, primitives, octrees, and height fields are solid by construction. Prefer solid whenever possible.

## 2. Data Flow

```text
YAML (world)
        |
        v
WorldDescription
        |
        v
CollisionModel  (built once; one pin.GeometryModel for robot links per
        |        robot_id, one shared pin.GeometryModel for world objects)
        |
        v
Scene  (wraps the WorldDescription; carries live object_poses, robot_states,
        |  attached objects, dynamic ACM)
        |
        +--- Scene.add_object(id, collision=spec, visual=spec, pose=T)  # perception
        +--- Scene.set_object_pose(id, T)                                # live pose
        |
        v
collision queries  (model, scene, q) -> ContactReport / DistanceReport / ...
```

The Coal `pin.GeometryModel` catalogue is the **single source of truth**. The planner reads from it. The UI reads from it via `CollisionModel.shapes_for(object_id)`. They cannot drift.

Three pose sources feed the per-query scratch buffer (`pin.GeometryData.oMg`):

| Geometry | Pose source |
|---|---|
| Robot link | per-robot FK via `pin.updateGeometryPlacements` |
| Free-standing world object | `Scene.object_poses[obj]` written directly into `oMg` |
| Attached object | FK on the parent frame, composed with `T_parent_obj` |

`pin.computeCollisions` doesn't know or care where each entry came from. Three pose sources, one query path.

## 3. Usage

### Setup

```python
import numpy as np
from algorithms.descriptions import WorldDescription
from algorithms.resolved import CollisionModel, KinematicModel, Scene
from algorithms.collision import is_in_collision, min_distance, clearance, check_edge_collision

world = WorldDescription.from_yaml("configs/worlds/franka_table_world.yaml")
collision_model = CollisionModel.from_world(world)
scene = Scene.from_world(world, collision_model)

system = world.robot("arm").robot_system
model = KinematicModel.from_robot_system(system)
home = system.named_joint_state("home")
q = np.array([home[name] for name in model.active_joint_names], dtype=float)
```

### Queries

```python
report = is_in_collision(model, scene, q)                 # discrete contact
gap = min_distance(model, scene, q)                       # nearest pair
near = clearance(model, scene, q, threshold=0.05)         # all pairs within 5 cm
edge = check_edge_collision(model, scene, q_a, q_b)       # sampled edge sweep
```

Each returns a typed report carrying the pair names, the distance / penetration, and a list of contacts if requested.

### Perception integration

```python
from algorithms.descriptions import BoxGeometrySpec, OctreeGeometrySpec, ConvexHullGeometrySpec

# Box obstacle at runtime
scene.add_object(
    "obstacle_box",
    collision=BoxGeometrySpec(type="box", size=(0.1, 0.3, 0.3)),
    pose=T_obstacle,
)

# Point cloud → octree (the native Coal answer)
scene.add_object(
    "scan",
    collision=OctreeGeometrySpec(type="octree", points=points_nx3.tolist(), resolution=0.01),
    pose=np.eye(4),
)

# Pose-only update at every perception tick
scene.set_object_pose("obstacle_box", T_obstacle_new)

# Remove a perception object
scene.remove_object("obstacle_box")
```

### Single source of truth for the UI

```python
for name in collision_model.object_names():
    for info in collision_model.shapes_for(name):
        # info.coal_shape is the exact object the planner queries
        # info.kind is a short tag for renderer dispatch
        render(info.kind, info.coal_shape, info.T_parent_shape)
```

### Multi-robot

Pass a `dict[robot_id, q]` to any query for composite-state checks. Cross-robot pairs are checked automatically.

```python
report = is_in_collision(left_model, scene, {"left_arm": q_left, "right_arm": q_right})
```

### Allowed collision matrix (ACM)

Three layers, three homes:

| Class | Example | Declared in |
|---|---|---|
| Robot-internal static | adjacent arm links | robot YAML `collision.allowed_pairs` |
| World static | a bin permanently on a table | world YAML `collision_matrix.rules` |
| Task-driven dynamic | EE allowed to touch a workpiece during grasp | runtime `Scene` API |

`scene.allow_collision(a, b)`, `scene.disallow_collision(a, b)`, and `scene.attach(obj, parent, T_parent_obj)` mutate the dynamic overlay. The full effective ACM is the static union minus dynamic disallows union dynamic allows.

## 4. Examples

| File | What it shows |
|---|---|
| `01_robot_collision_model.py` | Inspect robot collision shapes loaded from URDF. |
| `02_world_collision_objects.py` | Inspect world-object geometry from YAML. |
| `03_allowed_collision_matrix.py` | Inspect the static allowed-collision pairs. |
| `04_self_collision.py` | `is_in_collision` against a robot configuration. |
| `05_world_collision.py` | Move a world object at runtime and re-query. |
| `06_attached_object_collision.py` | Attach an object to a TCP and query it. |
| `07_min_distance_and_clearance.py` | `min_distance` and `clearance`. |
| `08_edge_collision.py` | Sampled joint-space edge collision. |
| `09_known_overlap_pair.py` | Two spheres intentionally overlapping. |
| `10_known_distance_pair.py` | Two spheres at a known separation. |
| `11_known_clearance_pair.py` | Two spheres at a known clearance threshold. |
| `12_perception_runtime_add.py` | Perception adds capsule, convex hull, box objects into a live scene. |
| `13_pointcloud_to_octree.py` | Point cloud wrapped as `coal.OcTree`; query distance from the robot. |
| `14_inspect_shapes_for_ui.py` | Iterate `shapes_for` exactly as a UI would. |

## 5. Common Errors

| Symptom | Cause | Fix |
|---|---|---|
| Robot at "home" reports collision | Adjacent link meshes overlap and aren't in `allowed_pairs` for a non-standard robot YAML. | Declare adjacent pairs as `allowed_pairs` (the bundled FR3 YAML already does this). |
| `set_object_pose` raises "object is currently attached" | You tried to teleport an attached object. | Call `scene.detach(id, T)` first, then `set_object_pose`. |
| `min_distance` returns negative numbers | Penetration depth is reported as a negative signed distance. | Expected behaviour for in-collision pairs. Check `is_in_collision` first if you want a binary decision. |
| Triangle mesh of a closed container fails to detect the robot inside | BVH meshes are surface only. | Use `ConvexHullGeometrySpec` for convex containers, or decompose. |
| Octree query slow | Resolution too fine for the workspace size. | Start at 1 cm; tune down only if needed. |
| `remove_object` raises "YAML-declared" | You tried to remove a static YAML object at runtime. | YAML objects are immutable; only `add_object` perception objects are removable. |
| `add_object` raises "already exists" | Duplicate object id. | `remove_object` first, then `add_object`. |
| Convex hull "swallows" a cavity | The hull eliminates concavities; a graspable mug becomes a solid blob. | Use `ConvexDecompositionSpec` (V-HACD) for concave assets. |

## 6. FAQs

**Q: Do I need a `Scene` to query collision?**
Yes. The collision pipeline reads runtime poses from `Scene.object_poses` and live robot state from `Scene.robot_states`. Even single-robot static queries go through the scene.

**Q: Can I add a runtime perception object without a YAML?**
Yes — `Scene.add_object(id, collision=..., visual=..., pose=...)`. The object lives only in the live scene; it never modifies the immutable `WorldDescription`.

**Q: How do I render the same geometry the planner uses?**
`CollisionModel.shapes_for(object_id)`. Returns the exact Coal objects with their parent / placement info. UI code that reads from this cannot drift from the planner.

**Q: What's the difference between `is_in_collision` and `min_distance`?**
`is_in_collision`: boolean per pair, with contact details if requested. Fast.
`min_distance`: signed distance for every pair, sorted. Slower, more information.

**Q: How do cross-robot collisions work?**
`geometry_entries` walks every robot in `scene.world.robots`, runs FK on each with that robot's `KinematicModel`, and writes their geometry into one shared entry table. Cross-robot pairs (left-arm link vs right-arm link) appear naturally because the pair-materialiser combinations() over all entries.

**Q: Can I have a robot's URDF mesh AND a perception-supplied collision shape for the same link?**
Not directly. The library treats robot collision geometry as fixed at `CollisionModel` build time. You can disable a robot link's collision and add a perception-supplied object as an attached object to that link's frame.

**Q: What happens if the YAML declares a collision mesh that doesn't exist on disk?**
Build fails fast with a clear `FileNotFoundError`. Mesh paths are resolved relative to the YAML file.
