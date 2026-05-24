# Plan

`algorithms` is a **generic robotics layer**. It is not specific to any application like bin-picking, assembly, welding, etc. Applications consume this layer; they do not live inside it.

This document is the canonical long-form planning reference for development. It owns: locked architectural decisions, the YAML schema, the module layout, the 10-stage pipeline, the methods inventory per stage, the build order, and open questions. If a decision here changes, update this file in the same commit as the code change.

The companion document [architecture.md](architecture.md) is the polished, public-facing summary of the locked design. When the two disagree, **this file wins**, and `architecture.md` should be updated to match.

Last updated: 2026-05-19.

---

## 1. Why a rewrite

The legacy and current designs mixed three concerns that must stay separate:

1. **Static physical description** — robots and worlds (URDF references, mounts, limits, geometry).
2. **Resolved heavy objects** — Pinocchio models, Coal collision geometry.
3. **Algorithm configuration and runtime state** — planner type, IK tolerance, object poses during execution.

When these mix, scene YAML becomes unstable across solver tuning, runtime updates require rewriting YAML, and multi-robot worlds are hard to express. v2 separates them by construction.

---

## 2. Architectural decisions

Locked in. Build to these.

### 2.1 Three-layer contract

```
YAML (named static facts)
    │
    ▼
Resolution layer (URDF + YAML → typed heavy objects)
    │
    ▼
Algorithms (NumPy arrays + Pinocchio model + Coal geometry + Scene state)
```

YAML stores: joint-name dicts, joint-name-to-limit mappings, 4x4 matrix lists, mesh paths, geometry processing instructions, chain definitions, named joint states.

YAML never stores: planner config, IK config, trajectory backend, dt, solver tolerances, runtime poses, dynamic collision allowances. Those are call-site or Scene-state concerns.

### 2.2 Data objects and operations

| Kind | Examples | Implementation |
|---|---|---|
| Resolved data objects | `KinematicModel`, `CollisionModel`, `Scene`, `Path`, `Trajectory` | classes or dataclasses |
| Stateless operations | `fk()`, `solve_ik()`, `plan_joint()`, `time_parameterize()` | functions |

There is no monolithic engine class. Algorithms stay in their own modules and
compose through explicit resolved data objects.

### 2.3 URDF is canonical; YAML overrides

For every field URDF supplies (joint position limits, velocity limits, effort limits, joint names, mimic relations, default collision geometry), URDF is the source of truth. YAML specifies overrides only for joints that need them.

For fields URDF cannot supply (acceleration limits, jerk limits, mount transforms, named TCPs, chain definitions, named joint states), YAML is the source.

Resolution layer validates at load time: if a chain needs trajectory generation, every joint in it must have acceleration and jerk available (from YAML). Otherwise raise.

### 2.4 One composed Pinocchio model per system

`KinematicModel.from_robot_system` uses `pin.appendModel(robot_model, gripper_model, parent_joint_id, mount_SE3)` to build **one** Pinocchio model covering arm + end-effector. The mount comes from `gripper.mount` in YAML.

One model means one FK call, one Jacobian over the full chain, IK on the combined chain works without stitching. Cost is identical to a smaller model.

### 2.5 KinematicModel is the cache boundary

`RobotSystemDescription` (pydantic, ~ms) → `KinematicModel` (heavy, ~100 ms, cached).

Algorithms accept `KinematicModel`, never re-resolve from YAML. Module-level LRU cache keyed on `(yaml_path, urdf_mtimes)` for auto-invalidation. `Scene` is never cached.

### 2.6 Mimic joints → 1 active DOF

A URDF `<mimic>` joint (e.g. `fr3_finger_joint2` mimicking `fr3_finger_joint1`) is exposed as **0 active DOF**. The driving joint is exposed; the follower is internal. Chains list only active joints.

At resolution time, parse `<mimic>` tags and store an expansion map. Inside FK/IK, `q_full = active_to_full @ q_active + offset_vec`. One matmul. Do not rely on Pinocchio's experimental mimic support.

### 2.7 Description vs Scene — the runtime split

`WorldDescription` is immutable, YAML-loaded, description-time. `Scene` is the runtime mutable container.

```python
@dataclass
class Scene:
    world: WorldDescription
    object_poses:  Dict[str, np.ndarray]   # mutable, 4x4 per object
    attached:      Dict[str, AttachedObject]
    robot_states:  Dict[str, np.ndarray]   # robot_id -> q
    collision_overlay: CollisionOverlay    # dynamic ACM, see §2.9
```

Perception (or any external state source) writes to `Scene`. Algorithms read `Scene`. Algorithms never accept `WorldDescription` directly.

**`Scene` is not middleware.** It is a typed in-process state container. It does not subscribe to ROS topics, queue messages, time-stamp updates, lock across threads, or handle network failures. Those are middleware concerns and live **outside** `algorithms` in application code. The expected pattern matches MoveIt's `PlanningScene` (state container, inside the library) vs `PlanningSceneMonitor` (ROS glue, outside). Keeping the engine transport-agnostic is what lets the same code run on ROS2, libfranka direct, plain TCP, or anything else.

```python
# Inside algorithms — pure state, no transport:
scene.set_object_pose("matka", T)

# Inside YOUR application — your chosen middleware writes into Scene:
node.create_subscription(PoseStamped, "/perception/matka",
                         lambda msg: scene.set_object_pose("matka", parse(msg)), 10)
```

### 2.8 Namespace rules

At `WorldDescription.from_yaml`:

- 1 robot → namespace may be null.
- Multiple robots → every namespace must be non-null, unique, match `^[a-z][a-z0-9_]*$`.

Validated at load time.

### 2.9 Collision allowances — three layers

Three classes of collision pair, three homes:

| Class | Example | Where |
|---|---|---|
| **Robot-internal static** | adjacent links of one arm; finger vs gripper body | robot YAML `collision.allowed_pairs` |
| **World static** | a bin permanently sits on a table | world YAML `collision_matrix.rules` |
| **Task-driven dynamic** | end-effector allowed to touch a workpiece during contact; attached object moves with the gripper | **Scene API at runtime** — never YAML |

The Scene API:

```python
scene.allow_collision(a, b, reason="contact phase")
scene.disallow_collision(a, b)
scene.attach(obj_id, parent_frame, T_parent_obj)   # auto-allows EE ↔ obj
scene.detach(obj_id, T_world_obj)                  # auto-revokes
```

This keeps YAML immutable while making dynamic ACM changes first-class.

### 2.10 Limits API

`KinematicModel` exposes, all returning NumPy in chain joint order:

```python
model.position_limits(chain_id)      -> (lower, upper)
model.velocity_limits(chain_id)      -> v
model.acceleration_limits(chain_id)  -> a
model.jerk_limits(chain_id)          -> j
model.effort_limits(chain_id)        -> tau
```

Sources per §2.3 (URDF + YAML override).

### 2.11 TCPs and chains are system-scoped

Defined once at the top level of the robot system YAML. Composed into world frames via the world robot's namespace. Never overridden per world instance.

### 2.12 Object type vocabulary

`Literal["workpiece", "obstacle", "fixture", "bin"]` in YAML, pydantic-enforced. `"attached"` is a Scene-only state, never a YAML value.

### 2.13 No algorithm config in YAML

Description YAML is for stable physical facts only. Solver knobs (planner type, IK tolerance, timeout, dt, smoothing iterations) are Python function arguments at the call site.

### 2.14 Collision data ownership: Coal vs Pinocchio

Two libraries, two roles. They are not interchangeable.

| Library | Role | What it owns |
|---|---|---|
| **Coal** (formerly HPP-FCL) | The actual collision math | Shape types (`coal.Box`, `coal.Sphere`, `coal.Cylinder`, `coal.Capsule`, `coal.Convex`, `coal.BVHModelOBBRSS`, `coal.OcTree`, `coal.HeightFieldOBBRSS`). GJK / EPA / distance / CCD implementations. |
| **Pinocchio** | The *rigging* — connecting shapes to robot joints + a query plumbing layer | `pin.GeometryObject`, `pin.GeometryModel`, `pin.GeometryData`. Knows which shape attaches to which joint frame, the local offset, the list of pairs to check, and how to push FK results into shape world poses. |

**Supported geometry inputs** (every variant maps to one native Coal type — the library does no algorithmic conversion):

| GeometrySpec variant | Coal type | Solid or surface | Best for |
|---|---|---|---|
| `BoxGeometrySpec` | `coal.Box` | solid | exact box dimensions |
| `SphereGeometrySpec` | `coal.Sphere` | solid | exact sphere |
| `CylinderGeometrySpec` | `coal.Cylinder` | solid | rods, drill bits |
| `CapsuleGeometrySpec` | `coal.Capsule` | solid | arm-like elongated obstacles |
| `MeshGeometrySpec` (file path) | `coal.BVHModelOBBRSS` | **surface** | CAD files (OBJ via Coal; STL/PLY/DAE/GLTF/3MF/OFF via trimesh fallback) |
| `MeshDataGeometrySpec` (in memory) | `coal.BVHModelOBBRSS` | **surface** | meshes already in memory; no /tmp roundtrip |
| `ConvexHullGeometrySpec` | `coal.Convex` | solid | convex objects (graspable workpieces, watertight convex CAD) |
| `OctreeGeometrySpec` | `coal.OcTree` | sparse occupancy | point clouds, voxel grids, dynamic clutter |
| `HeightFieldGeometrySpec` | `coal.HeightFieldOBBRSS` | terrain solid | top-down depth scans, table surfaces |

The surface vs solid split is load-bearing. A triangle BVH is a hollow shell: a point inside a closed mesh reports positive distance to the nearest triangle and the collision is missed. Convex hull, primitives, octree, and height field are all solid by construction. Prefer solid types whenever the geometry permits; use a mesh only when nothing else faithfully represents the object's shape.

Library does **not** convert between these. Perception (or the asset-prep pipeline) decides which variant to ship; the library wires it in.

Concretely:

* A `pin.GeometryObject` is a wrapper. It holds (`name`, `parentJoint`, `parentFrame`, `placement`, `mesh_path`, `mesh_scale`) **plus** a pointer to a Coal `CollisionGeometry`. Pinocchio does not reimplement collision math — it delegates to Coal.
* A `pin.GeometryModel` is the catalogue of `GeometryObject`s for one world (or one robot). It also carries the list of collision pairs to check.
* A `pin.GeometryData` is a **scratch buffer**. It holds the most-recent world-frame placement of each geometry object (`oMg[i]`) plus per-pair collision results. Mutated every query; never the source of truth.

Why use Pinocchio for collision rather than Coal alone? Because Coal doesn't know about robots. To use Coal with a robot you would have to run FK yourself, apply that FK to each shape's local offset, and maintain the pair list. `GeometryModel + GeometryData` is exactly that plumbing. Reimplementing it is wasted work.

### 2.15 `pin.GeometryModel` is a catalogue, not an FK pipeline

A common confusion: Pinocchio's documentation leans into the robot use case, which makes `pin.GeometryModel` sound like "the FK-driven collision pipeline." It isn't. It is a **catalogue type** — a list of `GeometryObject`s plus a list of collision pairs. Nothing about it forces FK.

A `GeometryObject` with `parent_joint = 0` (universe) is by construction static: `pin.updateGeometryPlacements` would set its `oMg[i] = oMjoint[0] @ placement = placement` — a fixed value. So putting world objects in `pin.GeometryModel` does not drag FK over them. At query time we bypass `updateGeometryPlacements` for those entries and write `gd.oMg[idx]` directly from `Scene.object_poses`.

This is why `CollisionModel` uses one consistent catalogue type (`pin.GeometryModel`) for both robot geometry and world geometry, even though the **pose sources** differ:

| Geometry origin | Pose source at query time | Mechanism |
|---|---|---|
| Robot link | query `q` → FK → `pin.Data.oMf` → `updateGeometryPlacements` | Pinocchio FK pipeline |
| Free-standing world object | `Scene.object_poses[obj]` → direct write into `gd.oMg[idx]` | bypass FK |
| Attached object | `Scene.attached[obj].parent_frame` → FK → `pin.Data.oMf` → compose with `T_parent_obj` → write into `gd.oMg[idx]` | FK on parent, then offset |

Three pose sources, one query path. `pin.computeCollisions` reads only `gd.oMg`; it neither knows nor cares where each entry came from. That's the win — one catalogue, one pair list, one query loop, three pose sources funneling into the same scratch buffer.

The takeaway for naming: think of `pin.GeometryModel` as **"the catalogue of shapes known to the world"** and `pin.GeometryData` as **"the scratch buffer holding their current poses for this query."** Neither is "the runtime state of where things are" — that is always `Scene`.

### 2.16 Runtime data flow: persistent state vs scratch buffers

This is the bit that confuses new readers. Be precise about it.

**Persistent state** (lives across queries; mutated by external code):

* `Scene.object_poses[obj_id]` — live 4x4 world pose of each free-standing object.
* `Scene.robot_states[robot_id]` — live `q` per robot (chain-ordered active DOF).
* `Scene.attached[obj_id]` — runtime attach record (parent frame + local offset).
* `Scene.collision_overlay` — dynamic ACM additions/removals.

**Scratch buffers** (per-query temp space; overwritten on every call):

* `pin.Data.oMf[frame_idx]` — world-frame placements of every Pinocchio frame after FK.
* `pin.GeometryData.oMg[geom_idx]` — world-frame placements of every geometry object after `updateGeometryPlacements`.

The flow for one collision query (Phase 4):

```
query q                                               (function input)
       │
       ▼
pin.forwardKinematics(model, data, q_full)            (writes pin.Data.oMf)
       │
       ▼
pin.updateGeometryPlacements(model, data, geom, gd)   (writes GeometryData.oMg[robot])
       │
       │   For world objects:
       │     Scene.object_poses[obj]  ──copy──▶  GeometryData.oMg[obj]
       │
       │   For attached objects:
       │     pin.Data.oMf[parent_frame] @ T_parent_obj  ──▶  GeometryData.oMg[obj]
       │
       ▼
Coal collision/distance call reads GeometryData.oMg
```

Key consequences:

* **`Scene` is the source of truth** for object poses and robot configurations.
* **`pin.Data` and `pin.GeometryData` are derived, ephemeral.** They cache the most-recent FK result and are overwritten on the next call. They are not where pose data "lives".
* The visualizer and the collision-avoidance logic read from the **same** `Scene.object_poses` (directly or via the GeometryData copy). They cannot disagree about where things are.
* Attached objects need no special geometry — only a different pose source. Same `pin.GeometryObject` in `CollisionModel.world_geom`, posed via FK on the parent frame instead of copied from `Scene.object_poses`.

### 2.17 Perception integration and runtime objects

Perception is **outside** the library (locked in §13). The library provides one boundary at which perception adds objects to the live scene; everything past that boundary uses the same code paths as YAML-declared objects.

The contract has three parts:

1. **Format.** Perception ships its output in one of the supported `GeometrySpec` variants from §2.14. The library does not run point-cloud filtering, voxelisation, mesh reconstruction, V-HACD on streaming data, or any other algorithmic conversion. Perception decides; the library wraps.

2. **Entrypoint.** A single Scene method:

   ```python
   scene.add_object(
       object_id,
       collision=<GeometrySpec or None>,
       visual=<GeometrySpec or None>,
       pose=<4x4 ndarray or None>,
   )
   ```

   Behind the scenes this allocates one `pin.GeometryObject` in `CollisionModel.world_geom`, indexes it, and writes the initial pose into `Scene.object_poses`. The visual spec (when supplied) lives on the Scene for UI introspection via `Scene.get_visual_spec(id)`; the planner ignores it.

3. **Lifecycle (Pattern A).** Geometry is built once at `add_object` time. Subsequent updates land in `Scene.object_poses` via the standard `set_object_pose`, identical to YAML-declared objects. For the rare case where the *shape itself* changes mid-run, the caller does `remove_object` followed by a fresh `add_object`.

   `remove_object` refuses to remove YAML-declared objects — YAML is for static physical facts; runtime mutation is for perception inputs only.

This pattern keeps `WorldDescription` immutable, keeps `Scene` as the authoritative runtime state container, and keeps `CollisionModel` as the single source of truth for shapes. No new layer is introduced. Multi-robot worlds work without changes — every perception object lives under the universe joint exactly like YAML world objects.

### 2.18 UI integration via the collision catalogue

A visualizer must show **exactly what the planner uses**. The library provides one accessor that returns the same Coal shapes the planner queries:

```python
collision_model.shapes_for(object_id) -> list[ShapeInfo]
# ShapeInfo: name, coal_shape, owner, parent_joint, T_parent_shape, kind
```

`kind` is a short string tag (`"box"`, `"sphere"`, `"cylinder"`, `"capsule"`, `"convex_hull"`, `"octree"`, `"height_field"`, `"mesh"`) that lets UI code dispatch to its renderer per shape category. The composition of the world pose is identical to what `collision/_runtime.py` does at query time:

```python
# world objects
T_world_shape = scene.object_poses[name] @ info.T_parent_shape
# robot links
T_world_link  = fk(scene, robot_id, q, info.parent_joint)
T_world_shape = T_world_link @ info.T_parent_shape
```

**Rule (load-bearing): visualisation reads from `shapes_for`. It does not construct its own shapes from visual meshes, declared dimensions, or inferred bounding boxes.** If the UI builds an independent representation it will drift from the planner's view as soon as anything in the catalogue changes (decomposition, processing override, perception input). The catalogue is the contract.

`Scene.get_visual_spec(id)` provides the matching visual hook: it returns the perception-overlayed visual when present, falling back to the YAML-declared visual when not. UI code reads both — `shapes_for` for collision rendering, `get_visual_spec` for high-fidelity visuals — but never tries to reconstruct the collision shape from the visual one.

---

## 3. YAML schema (v2)

### 3.1 Robot system

```yaml
schema: dexsent.algorithms.robot_system
version: 2
id: franka_fr3_with_franka_hand
name: Franka FR3 With Franka Hand

robot:
  id: franka_fr3
  urdf_path: ../../../assets/robots/franka_fr3/urdf/franka_fr3.urdf
  package_dirs: []
  base_frame: base

  joint_limits:
    # Optional overrides only. Missing values fall back to URDF.
    # Acceleration and jerk live here too (URDF does not carry them);
    # they are required for any joint used in a chain that needs trajectory generation.
    fr3_joint4:
      position: [-2.8, -0.1]
      velocity: 1.5
      acceleration: 12.0
      jerk: 5000.0

  collision:
    enabled: true
    source: urdf
    allowed_pairs: []   # robot-internal static allowances only
    disabled_links: []

gripper:
  id: franka_hand
  urdf_path: ../../../assets/grippers/franka_hand/urdf/franka_hand.urdf
  package_dirs: []
  root_frame: fr3_hand

  mount:
    parent_frame: fr3_link8
    child_frame: fr3_hand
    matrix:
      - [1.0, 0.0, 0.0, 0.0]
      - [0.0, 1.0, 0.0, 0.0]
      - [0.0, 0.0, 1.0, 0.05]
      - [0.0, 0.0, 0.0, 1.0]

  joint_limits:
    fr3_finger_joint1:
      position: [0.0, 0.04]
    # fr3_finger_joint2 is a URDF <mimic> — not declared.

  collision:
    enabled: true
    source: urdf
    allowed_pairs: []

tcps:
  - id: robot_tcp
    transform:
      parent_frame: fr3_link8
      child_frame: robot_tcp
      matrix: ...
  - id: hand_tcp
    transform:
      parent_frame: fr3_hand
      child_frame: fr3_hand_tcp
      matrix: ...

kinematic_chains:
  - id: arm
    base_frame: base
    tip_frame: fr3_link8
    tcp_frame: robot_tcp
    joints: [fr3_joint1, fr3_joint2, fr3_joint3, fr3_joint4, fr3_joint5, fr3_joint6, fr3_joint7]

  - id: arm_with_gripper
    base_frame: base
    tip_frame: fr3_hand_tcp
    tcp_frame: fr3_hand_tcp
    joints: [fr3_joint1, fr3_joint2, fr3_joint3, fr3_joint4, fr3_joint5, fr3_joint6, fr3_joint7, fr3_finger_joint1]

named_joint_states:
  home:
    joints:
      fr3_joint1: 0.0
      fr3_joint2: -0.7853981633974483
      fr3_joint3: 0.0
      fr3_joint4: -2.356194490192345
      fr3_joint5: 0.0
      fr3_joint6: 1.5707963267948966
      fr3_joint7: 0.7853981633974483
      fr3_finger_joint1: 0.01
```

### 3.2 World

```yaml
schema: dexsent.algorithms.world
version: 2
id: example_world
world_frame: world

robots:
  - id: arm
    robot_system: ../robots/franka_fr3_with_franka_hand.yaml
    namespace: null   # required non-null if more than one robot
    base_pose:
      parent_frame: world
      child_frame: base
      matrix: [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]

objects:
  - id: matka
    type: workpiece   # workpiece | obstacle | fixture | bin
    pose:
      parent_frame: world
      child_frame: matka
      matrix: [[1,0,0,0.45],[0,1,0,0],[0,0,1,0.12],[0,0,0,1]]
    visual:
      enabled: true
      geometry:
        type: mesh
        path: ../../../assets/objects/matka/visual/model.obj
        scale: [1.0, 1.0, 1.0]
    collision:
      enabled: true
      geometry:
        type: mesh
        path: ../../../assets/objects/matka/collision/model.stl
        scale: [1.0, 1.0, 1.0]
      processing:
        type: convex_decomposition
        max_hulls: 16

collision_matrix:
  # Static physical facts ONLY (things that never change at runtime).
  # Task-driven allowances live in Scene, not here.
  default_action: check
  rules: []
```

---

## 4. Module layout

```
algorithms/
├── descriptions/          # Layer 1: YAML pydantic models, no computation
│   ├── transforms.py
│   ├── robot_system.py
│   └── world.py
│
├── resolved/              # Layer 2: built once from descriptions, cached
│   ├── kinematic_model.py # Composed Pinocchio model + chain slicing
│   │                      # + mimic expansion + limits API
│   ├── collision_model.py # Coal geometry wrapped by pin.GeometryModel + static ACM
│   │                      # (Coal owns the shape math; Pinocchio owns the rigging)
│   ├── scene.py           # Mutable runtime state + dynamic ACM
│   └── geometry_cache.py  # Content-addressed disk cache
│
├── kinematics/
│   ├── fk.py
│   ├── jacobian.py
│   ├── singularity.py     # manipulability, condition number, min σ
│   └── ik/                # Multiple backends, pluggable (DLS, LM, optional analytical)
│       ├── solver.py      # DLS / LM solvers; returns candidate q
│       ├── validator.py   # joint limits + collision + singularity filter on candidates
│       └── result.py      # IKResult(q, validity_report, iterations, status)
│
├── collision/
│   ├── checker.py         # discrete-state collision query
│   ├── distance.py        # min distance + clearance
│   ├── continuous.py      # swept-volume / CCD between two configurations
│   └── attached_object.py # collision handling for objects rigidly attached at runtime
│
├── planning/
│   ├── joint_space/       # OMPL adapter: RRT, RRT-Connect, PRM, BIT*, ...
│   ├── cartesian/         # straight-line, arc (linear / circular in TCP space)
│   ├── state_validity.py  # in-planner state checker (collision + joint limits)
│   └── validator.py       # post-plan advisory: clearance, singularity, branch-jump
│
├── optimization/
│   ├── shortcut.py        # random shortcut smoothing
│   └── spline.py          # B-spline / quintic fit through waypoints
│
├── trajectory/
│   ├── time_parameterize.py # backend selected at implementation time
│   ├── sampler.py         # Trajectory.at(t) -> (q, qd, qdd)
│   └── validator.py       # collision over time, TCP speed, controller compat
│
├── primitives/            # Generic motion primitives (application-independent)
│   ├── move_j.py          # joint-space goto
│   ├── move_l.py          # cartesian straight-line
│   ├── move_c.py          # cartesian arc
│   ├── approach.py        # straight-line approach to a target pose
│   ├── retreat.py         # straight-line retreat from a pose
│   └── via_motion.py      # smooth motion through a sequence of waypoints
│
└── integrations/          # Optional thin adapters for non-core formats, if needed
```

**Out of scope for `algorithms`:** grasp planning, task sequencing, perception,
behavior trees, application state machines, controller drivers, transport
middleware, watchdogs, and recording. Those belong to application layers built
on top of this engine.

---

## 5. The 10-stage pipeline

| # | Stage | What happens | Module |
|---|---|---|---|
| 1 | FK | `q → T_world_frame` for any frame | `kinematics/fk.py` |
| 2 | IK | `T_target, q_seed → q_solution` | `kinematics/ik/` |
| 3 | IK validation | joint limits + collision filter on IK candidates | inside `kinematics/ik/` — solutions returned with validity flags |
| 4 | Path planning | joint-space (OMPL) or cartesian (linear/arc); planner calls `planning/state_validity.py` on every sample | `planning/joint_space/`, `planning/cartesian/`, `planning/state_validity.py` |
| 5 | Path validation | post-plan advisory: clearance, singularity, branch-jump | `planning/validator.py` |
| 6 | Path optimization | shortcut + spline smoothing | `optimization/` |
| 7 | Trajectory planning | `Path → Trajectory(q(t), qd(t), qdd(t))` | `trajectory/time_parameterize.py` |
| 8 | Trajectory validation | dense-time collision, TCP speed/accel, v/a/j envelope, command-rate compatibility | `trajectory/validator.py` |
| 9 | Controller integration | `Trajectory → robot wire format` | application layer, outside `algorithms`. **Handoff contract: §6.14.** |
| 10 | Runtime monitoring | tracking error, watchdog, live collision, recording | application layer, outside `algorithms`. **Handoff contract: §6.14.** |

### 5.1 Clarifications on the 10 stages

**Singularity check belongs in two places** — inside IK (filter solutions near singularity) and inside path validator (catch a path that crosses a singularity). Not in trajectory optimization; singularity is a kinematic property of `q`, not a timing property.

**In-planner vs post-plan validation.** Geometric checks (collision, joint limits) are not a separate step — the planner calls a state-validity-checker on every sample. The post-plan validator runs advisory checks only (clearance margin, singularity, branch-jump). For trajectories, the time-parameterization backend enforces its supported limits; the validator checks TCP speed, dense-time collision, and controller compatibility.

**Path optimization vs trajectory optimization.** Smoothing (shortcut + spline) operates on the joint path with no timing. Trajectory optimization (CHOMP/STOMP) combines smoothing with time parameterization under a cost function. v2 keeps trajectory-backend selection out of YAML; CHOMP/STOMP is out of scope until workload data proves it is required.

**Execution is outside the engine.** `algorithms` may produce a validated `Trajectory`, but converting it to robot-specific wire formats, running the streaming loop, and monitoring hardware belong to the application layer. The engine should expose clear data structures and validators; it should not own controller protocols.

**Attached-object handling** — when an object becomes rigidly attached to a robot link at runtime (e.g., grasped, clamped, picked up by a tool changer), its collision geometry moves with that link. This is handled in `Scene` (attach/detach) and `collision/attached_object.py`. It is a property of the engine, not of any specific application.

### 5.2 What flows between modules

| From → To | Data type | Shape |
|---|---|---|
| description → resolved | YAML files | text |
| resolved → kinematics | `KinematicModel`, q | object + `np.ndarray (dof,)` |
| kinematics IK → planning | q_goal | `np.ndarray (dof,)` |
| planning → optimization | `Path` | `np.ndarray (N, dof)` |
| optimization → trajectory | `Path` (smoothed) | `np.ndarray (N, dof)` |
| trajectory → primitives | `Trajectory` | callable `(t) -> (q, qd, qdd)` + duration |
| primitives → application | `Trajectory` (or sequence) | engine output consumed by caller |
| application → robot | streamed setpoints | outside `algorithms` |

**Everything inside the engine is NumPy.** Joint-name dicts appear only at the YAML and external-API boundaries.

### 5.3 Composition example

A generic "move from current state to a target pose with planning and obstacle avoidance":

```python
# Build once at process start
system   = RobotSystemDescription.from_yaml(...)
world    = WorldDescription.from_yaml(...)
model    = KinematicModel.from_robot_system(system)        # cached
coll_mdl = CollisionModel.from_world(world, model)         # cached
scene    = Scene.from_world(world, coll_mdl)

# At call time — perception or upstream code updates Scene
scene.set_object_pose("workpiece", T_world_object)
scene.set_robot_state("arm", q_current)

# Algorithm composition
q_goal  = ik.solve(model, scene, T_target, q_seed=q_current, chain_id="arm_with_gripper")
path    = planning.joint_space.plan(model, scene, q_current, q_goal, planner="RRTConnect", timeout=2.0)
path    = optimization.shortcut(path, model, scene)
path    = optimization.spline_fit(path)
planning.validator.check(model, scene, path)
traj    = trajectory.time_parameterize(path, model.kinematic_limits("arm_with_gripper"))
trajectory.validator.check(traj, model, scene)

# Application code decides how to execute or simulate the trajectory.
application.execute(traj)
```

Every step is a separate function that takes `model` and `scene` plus its specific inputs. The same composition pattern applies to any application — pick/place, machining, welding, inspection — by varying which target poses are chosen and which primitives are sequenced.

---

## 6. Methods inventory per stage

Catalog of algorithm options at each stage. Not all are required for v1 — this is the complete option space so we know what is available without having to re-research.

### 6.0 Glossary

The motion-planning vocabulary, locked. Every code file, every doc, every example uses these terms exactly. Confusing two of these terms is the most common source of design mistakes in motion-planning code, so this section is non-negotiable.

#### The three layers of motion data

```text
Layer 1: Configurations & Poses     one atom, one instant, no motion
Layer 2: Path                       N waypoints, ordered, geometry only
Layer 3: Trajectory                 function t -> (q(t), qd(t), qdd(t))
```

A **path** has geometry but no time. A **trajectory** has both. Path planning and path optimization both live in Layer 2. Time parameterization is the only step that bridges Layer 2 to Layer 3.

#### Locked vocabulary

| Term | Means | Does **not** mean |
|---|---|---|
| **Configuration** `q` | One joint vector at one instant. Length = `len(model.active_joint_names)`. | A state with velocity. |
| **Pose** `T` | One 4x4 homogeneous transform. | A motion. |
| **Waypoint** | A single `q` (or `(T, q)` in Cartesian) on a planned route. | A point with velocity or timestamp. |
| **Segment** | The geometric link between two consecutive waypoints. | A timed motion. |
| **Path** | Ordered sequence of waypoints; geometry only. | A trajectory. A time-stamped sequence. A single primitive shape. |
| **Trajectory** | Function `t -> (q(t), qd(t), qdd(t))` over `[0, duration]`. | A path. A list of waypoints with times. |
| **Sample** | One `(t, q, qd, qdd)` evaluated from a trajectory. | A waypoint. |
| **Time parameterization** | The process of turning a path into a trajectory. | Optimization. Smoothing. |
| **Optimization (smoothing)** | A geometric pass: Path in, Path out. | Time parameterization. |
| **Planner** | Algorithm that takes `(q_start, q_goal)` and produces a Path. | Trajectory generator. |
| **Backend** | Concrete implementation of a planner or time parameterizer. | The user-facing function. |
| **Validity function** | `q -> bool` combining joint limits + collision. Built once, called per sample. | A validator. |
| **Validator** | Function that takes a Path or Trajectory and returns a Report. | A planner. The validity function. |
| **PathPlanResult** | `(status, path, diagnostics)` returned by `plan_joint` / `plan_cartesian`. | A Path. |
| **TrajectoryResult** | `(status, trajectory, diagnostics)` returned by `time_parameterize`. | A Trajectory. |
| **State validity contract** | The signature `q -> bool` planners call on every sample. | A subscription or service. |

#### Path — invariants

A `Path` carries `waypoints: (N, dof)`, `joint_names`, optional `cartesian_waypoints: (N, 4, 4)` populated only by `plan_cartesian`, and a `metadata` dict.

1. `N >= 2`.
2. Every waypoint inside joint limits with margin.
3. Every segment collision-free at the planner's sampling resolution.
4. Adjacent waypoints satisfy a max-step constraint from the planner.

A path has **no velocity, no acceleration, no timestamp**. Until time parameterization runs, talking about "the velocity at waypoint 5" is a category error.

#### Trajectory — invariants

A `Trajectory` carries `duration`, `joint_names`, `backend_used`, and a private representation. Public access is through `at(t)` and `sample(dt)`.

1. `at(0)` equals the path start; `at(duration)` equals the path goal.
2. `q(t)` is continuous; Ruckig backend also gives continuous `qd(t)`.
3. `|qd(t)| <= v_max`, `|qdd(t)| <= a_max` everywhere (validator allows small numerical slack).
4. The trajectory's geometric path equals the path it was built from. Time parameterization re-times, doesn't re-route.

#### The pipeline (single canonical flow)

```text
   q_start, q_goal
        |
        v   PHASE 6a: Path Planner   (OMPL RRTConnect or StraightLine)
   Path of N waypoints
        |
        v   PHASE 6b: Path Validator  (advisory; planner already checked basics)
   Path validated
        |
        v   PHASE 6c: Path Optimizer  (shortcut + spline; Path -> Path)
   Path of M waypoints (M <= N typical)
        |
        v   PHASE 6d: Time Parameterizer  (Ruckig or Polynomial; Path -> Trajectory)
   Trajectory: function t -> (q, qd, qdd)
        |
        v   PHASE 6e: Trajectory Validator  (v/a/j envelopes + dense-time collision)
   Validated trajectory
        |
        v   trajectory.sample(dt) for streaming controllers
   (q[t], qd[t], qdd[t]) at the controller tick
```

Each phase has one input type, one output type, one validator. The boundaries are sharp.

#### Waypoint vs Sample (the single most important distinction)

| | Waypoint (path atom) | Sample (trajectory atom) |
|---|---|---|
| Has position `q`? | yes | yes |
| Has velocity `qd`? | **no** | yes |
| Has acceleration `qdd`? | **no** | yes |
| Has timestamp `t`? | **no** | yes |
| Defined before parameterization? | yes | no |
| What it answers | "where" | "where, when, how fast, accelerating how" |

A path of 10 waypoints, after Ruckig at `dt = 0.01` for a 2-second move, becomes a trajectory of ~200 samples. The 10 path waypoints don't appear as marked points on the trajectory — they constrained the geometry; the timing law is now smooth across them.

#### Interior-waypoint behaviour (locked default)

The library's default is **pass-through**: the time parameterizer keeps non-zero velocity at interior waypoints so the robot doesn't pause at every via point. Ruckig handles this by chaining segments with target-velocity matching; the polynomial backend uses Catmull-Rom finite differences for interior velocities. Rest-to-rest behaviour is opt-in via `TimeParameterizationOptions`.

### 6.1 FK
- Pinocchio recursive FK (`forwardKinematics` + `updateFramePlacements`).
- KDL recursive solver (alternative; less performant than Pinocchio).
- Custom DH-based FK (only for analytical work; never in production).

### 6.2 Jacobian
- Pinocchio body / world / local-world-aligned Jacobian (`computeJointJacobian`, `getFrameJacobian`).
- Geometric vs analytical Jacobian (rotation parameterization choice).
- Numerical Jacobian via finite differences (verification only).

### 6.3 IK

Implemented v1:

- **GenericConstrainedIK** — default backend. Multi-start bounded nonlinear
  least-squares using SciPy `least_squares(method="trf")`.
- **DLSIK** — opt-in damped least-squares fallback/debug backend.
- **QPVelocityIK** — Cartesian velocity IK for servo-style `qdot` output via
  `solve_velocity(...)`; it does not use the pose-IK `IKResult` path.
- **Analytical registry** — OPW and spherical-wrist 6R interfaces exist.
  They intentionally return `INVALID_INPUT` until robot-specific parameters
  or a compatible 6R structure are registered.

Dispatch order:

1. Explicit `backend="opw"` → OPW analytical backend.
2. Explicit `backend="spherical_wrist_6r"` → spherical-wrist analytical backend.
3. Registered robot-specific analytical solver → that solver.
4. Explicit `backend="dls"` → DLSIK.
5. No backend hint → GenericConstrainedIK.
6. Realtime Cartesian servoing → `solve_velocity(...)`, not `solve(...)`.

Every pose-IK candidate passes through `validator.validate(...)` before it can
be returned as `IKStatus.SUCCESS`.

`IKResult.status` is enumerated, never a boolean:

`SUCCESS`, `INVALID_INPUT`, `UNREACHABLE`, `MAX_ITERATIONS`, `TIMEOUT`,
`JOINT_LIMIT_VIOLATION`, `POSE_ERROR_TOO_HIGH`, `SINGULARITY_RISK`,
`FINAL_COLLISION`, `CONSTRAINT_VIOLATION`, `NO_VALID_CANDIDATE`,
`NUMERICAL_FAILURE`.

Future backend candidates:

- IPOPT / Ceres / NLopt behind the existing `NonlinearIKBackend` interface.
- Robot-specific closed-form solvers registered per robot id.
- TRAC-IK-style wrapper if field data shows the SciPy prototype is insufficient.

### 6.4 Singularity metrics

Computed from the Jacobian J:
- **Manipulability index** (Yoshikawa) = √det(J Jᵀ). Volume of the manipulability ellipsoid.
- **Condition number** = σ_max / σ_min. Sensitivity to inputs.
- **Inverse condition number** = σ_min / σ_max. More numerically stable.
- **Minimum singular value** σ_min. Direct distance-to-singularity proxy.
- **Dexterity / isotropy index** — ratio of ellipsoid axes.

All are one SVD of J. Cheap.

### 6.5 Collision

Broadphase (cull non-colliding pairs):
- AABB tree / BVH (what Coal uses).
- Sweep-and-prune.
- Spatial hashing.

Narrowphase (exact collision between two shapes):
- **GJK + EPA** — convex shape distance + penetration depth.
- **MPR** (Minkowski Portal Refinement) — alternative to EPA.
- Primitive-primitive analytical (box-box, sphere-sphere, capsule-capsule).

Distance queries:
- Signed distance (with normal).
- Closest-point pair.
- Clearance to a witness set.

Continuous collision detection (between configurations):
- **Linear shape cast** — sweep a shape along a linear trajectory.
- **Conservative advancement** — iteratively step until collision.
- **Bilateral advancement** — from both endpoints inward.
- **Sub-sampled discrete check** with adaptive density.

Coal provides broadphase + GJK/EPA + distance + linear-cast CCD.

### 6.6 Joint-space path planning

Sampling-based (all available via OMPL):
- **RRT** — single-tree, biased toward goal.
- **RRT-Connect** — bidirectional; typically fastest for free-space connection.
- **BiRRT** — bidirectional RRT variant.
- **RRT*** — asymptotically optimal.
- **PRM**, **PRM*** — roadmap; reuse across queries.
- **LazyPRM** — defer edge validation.
- **BIT*** (Batch Informed Trees) — fast asymptotic optimality.
- **FMT*** (Fast Marching Trees).
- **SBL**, **EST**, **KPIECE**, **LBKPIECE** — alternative samplers, sometimes better in narrow passages.
- **AIT***, **ABIT*** — anytime informed variants.

Search-based (for low-DOF or precomputed graphs):
- A*, weighted A*, ARA*, ANA* — on a discretized grid or roadmap.

Trajectory optimization-based (covered separately in 6.8).

### 6.7 Cartesian path planning

- **MoveL / linear** — TCP straight line in position; orientation interpolated (slerp).
- **MoveC / circular** — TCP arc through a via point.
- **MoveS / spline** — TCP follows a Cartesian spline (quintic, B-spline) through poses.
- **Blended motion** — corner blending between linear/arc segments to avoid stops.
- **Constrained Cartesian** — TCP confined to a manifold (line, plane, axis); planner respects the constraint while resolving redundancy.

Implementation pattern: sample TCP path → IK at each sample with previous q as seed → check joint-space continuity (branch-jump detection) → return joint path.

### 6.8 Path optimization (geometric, no timing)

- **Random shortcut smoothing** — pick two waypoints, try to connect directly; iterate.
- **Adaptive shortcut** — bias toward longer shortcuts first, fall back to shorter.
- **Cubic spline fitting** — C¹ continuity.
- **Quintic spline fitting** — C² continuity.
- **B-spline fitting** — variable smoothness via degree + knot vector.
- **Time-elastic band (TEB)** — local optimization with obstacle distance as cost.

### 6.9 Trajectory optimization (path + timing jointly)

- **CHOMP** — covariant gradient descent on smoothness + obstacle cost.
- **STOMP** — stochastic optimization, gradient-free.
- **TrajOpt** — sequential QP with continuous-time collision constraints.
- **GPMP2** — Gaussian process motion planner.

Out of scope for v1. Listed for completeness.

### 6.10 Trajectory generation (time parameterization of a fixed path)

Time-optimal under kinematic limits:
- **Jerk-limited online parameterization** — backend selected when implemented; not a schema commitment.
- **TOTG / TOPP-RA** — time-optimal path parameterization on a fixed geometric path; respects v/a limits but typically not jerk.
- **ISP** (Iterative Spline Parameterization) — used in MoveIt as default; legacy choice.

Polynomial profiles:
- **Cubic** — C¹ continuity, simple.
- **Quintic** — C² continuity (zero accel at endpoints).
- **Septic** — C³ continuity (zero jerk at endpoints).

Heuristic profiles:
- **Trapezoidal** velocity profile — bang-bang acceleration.
- **S-curve** — jerk-limited trapezoidal.
- **Smooth quintic-through-waypoints** — single continuous quintic spline through all waypoints with finite-difference interior velocities (no zero-velocity stops at viapoints).

Backend selection is an implementation detail. The public contract is a timed trajectory plus validators for limits and collision.

### 6.11 Validators

IK candidate validation:

- Backend convergence status.
- Finite active-q vector.
- Position limits with configured margin.
- FK position/orientation error against `PoseTarget` tolerances.
- Resolved target frame existence.
- Singularity thresholds (`min_sigma_limit`, `condition_number_limit`).
- Final collision when a `Scene` with `CollisionModel` is supplied.

Path-level (geometric, post-plan):
- Discrete state validity (collision + joint limits) at fine resolution.
- Continuous collision over each segment.
- Clearance margin against obstacles.
- Singularity metric along the path.
- Branch-jump detection (large Δq with small ΔTCP).
- Joint-velocity envelope estimate (Δq / Δt) if a nominal speed is given.

Trajectory-level (timed):
- v / a / j envelope check.
- TCP Cartesian speed / acceleration.
- Dense-time collision (at controller rate).
- Command-rate compatibility (does the trajectory have enough resolution at the controller's tick?).
- Effort / torque envelope (if dynamics available).

### 6.12 Controller output compatibility

`algorithms` does not execute robot commands, but its trajectory and
primitive outputs must be easy to adapt to common controller command shapes:

- **Joint trajectory action** — full `Trajectory` sent once; common on
  research robots and ROS-based stacks.
- **Streaming joint angles** — `q(t)` sampled at fixed rates such as
  100, 125, 250, 500, or 1000 Hz.
- **Streaming joint position / velocity / acceleration** — `q(t)`, `qd(t)`,
  and `qdd(t)` for controllers with feedforward support.
- **Cartesian pose / pose+twist streaming** — FK-derived TCP pose and twist
  for Cartesian impedance or servo modes.
- **Joint velocity command** — `qd(t)` or differential-IK output for velocity
  servoing.
- **Joint torque / effort command** — future dynamics-backed output; only
  valid once dynamics and effort validation are implemented.

The controller adapter lives outside `algorithms`. The engine must expose
enough typed data for that adapter to be deterministic: trajectory sampling,
limits, frame ids, TCP transforms, validation results, and compatibility
metadata such as required sample rate or derivative availability.

### 6.13 Application execution concerns

The engine does not implement robot drivers, streaming loops, watchdogs, or
hardware safety state machines. Applications that consume `algorithms`
outputs may need:

- ROS actions, libfranka loops, RTDE streams, EtherCAT processes, or custom
  robot transports.
- Position, velocity, impedance, force, or torque control mode selection.
- Tracking-error monitoring, heartbeat monitoring, E-stop integration, and
  timeout handling.

Those concerns should stay above the library. The engine's responsibility is to
produce typed descriptions, resolved models, kinematic/collision queries,
planned paths, trajectories, and validation results.

### 6.14 The handoff contract for stages 9 and 10

Stages 9 (controller integration) and 10 (runtime monitoring) live in
application code, not in `algorithms`. They are still part of the
architecture because the library must hand off enough typed data for
them to be built deterministically. This section locks **what the
engine exposes**, not **how the application uses it**.

#### Stage 9 — controller integration (handoff surface)

What an application needs from `algorithms` to drive a real robot:

| What | Type | Source | Why it matters |
|---|---|---|---|
| Trajectory samples | `Trajectory.at(t) -> (q, qd, qdd)` | `trajectory/sampler.py` | Streaming controllers tick at fixed rates; sampling must be deterministic and side-effect free. |
| Trajectory metadata | `Trajectory.duration`, `start_time`, `end_time` | `trajectory/` | Adapter sequencing, action goal duration. |
| Joint name order | `KinematicModel.active_joint_names` | `resolved/` | Wire-format slots are robot-specific; the adapter maps engine-order to wire-order using this list. |
| Joint limits | `KinematicModel.position_limits`, `velocity_limits`, `acceleration_limits`, `jerk_limits`, `effort_limits` | `resolved/` | Adapter clamps or rejects out-of-envelope commands before they reach the driver. |
| TCP metadata | `RobotSystemDescription.tcp(...)` | `descriptions/` | Cartesian streaming modes need the TCP frame id and offset. |
| Frame metadata | resolved-model frame names + `system.robot.base_frame` | `resolved/` | Adapter knows what `q` and TCP poses are expressed relative to. |
| Validation results | `TrajectoryValidationReport` from 6e | `trajectory/validator.py` | Adapter refuses to stream a trajectory that failed validation; the report says **which** check failed. |
| Compatibility metadata | required sample rate (Hz), derivative availability (`has_velocity`, `has_acceleration`), recommended controller mode | `trajectory/` | Stops an application from sending a 10 Hz trajectory to a 1 kHz controller, or feedforward-required commands to a position-only driver. |

What the engine **does not** provide and the application must own:

- Driver bindings (libfranka, RTDE, EtherCAT, ROS action clients).
- Controller mode selection (position / velocity / impedance / force).
- Wire-format encoding (CAN frames, ROS messages, ProtoBuf, etc.).
- Synchronisation across robots at the wire level.
- E-stop, recovery, hand-guiding, and other safety interlocks.

The handoff rule: an adapter is correct if, given the typed handoff
above, it can construct every wire-format field without reading
`algorithms` source.

#### Stage 10 — runtime monitoring (handoff surface)

Runtime monitoring is the loop that compares **what the controller is
doing** to **what the trajectory said it would do** and reacts. The
loop lives outside the library; the comparison primitives live inside.

| What | Type | Source | Why it matters |
|---|---|---|---|
| Reference samples | `Trajectory.at(t) -> (q, qd, qdd)` | `trajectory/sampler.py` | The "expected" side of the tracking comparison. |
| Live measured state | written into `Scene.robot_states[robot_id]` | `resolved/scene.py` | The "actual" side. Middleware writes it; the engine treats it as a typed runtime input. |
| Tracking-error metric | `q_measured - q_reference`, `‖p_measured - p_reference‖` via FK | application code calling `kinematics/fk` | Library exposes FK + sampler; the metric is one line of application code. |
| Live collision check | `is_in_collision(model, scene, q_measured)` | `collision/` | Detects unexpected contact during execution. Same call shape as planning-time. |
| Live distance margin | `min_distance(model, scene, q_measured)` | `collision/` | Watchdogs on shrinking clearance. |
| Live limits margin | `KinematicModel.position_limits` vs current `q_measured` | `resolved/` | Detect drift toward limits before the controller clamps. |
| Trajectory residual time | `Trajectory.duration - t_elapsed` | `trajectory/` | Timeout decisions. |

What the engine **does not** provide and the application must own:

- The monitoring loop itself (frequency, thread/process, priority).
- Tolerance thresholds (these are task-specific and belong to call
  sites, not YAML — same rule as §2.13).
- Reaction policies (stop, slow, replan, recover, alert).
- Logging / recording transport (MCAP, ROS bag, CSV).
- Heartbeat and watchdog timers.
- E-stop integration and safety-rated channels.

The handoff rule: a monitor is correct if it can be implemented as a
pure consumer of `(Trajectory, Scene, KinematicModel, CollisionModel)`
plus its own timer source and reaction policy.

#### Why this is in the architecture even though it is "out of scope"

The line between "library" and "application" is exactly the line where
the handoff surface lives. Documenting that surface is part of locking
the library's contract — without it, every integrator re-derives it
from the source and the contract drifts. Stages 9 and 10 are documented
here so the engine knows what it owes, and integrators know what they
get.

### 6.15 Attached-object handling

State changes:
- Attach: add object to a robot link, transfer collision geometry into the moving frame, auto-allow collisions between the link and the object.
- Detach: revert; object becomes a world-frame entity again.

Collision implications:
- Attached object's geometry must be transformed by FK at query time.
- Allowed-pairs overlay applies until detach.

---

## 7. Library inventory

All Python at the API layer; speed comes from the C++ libraries underneath.

| Concern | Library | Why |
|---|---|---|
| Kinematics, FK, Jacobian, dynamics | Pinocchio (C++) | µs-per-call. Composed model via `appendModel`. |
| Collision | Coal (C++, formerly HPP-FCL) | Broadphase + narrowphase + distance + linear CCD. |
| Sampling-based planning | OMPL (C++) | All planners listed in 6.6. |
| Time parameterization | Backend TBD | Should respect configured velocity, acceleration, and jerk limits. |
| Mesh I/O, V-HACD decomposition | Trimesh, Open3D | Mesh loading and convex decomposition. |
| Rotations, math (boundaries only) | SciPy | Used at API and validation boundaries where useful. |
| Schema | Pydantic, PyYAML | All YAML loading and validation. |

---

## 8. Build order

Each numbered phase is a module boundary. Sub-phases (5a, 5b, 6a, …) are
ordered deliverables inside that boundary. A sub-phase only lands when
its predecessors land — validators are first-class deliverables, not an
afterthought. Pipeline-stage column maps each deliverable back to
§5's 10-stage pipeline so nothing is silently dropped.

| Phase | Sub-phase | Module | Deliverable | Pipeline stage | Status |
|---|---|---|---|---|---|
| 1 | — | `descriptions/` | Schema v2 loads to typed objects. | — (infrastructure) | **Done** |
| 2 | — | `resolved/` | Composed Pinocchio model, mimic expansion, limits API, Scene with dynamic ACM. The unblocker. | — (infrastructure) | **Done** |
| 3 | — | `kinematics/fk`, `jacobian`, `singularity` | Pure functions on `KinematicModel`. | 1 | **Done** |
| 4 | — | `collision/` | Coal-backed queries on `Scene`, including attached objects. | (consumed by 3, 5b, 6b, 6e) | **Done** |
| 5 | 5a | `kinematics/ik/solver.py`, `backends/`, `constraints/`, `costs/` | Analytical interfaces (OPW, spherical-wrist 6R, registry), `GenericConstrainedIK` (multi-start bounded NLS, default), optional DLS, `QPVelocityIK` for servoing. | 2 | **Done** |
|   | 5b | `kinematics/ik/validator.py`, `result.py` | Mandatory post-solve checks (finite, joint bounds, pose error, singularity), optional final collision; `IKResult` wraps `q` with an enumerated status + diagnostics. | 3 | **Done** |
| 6 | 6a | `planning/joint_space.py`, `planning/cartesian.py`, `planning/state_validity.py`, `planning/backends/{ompl,straight_line}.py` | `plan_joint` (OMPL RRTConnect default + straight-line backend); `plan_cartesian` (straight TCP line + IK continuity check); single + composite state validity factories; multi-robot collision lift in `collision/_runtime.py`. | 4 | **Done** |
|   | 6b | `planning/validator.py` | `validate_path` post-plan advisory: joint limits, continuous collision, clearance, singularity, branch-jump, velocity-envelope estimate. | 5 | **Done** |
|   | 6c | `optimization/shortcut.py`, `optimization/spline.py` | `shortcut_smooth` (random shortcut with collision-aware validity check), `remove_redundant_waypoints`, `spline_fit` (cubic/quintic with Catmull-Rom interior velocities). Path in, Path out — purely geometric. | 6 | **Done** |
|   | 6d | `trajectory/time_parameterize.py`, `trajectory/trajectory.py`, `trajectory/backends/{polynomial,ruckig_backend}.py` | `time_parameterize` — pass-through by default (Catmull-Rom interior velocities; robot does NOT stop at waypoints). Two backends: PolynomialBackend (C^2 quintic, always available) and RuckigBackend (jerk-limited, opt-in, local-only — no cloud API). `Trajectory.at(t)` + `sample(dt)` for streaming controllers. | 7 | **Done** |
|   | 6e | `trajectory/validator.py` | `validate_trajectory` — dense-time joint-limits, v/a/j envelopes, collision, optional TCP linear/angular speed, controller-rate compatibility. | 8 | **Done** |
| 7 | — | `primitives/{move_joint,move_l,approach,retreat,via_motion}.py` | Five primitives shipped: `move_joint` (joint-space goto), `move_l` (linear Cartesian), `approach` (linear descent), `retreat` (linear lift-off), `via_motion` (pass-through across N via-points). Each composes 5b → 6a → 6b → 6c → 6d → 6e and returns a validated `Trajectory`. `move_c` (arc) reserved for future. | (composes 2–8) | **Done** |
| 8 | 8a | (application layer, outside library) | Controller adapter consumes `Trajectory` + compatibility metadata from §6.12. `algorithms` exposes the typed handoff surface; no driver code in the library. | 9 | Out of scope for library |
|   | 8b | (application layer, outside library) | Runtime monitoring consumes `Trajectory.sampler` + validation reports from 6e + live `Scene` updates. `algorithms` exposes the comparison primitives (sampled reference vs measured); no monitor loop in the library. | 10 | Out of scope for library |

Phases 1–3 are pure infrastructure with no algorithmic risk. Once they
land, every later sub-phase is a single bounded module with a clear
input/output contract. Validators (5b, 6b, 6e) are deliverables, not
optional polish — a sub-phase that ships its solver/planner/trajectory
without its validator is **not** considered complete.

Phase 8 is outside the library. It is included in the table only to
make the engine's responsibility to its consumers explicit: §6.12 and
§6.14 define what the library must hand off so 8a/8b can be built
without re-reading source.

---

## 9. Open questions

Flag and revisit when relevant; not blocking v1.

### Resolved during Phase 4 collision implementation

- **World mesh loading.** Mesh world objects are loaded eagerly in
  `CollisionModel.from_world` through `coal.MeshLoader`; the placeholder
  `coal.Box(1,1,1)` is gone. Optional mesh processing is routed through
  `resolved/geometry_cache.py`, keyed by mesh bytes, scale, and processing
  configuration.
- **Collision query layering.** `collision/` contains only low-level geometry
  queries. Joint limits, singularity, custom constraints, planning validity,
  and trajectory-time checks belong to future `validity/`, `planning/`, and
  `trajectory/` layers.

### Resolved during FK/Jacobian implementation

- **Per-call Pinocchio data.** `KinematicModel` does not store mutable `pin.Data`.
  FK and Jacobian calls allocate fresh scratch data per query, so cached
  `KinematicModel` instances are safe to share across multiple world robots and
  interleaved calls.

### Long-term / nice-to-have

- **Dynamics support.** Pinocchio gives RNEA / ABA for free. Decide later whether torque-aware planning is needed; if yes, `model.effort_limits` becomes load-bearing and a `dynamics/` module joins `kinematics/`.
- **Multi-arm coordinated motion.** Plan two arms simultaneously vs sequentially — affects `planning/` API shape.
- **Async planning.** Run planner in background while the application handles another task. Out of scope for v1.
- **Recorder/export format.** CSV vs MCAP vs Parquet — decide only if snapshot or trajectory logging becomes part of this library.
- **Additional IK backends.** TRAC-IK, EAIK, analytical solvers — add only if Pinocchio DLS/LM is empirically insufficient.
- **Trajectory optimization** (CHOMP/STOMP). Implement only if shortcut + time parameterization is empirically insufficient on real workloads.
