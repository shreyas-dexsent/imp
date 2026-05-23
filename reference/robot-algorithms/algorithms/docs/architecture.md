# Architecture

This document describes the design of `algorithms`, the contracts each
layer must satisfy, and the rationale behind the architectural decisions.

The implementation is locked at the level of these contracts. Collision,
planning, optimization, trajectory generation, and motion primitives are built
on top of the resolved-object layer rather than by redesigning it.

> This is the polished public-facing summary. The canonical long-form
> planning document — including the YAML schema, the 10-stage pipeline,
> methods inventory per stage, the build order with sub-phases, and open
> questions — is [plan.md](plan.md). When the two disagree, `plan.md`
> wins.

## 1. Three-layer separation

```
YAML files
    │
    │  parsing + validation, no computation
    ▼
Layer 1: descriptions/  ──  RobotSystemDescription, WorldDescription
    │
    │  one-time URDF parsing, Pinocchio model construction,
    │  Coal geometry, mimic analysis, limit resolution
    ▼
Layer 2: resolved/      ──  KinematicModel, CollisionModel, Scene
    │
    │  stateless operations on resolved objects
    ▼
Layer 3: operations     ──  kinematics/, collision/, planning/,
                            optimization/, trajectory/, primitives/
```

**Layer 1 — descriptions.** Pydantic models parsed from YAML. Pure data
with validation. No URDF reading, no NumPy work, no Pinocchio / Coal
interaction. Stable across solver tuning because no algorithm
configuration lives here.

**Layer 2 — resolved.** Heavy typed objects built from descriptions.
This is the expensive layer — URDF parsing, Pinocchio model
construction, mimic-joint analysis, joint limit resolution, Coal
geometry, mesh decomposition. Built once at process start, cached, and
reused by every operation.

**Layer 3 — operations.** Stateless functions on resolved objects.
Each operation takes a resolved object (and optionally a `Scene`) plus
inputs, and returns NumPy arrays. No operation stores state, holds a
long-lived `pin.Data`, or caches results.

## 2. Description data is static; runtime state lives in `Scene`

`WorldDescription` is immutable after load. The pose declared for an
object in YAML is the **description-time default**. At runtime the
authoritative pose lives in `Scene.object_poses`, which perception (or
any external state source) writes into. Operations always read from
`Scene`, never from `WorldDescription` directly.

| State category | Lifetime | Where |
|---|---|---|
| Physical facts (URDF references, mounts, joint limits, geometry) | Build time | YAML + `descriptions/` |
| Resolved heavy data (Pinocchio model, Coal shapes) | Process | `KinematicModel`, `CollisionModel` |
| Live runtime state (poses, joint configurations, attachments) | Tick | `Scene` |

This split has two consequences:

1. **YAML is never edited at runtime.** Perception, control, and planning
   all flow through `Scene`. If a piece of information would need to
   change at runtime, it does not belong in YAML.
2. **The collision query layer reads runtime poses from
   `Scene` and pushes them into Pinocchio's `GeometryData` scratch
   buffer on every call.** The scratch buffer never persists state.

## 3. One composed Pinocchio model per robot system

`KinematicModel.from_robot_system` builds **one** Pinocchio model that
covers the robot and its (optional) gripper. The composition uses
`pin.appendModel` with the YAML-declared mount transform.

A single model means:

* One FK call evaluates the whole chain.
* One Jacobian covers the combined `arm + finger` chain.
* IK over the combined chain (Phase 5) is a single nonlinear problem,
  not a stitched pair.

## 4. Mimic joints become one active degree of freedom

A URDF `<mimic>` joint (typically the second finger on a parallel-jaw
gripper) is exposed as **zero active DOF**. The driving joint is
exposed; the follower is internal.

A linear expansion map handles the conversion:

```
q_full = active_to_full @ q_active + active_offset
```

Operations apply this internally:

* FK calls `model.expand(q_active)` before invoking Pinocchio.
* Jacobian folds Pinocchio's full-DOF columns back to active-DOF columns
  via `J_full @ active_to_full`.

Users only ever see active DOF. For Franka FR3 + Franka hand, that is
**8** active joints (7 arm + 1 active finger), not 9.

## 5. URDF is canonical; YAML overrides

Limit resolution:

* **Position, velocity, effort.** URDF supplies the default. YAML
  `joint_limits` overrides per joint where present.
* **Acceleration, jerk.** URDF does not carry these. YAML is the source.
  For any joint that participates in a kinematic chain (and therefore
  in trajectory generation downstream), YAML must supply both.
  Resolution raises at build time if missing.

The resolution layer enforces this contract; downstream operations
assume the limits are present and well-defined.

## 6. `KinematicModel` is the cache boundary

```
RobotSystemDescription (pydantic, ~ms)
       │
       │  cached: keyed on (yaml_path, urdf_mtimes)
       ▼
KinematicModel  (heavy, ~100 ms)
       │
       ▼  every operation
fk(...), jacobian(...), ...
```

The cache is module-level. The key includes URDF file mtimes so editing
a URDF auto-invalidates the cache on the next call. The cache is sized
at 16 entries (LRU), enough for any realistic robot cell.

A `KinematicModel` may back multiple world robots of the same type. The
cache returns one instance for two world robots that share a YAML.

## 7. `pin.Data` is per-call, never stored

A `KinematicModel` deliberately does **not** store a `pin.Data` companion
buffer. Two world robots backed by the same cached model would otherwise
share one mutable `pin.Data`, and interleaved FK calls would clobber
each other.

Every operation allocates a fresh `pin.Data` via
`model.pin_model.createData()`. The cost is sub-microsecond. This is
the same pattern `pinocchio.RobotWrapper` follows internally.

## 8. Multi-robot worlds require namespaces

A world with one robot may use a null namespace. A world with more than
one robot requires every robot to have a non-null, unique namespace
matching the regex `^[a-z][a-z0-9_]*$`. The namespace is prepended to
geometry and frame names in collision data and other places where name
collisions would otherwise occur.

Frame names in the kinematic API stay resolved-model local (e.g.
`"fr3_link8"` or `"fr3_hand_tcp"`). They may come from URDF frames or
YAML-injected TCP frames. The world-level
`fk(scene, robot_id, q, frame_id)` composes the robot's `base_pose` into
the result. Callers never need to write `"left/fr3_link8"` by hand for
FK queries.

## 9. Collision data ownership: Coal vs Pinocchio

Two libraries do two jobs:

| Library | Role |
|---|---|
| Coal (formerly HPP-FCL) | Shape data and collision math (GJK, EPA, distance, CCD). |
| Pinocchio | The rigging — which shape attaches to which joint frame, what the local offset is, what pairs to check. Pushes FK results into geometry world poses. |

`pin.GeometryModel` is a **catalogue type**: a list of `GeometryObject`s
plus a list of collision pairs. It does not force FK. A `GeometryObject`
with `parent_joint = 0` (universe) is by construction static — its
runtime pose comes from the `Scene`, not from FK.

The collision query layer populates per-query geometry placements from
three sources before invoking Coal:

| Geometry origin | Pose source | Mechanism |
|---|---|---|
| Robot link | query `q` → FK | `pin.updateGeometryPlacements` |
| Free-standing world object | `Scene.object_poses[obj]` | direct write into `gd.oMg[idx]` |
| Attached object | FK on parent frame, composed with `T_parent_obj` | FK + matrix multiply, then write |

The implementation materialises active name pairs, writes current
placements into per-call scratch objects, and invokes Coal's narrowphase
for collision or distance. Coal only sees shapes plus current transforms;
it neither knows nor cares where each transform came from. Three pose
sources, one query path.

Mesh world objects are loaded eagerly in `CollisionModel.from_world`.
If a mesh declares geometry processing, the result is routed through the
content-addressed geometry cache before Coal loading. The old placeholder
mesh path is gone.

## 10. Three-layer Allowed Collision Matrix (ACM)

Collision allowances live at the layer where the pair makes sense:

| Class | Example | Where |
|---|---|---|
| Robot-internal static | gripper fingertip ↔ palm | robot YAML `collision.allowed_pairs` |
| World static | bin permanently sits on table | world YAML `collision_matrix.rules` |
| Task-driven dynamic | end-effector ↔ workpiece during grasp | Scene runtime |

The first two land in `CollisionModel.static_allowed_pairs` at build
time. The third lives in `Scene.collision_overlay` and is mutated at
runtime via `scene.allow_collision`, `scene.disallow_collision`, and
implicitly via `scene.attach` / `scene.detach`.

`scene.is_pair_allowed(a, b)` returns the effective allowance,
combining the static rules and the dynamic overlay. Precedence (highest
first): dynamic disallow, dynamic allow, static allow.

## 11. `Scene` is a state container, not middleware

`Scene` is a typed in-process state container. It does not subscribe to
ROS topics, queue messages, time-stamp updates, lock across threads, or
handle network failures. Those are middleware concerns and belong to
application code that writes into the scene.

The analogy with MoveIt:

| MoveIt | `algorithms` | Where it lives |
|---|---|---|
| `planning_scene::PlanningScene` | `Scene` | inside the library |
| `PlanningSceneMonitor` | application middleware glue | outside the library |

Keeping `Scene` transport-agnostic is what lets the same engine run on
ROS2, libfranka direct, plain TCP, or any other transport.

## 12. Stateless Operation Contract

Every operation in `kinematics/` follows the same contract:

* Takes a resolved object (`KinematicModel`) and optionally a `Scene`.
* Takes inputs as NumPy arrays.
* Returns NumPy arrays.
* Allocates any scratch buffers (`pin.Data`) per call.
* Holds no state of its own; caches no results.

This contract makes the operations trivially correct in multi-robot and
concurrent settings — there is no shared mutable state to corrupt.

The same contract extends to collision and planning operations. Future
operations may add input arguments such as a `CollisionModel` for collision
queries or planner-specific kwargs for `plan_joint`, but the stateless
shape is fixed.

## 13. IK Solver Dispatch and Validation

Pose IK lives in `kinematics/ik/` and follows the same stateless
operation contract as FK, Jacobian, singularity, and collision:

```python
result = solve(model, target, q_seed, options=..., scene=...)
```

The dispatch order is fixed:

1. Explicit OPW request.
2. Explicit spherical-wrist 6R request.
3. Registered robot-specific analytical solver.
4. Explicit DLS request.
5. Default `GenericConstrainedIK`.

OPW and spherical-wrist backends currently provide the interface and
structured unsupported-robot response. They need robot-specific
parameters or a matching robot structure before they can return branches.

`GenericConstrainedIK` is the v1 default. It uses deterministic
multi-start bounded nonlinear least-squares over the active-q vector.
Every candidate passes through `validator.validate(...)` before it can
be returned as `IKStatus.SUCCESS`.

Success guarantees:

* finite active `q`
* active joint position limits with margin
* final FK pose within `PoseTarget` tolerance
* singularity threshold check when enabled
* final collision check when a `Scene` with a `CollisionModel` is supplied

Success does not guarantee path collision-freeness, velocity,
acceleration, jerk, torque, controller compatibility, human-safe
execution, or cycle-time optimality. Those are validity, planning,
trajectory, and application concerns.

Realtime Cartesian servoing uses `solve_velocity(...)`, which returns
`qdot` directly and does not return `IKResult`.

## 14. What is NOT in `algorithms`

These are deliberately out of scope:

* **Application-level concerns** — grasp planning, task sequencing,
  perception, behaviour trees. These live in application layers built
  on top of `algorithms`.
* **Middleware glue** — ROS / DDS / EtherCAT / libfranka bindings.
* **Visualization** — meshcat / RViz / custom viewers. Read from
  `Scene` and `descriptions` directly, build viewers outside the engine.
* **Algorithm configuration in YAML** — planner type, IK tolerance, dt,
  smoothing iterations. These are call-site arguments.

## 15. Controller Output Compatibility

`algorithms` does not implement robot drivers or streaming loops. It should,
however, keep its trajectory and primitive outputs compatible with common
controller command shapes:

* Joint trajectory action: a full `Trajectory` sent once.
* Streaming joint positions: `q(t)` sampled at a fixed rate.
* Streaming joint position / velocity / acceleration: `q(t)`, `qd(t)`,
  and `qdd(t)` when derivatives are available.
* Cartesian pose / pose+twist streaming: TCP pose and twist derived from
  FK and Jacobians.
* Joint velocity command: velocity-servo or differential-IK outputs.
* Joint torque / effort command: future dynamics-backed output.

The adapter that turns those outputs into ROS actions, libfranka calls,
RTDE setpoints, EtherCAT packets, or proprietary messages lives outside the
library. The library contract is to provide deterministic sampled data,
limits, frame metadata, TCP metadata, and validation results.

## 16. Layering: Collision vs Validity vs Planning vs Trajectory

The collision package is intentionally low level:

```text
collision/   = geometry queries
validity/    = state, edge, and path acceptability
planning/    = search using validity
trajectory/  = time and execution validation
```

`collision/` answers only geometry questions:

* Is this q in contact?
* What is the closest active pair?
* Which active pairs are below a clearance threshold?
* Does this sampled joint-space edge hit anything?

It does not check joint limits, singularity, velocity, acceleration, jerk,
torque, controller compatibility, or human-safe execution. Future validity
functions compose collision with those checks.

## 17. Multi-robot collision and path planning

### What the stack handles today

`descriptions/`, `resolved/`, and `kinematics/` are already multi-robot.
A `WorldDescription` may declare N robots, each with a unique namespace
and `base_pose`; `Scene.robot_states[robot_id]` carries one `q` per
robot; `fk(scene, robot_id, q, frame)` composes the per-robot base pose
correctly. `CollisionModel` already namespaces robot geometry names so
two FR3s in one world never collide on a name like `fr3_link5_0`.

The collision query layer (`collision/`) is the **only** part of the
stack that currently narrows to a single robot. The narrowing lives in
one place: `collision._runtime._single_robot_base_pose`, which raises if
`len(scene.world.robots) != 1`. Multi-robot collision checking and
multi-robot path planning are therefore **not yet supported** as a
single call. Today they have to be approximated by:

1. Treating one robot as "the planning robot" and freezing the others
   into the scene as static obstacles via `Scene.object_poses` (loses
   inter-robot coupling — the other arm cannot react).
2. Or running per-robot queries with the other robots' meshes copied in
   as world objects (works for one-shot checks; will not scale to a
   true joint planner).

Neither is a true multi-robot collision query. Both are workarounds.

### Where multi-robot belongs in the layering

Multi-robot is **not** a planner concern; it is a collision-query
concern that the planner then consumes. The split:

| Concern | Layer | Form |
|---|---|---|
| Joint composite state of N robots | `Scene` (already exists) | `Scene.robot_states: dict[robot_id, q]` |
| Geometry placement for all robots + world + attached | `collision/_runtime` | one `oMg` buffer with every robot's links written |
| "Is the composite state in collision?" | `collision/is_in_collision` | one call, all pairs |
| "Plan a collision-free joint path" | `planning/` (Phase 6) | calls validity, which calls collision |

The architectural rule: **the planner never iterates over robots.** It
queries one collision function with a composite state, and that one
function knows how to place every robot's geometry into one shared
buffer. This keeps the planner agnostic to robot count (1, 2, or N) and
keeps the stateless-operation contract intact.

### Required changes to lift the single-robot gate

These are small and localised. They do not touch any layer-boundary
contract:

1. **Composite state object.** Introduce a typed `SceneState`
   (or accept `Scene.robot_states` directly) that maps `robot_id -> q`.
   Callers pass one of those instead of a bare `q`.
2. **`_runtime.geometry_entries` walks all robots.** Replace the
   single-robot guard with a loop:
   * for each `world_robot` in `scene.world.robots`, fetch its cached
     `KinematicModel`, allocate a per-robot `pin.Data` and
     `pin.GeometryData`, run FK with that robot's `q`, and write
     `T_world_base[robot] @ oMg[link]` into the entry table.
   * Attached objects: look up `attached.robot_id` (new field on
     `AttachedObject`) to pick the correct parent FK.
3. **`active_pairs` cross-robot filter.** The existing single-robot ACM
   already disallows robot-internal pairs the user marks as "always
   ignored". For multi-robot, a fresh class of pairs appears:
   `robot_A.link_i  vs  robot_B.link_j`. By default these are checked.
   YAML `collision_matrix.rules` already supports namespaced names, so
   no schema change is needed.
4. **`is_in_collision`, `min_distance`, `clearance`, `check_edge_collision`**
   take the composite state. The body is unchanged — the work is all
   in `_runtime`.
5. **Edge collision sampling** interpolates each robot's `q`
   independently between `q_a[robot]` and `q_b[robot]` with the same
   alpha schedule (`max_joint_step` becomes per-robot, broadphase still
   shared).

What does **not** change: `KinematicModel`, `CollisionModel`, the ACM,
the geometry cache, FK, Jacobian. The cache contract ("one
`KinematicModel` may back many world robots of the same type") was
designed for exactly this case.

### Path planning over multiple robots

Once multi-robot collision checking exists, joint path planning splits
into three increasingly cooperative strategies. All three sit in
`planning/` (Phase 6) and call into the same multi-robot collision
function.

| Strategy | What it plans over | Use when |
|---|---|---|
| Prioritised / sequential | One robot at a time, others frozen | Robots rarely interact; cheap |
| Decoupled with rechecks | Per-robot plans + a composite collision re-check pass | Robots share workspace but goals are independent |
| Composite-space | `q = concat(q_robot_1, ..., q_robot_N)` planned jointly | True cooperative motion (handoff, dual-arm grasp) |

The first two are layered application policies on top of
single-robot planning + the new multi-robot collision call. Only the
third requires a planner that natively understands a composite
configuration space; OMPL handles this as a Cartesian-product
`StateSpace`, so no library swap is needed.

### Plan summary (during Phase 6)

The cleanest sequencing is:

1. Replace the single-robot gate in
   `collision/_runtime.py` with the multi-robot walk above. Add tests
   that mirror the existing `test_fk_world_independent_per_robot_in_multi_world`
   pattern, but for collision: two FR3s, same `q`, different
   `base_pose`, verify they collide when placed too close and don't
   when placed apart.
2. Then build `planning/` with the composite-state signature from day
   one — never ship a single-robot planner API that a multi-robot
   version has to break.

Doing it in that order means the planner is born multi-robot. Doing it
the other way around (single-robot planner first, multi-robot retrofit
later) is the path that has historically forced rewrites in
MoveIt-class libraries.

## 18. Build Phases

| Phase | Modules | Status |
|---|---|---|
| 1 | `descriptions/` | Done |
| 2 | `resolved/` (KinematicModel, CollisionModel, Scene, geometry_cache) | Done |
| 3 | `kinematics/` (fk, jacobian, singularity) | Done |
| 4 | `collision/` (is_in_collision, min_distance, clearance, sampled edge CD) | Done |
| 5 | `kinematics/ik/` (GenericConstrainedIK, DLS, QP velocity IK, analytical interfaces, validator) | Done |
| 6 | `planning/`, `optimization/`, `trajectory/` | Done |
| 7 | `primitives/` (generic motion primitives) | Done |
| 8 | Application integration | Out of scope for `algorithms`; implemented above the library |

The locked contracts at the layer boundaries are stable: descriptions
parse YAML, resolved objects are built once and cached, operations are
stateless. Future work should extend the operation set without changing
those contracts.
