# YAML schema reference

`algorithms` reads two kinds of YAML file:

* **Robot system YAML** — declares one robot, an optional gripper, the
  joint limits, the named TCPs, the kinematic chains, and the named
  joint states.
* **World YAML** — places robots in a world, declares world objects
  (workpieces, fixtures, bins, obstacles), and sets static collision
  rules.

Both schemas are validated by pydantic at load time. Unknown fields are
rejected. Path references are relative to the YAML file's location.

## Robot system schema

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
    # Optional per-joint overrides on top of URDF.
    # Acceleration and jerk MUST be supplied for any joint that
    # appears in a kinematic chain (URDF does not carry them).
    fr3_joint1: { acceleration: 15.0, jerk: 7500.0 }
    fr3_joint2: { acceleration: 7.5,  jerk: 3750.0 }
    # ... one entry per joint that participates in a trajectory chain ...
    fr3_joint4:
      position: [-2.8, -0.1]    # narrow the URDF range
      velocity: 1.5             # narrow the URDF velocity limit
      acceleration: 12.0
      jerk: 6250.0

  collision:
    enabled: true
    source: urdf
    allowed_pairs:
      - { a: fr3_link7, b: fr3_link8, reason: "adjacent links" }
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
      acceleration: 1.0
      jerk: 100.0
    # fr3_finger_joint2 is a URDF <mimic> joint — do NOT declare it.

  collision:
    enabled: true
    source: urdf
    allowed_pairs: []
    disabled_links: []

tcps:
  - id: robot_tcp
    transform:
      parent_frame: fr3_link8
      child_frame: robot_tcp
      matrix: [[1,0,0,0],[0,1,0,0],[0,0,1,0.12],[0,0,0,1]]
  - id: hand_tcp
    transform:
      parent_frame: fr3_hand
      child_frame: fr3_hand_tcp
      matrix: [[1,0,0,0],[0,1,0,0],[0,0,1,0.1034],[0,0,0,1]]

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
    joints:
      - fr3_joint1
      - fr3_joint2
      - fr3_joint3
      - fr3_joint4
      - fr3_joint5
      - fr3_joint6
      - fr3_joint7
      - fr3_finger_joint1
    # Mimic followers omitted; resolution expands them internally.

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

### Field reference

#### `robot`

| Field | Type | Notes |
|---|---|---|
| `id` | string | Identifier referenced elsewhere. |
| `urdf_path` | string | Relative or absolute. Relative paths resolve against the YAML's directory. |
| `package_dirs` | list[string] | Additional URDF mesh search roots (defaults to the URDF's parent dir). |
| `base_frame` | string | Frame considered the robot's base. `fk_local` returns poses relative to this. |
| `joint_limits` | dict | Map from joint name to optional overrides. See below. |
| `collision` | object | Within-robot static collision configuration. |

#### `joint_limits[joint_name]`

| Field | Type | URDF source? | Required in YAML? |
|---|---|---|---|
| `position` | `[lower, upper]` | yes | no — override only |
| `velocity` | float | yes | no — override only |
| `effort` | float | yes | no — override only |
| `acceleration` | float | **no** | **yes**, for any joint used in a chain |
| `jerk` | float | **no** | **yes**, for any joint used in a chain |

The resolution layer raises at build time if a chain joint lacks
`acceleration` or `jerk`.

#### `collision`

| Field | Type | Notes |
|---|---|---|
| `enabled` | bool | Default `true`. |
| `source` | `"urdf"` | Only URDF collision geometry is supported in v2. |
| `allowed_pairs` | list[{a, b, reason?}] | Within-robot pairs that may be in contact by design. |
| `disabled_links` | list[string] | Link names whose collision geometry is omitted entirely. |

Task-driven dynamic allowances (end-effector ↔ workpiece during a grasp)
belong to the runtime `Scene`, not to YAML.

#### `gripper`

Same as `robot`, plus:

| Field | Type | Notes |
|---|---|---|
| `root_frame` | string | The gripper's root link name (must exist in the gripper URDF). |
| `mount` | TransformSpec | Static transform from the robot's parent frame to `root_frame`. |

#### `tcps`

A list of named Tool Center Point transforms. Each TCP is a
`TransformSpec` relative to a parent frame. Multiple TCPs may coexist
(e.g. a fingertip TCP and a palm TCP).

#### `kinematic_chains`

Each chain defines an ordered group of active DOF. `q` vectors for the
chain are NumPy arrays in `joints` order.

| Field | Type | Notes |
|---|---|---|
| `id` | string | No default; every chain must be named explicitly. |
| `base_frame` | string | First frame of the chain. |
| `tip_frame` | string | Last frame of the chain. |
| `tcp_frame` | string, optional | TCP id this chain effectively targets. |
| `joints` | list[string] | Active joints in order. Mimic followers are NOT listed. |

#### `named_joint_states`

Map from name (e.g. `"home"`, `"safe_pose"`) to a `{joints: {...}}` dict.
`joints` is a `{joint_name: value}` map. The legacy parallel-list form
(`names: [...], positions: [...]`) is rejected by the v2 schema.

## World schema

```yaml
schema: dexsent.algorithms.world
version: 2
id: franka_table_world
name: Franka Table World
world_frame: world

robots:
  - id: arm
    robot_system: ../robots/franka_fr3_with_franka_hand.yaml
    namespace: null
    base_pose:
      parent_frame: world
      child_frame: base
      matrix: [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]

objects:
  - id: matka
    type: workpiece     # one of: workpiece | obstacle | fixture | bin
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
      origin:
        # Optional offset from object frame to collision-mesh frame.
        # Useful when the collision mesh was authored or decomposed
        # with a different centre than the visual mesh.
        parent_frame: matka
        child_frame: matka_collision
        matrix: [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
      processing:
        type: convex_decomposition
        max_hulls: 16

collision_matrix:
  default_action: check
  rules:
    - { a: floor, b: wall_a, action: allow, reason: "permanent contact" }
```

### Field reference

#### `robots[i]` (a world robot)

| Field | Type | Notes |
|---|---|---|
| `id` | string | Unique per world. |
| `robot_system` | string | Path to a robot-system YAML. Sub-loaded at world load. |
| `namespace` | string, optional | Required non-null and unique if the world has more than one robot. Regex: `^[a-z][a-z0-9_]*$`. |
| `base_pose` | TransformSpec, optional | Where the robot's base sits in the world. Defaults to identity. |

#### `objects[i]` (a world object)

| Field | Type | Notes |
|---|---|---|
| `id` | string | Unique per world. |
| `type` | `workpiece` \| `obstacle` \| `fixture` \| `bin` | Application-level tag. `"attached"` is **not** a valid YAML value — it is a runtime-only Scene state. |
| `pose` | TransformSpec | Description-time default pose. Runtime updates go through `Scene.set_object_pose`. |
| `visual` | VisualSpec, optional | Visual geometry for rendering. |
| `collision` | CollisionGeometrySpec, optional | Collision geometry. |

#### `visual` and `collision`

Both wrap a geometry variant and an optional `origin` offset.

| Field | Type | Notes |
|---|---|---|
| `enabled` | bool | Default `true`. |
| `geometry` | discriminated union | One of mesh / box / sphere / cylinder. See below. |
| `origin` | TransformSpec, optional | Offset from the object's local frame to the geometry's frame. Default identity. |
| `processing` (collision only) | ProcessingSpec, optional | Currently only `convex_decomposition`. |

Visual and collision geometry are **independent**: they can be different
meshes with different scales and different origin offsets. This matches
production asset pipelines where a high-poly visual mesh and a
simplified collision mesh do not always share a centre.

#### Geometry variants

```yaml
# Mesh
{ type: mesh, path: "...", scale: [1.0, 1.0, 1.0] }

# Box
{ type: box, size: [0.4, 0.3, 0.2] }      # axis-aligned

# Sphere
{ type: sphere, radius: 0.05 }

# Cylinder
{ type: cylinder, radius: 0.05, length: 0.12 }  # along local z axis
```

#### `collision_matrix`

| Field | Type | Notes |
|---|---|---|
| `default_action` | `check` \| `allow` | Default for any pair not listed explicitly. |
| `rules` | list[{a, b, action, reason?}] | Per-pair static rules. |

Only **static physical-fact** allowances belong here. Task-driven
dynamic allowances belong to the runtime Scene overlay.

## Common pitfalls

1. **Forgetting acceleration and jerk** for chain joints. The build
   raises with a clear error listing the missing joints.

2. **Declaring mimic followers in chain joint lists.** URDF `<mimic>`
   joints are not active DOF; including them in the chain produces an
   invalid mapping.

3. **Forgetting the namespace** in a multi-robot world. The world load
   raises before any geometry is built.

4. **Putting algorithm configuration in YAML** (planner timeout,
   IK tolerance, smoothing iterations). These belong to call-site
   arguments. Description YAML must remain stable across solver tuning.

5. **Editing object poses in YAML at runtime.** The YAML pose is the
   description-time default. Runtime updates go through
   `Scene.set_object_pose`; YAML stays immutable.
