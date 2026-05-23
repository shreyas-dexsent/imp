# Forward Kinematics

## 1. Description

Forward kinematics (FK) maps a joint vector `q` to the pose of a frame.

```text
Input:  KinematicModel, q (joint vector), frame_id
Output: 4x4 homogeneous transform T_parent_frame
```

`algorithms` exposes two FK entry points:

| Function | Parent frame | Use when |
|---|---|---|
| `fk_local(model, q, frame_id)` | the robot's declared base frame | you only care about geometry within one robot |
| `fk(scene, robot_id, q, frame_id)` | the world frame | the robot is placed in a world (one or many robots) |

Both have batch variants that compute many frames in one Pinocchio pass:

- `fk_local_many(model, q, frame_ids)` → `{frame_id: T_base_frame}`
- `fk_many(scene, robot_id, q, frame_ids)` → `{frame_id: T_world_frame}`

Frames can come from three places: the robot URDF, the gripper URDF, and YAML-injected TCPs. FK does not distinguish — once the `KinematicModel` is built, every frame is queryable.

## 2. Data Flow

```text
YAML (robot system)              YAML (world)
        |                             |
        v                             v
RobotSystemDescription         WorldDescription
        |                             |
        v                             v
KinematicModel  <--cached--    Scene  (Scene wraps the world; KinematicModel
        |                             |  is shared across robots of the same type)
        |                             |
        +------- fk_local(...)        +------- fk(scene, robot_id, ...)
```

YAML files declare:

- The URDF path the robot lives in.
- The base frame the engine should treat as the robot base.
- Static TCP frames declared as `parent_frame → child_frame` transforms.
- Kinematic chains (groups of active joints).
- Named joint states stored as `{joint_name: value}` dicts.
- An optional gripper attached by a fixed mount transform.

`KinematicModel.from_robot_system(...)` reads the YAML and URDF once, composes robot + gripper into one Pinocchio model, expands `<mimic>` joints into an internal expansion matrix, and caches the result. Subsequent calls return the cached instance.

`Scene.from_world(world)` builds a runtime container holding the world description and live robot / object state. FK never mutates the scene; it allocates a fresh Pinocchio `Data` per call and runs `forwardKinematics` + `updateFramePlacements`.

## 3. Usage

### Setup

```python
import numpy as np
from algorithms.descriptions import RobotSystemDescription, WorldDescription
from algorithms.kinematics import fk_local, fk
from algorithms.resolved import KinematicModel, Scene

# Robot-only (no world)
system = RobotSystemDescription.from_yaml("configs/robots/franka_fr3_robot_only.yaml")
model = KinematicModel.from_robot_system(system)
home = system.named_joint_state("home")
q = np.array([home[name] for name in model.active_joint_names], dtype=float)

T_local = fk_local(model, q, "robot_tcp")  # base -> robot_tcp

# Robot placed in a world
world = WorldDescription.from_yaml("configs/worlds/franka_table_world.yaml")
scene = Scene.from_world(world)
system = world.robot("arm").robot_system
model = KinematicModel.from_robot_system(system)
home = system.named_joint_state("home")
q = np.array([home[name] for name in model.active_joint_names], dtype=float)

T_world = fk(scene, "arm", q, "fr3_hand_tcp")  # world -> fr3_hand_tcp
```

### Active-joint order (mandatory)

`q` must be a NumPy array in `model.active_joint_names` order. Convert named-joint-state dicts via:

```python
q = np.array([named[name] for name in model.active_joint_names], dtype=float)
```

This order is shared by every algorithm in the library (FK, Jacobian, IK, collision, planning, trajectory).

### Mimic joints

A URDF `<mimic>` joint (e.g. the Franka hand's second finger) is hidden from `model.active_joint_names`. The user sets the driving joint; the follower is expanded internally via `q_full = active_to_full @ q_active + offset`.

### Multi-robot worlds

```python
T_left = fk(scene, "left_arm", q_left, "fr3_hand_tcp")
T_right = fk(scene, "right_arm", q_right, "fr3_hand_tcp")
```

Frame names stay local (no `"left/fr3_hand_tcp"` namespacing on the FK side).

### Performance

`KinematicModel.from_robot_system` is cached on `(yaml_path, urdf_mtimes)`. Editing a URDF auto-invalidates the cache. FK itself allocates a fresh `pin.Data` per call (sub-microsecond cost), so concurrent / interleaved multi-robot queries do not corrupt each other.

| Operation | Cost |
|---|---|
| `KinematicModel.from_robot_system` (first call) | ~100 ms |
| `KinematicModel.from_robot_system` (cached) | < 1 µs |
| `fk_local(...)` per call | ~10 µs |
| `fk_local_many(...)` per call | ~15 µs + Coal lookups |

## 4. Examples

Run from the repository root:

| File | What it shows |
|---|---|
| `01_robot_only_fk.py` | Minimum useful call: `fk_local(model, q, frame_id)`. |
| `02_robot_with_static_tcp.py` | YAML-declared TCP appears as a real resolved-model frame. |
| `03_robot_with_actuated_gripper.py` | Robot + gripper composed into one model; mimic finger hidden from `active_joint_names`. |
| `04_world_frame_fk.py` | `fk(scene, robot_id, q, frame_id)` returns world-frame pose. |
| `05_two_robot_world_fk.py` | Two robots, two world poses from the same `q`. |
| `06_batch_fk_many_frames.py` | Batched FK over many frames in one Pinocchio pass. |

## 5. Common Errors

| Symptom | Cause | Fix |
|---|---|---|
| `KeyError: 'frame_id'` | Frame name not in the resolved kinematic model. | Use exactly the URDF link name, TCP id from YAML, or composed name like `fr3_hand_tcp`. |
| `ValueError: q has shape (n,) but expected (m,)` | `q` is in the wrong order or has the wrong DOF. | Build `q` from `model.active_joint_names`, never from the YAML named-state dict directly. |
| Mimic joint silently ignored | You passed `q` in URDF joint order, including the mimic follower. | Use `model.active_joint_names`; the follower is internal. |
| Two FR3s give the same world pose | You forgot to pass each robot's `base_pose` in YAML or you used `fk_local` instead of `fk`. | Use `fk(scene, robot_id, ...)`; world poses come from `WorldRobotDescription.base_pose`. |
| `T` not orthonormal in your test | You compared via `==` instead of `np.allclose`. | Use `np.testing.assert_allclose(T_a, T_b, atol=...)`. |

## 6. FAQs

**Q: Why must `q` be in `active_joint_names` order? Why not pass a dict?**
The library is stateless and consumes NumPy arrays everywhere. Dicts appear only at the YAML boundary. Converting is one line; doing it explicitly keeps every operation predictable.

**Q: Can I call FK without building a `Scene`?**
Yes. Use `fk_local(model, q, frame_id)`. The base frame is the robot's own declared base; no world placement is involved.

**Q: How do I get the TCP pose in world coordinates for a multi-robot world?**
`fk(scene, "left_arm", q_left, "fr3_hand_tcp")`. The world transform `T_world_base[left_arm]` from YAML is composed automatically.

**Q: Is `KinematicModel.from_robot_system` thread-safe?**
The cache lookup is not synchronised. Build once at startup; share the resulting `KinematicModel` across threads. FK itself is safe (per-call scratch data).

**Q: Does FK respect joint limits?**
No. FK is geometry; it evaluates whatever `q` you supply. Limits are enforced by IK, the path validator, and the trajectory validator.
