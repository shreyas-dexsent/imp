# DexSent VGR Robot Controller 2.5D

A robot-agnostic **motion control service** for Vision-Guided Robotics (VGR). It exposes a stable control/state contract to the **Orchestrator**, while hiding robot-specific details behind adapters.

**Primary design goal**
> Any robot can be used by implementing a new adapter, without changing orchestrator logic.

This repo includes a UR5e MuJoCo adapter. For the best visuals, use the MuJoCo Menagerie UR5e model (recommended).

## Start Robot Controller (Quick)

```bash
cd C:\Users\asus\dexsent\vgr\dexsent-vgr-robot-controller-2.5d
python app.py --config configs/robot.local.json
```

If you have not prepared the UR5e model yet, run `python tools/setup_ur5e_mjcf.py` first.

## Running (Development)

### 1) Create Conda Env (vgr-robot)

```bash
conda create -n vgr-robot python=3.11 -y
conda activate vgr-robot
pip install -r requirements.txt
```

### 2) Start Robot Controller Node

```bash
cd C:\Users\asus\dexsent\vgr\dexsent-vgr-robot-controller-2.5d
python app.py --config configs/robot.local.json
```

If you see `command_plane_addr_in_use`, stop the previous controller process or change the port in `configs/robot.local.json`.

### UR5e MJCF Setup (MuJoCo Menagerie, Recommended)

Download the MuJoCo-native UR5e model:

```bash
python tools/setup_ur5e_mjcf.py
```

### UR5e URDF Setup (Alternative)

Download UR5e URDF + meshes and generate `ur5e.urdf`:

```bash
python tools/setup_ur5e_urdf.py
```

### MuJoCo Viewer (Optional)

Enable the viewer in `configs/robot.local.json`:

```json
{
  "viewer": {
    "enabled": true,
    "rate_hz": 30,
    "step_physics": false
  }
}
```

### 3) Drive via Orchestrator

In VIM-303 mode, orchestrator is the only intended client.

---
## Position in the VIM-303 Architecture

**Authority model**
> The **Orchestrator** is the only external authority. Robot Controller executes motion requests and streams state.

```
Orchestrator  --->  Robot Controller  --->  Robot HW / Simulation
                (commands)             (execution)
Orchestrator  <---  Robot Controller
                (state + events)
```

Robot Controller is analogous to Camera Core:
- Camera Core abstracts cameras
- Robot Controller abstracts robots

---

## What Robot Controller Does (and Does Not)

### ✅ Responsibilities
- Provide a **standard command interface** to:
  - Move TCP to a pose
  - Move joints
  - Operate end-effector actions (open/close gripper)
  - Stop / abort motion
- Provide a **standard state interface**:
  - Robot mode/status
  - TCP pose
  - Joint positions/velocities
  - Active motion status
- Enforce robot-level safety behavior:
  - speed clamps
  - motion timeouts
  - stop-on-error
  - robot-specific protective stop handling
- Support **teach/record** helpers (optional):
  - read current TCP pose
  - jog small increments (if supported)

### ❌ Non-responsibilities
- No vision computation
- No task logic (palletizing/sorting/tending)
- No recipe library
- No station calibration ownership (orchestrator owns hand-eye, camera intrinsics)
- No template/object library

---

## Deployment Mode

### Recommended: Service Mode (camera-core style)
- Run Robot Controller as a separate process on the station PC.
- Bind command endpoints to **localhost** by default.
- Orchestrator connects as a client.

### Alternative: In-Process Library Mode
- Useful for early simulation prototypes.
- Same adapter and core APIs, but directly imported.

This README documents **service mode**.

---

## Interfaces (External Contracts)

Robot Controller exposes two channels:

1) **Command Plane** (Orchestrator → Robot Controller)
- Request/response for discrete actions

2) **State/Event Plane** (Robot Controller → Orchestrator)
- Continuous state snapshots
- Asynchronous motion lifecycle events

The wire protocol can be either:
- HTTP (FastAPI) + WebSocket
- ZMQ REQ/REP (commands) + ZMQ PUB (state/events)

**Recommendation for VGR stack consistency:** ZMQ for commands + PUB for state.

---

## Data Types

### PoseSE3
- `position_m: [x, y, z]`
- `quat_xyzw: [qx, qy, qz, qw]`
- `frame: "base"` (robot base frame)

### MotionProfile
- `speed: "slow" | "normal" | "fast"`
- `max_lin_vel_mps`
- `max_ang_vel_rps`
- `max_joint_vel_scale` (0..1)
- `timeout_s`

### RobotState
- `timestamp_ns`
- `mode: IDLE | MOVING | ERROR | ESTOP | PROTECTIVE_STOP | DISCONNECTED`
- `tcp_pose: PoseSE3`
- `q: [..]` joints
- `dq: [..]` joint velocities
- `active_motion_id: str | null`

---

## Command Plane: Requests

### 1) Connect / Disconnect
```json
{ "cmd": "CONNECT" }
```
```json
{ "cmd": "DISCONNECT" }
```

### 2) Move TCP (Cartesian)
```json
{
  "cmd": "MOVE_TCP",
  "motion_id": "m-001",
  "target": {
    "frame": "base",
    "position_m": [0.40, 0.00, 0.55],
    "quat_xyzw": [0, 0, 0, 1]
  },
  "profile": { "speed": "normal", "timeout_s": 10.0 }
}
```

Note: the MuJoCo UR5e adapter currently solves position-only IK (orientation is best-effort).

### 3) Move Joints
```json
{
  "cmd": "MOVE_JOINTS",
  "motion_id": "m-002",
  "q": [0.0, -0.5, 0.0, -2.0, 0.0, 1.6, 0.8],
  "profile": { "speed": "slow", "timeout_s": 15.0 }
}
```

### 4) Gripper
```json
{ "cmd": "GRIPPER_OPEN" }
```
```json
{ "cmd": "GRIPPER_CLOSE" }
```

### 5) Stop / Abort
```json
{ "cmd": "STOP" }
```
```json
{ "cmd": "ABORT", "motion_id": "m-001" }
```

### Command Responses
All commands respond with:
```json
{
  "ok": true,
  "message": "accepted",
  "motion_id": "m-001",
  "robot_mode": "MOVING"
}
```
or
```json
{
  "ok": false,
  "message": "reject",
  "reason": "robot_in_error"
}
```

---

## State/Event Plane

### State Snapshot (periodic)
Robot Controller publishes `ROBOT_STATE` at a configured rate (e.g., 20–50 Hz):

```json
{
  "event": "ROBOT_STATE",
  "timestamp_ns": 123456789,
  "mode": "MOVING",
  "tcp_pose": {
    "frame": "base",
    "position_m": [0.41, 0.01, 0.52],
    "quat_xyzw": [0, 0, 0, 1]
  },
  "q": [..],
  "dq": [..],
  "active_motion_id": "m-001"
}
```

### Motion Lifecycle Events (asynchronous)
```json
{ "event": "MOTION_STARTED", "motion_id": "m-001" }
```
```json
{ "event": "MOTION_SUCCEEDED", "motion_id": "m-001", "elapsed_s": 2.43 }
```
```json
{ "event": "MOTION_FAILED", "motion_id": "m-001", "reason": "timeout" }
```

The orchestrator uses these events to implement deterministic task state machines.

---

## Safety Model

Safety is enforced at two levels:

### 1) Orchestrator-level safety (high-level)
- workspace bounds checks
- recipe constraints
- approach/retreat sequencing

### 2) Robot Controller safety (robot-level)
- hard speed clamps
- motion watchdog timeout
- stop on communication failure
- map robot faults to standardized modes (`PROTECTIVE_STOP`, `ERROR`, etc.)

**Hard rule**
> On any adapter exception or motion failure, Robot Controller transitions to `ERROR` and refuses new motions until cleared (via explicit recover action).

---

## Teach / Waypoint Support (Optional)

Robot Controller does not own the Pose Library; **Orchestrator stores poses** under station data.

Robot Controller can provide primitives to support UI-driven pose capture:
- `GET_TCP_POSE` (read current TCP)
- `JOG_TCP_DELTA` (small delta move)

This enables operator workflows:
1) Jog robot to desired pose
2) UI calls Orchestrator: `SAVE_POSE(name, current_tcp_pose)`
3) Orchestrator writes `data/station/pose_library/poses.yaml`

---

## Adapters (Robot-Agnostic Design)

Robot Controller implements a common adapter interface:

**Adapter responsibilities**
- connect/disconnect
- get_state
- execute motion primitives
- execute gripper actions
- stop/abort

### Planned adapters
- `mujoco_ur5e` (default)
- `franka_fr3` (libfranka)
- `ur_rtde` (UR RTDE)

Each adapter translates the stable contract into robot-native APIs.

---

## Configuration

`configs/robot.local.json` (example):

```json
{
  "robot": {
    "robot_id": "robot_01",
    "type": "mujoco_ur5e",
    "model_path": "robot_controller/assets/models/ur5e_mjcf/ur5e.xml",
    "home_q": [0.0, -1.57, 1.57, -1.57, -1.57, 0.0]
  },
  "io": {
    "command_plane": { "mode": "reqrep", "endpoint": "tcp://127.0.0.1:5571" },
    "state_plane":   { "mode": "pub",   "endpoint": "tcp://127.0.0.1:5572", "rate_hz": 30 }
  },
  "safety": {
    "max_lin_vel_mps": 0.5,
    "max_ang_vel_rps": 1.0,
    "motion_timeout_s": 15.0
  },
  "viewer": { "enabled": true, "rate_hz": 30, "step_physics": false },
  "logging": { "level": "INFO" }
}
```

---

## Repository Structure (Reference)

```
dexsent-vgr-robot-controller-2.5d/
├── app.py
├── configs/
│   └── robot.local.json
├── robot_controller/
│   ├── core/
│   │   ├── controller.py          # command handling + motion lifecycle
│   │   ├── adapter_base.py        # required adapter interface
│   │   ├── models.py              # PoseSE3, RobotState, MotionProfile
│   │   └── safety.py              # clamps, watchdog, standardized faults
│   ├── io/
│   │   ├── command_plane/
│   │   │   └── server.py          # REQ/REP or HTTP server
│   │   └── state_plane/
│   │       └── publisher.py       # PUB or WS state stream
│   ├── adapters/
│   │   ├── mujoco_panda/
│   │   │   ├── adapter.py
│   │   │   ├── ik.py
│   │   │   └── sim.py
│   │   ├── franka_fr3/            # later
│   │   └── ur_rtde/               # later
│   └── common/
│       └── time.py
├── test/
│   ├── adapter_contract_test.py
│   └── motion_lifecycle_test.py
└── README.md
```

## Current Skeleton (Implemented)

```
robot_controller/
  adapters/
    mujoco_ur5e/
      adapter.py
  assets/
    models/ur5e/ur5e_minimal.xml
    models/ur5e_mjcf/ur5e.xml
  core/
    adapter_base.py
    controller.py
    models.py
  io/
    command_plane/server.py
    state_plane/publisher.py
configs/
  robot.local.json
  robot.fr3.json
app.py
```

---

## License

DexSent Robotics Pvt Ltd. Proprietary.
