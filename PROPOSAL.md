# PROPOSAL — Why we rebuild VGR as `imp`

> **Audience:** the team. This document makes the engineering case for
> moving the existing VGR codebase onto a single, typed, observable
> substrate (`imp`) — *without throwing away any of the work already in
> VGR*. The full spec is in [`README.md`](./README.md); the phased build
> plan is in [`PLAN.md`](./PLAN.md). This document explains *why* before
> *how*.

---

## 1. TL;DR

VGR works. The team built a vision-guided pick system that runs robots,
sees objects, calibrates cameras, and ships pose estimates to motion.
The algorithms are good — Pinocchio FK/IK, Coal collision, MegaPose,
PPF-ICP, the OnRobot/Robotiq gripper integrations. **None of that is
being thrown away.**

What VGR doesn't have is the *substrate* a robotics platform needs once
you go beyond one customer, one cell, one developer. Today, debugging a
field failure means SSHing in, reading source, and grepping logs. Adding
a second camera means editing five config files with hand-picked port
numbers. Vision and motion talk over opaque ZMQ frames, so a schema
change in one silently breaks the other at runtime. There is no `imp
topic hz` equivalent, no late-joiner state, no plugin contract, no CLI,
no cross-platform installer.

These are not "polish" gaps — each one becomes a real cost the first
time a customer asks "why did it fail at 3am?". `imp` adds the
substrate underneath the algorithms VGR already has. The flagship
application is still VGR (vision-guided robotics); the platform just
makes the next ten applications cost a tenth of the first.

**The bet is on one decision: the substrate.** We chose Eclipse Zenoh
because it gives us — natively — typed pub/sub + queryable services +
durable storage + zero-copy SHM + cross-host routing + admin/introspection,
all of which VGR currently hand-rolls (badly, in places). Zenoh is just
the daemon + transport layer; if it ever stops fitting we replace
`imp-bus` (the thin wrapper) and the rest of the platform is untouched.
Section 6 below makes the Zenoh case in detail.

---

## 2. What VGR does well (the baseline we're preserving)

Before any criticism: VGR's algorithms and integrations are genuinely
good engineering, and **every one of them carries forward into `imp`**.

| VGR delivers | Where it lives | imp's plan |
|---|---|---|
| 6-D pose estimation: MegaPose, PPF-ICP, template / SIFT, cuboid, blob, YOLO seg | `reference/dexsent-vgr-vision-engine-*` | Wrapped as `modules/perception-*` (algorithm preserved, ZMQ plumbing replaced) |
| Motion: FK, IK, collision, planning, trajectory (Pinocchio / Coal / OMPL / Ruckig) | `reference/robot-algorithms` | Vendored intact as `modules/motion-core` — **226 tests pass verbatim** |
| Camera drivers: RealSense D435i + D405, Basler, FLIR, UVC | `reference/dexsent-vgr-camera-core-*` | `hal/camera-*` |
| Robot adapters: MuJoCo UR5e, Franka FR3, xArm, UR-RTDE planned | `reference/dexsent-vgr-robot-engine-*` | `hal/robot-*` |
| Calibration: intrinsics, hand-eye, TCP, target | `reference/.../orchestrator/vision/` | `jobs/calibration-*` + `services/calibration-*` |
| Grasp library / candidate / feasibility | `reference/.../robot_engine/planning/` | `modules/motion-grasp-library` (already ported) |
| Object library, pose library, scene + obstacle YAMLs | `reference/data-master/` | `workspace/stations/<id>/processes/<id>/...` |
| Run logging + debug artifacts | `reference/.../orchestrator/runs/` | `runs/<id>/` + Zenoh bag |
| UI surfaces: Operator console, RobotViz, Grasp Studio, Gripper Studio, Calibration Wizard, Perception Debug, Frame Editor, Gizmo Editor, Pose Library, Object Browser, Asset Manager, Run Monitor | `reference/.../orchestrator/ui/` | `ui/views/*` (every surface preserved) |
| Robot/gripper asset catalog (Franka, UR, KUKA, Fanuc, xArm; Robotiq, Franka Hand) | `reference/.../robot_asset_catalog/` | `catalog/` (shipped with install) |
| Offline training: YOLO seg, LabelMe→YOLO, capture-from-realsense | `reference/data-master/extras/seg_obj/` | `tools/train/` |

**This list is the floor.** Anything in VGR that a customer relies on
must survive the move. `imp` is a re-foundation, not a rewrite of the
work that's already shipped.

---

## 3. What VGR is structurally missing

These are the gaps that don't show up in a feature checklist but show up
the day after deployment. Each one is something the team has either
already hit or will hit at scale.

### 3.1 No typed wire contract — vision and motion can disagree silently

ZMQ moves *opaque byte frames*. The publisher serializes whatever it
wants; the subscriber decodes whatever it expects. When vision adds a
field to `Pose6D`, motion doesn't know — until the next pick crashes
with a key error. There's no compile-time check, no runtime schema
validation, no version negotiation. The bug surfaces on the robot, not
in CI.

In `imp` every topic carries a Protobuf schema tag (e.g.
`schema=imp.Pose6D/1`). Subscribers reject mismatches at the edge.
Schema bumps are explicit and visible in `imp doctor`. *The class of bug
where vision and motion silently disagree disappears.*

### 3.2 No discovery — every connection is a hand-picked port

```text
reference/dexsent-vgr-camera-core-*/config/*.yaml      → 5555, 5556
reference/dexsent-vgr-vision-engine-*/configs/*.yaml   → 5561, 5571
reference/dexsent-vgr-application-orchestrator-*/...   → 5572, 8210
```

Six configs, six hand-picked ports. Adding a second camera = editing
those configs in lockstep. Putting a second instance on the same machine
= a port collision. Moving anything across a host boundary = the
WebSocket+base64 relay hack already in the codebase. **There is no
discovery: every node finds every other node by literal IP:port.**

In `imp`, nodes scout and find each other automatically. Adding a
camera is `imp up`; no config file knows the port number.

### 3.3 No introspection — debugging means reading source

VGR has no equivalent of `ros2 topic hz` or `ros2 node list`. To answer
"is the camera publishing? at what rate? is anyone subscribed? what's
the schema?" you read the publisher's code, read the subscriber's code,
add a `print`, redeploy, and stare at a terminal.

In `imp`, the substrate's admin space already lists every session,
publisher, and subscriber. `imp graph` draws the live data-flow map with
per-edge rates. `imp topic hz <ke>` and `imp topic echo <ke>` are one
command. *Field debugging stops requiring a developer with the source
tree.*

### 3.4 No late-joiner state — the UI sees nothing until the next publish

A monitoring dashboard that connects mid-run sees the *next* frame, not
the *current* one. VGR works around this with bespoke
`vgr_result_<id>` SHM segments (literally commented as "durable path
because large payloads can be missed by the live subscriber"). Every
single piece of state that needs to survive a reconnect is hand-rolled.

In `imp`, Zenoh storage + `query()` is built in. The UI queries the
current calibration, the last pose, the active scene at startup; the
panels render instantly without waiting for the next publish.

### 3.5 No native zero-copy for large payloads

VGR's camera-core moves images by SHM triple-buffer — a custom
synchronization protocol on top of POSIX shared memory. Adding another
large-payload channel (point cloud, segmentation mask, depth tensor)
means writing it again. The relay path for remote compute is a
WebSocket+base64 hack (literally documented as such in the README).

In `imp`, Zenoh native SHM kicks in automatically when peers are on the
same host; the same `publish(frame)` API switches to network transport
when they're not. *One transport, one API, no hand-rolled SHM.*

### 3.6 No no-code task composition

Every task in VGR is a Python file in `orchestrator/tasks/`
(`bin_picking.py`, `pallatizing.py` [sic], `pick_place.py`,
`follow_object.py`, `_pick_runtime.py`). Each one is bespoke
orchestration code. Want a new pick variant? Write a new Python file.
Want an integrator to add one? They need full source access and the
ability to modify the orchestrator.

In `imp`, a task is a **YAML graph**. Nodes are registered modules;
edges are typed channels. The Task Composer UI edits the graph. A new
task is a new YAML file, no code. *Customers and field engineers can
compose tasks without touching source.*

### 3.7 No plugin contract — every new device requires reading source

VGR has no SDK. To add a new camera, you read `camera_core`'s drivers
folder, copy the closest match, and rewire `config/*.yaml`. To add a new
robot, you read `robot_engine/adapters/`, do the same. To add a new
perception module, you read `vision_engine/modules/`, copy a sibling,
and figure out the ZMQ control protocol from another module's source.

In `imp`, an integrator installs `imp-sdk`, decorates a function with
`@module` / `@hal_device` / `@service` / `@job`, registers it under an
entry-point group, and `imp doctor` lists their plugin. **They never
read `imp` source.** This is the difference between a product and a
project.

### 3.8 No cross-platform symmetry

VGR is implicitly Linux-first. Windows works "if you set it up right".
Customers on Windows operator stations are a port project, not a
deployment.

In `imp`, Windows and Linux are first-class targets with the same
installers (MSI vs DEB/RPM/AppImage), same SDK, same CLI, same `imp
doctor` parity check.

### 3.9 No CLI — operations needs Python

VGR is operated by running scripts and watching logs. There is no
operator-facing command-line tool. A field engineer reproducing an issue
runs `python -m something`, reads stack traces, and edits configs.

`imp` ships `imp` (Linux) / `imp.exe` (Windows): `imp up`, `imp task
run`, `imp topic hz`, `imp graph`, `imp doctor`, `imp bag record`. This
is the ROS-style debug loop, but as one binary that the installer drops
on the PATH.

### 3.10 No bag record / replay — every bug needs the original hardware

VGR has no equivalent of `ros2 bag record/play`. To reproduce a
perception bug, you need the camera, the lighting, the object on the
table, and the patience to retry. There's no way to capture a real
session and replay it into a debugger.

In `imp`, `imp bag record imp/<station>/perc/**` captures every topic.
`imp bag play` replays it into perception with the hardware off. **The
loop "record a real failure → replay → fix → diff" exists.** This is
how serious robotics teams debug.

### 3.11 No architectural layers — four repos, no contract between them

VGR is four top-level repositories:

```text
reference/dexsent-vgr-camera-core-2.5d-main/
reference/dexsent-vgr-vision-engine-2.5d-main/
reference/dexsent-vgr-robot-engine-main/
reference/dexsent-vgr-application-orchestrator-main/
```

The contract between them is "whatever the orchestrator code imports".
There is no module boundary, no plugin boundary, no Hardware Abstraction
Layer in the sense the term is used in robotics. The orchestrator
imports vision-engine which imports camera-core. Changing a camera
driver means understanding the orchestrator.

In `imp`, the layers are explicit, named, and have typed contracts:

```text
HAL          (drivers, vendor SDKs)        — topics only, nothing else
Modules      (pure typed functions)        — perception, motion, spatial
Services     (synchronous queryables)      — bounded ops, pose lookup, asset get
Jobs         (long-running cancelable)     — calibration, run-task
Task layer   (graph compiler + runtime)    — composes modules + services
Operations   (supervisor, CLI, graph)      — lifecycle + introspection
UI           (one app, thin views)         — visualizes topics, triggers services
```

A perception developer touches `modules/perception-*`. A driver
developer touches `hal/*`. **They don't read each other's code, and
they can't break each other accidentally.**

---

## 4. What each gap costs in real robotics deployment

The structural gaps in §3 are abstract. Here's what they cost in
scenarios the team will actually face.

| Scenario | VGR cost (today) | imp cost |
|---|---|---|
| **"Why did it fail at 3am at customer X?"** | SSH in, read source, grep logs, possibly fly out. No bag of the run. | `imp run show <id>`, replay the bag offline, fix in the office. |
| **Vision team bumps `Pose6D` to add quaternion confidence** | Motion runtime crashes on next pick, no compile-time check. | Schema version bump caught at load; `imp doctor` flags the drift. |
| **Customer puts a second camera on the cell** | Edit 5 config files with new hand-picked ports; pray no collision. | `imp up`; the new HAL node is auto-discovered. |
| **Customer wants perception on a separate GPU host** | Use the WebSocket+base64 hack, or rebuild the transport. | `placement.yaml` moves the module; topology change, no code change. |
| **Customer wants the operator UI on a Windows tablet** | Port effort: weeks. | MSI installer; same SDK, same UI. |
| **New integrator joins the team** | Two weeks reading code to add their first perception module. | `imp-sdk` docs + `examples/`; first module in a day. |
| **Calibration drift suspected** | Read hand-eye JSON; compare to log; compute manually. | `imp topic echo imp/<st>/tf`; `imp graph` shows the live edge with timestamp. |
| **New "follow-object" task variant requested** | Write a new Python orchestration file; redeploy. | New YAML in Task Composer; no code change. |
| **Robot RT loop starves under perception load** | Profile by hand; reshuffle threads; hope it holds. | Robot HAL is its own process; rate-decoupled scheduling is built in. |
| **A perception module needs to live in the cloud** | Not possible without significant rework. | Add a router; same task graph, deployment-only change. |
| **A topic is dropping at unknown rate** | Add print statements, redeploy, watch terminal. | `imp topic hz imp/<st>/perc/**`. |
| **Reproduce a one-in-a-thousand pose failure** | Hope it happens again. | `imp bag play` the captured session into a debugger. |
| **Add a new gripper from a new vendor** | Read `gripper-*` drivers in source; copy the closest one. | `@hal_device` against `imp-sdk`; new wheel ships without imp recompile. |
| **Ship a sealed binary to a customer who shouldn't see source** | No story today. | Operator install (Nuitka + Rust); developer install adds the SDK headers only. |
| **Audit which schema versions are running across hosts** | Read source on every host. | `imp doctor` reports schema drift in one call. |

Each row above is a future support ticket, a future deployment
delay, or a future onboarding bottleneck. The platform investment is
about *not paying these costs forever*.

---

## 5. The fix — `imp` architecture

### 5.1 One picture

```text
┌─────────────────────────────────────────────────────────────────────┐
│                     UI (one app, thin views)                        │
│  Dashboard · Operator · RobotViz · Task Composer · Calibration ...  │
└──────────────────────────────┬──────────────────────────────────────┘
                               │  subscribe + storage-query + svc-call
┌──────────────────────────────┴──────────────────────────────────────┐
│  Operations    Supervisor (lifecycle) │ Graph svc │ imp CLI         │
├──────────────────────────────────────────────────────────────────────┤
│  Task layer    Graph Compiler  │ Task Runtime  │ Sequence FSM        │
├──────────────────────────────────────────────────────────────────────┤
│  Services      tf.lookup · asset.get · grasp.define · run.* · ...    │
│  Jobs          calibration.* · object.init · run-task · ...          │
├──────────────────────────────────────────────────────────────────────┤
│  Modules       perception-*   │ motion-*    │ spatial-*              │
├──────────────────────────────────────────────────────────────────────┤
│  HAL           camera-* · robot-* · gripper-* · plc-*                │
├──────────────────────────────────────────────────────────────────────┤
│              ZENOH (pub/sub + query + storage + SHM)                 │
│   keyexprs · schemas · QoS · routers · admin space · multi-host      │
└──────────────────────────────────────────────────────────────────────┘
                               │
                              [HARDWARE]
```

**Every layer talks via topics.** Nothing else. The CLI sees what the UI
sees. The supervisor sees what the integrator sees. There is no
back-channel, no second API, no privileged path.

### 5.2 The folder tree

The repository layout itself enforces the layering. Each architectural
concept = one top-level folder. There is no `plugins/` umbrella — `hal/`,
`modules/`, `services/`, `jobs/`, `ui/` are siblings, so an integrator
opens the repo and immediately knows where their new code goes.

```text
imp/
├── crates/                  # sealed Rust core (contracts + runtimes only)
│   ├── core bus schemas
│   ├── hal-contract module-contract service-contract
│   ├── tasks supervisor workspace
│   ├── cli ui-host installer
│
├── hal/                     # device drivers (one folder per device)
│   ├── camera-realsense camera-basler-gige camera-flir-gige camera-uvc
│   ├── robot-mujoco-ur5e robot-ur-rtde robot-franka-fr3 robot-xarm
│   ├── gripper-onrobot gripper-robotiq gripper-franka-hand
│   └── plc-modbus
│
├── modules/                 # functional modules (one folder per capability)
│   ├── perception-{yolo,megapose,ppf-icp,template,template-sift,feature,
│   │                 blob,cuboid,opt-sift,track,fusion,preview}
│   ├── motion-{core,pinocchio,coal,ompl,cartesian,path-processor,
│   │            ruckig,grasp-library}
│   └── spatial-{tf,transform}
│
├── services/                # synchronous queryables (one folder per service)
│   ├── calibration-{tcp,target} grasp-define scene-define
│   ├── pose-library tf-lookup asset-get
│   └── run-store robot-digital-twin
│
├── jobs/                    # long-running cancelable ops (one folder per job)
│   ├── calibration-{intrinsics,hand-eye,samples}
│   ├── object-init
│   └── run-task
│
├── catalog/                 # pre-configured assets shipped with the install
│   ├── robots/ grippers/ objects/
│
├── sdk/                     # public plugin/task-author surface
│   ├── py/                  # imp_sdk (pip-installable, sealed)
│   └── rs/                  # imp-sdk (cargo-installable, sealed)
│
├── ui/                      # ONE TypeScript app
│   ├── app/                 # shell: routing, layout, view registry
│   ├── lib/                 # bus (Zenoh) · schemas · viewer3d · forms
│   └── views/               # one thin folder per surface (compose lib/*)
│
├── docs/                    # user · developer · architecture · reference
├── examples/                # runnable example workspaces (templates)
└── tools/                   # build · train · dev (codegen runners, etc.)
```

A camera driver lives next to other camera drivers. A perception module
lives next to other perception modules. **There is no possible question
about where new code goes.**

### 5.3 The operating model — Station → Process → Task → Run

Carried forward from VGR's mental model (the team already thinks this
way), now reflected in the workspace layout:

| Entity | Owns | Workspace path |
|---|---|---|
| **Station** | one physical cell — cameras, robots, calibration inventory | `stations/<station_id>/` |
| **Process** | one workflow under a station (one task type) — gripper, robot, object library, poses | `stations/<station_id>/processes/<process_id>/` |
| **Task** | a saved configuration — graph + parameters + asset mappings | `…/processes/<process_id>/tasks/<task_id>.yaml` |
| **Run** | one execution instance — timeline, log, bag, artifacts | `runs/<run_id>/` |

The CLI, the UI, and every service operates on these four IDs uniformly.

### 5.4 What each module looks like (the SDK shape)

```python
# A new perception module: 12 lines.
from imp_sdk import module, Frame, Detections

@module(inputs={"frame": Frame}, outputs={"dets": Detections}, rate_hz=10)
def my_detector(frame: Frame, p: MyParams) -> Detections:
    ...
    return Detections(...)

# pyproject.toml — entry-point registration
[project.entry-points."imp.modules"]
my-detector = "my_detector:my_detector"
```

The author ships their wheel. `imp` discovers it via
`importlib.metadata` at startup. No `imp` source is touched. **A new
perception module ships in hours, not weeks.**

---

## 6. Why Zenoh — and why it's the reversible choice

### 6.1 What we need from the substrate

These properties are non-negotiable for the platform's promises:

| Property | Why we need it |
|---|---|
| Typed pub/sub | Catch vision/motion contract drift at the edge, not on the robot |
| Discovery | Adding a node doesn't require editing N other configs |
| Services + long-running jobs | Bounded request/response + cancelable calibration sweeps |
| Live introspection | `imp graph` / `imp topic hz` are first-class |
| Zero-copy SHM (same host) | Frames + clouds + tensors without `cp` |
| Cross-host transport (optional) | Customer X wants split deployment without code change |
| Cross-platform (Win + Linux first-class) | The customer chooses the OS, not us |
| Embeddable in a sealed product | We ship binaries, not Python trees |

### 6.2 The substrate comparison

| Property | Raw ZMQ + SHM (what VGR has) | ROS 2 / DDS | Eclipse Zenoh |
|---|---|---|---|
| Typed topics + schemas | manual | ✅ native | ✅ native |
| Services + jobs | manual | ✅ + actions | ✅ queryables + jobs |
| Discovery | none (hard-coded ports) | ✅ | ✅ admin space |
| Zero-copy SHM on one host | hand-rolled triple buffer | partial | ✅ native |
| Cross-platform | OK | ⚠ Windows second-class | ✅ first-class everywhere |
| Cross-host / WAN | manual relays + base64 | ❌ DDS multicast WAN-hostile | ✅ routers, NAT-native |
| Embed in sealed product | OK | ⚠ heavy to seal | ✅ small, EPL-2.0 / Apache-2.0 |
| Verdict | a brilliant socket library, not middleware | strong on Linux LAN; awkward elsewhere | **fits every row** |

### 6.3 The structural argument: VGR already pays the ZMQ tax

The strongest evidence that ZMQ is the wrong layer for this system is
that **VGR already hand-builds the workarounds for ZMQ's gaps**, and
each workaround is something Zenoh provides natively:

| VGR workaround (in `reference/`) | The ZMQ gap it patches | Zenoh's native answer |
|---|---|---|
| SHM triple buffers for frames (ZMQ carries only `FRAME_READY`, never the image) | no zero-copy large-payload transport | SHM transport carries the payload; one publish |
| `vgr_result_<id>` SHM as "durable path because large payloads can be missed by the live subscriber" | PUB/SUB drops; no durability/last-value | reliable QoS + storage; query the latest result |
| WebSocket + base64 relay for remote compute | no WAN/NAT traversal | a `zenohd` router federates any two sides |
| hard-coded ports `5555/5556/5561/5571/5572/8210` across four configs | no discovery; manual N×M wiring | scouting — add a node, it's found |

Each of these is custom code that needs maintenance, that has bugs, and
that no one outside the original author can debug. The Zenoh decision
deletes them.

### 6.4 Why Zenoh specifically (vs ROS 2)

ROS 2 abstracts its transport behind the RMW interface. Its DDS default
hit well-known walls in industrial use — discovery that doesn't scale,
reliance on multicast (disabled on most enterprise/cloud networks), and
WAN-hostility. The ROS community's answer is **`rmw_zenoh`**, developed
with Open Robotics and now an officially supported RMW. Zenoh is an
**Eclipse Foundation** project from **ZettaScale**, led by the same
people who co-authored DDS — built specifically to fix DDS's transport
limits.

So picking Zenoh is not picking against ROS 2. It's picking the
transport ROS 2 itself is moving toward, *without* yet committing to the
weight of the ROS 2 ecosystem.

### 6.5 The lock-in concern (and why it doesn't apply)

> "If we standardize on Zenoh and it stops working for us, are we
> stuck?"

**No, and this was a deliberate design choice.** Three reasons:

1. **Zenoh is just a daemon + a transport.** Application code in `imp`
   talks to `imp_sdk.bus` (a thin wrapper, ~150 lines of Python). It
   does not import `zenoh` directly. The day we want a different
   transport, we replace `imp_sdk.bus` and the bus crate in Rust. The
   modules, services, jobs, tasks, UI, CLI — none of them change.

2. **The keyexpr namespace + Protobuf schemas + QoS classes are
   substrate-agnostic.** They are the contracts. Any transport that can
   move typed bytes with QoS hints can sit underneath them.

3. **The exit door is open and pre-mapped.** If we ever standardize on
   Linux + MoveIt 2 + tf2, the migration is "switch to ROS 2 with
   `rmw_zenoh`". Same transport substrate, plus the ROS ecosystem. We
   keep the keyexpr layout and the schemas; we replace the runtime
   shells with ROS nodes. This is documented in the spec
   ([`README.md` §2](./README.md#2-why-zenoh--the-substrate-decision))
   as "the one reversible decision here."

In other words: Zenoh is a *correct* default today and a *cheap* exit
tomorrow.

---

## 7. Migration plan — nothing gets lost

The build is staged so each phase is shippable on its own and **every
VGR feature lands before the rebuild is called complete**.

| Phase | Scope | Status |
|---|---|---|
| **P0** | Scaffold + doc-rule | done |
| **P1** | Substrate: Zenoh + Protobuf + key/QoS conventions | done |
| **P2** | Workspace (Station→Process→Task→Run) + HAL framework + first sim robot | done |
| **P3** | Motion stack (FK/IK/collision/plan/trajectory) + spatial (tf/transform) + Scene-fill seam | done |
| **P4** | Packaging + entry-point discovery + tests + CI + smoke gate | done |
| **P5** | Task layer (Graph Compiler + Task Runtime) + end-to-end sim chain | in progress |
| **P6** | Perception modules + calibration → vision-guided pick in sim | next |
| **P7** | Supervisor + full CLI + introspection + bag record/play | |
| **P8** | Services/jobs complete + C++/TS codegen + UI app | |
| **P9** | Remaining drivers + task templates + Nuitka sealing + Win/Linux installers | |
| **P10** | Hardening: TLS/ACL, time discipline, RT determinism, multi-host | |

See [`PLAN.md`](./PLAN.md) for the full per-phase Definition of Done
(README + examples + automated tests + CI green + no `reference/` leaks
+ phase is independently shippable). The DoD applies to every phase —
this is what makes the rebuild production-grade rather than just
functional.

### Carry-forward guarantee

A phase is not done until the VGR features in its scope are reproducible
in `imp`. Concretely:

- **P3** carried Pinocchio FK/IK, Coal collision, OMPL planning, Ruckig
  trajectory, and grasp library — *all 226 motion-core tests pass
  verbatim* against the vendored library.
- **P6** will carry every perception module VGR has (MegaPose, PPF-ICP,
  template, template-SIFT, feature, blob, cuboid, opt-SIFT, YOLO seg,
  preview, track, fusion), plus all calibration types (intrinsics,
  hand-eye, TCP, target, samples).
- **P8** will carry every UI surface (Operator console, RobotViz, Task
  Composer, Calibration Wizard, Grasp Studio, Gripper Studio, Object
  Browser, Pose Library Editor, Scene/Frame Editor, Gizmo Editor,
  Perception Debug, Run Monitor, Asset Manager, Dashboard).
- **P9** ships the pre-configured robot/gripper catalog VGR already
  curates (Franka FR3, UR3/5/10e, KUKA KR16, Fanuc CRX-10iA, xArm;
  Robotiq 2F/Hand-E, Franka Hand, OnRobot).

When **P9 ships**, `reference/` is deletable — every working capability
has a typed, tested, documented home in `imp/`. The CI guard
([`tools/dev/check_no_reference_leak.py`](./platform/tools/dev/check_no_reference_leak.py))
already enforces that no `imp` code links back into `reference/`.

---

## 8. Common objections, answered

### 8.1 "We already shipped VGR. Why touch it?"

We're not retiring VGR's algorithms — we're moving them onto a
substrate that scales to ten customers without ten copies of the
codebase. Every algorithm that ships today ships in `imp`. The
investment is in the layer underneath.

### 8.2 "Rebuilding will take months."

The build is staged (P0–P10) and *every phase is independently
shippable*. P0–P4 are already done; P5 is in progress; P6 lands the
first vision-guided pick. We don't lose months of shipping; we ship
incrementally with `imp` running alongside VGR until VGR is no longer
the production path.

### 8.3 "Zenoh is unfamiliar. What if it has bugs?"

Zenoh is an Eclipse Foundation project, used in production by
ZettaScale's customers, and chosen by Open Robotics as the next-gen ROS
2 RMW. The bug surface is smaller than the ZMQ + custom-SHM +
WebSocket-relay + custom-discovery stack VGR currently maintains
in-house. And our `imp_sdk.bus` wrapper means the day we want to swap
the substrate, application code doesn't change.

### 8.4 "We don't need a CLI / no-code / plugin SDK — we're the only users."

True today. Not true once we sign customer #2, or hire integrator #3.
The cost of building the substrate is paid once; the cost of *not*
having it is paid every time someone new touches the system. Build it
when it's cheap (now), not after we've shipped three copies of the
codebase.

### 8.5 "Protobuf schemas are friction during prototyping."

Schemas live in one place (`crates/schemas/proto/`) and code-gen for
Python/Rust/C++/TS. Adding a field is one edit + a re-codegen + bump
the version. The friction is exactly the same as adding a Pydantic
field today — and the upside is integration errors caught at the edge.

### 8.6 "What about ROS 2?"

Possible long-term direction; the spec calls this out as the **one
reversible decision** in the architecture. If we standardize on Linux
and want MoveIt 2 + tf2, we move to ROS 2 with `rmw_zenoh`. Same
transport, plus the ecosystem. We don't pre-commit to it because (a)
ROS 2 is Linux-first and we have Windows customers, (b) ROS 2 is heavy
to seal for a closed product, and (c) we don't yet need MoveIt — our
motion stack is good and tested. But the door is open.

### 8.7 "Will this slow down feature work?"

Short-term: P5–P6 is foundation work that doesn't visibly add features
to VGR. Medium-term: the first vision-guided pick on `imp` lands in P6,
matching VGR's flagship. Long-term: every new task is YAML + a SDK
plugin instead of orchestrator surgery — feature velocity goes *up*,
not down.

### 8.8 "What if Zenoh dies as a project?"

(a) Eclipse projects don't die quickly; (b) ZettaScale ships it
commercially; (c) we'd rebuild on a different transport in weeks, not
months, because the wrapper is thin and the schemas + keyexpr layout
are substrate-agnostic.

---

## 9. The ask

Land the platform substrate (the work in P0–P5, mostly done) and the
operations layer (P7), then build VGR's features on top in P6 and P8.
**Treat the current VGR codebase as the source of truth for the
algorithms and integrations**, not for the architecture. By the end of
P9, `reference/` is deletable and we have a sealed installable product
with a plugin SDK, a CLI, a no-code UI, and a Win+Linux story — without
having lost a single VGR capability.

The platform investment is real. So is the cost of *not* making it. Each
gap in §3 is paying that cost slowly today. We can keep paying it, or
we can build the substrate once.
