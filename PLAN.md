# imp — build plan & status

> **Canonical spec:** [`README.md`](./README.md). This document tracks *how* imp gets
> built: current status, an honest architecture re-evaluation, a per-phase
> Definition of Done, and the detailed phased plan to a production-grade release.
> The original **rev-2 plan is preserved verbatim** in the appendix (§6); the
> detailed plan in §4 supersedes its brief "Build / phase order" list.

---

## 1. Current status (as of 2026-05-25)

Built on `main`. Toolchain proven in-container: Rust 1.94 + zenoh 1.9,
Python 3.11 + eclipse-zenoh 1.9, Pinocchio 4 + coal + ompl + open3d,
MuJoCo 3.8. CI runs Rust + Python (lite) + Python (heavy) on every PR.

| Phase | Scope | State | Evidence |
|---|---|---|---|
| P0 | Scaffold + README §9–§22 edits + doc-rule | **done** (PR #1) | 60 plugin folders, `check_docs.py` green |
| P1 | Substrate: `crates/{core,bus,schemas,*-contract,cli}` | **done** (PR #2) | cross-language Zenoh round-trip; schema-tag reject |
| P2 | Workspace+config (`crates/workspace`) + HAL framework + `hal/robot-mujoco-ur5e` | **done** (PR #3) | workspace tests; UR5e command→motion→state over the bus |
| P3 | Motion stack + spatial + Scene-fill | **done** (PRs #4–#7 + commit 9cc0ca9) | 226 motion-core tests; FK/IK/coal/ompl/ruckig over bus; spatial-tf + spatial-transform + motion-cartesian + motion-path-processor + motion-grasp-library landed; **FK is world-frame, collision routes perception Pose6D into Scene** (closes D3); the Scene-fill integration test in `platform/tests/test_scene_fill.py` is green in CI |
| P4 | Quality foundation: packaging + tests + CI | **code-complete** (this branch) | `tools/dev/install_dev.sh` (editable install of every package); `imp_sdk.discover` + lazy `__init__` (closes D6); promoted `verify_*.py` -> `platform/tests/test_modules_bus.py`; `platform/tests/test_smoke_motion_chain.py` smoke gate; `tools/dev/check_no_reference_leak.py` guard; `.github/workflows/ci.yml` runs Rust (build/test/clippy/fmt) + lite Python (doc rule + no-ref-leak + 34 pure-library tests) + heavy Python (motion-core 226 + bus modules + Scene-fill + smoke gate) |

**Verified capabilities today:** typed pub/sub with schema rejection
(Rust↔Python); a sim robot HAL (state + joint/trajectory commands); the
full per-op motion chain **FK · IK · collision · plan · trajectory** wrapping
`robot-algorithms`, each cross-checked against the library's direct call;
**a topic-driven tf frame graph** (`spatial-tf`); **camera→base lifting**
through tf with optional FK-injected `base→tcp` (`spatial-transform`);
**Cartesian planning + shortcut/spline post-processing**; **grasp library
+ `synthesize_grasps`** (orchestrator port); **Scene-fill seam wired** —
perception `Pose6D` mutates `Scene.object_poses` inside the collision
module before each query, and FK composes through the world's `base_pose`.

**Not yet built:** perception, the task layer, services/jobs, supervisor,
UI, C++/TS codegen, packaging/CI (P4), hardening.

---

## 2. Architecture re-evaluation

### 2.1 What is on-architecture (keep going)

- **Substrate matches §6/§7:** one Protobuf IDL → Rust+Python; `imp/<station>/…`
  keyexpr; QoS classes; `schema=` attachment with reject-on-mismatch.
- **`motion-core` is the canonical motion source (§9/§22):** robot-algorithms
  vendored intact, its 226-test suite passing — zero algorithm drift.
- **Compute-Runtime pattern (§9):** `subscribe → keep latest → compute → publish`
  is real and the thin `motion-*` wrappers add no drift (verified to 1e-6).
- **HAL contract (§8)** and **workspace model (§14)** realized and tested.

### 2.2 Divergences & debts (must be resolved, tracked below)

- **D1 — Contract split-brain.** `crates/{hal,module,service}-contract` are Rust
  trait *stubs* nothing uses; the working runtimes are Python (`imp_sdk`). The
  interface descriptor + lifecycle exist twice (Rust + Python) with no shared
  source. → Decide the polyglot contract story and generate one canonical
  descriptor both languages consume.
- **D2 — Compute Runtime lives in the SDK, not the sealed core.** Pragmatic for
  Python modules, but §9/§19 intend a sealed runtime. Needs an explicit decision
  (sealed Python runtime vs. SDK-provided) so "sealed product" still holds.
- **D3 — The Scene-fill seam is unproven.** The headline §9 claim — the runtime
  fills `Scene` from topics each tick (object poses from perception, attach/detach
  on grasp, dynamic ACM) — is *not yet demonstrated*. FK uses `fk_local` (no
  Scene); collision passes `q` directly. This is the core promise and must be
  shown end-to-end.
- **D4 — No integration / no task layer.** Everything runs as independent
  processes wired by hand. The "platform" (compose modules into a graph, run a
  task) does not exist yet — this is the highest-leverage gap.
- **D5 — Verification is ad-hoc.** `verify_*.py` are run by hand; there is no
  Python test suite, no CI, no regression protection. Not production-grade.
- **D6 — Packaging is unreal.** `PYTHONPATH` hacks everywhere; entry-point plugin
  discovery (`imp.hal`/`imp.modules`) is declared but never exercised.
- **D7 — No `spatial-*`/tf.** Perception emits camera-frame poses; motion needs
  base frame. Without `transform`/`tf` the perception→motion chain cannot close.
- **D8 — No perception.** The product is *vision*-guided; zero perception modules.
- **D9 — No services/jobs, no supervisor, no UI, no C++/TS codegen.** Nodes are
  launched manually; nothing orchestrates lifecycle from `deployment.yaml`.
- **D10 — Hardening untouched (§20):** SHM/`BlobRef` for big payloads, real-time
  determinism on the robot loop, TLS/ACL/signing/licensing, time sync, multi-host.

### 2.3 Recommended immediate next steps (the correct order)

Prove and protect the architecture **before** adding breadth:

1. **Finish P3** — `spatial-transform` + a real **Scene-fill demonstration**
   (perception pose → `Scene.object_poses`, grasp attach/detach, world-frame
   FK/collision). Closes D3, D7.
2. **P4 Quality foundation** — packaging + entry-point discovery + promote every
   `verify_*` to a pytest/cargo test + GitHub Actions CI. Closes D5, D6; protects
   everything after.
3. **P5 Task layer** — `crates/tasks` Graph Compiler + Task Runtime, then run the
   motion chain end-to-end in sim as a task graph. Closes D4 — the "it's a
   platform" milestone.

Then perception (P6) → ops (P7) → services+UI (P8) → drivers+packaging (P9) →
hardening (P10).

---

## 3. Definition of Done — applies to *every* phase

A phase is not "done" until **all** of these hold (this is what makes each phase
production-grade, not just functional):

1. **Code + per-folder `README.md`** updated (status, usage, migration source).
2. **Automated tests** committed and green in CI: Rust via `cargo test`, Python via
   `pytest`. Any `verify_*` script is promoted to a CI test — no manual checks.
3. **Examples**: each new `crates|hal|modules|services|jobs` plugin ships a runnable
   `examples/` (existence enforced by `check_docs.py`; smoke-run in CI where feasible).
4. **Docs**: `docs/user` and/or `docs/developer` updated for the new capability.
5. **Quality gates green**: `cargo fmt --check`, `cargo clippy -D warnings`, Python
   lint/format, the doc rule, and the **no-`reference/`-leak** grep.
6. **Shippable**: the phase runs on its own and leaves `main` releasable.

---

## 4. Detailed phased plan (v2) — supersedes the rev-2 "Build / phase order"

Two phases are **new** vs. the original eight: **P4 (Quality foundation)** and
**P10 (Production hardening)**. Perception is consolidated into **P6** (it was
split across the old "first perception" in P3 and "remaining vision" in P7).

### P0 — Scaffold & spec alignment — **DONE**
Tree, per-folder README+examples, root manifests, `check_docs.py`, README §9–§22
edits.

### P1 — Substrate & contracts — **DONE**
`crates/{core,bus,schemas,hal-contract,module-contract,service-contract,cli}`;
Protobuf→Rust+Python; keyexpr/QoS/schema-tag; `imp topic echo|hz`, `version`.
*Carried debt → P4:* C++/TS codegen; unify Rust contract ↔ Python descriptor (D1).

### P2 — Workspace, config & HAL framework — **DONE**
`crates/workspace` (Station→Process→Task→Run, schema-validated, trash backup);
`imp station|process`; `imp_sdk.hal`; `hal/robot-mujoco-ur5e`.

### P3 — Motion stack, spatial & the Scene-fill seam — **CODE-COMPLETE**
- **Done (prior PRs):** `motion-core` (226 tests), `imp_sdk.module` Compute
  Runtime, `motion-pinocchio` FK+IK, `motion-coal`, `motion-ompl`, `motion-ruckig`.
- **Done (this branch):** `spatial-tf` (TfGraph library + bus-resident
  module, 12 pytest); `spatial-transform` (eye-to-hand + eye-in-hand modes,
  pure-math `lift_pose` helpers, 8 pytest); `motion-cartesian`
  (`plan_cartesian` wrapper); `motion-path-processor` (`shortcut_smooth` +
  `spline_fit` wrapper); `motion-grasp-library` (`GraspLibrary` +
  `synthesize_grasps` port from `reference/.../robot_engine/planning/`,
  8 pytest); **FK switched to world-frame `fk(scene,…)`** (closes D3 for FK);
  **collision routes a perception `Pose6D` into `Scene.set_object_pose`
  each tick** before querying (closes D3 for collision).
- **Headline demo:** `platform/tests/test_scene_fill.py` proves (1) FK
  publishes world-frame `Pose6D`, (2) a topic-driven object pose change
  flips the collision verdict, (3) `Scene.attach` registers the dynamic
  ACM allowance so EE↔attached-object contacts are suppressed. Gated on
  `pinocchio + coal + zenoh`; skips cleanly on hosts without them.
- **Deferred to P5 (task layer):** grasp-event schema (`Scene.attach` is
  in-process only for now) and end-to-end task chaining of the modules.
- **Deferred to P4:** the `verify_*.py` → pytest promotion + CI wiring
  (the new pure-library tests are already pytest; the heavy ones need
  the container env P4 sets up).

### P4 — Quality foundation: packaging, tests & CI — **CODE-COMPLETE**
Closes debts **D5** (ad-hoc verification) and **D6** (PYTHONPATH-based packaging).
- **Packaging:** `platform/tools/dev/install_dev.sh` does a single-shot
  editable install of every Python package (sdk + motion-core + every
  module + HAL) in dependency order. `imp_sdk.discover` exposes the
  entry-point plugin discovery layer (`imp.hal` / `imp.modules` /
  `imp.services` / `imp.jobs`) the runtime + CLI sit on; tested with
  6 unit tests (pure-stdlib, runs without zenoh). `imp_sdk/__init__.py`
  is now lazy (PEP 562) so pure-Python tooling — `discover`, `keyexpr`,
  the doc rule — works on hosts without the heavy native env.
- **Test suites:** every `verify_*.py` shell-script promoted to a pytest
  in `platform/tests/test_modules_bus.py` (one bus round-trip per
  module). New shared `imp_sdk.testing.module_under_test` helper bringsup
  a module in a daemon `ModuleNode` thread for test-side publish/recv.
  Pure-library suites (12 + 8 + 8 + 6 = 34 tests) run on every PR; bus
  suites (`test_modules_bus.py`, `test_scene_fill.py`,
  `test_smoke_motion_chain.py`) run in the heavy lane with full env.
- **CI:** [`.github/workflows/ci.yml`](.github/workflows/ci.yml) — three jobs:
  *lite* (Ubuntu, no native deps): `check_docs.py` + `check_no_reference_leak.py`
  + the 34 pure-library tests. *heavy* (Ubuntu + micromamba):
  `tools/dev/environment.yml` env → `install_dev.sh` → motion-core 226
  tests + bus round-trips + Scene-fill + smoke gate. *rust*:
  `cargo fmt --check`, `clippy -D warnings`, `build`, `test` across the
  workspace.
- **Smoke gate:** `platform/tests/test_smoke_motion_chain.py` wires a
  synthetic perception `Pose6D` → spatial-transform → IK → ompl plan
  and asserts the final path's TCP matches the target. Marks the
  motion-chain shippable end-to-end.
- **No-reference-leak guard:** `tools/dev/check_no_reference_leak.py`
  blocks any code-file `reference/` path or `from/import reference`
  reaching `platform/`; doc mentions (e.g. README "Migrates from
  reference:") are explicitly allowed.
- **DoD:** every PR runs the three CI jobs; zero manual `verify_*.py`
  runs needed.

### P5 — Task layer & end-to-end sim chain — *(orig P4)*
- `crates/tasks`: Graph Compiler (schema+wiring validation, placed instances) +
  Task Runtime (sequence FSM reacting to `reject_reason`/events) + `task.yaml`
  schema in the workspace; `jobs/run-task` + `run-store`.
- **End-to-end in sim:** a task graph driving PoseTarget → `spatial-transform` →
  IK → plan → trajectory → `hal/robot-mujoco-ur5e`, collision-gated, with events
  on `ctrl/runs/<id>/events` and artifacts in `runs/<id>/`.
- **Tests:** `imp task validate/run`; a headless CI integration test that runs the
  chain and asserts the sim robot reaches the target. Closes **D4**.

### P6 — Perception, tf & calibration → vision-guided pick — *(consolidated)*
- **Calibration + tf first:** `jobs/calibration-{intrinsics,hand-eye,samples}`,
  `services/calibration-{tcp,target}`; `spatial-tf` consumes calibration (hand-eye
  as a tf edge).
- **Perception:** `perception-ppf-icp` (first, model-based), `perception-yolo`
  (detect/segment), `perception-megapose`; `object-init` job; `grasp-define`.
- **Verification:** `imp bag record`→`bag play` + synthetic RGB-D + reference
  fixtures; unit-test each ported pure fn vs. reference outputs.
- **Milestone:** a full **vision-guided pick in sim** (perception→transform→grasp→
  motion chain from P5). Closes **D7, D8**.

### P7 — Operations: supervisor, CLI, introspection, bagging — *(orig P5)*
- `crates/supervisor` (lifecycle FSM, `deployment.yaml`, restart policy,
  heartbeats); full `imp` CLI (`up/down`, `node`, `lifecycle`, `graph`,
  `service/job`, `bag`, `param`, `deploy`, `doctor`); introspection/Graph service.
- **Tests:** `imp up` launches a deployment; supervisor restarts a killed node;
  `imp doctor` flags schema drift/clock skew/dead topics. Closes D9 (ops half).

### P8 — Services/jobs complete & UI — *(orig P6, + C++/TS codegen)*
- Remaining services/jobs: `pose-library`, `tf-lookup`, `asset-get`,
  `scene-define`, `robot-digital-twin` (`shapes_for` serialization).
- **Schema C++/TS codegen** (closes that half of D1).
- `ui/` app (`lib/{bus,schemas,viewer3d,forms,components}`) + `crates/ui-host`
  (bundle server + `zenoh-bridge-ws` + Tauri shell); views in priority order
  RobotViz (via `shapes_for`) → Calibration Wizard → Task Composer → Grasp Studio
  → Run Monitor → Perception Debug → rest.
- **Tests:** views render from topics+storage with no REST; headless-browser UI
  smoke in CI; a service-from-button path. Closes D9 (UI half).

### P9 — Remaining drivers, templates, SDK & packaging/sealing — *(orig P7+P8)*
- Drivers: `camera-{realsense,basler-gige,flir-gige,uvc}`,
  `robot-{ur-rtde,franka-fr3,xarm}`, grippers, `plc-modbus` (hardware-gated → HIL
  test lane).
- Templates: `examples/{bin-picking-ppf,bin-picking-megapose,follow-object,
  palletizing,conveyor-sort}`; `tools/train` port.
- Sealing/packaging: finalize `sdk/{py,rs}`, **Nuitka** sealing, `crates/installer`
  (MSI/DEB/RPM/AppImage), `imp-router`; Win+Linux symmetry.
- **Tests:** operator + developer installers build on Win+Linux; the installed
  product runs an example workspace.

### P10 — Production hardening & release readiness — **NEW** *(spec §20)*
- **Big payloads:** Zenoh **SHM** + `BlobRef` blob store for frames/clouds/tensors.
- **Determinism:** process-isolate the robot real-time path; pre-dispatch
  trajectory validation (collision/limits/reachability).
- **Security:** TLS + Zenoh access-control on every link; signed assets/models;
  licensing gate at `configure()`.
- **Time:** NTP/PTP discipline; `imp doctor` clock-skew checks.
- **Fault model:** cancelable/time-bounded jobs+motions; supervisor restart;
  robot HAL fail-safe to protective stop.
- **Multi-host + parity:** LAN/WAN router topologies; Win+Linux release builds +
  smoke; soak/regression suite; observability (metrics, logs, per-run bag).
- **DoD:** a hardening checklist passes; signed, reproducible release artifacts.
  Closes **D10**.

---

## 5. Phase → spec traceability

| Phase | README sections |
|---|---|
| P1 | §2 §3 §6 §7 |
| P2 | §8 §14 |
| P3 | §9 §22 |
| P4 | §17 §19 (packaging, SDK) |
| P5 | §5 §11 |
| P6 | §9 (perception) §10 (calibration) |
| P7 | §12 §16 |
| P8 | §10 §13 |
| P9 | §8 §19 (install) §22 |
| P10 | §18 §20 |

---

## 6. Appendix — rev-2 plan (unmodified)

*(Verbatim; its "Build / phase order" is superseded by §4 above.)*

# Plan: Build `imp` in `platform/` by migrating the VGR `reference/` codebase (rev 2)

## Context

Repo holds `README.md` (complete spec for **imp** — a Zenoh-based, topic-driven robotics middleware +
no-code VGR platform) and `reference/` (frozen prior code). Goal: stand up a fresh `platform/` tree that
implements the README exactly, porting **every** working capability out of `reference/` so that when
`reference/` is deleted, nothing is lost. Do not modify or import from `reference/`.

**This revision (rev 2)** responds to five decisions from review:

1. **`services/` mixed services + jobs** — confusing. Split into two sibling top-level folders
   `services/` (synchronous queryables) and `jobs/` (long-running cancelable ops), matching README §10's
   two interaction styles and the `imp.services` / `imp.jobs` entry-point groups already in README §17.
2. **New `reference/robot-algorithms/`** (added on `origin/main`, commit "add robot algo in reference",
   279 files) — a clean, locked, tested, **transport-agnostic** robotics library. Confirmed decisions:
   it is the **canonical source** for `modules/motion-*` + `modules/spatial-*`, and its two
   pydantic-validated YAML schemas become imp's **canonical robot/world/scene config**.
3. **Where do real deploy tasks live?** Not in `crates/tasks` (that's the engine). Deployed tasks are
   **workspace YAML** under `processes/<id>/tasks/<id>.yaml`, authored in Task Composer, run via
   services — functionally equivalent to the reference, cleaner architecture. `examples/` = tutorials.
4. **UIs are disconnected/unstructured** — replace the pile of standalone HTML + 125 REST endpoints +
   polling with **one `imp-ui` app**: a single Zenoh data layer, shared 3D viewer + schema-driven forms,
   and one view-contract (subscribe + storage-query + service-call). New top-level `ui/`.
5. **Tree is editable where sensible** — changes below are deliberate, not cosmetic.

The user also asked that these be **reflected in `README.md`**. I'm in plan mode (can only edit the plan
file), so §"README.md edits to apply" below specifies every change precisely; they are applied as
execution step 0 (before any code).

## Decisions (confirmed)

- **Folder:** `platform/` = the `imp` root (README §19). Sibling to `reference/`; promotable to repo root
  after `reference/` is deleted.
- **Scope:** full polyglot — sealed Rust/C++ core (`crates/`), Python modules/services/SDK + the
  robot-algorithms motion library, **TypeScript/Three.js UI** (`ui/`).
- **services/jobs:** split into `services/` + `jobs/`.
- **Motion source of truth:** `reference/robot-algorithms/` (orchestrator's embedded `robot_engine/` is
  superseded; mined only for grasp planning, which robot-algorithms intentionally excludes).
- **Robot/world/scene config:** adopt robot-algorithms' `dexsent.algorithms.robot_system` v2 and
  `dexsent.algorithms.world` v2 YAML schemas (pydantic descriptions) as imp's canonical formats.

## What `reference/robot-algorithms/` is (and why it changes the plan)

A `pip`-installable Python package `algorithms` with a **locked three-layer architecture** (its
`docs/architecture.md` + `docs/plan.md`): "Applications consume this layer; they do not live inside it…
It does not own ROS nodes, drivers, sessions, UI sync, bagging, or middleware. Application code writes
live state into `Scene`." This is *exactly* imp's "functions, not engines" principle (README §9) — so
imp becomes the application layer that wraps it with the Zenoh topic contract.

| Layer | Package | Contents | imp role |
|---|---|---|---|
| **1. descriptions** | `algorithms/descriptions/` | pydantic models from YAML: `RobotSystemDescription`, `WorldDescription` (+ `TransformSpec`, geometry specs, `JointLimitsSpec`, `KinematicChainSpec`, `CollisionMatrixSpec`, …). No computation. | imp **config schema** for robots/grippers/worlds (config kind, schema-validated, version-pinned — README §14) |
| **2. resolved** | `algorithms/resolved/` | heavy built-once objects: `KinematicModel` (composed Pinocchio model + mimic expansion + limits, LRU-cached on yaml+urdf mtimes), `CollisionModel` (Coal shapes + static ACM + `shapes_for()`), **`Scene`** (mutable live state: `object_poses`, `attached`, `robot_states`, `collision_overlay`), `geometry_cache` | built at module `activate()`; **`Scene` is the live-state seam** the Compute Runtime fills from subscribed topics |
| **3. operations** | `kinematics/` (fk, jacobian, singularity, ik w/ DLS+generic+QP+analytical OPW/6R backends, constraints, costs, validator), `collision/`, `planning/` (joint_space, cartesian, backends: ompl+straight_line, validator, state_validity), `optimization/` (shortcut, spline), `trajectory/` (time_parameterize, backends: ruckig+polynomial, validator), `primitives/` (move_joint, move_l, approach, retreat, via_motion, bin_pick) | stateless NumPy fns | wrapped as the `@module` bodies of `modules/motion-*` |
| assets | `assets/` | robots (franka_fr3, kuka_kr16, fanuc_crx10ia), grippers (franka_hand, robotiq_2f_c2), objects (matka), assemblies | → imp `catalog/` (canonical) |
| configs | `configs/robots/*.yaml`, `configs/worlds/*.yaml` | example robot-system + world YAML | → example workspaces + catalog seeds |
| examples/tests/docs | extensive per-capability | runnable examples + full pytest suite + architecture/yaml_schema/plan docs | → per-module `examples/` + tests + `docs/` |

### Answering "how do the model + pydantics + YAMLs fit, and how does data flow to the UI?"

The data path is a single structured chain — **descriptions(YAML) → resolved(models) → Scene(live) →
typed Zenoh topic/query → UI** — with no second API:

1. **YAMLs** (robot-system, world) live in the workspace as config (validated at load by the carried-over
   **pydantic descriptions**). They store only static facts (URDF refs, mounts, limits, chains, named
   states, geometry, static ACM). Algorithm config (planner/IK tolerances) and runtime poses are **never**
   in YAML (architecture rule §2.7).
2. **Resolved models** (`KinematicModel`/`CollisionModel`) are built once at a motion module's
   `activate()` and cached. The **Compute Runtime** owns a `Scene` and, each tick, writes subscribed
   topic data into it: `hal/<robot>/state` → `Scene.robot_states`; perception `Pose6D` → 
   `Scene.object_poses`; grasp events → `Scene.attach/detach` + dynamic ACM overlay. (This is precisely
   the "your middleware writes into Scene" pattern the library's `docs/plan.md` shows for ROS — imp uses
   Zenoh instead.) Then it calls the stateless op and publishes a typed result (`Pose6D`, `Trajectory`,
   collision validity, …).
3. **UI** reads the *same* resolved truth. The library's `examples/collision/14_inspect_shapes_for_ui.py`
   is the model: a `services/robot-digital-twin` + scene/tf publisher serializes
   `CollisionModel.shapes_for(name)` (the exact shapes the planner queries — meshes, primitives, convex
   hulls, octrees) plus current `Scene` state into typed messages. **RobotViz** / **Scene-Editor**
   subscribe to `hal/<robot>/state` + `tf` + a `scene` query and render via the shared `viewer3d`,
   composing each shape's world pose exactly as the collision query does (`Scene.object_poses[obj] @
   T_parent_shape`, or `fk(...) @ T_parent_shape` for robot links). Schema-driven, one source of truth.

## README.md edits to apply (execution step 0)

| README section | Edit |
|---|---|
| **§9 Functional modules** | Add a paragraph: motion/spatial modules **wrap the `robot-algorithms` three-layer library**; the resolved `KinematicModel`/`CollisionModel` is built at `activate()`, and `Scene` is the live-state object the Compute Runtime fills from topics each tick. Update the "from the VGR reference" line to name robot-algorithms as the motion source. |
| **§10 Services and jobs** | State explicitly that services (sync queryables) and jobs (long-running) — though one descriptor — live in **separate top-level folders `services/` and `jobs/`**, mirroring the `imp.services`/`imp.jobs` entry-point groups. Keep the op table but add a "Folder" column. |
| **§11 Task layer** | Add a short subsection "Where deployed tasks live": deployed tasks are **workspace YAML** (`processes/<id>/tasks/<id>.yaml`), authored in Task Composer or by hand, loaded by the Task Runtime; `crates/tasks` is the engine; `examples/` are tutorials/templates — the runtime knows no fixed task list. |
| **§13 UI integration** | Replace the loose surface list framing with the **structured `imp-ui` architecture**: one app; one `bus/` data layer; shared `viewer3d/` + `forms/` + generated `schemas/`; each view = subscribe + storage-query + service-call, owns no logic. Add the `shapes_for` single-source-of-truth pattern for all 3-D views. |
| **§14 Workspace/assets/config** | Make robot-algorithms YAML canonical: add config kinds `robot_system.yaml` (`dexsent.algorithms.robot_system/2`) and `world.yaml`/scene (`dexsent.algorithms.world/2`). Robot/Gripper/Scene asset rows now reference these schemas. Document the **description-vs-runtime split** (static YAML vs `Scene`/Zenoh-storage live state) and the **three-layer ACM** (robot YAML `allowed_pairs`, world YAML `collision_matrix`, runtime Scene overlay). |
| **§19 Source layout** | (a) split `services/` → `services/` + `jobs/`; (b) add `modules/motion-core/` (the robot-algorithms port: descriptions + resolved + ops) that the thin `motion-*` plugins wrap; (c) rename `crates/ui/` → `crates/ui-host/` and add top-level **`ui/`** (TS app); (d) note `catalog/` is sourced from robot-algorithms `assets/`. |
| **§22 From the VGR reference** | Add a `robot-algorithms` row to the keep/replace table (keep: whole three-layer motion library + YAML schemas + catalog assets; replace: nothing — it's already transport-agnostic, just wrap with Zenoh). Note grasp_library/grasp_feasibility still come from the orchestrator (robot-algorithms excludes them by design). |

## Target file tree — `platform/` (rev 2)

`←` = dominant reference source; `NEW` = no reference code. Every `crates/*`, `hal/*`, `modules/*`,
`services/*`, `jobs/*` ships its own `README.md` + `examples/` (CI-enforced).

```
platform/
├── Cargo.toml · pyproject.toml · README.md · LICENSE
│
├── crates/                         # sealed Rust core
│   ├── core/                       # ids, errors, schema registry, plugin discovery      NEW
│   ├── bus/                        # Zenoh wrapper, key conventions, QoS classes          NEW (replaces ZMQ ipc/ + SHM triple-buffer)
│   ├── schemas/                    # Protobuf IDL + generated Rust/Py/C++/TS bindings      NEW
│   ├── hal-contract/               # HAL trait + base device node + lifecycle             NEW
│   ├── module-contract/            # module trait + Compute Runtime + scheduler            NEW (owns the Scene-fill loop for motion)
│   ├── service-contract/           # service + job traits + dispatcher                     NEW
│   ├── tasks/                      # Graph Compiler + Task Runtime + sequence FSM          NEW (replaces orchestrator/tasks/*)
│   ├── supervisor/                 # lifecycle FSM, deployment manifest, restart policy    NEW
│   ├── workspace/                  # loader, schema validation, asset resolver, trash/     ← orchestrator/storage/*
│   ├── cli/                        # `imp` binary                                          NEW
│   ├── ui-host/                    # serve UI bundle + native (Tauri) shell + zenoh-bridge-ws  ← (host for ui/)
│   └── installer/                  # MSI / DEB / RPM / AppImage                            NEW
│
├── hal/                            # device drivers (Python; perf loops Rust/C++)
│   ├── camera-realsense/           ← camera_core/drivers/realsense_d435i.py (D435i+D405)
│   ├── camera-basler-gige/         ← camera_core/drivers/basler_gige.py
│   ├── camera-flir-gige/           ← camera_core/drivers/flir_blackfly_gige.py (un-stub PySpin)
│   ├── camera-uvc/                 ← camera_core/drivers/webcam_uvc.py
│   ├── robot-mujoco-ur5e/          ← robot_controller/adapters/mujoco_ur5e/ (+ UR5e MJCF)
│   ├── robot-ur-rtde/              NEW (reference: planned only)
│   ├── robot-franka-fr3/           ← robot_controller/adapters/franka_fr3/
│   ├── robot-xarm/                 ← robot_controller/adapters/xarm/
│   ├── gripper-onrobot/            NEW
│   ├── gripper-robotiq/            ← xArm gripper paths + catalog robotiq_2f_c2
│   ├── gripper-franka-hand/        ← franka_fr3 gripper control + catalog franka_hand
│   └── plc-modbus/                 NEW
│
├── modules/                        # functional modules
│   ├── motion-core/                ← robot-algorithms (descriptions + resolved + Scene + ops) — shared pkg `imp-motion-core`
│   ├── perception-yolo/            ← vision_engine/modules/object_proposals/
│   ├── perception-megapose/        ← modules/megapose_bin_picking/ + third_party/megapose6d/ + weights/
│   ├── perception-ppf-icp/         ← modules/ppf_icp_bin_picking/
│   ├── perception-template/        ← modules/template_matching/
│   ├── perception-template-sift/   ← modules/tamplate_matching_sift/ (fix name)
│   ├── perception-feature/         ← modules/feature_matching/
│   ├── perception-blob/            ← modules/blob_detection/
│   ├── perception-cuboid/          ← modules/cuboid_pose_6d/ (+ temporal_filter.py)
│   ├── perception-opt-sift/        ← modules/opt_sift/
│   ├── perception-track/           NEW (extract Kalman from cuboid/temporal_filter.py, generalize)
│   ├── perception-fusion/          NEW (multi-view; not in reference)
│   ├── perception-preview/         ← modules/camera_preview/
│   ├── motion-pinocchio/           ← motion-core kinematics/ (fk, jacobian, singularity, ik+backends)   [thin @module wrapper]
│   ├── motion-coal/                ← motion-core collision/
│   ├── motion-ompl/                ← motion-core planning/{joint_space,backends/ompl}
│   ├── motion-cartesian/           ← motion-core planning/{cartesian,backends/straight_line}
│   ├── motion-path-processor/      ← motion-core optimization/{shortcut,spline}
│   ├── motion-ruckig/              ← motion-core trajectory/{time_parameterize,backends/ruckig,validator}
│   ├── motion-grasp-library/       ← orchestrator/robot_engine/planning/{grasp_library,grasp_candidate,grasp_feasibility}.py  (NOT in robot-algorithms)
│   ├── spatial-tf/                 ← motion-core descriptions/transforms + world base_pose composition (+ tf edges over Zenoh)  NEW glue
│   └── spatial-transform/          ← motion-core transforms + fk world composition + motion primitives frame offsets
│       # robot-algorithms primitives/ (approach/retreat/move_l/via/bin_pick) → reusable task sub-graphs in examples/ + motion-primitives helpers
│
├── services/                       # synchronous queryables
│   ├── calibration-tcp/            ← orchestrator/vision/calibration.py (TCP offset)
│   ├── calibration-target/         ← calibration.py + data target.json (charuco)
│   ├── grasp-define/               ← grasp_authoring.json model + Grasp Studio backend
│   ├── scene-define/               ← world.yaml authoring + obstacles
│   ├── pose-library/               ← orchestrator/storage/pose_store.py
│   ├── tf-lookup/                  ← motion-core transforms / Scene query
│   ├── asset-get/                  ← orchestrator/storage/object_store.py
│   └── robot-digital-twin/         ← orchestrator robot-digital-twin + CollisionModel.shapes_for() serialization
│
├── jobs/                           # long-running cancelable ops
│   ├── calibration-intrinsics/     ← calibration.py (intrinsics)
│   ├── calibration-hand-eye/       ← calibration.py (ArUco/charuco wizard)
│   ├── calibration-samples/        ← calibration.py (raw samples / re-solve)
│   ├── object-init/                ← assets/mesh_converter + .stp→.obj tessellation
│   └── run-task/                   ← orchestrator/core/executor.py (task run as a job) + run-store
│
├── catalog/                        ← robot-algorithms assets/ (robots, grippers, objects, assemblies)
│
├── sdk/{py,rs}/                    # public plugin/task author surface (@hal_device/@module/@service/@job)  NEW
│
├── ui/                             # single TypeScript app (NEW structure; ports reference UI logic)
│   ├── app/                        # shell: routing, layout, view registry, global status (LEDs from topics)
│   ├── lib/
│   │   ├── bus/                    # ONE Zenoh client (zenoh-ts / zenoh-bridge-ws): subscribe/query/call/job   (replaces api.js + polling)
│   │   ├── schemas/                # generated TS types from Protobuf (schema-driven render + forms)
│   │   ├── viewer3d/               # shared Three.js + OCCT scene, URDF/STL/OBJ/DAE loaders, gizmo/TransformControls  ← ui/assets/operator/bin-picking/shared/*
│   │   ├── forms/                  # schema-driven parameter forms from Interface descriptors
│   │   └── components/             # shared widgets (panels, tables, timeline, charts)
│   └── views/                      # one thin folder per surface (compose lib/*; own no logic)
│       ├── dashboard/              ← index.html               · operator/          ← operator*.html
│       ├── robotviz/               ← operator_robotviz.html   · task-composer/     NEW (no-code graph editor)
│       ├── calibration-wizard/     ← calibration.html         · grasp-studio/      ← grasp-studio.js
│       ├── gripper-studio/         ← gripper-studio.js         · gizmo/             ← gizmo-editor.js
│       ├── object-browser/         ← operator_objects.js      · pose-library/      ← operator_waypoints.js
│       ├── scene-editor/           NEW                         · perception-debug/  ← perception_debug.html
│       ├── run-monitor/            ← operator_monitor.js       · asset-manager/     ← operator_assets.js
│
├── docs/{user,developer,architecture,reference}/   ← incl. robot-algorithms docs/* + camera-core/vision docs/*
│
├── examples/                       # runnable example workspaces (tutorials; each a full workspace/)
│   ├── bin-picking-ppf/ · bin-picking-megapose/ · follow-object/ · palletizing/ · dummy-testing/ · conveyor-sort(NEW)/
│
└── tools/
    ├── build/   NEW   ├── train/   ← data-master/extras/seg_obj/   └── dev/   NEW
```

## Migration mapping — keep algorithm, replace transport (rev 2 deltas)

Unchanged rows from rev 1 (camera drivers, perception modules, calibration math, run logging, workspace
stores, UI viewer logic, training pipeline) still hold. **Changed by this revision:**

| Area | → destination | Keep | Replace |
|---|---|---|---|
| `robot-algorithms` (whole) | `modules/motion-core/` + thin `motion-*` plugins | **everything** — descriptions, resolved, all ops, tests, examples, docs | nothing internal; add a Zenoh Compute-Runtime adapter that fills `Scene` from topics and publishes typed results |
| orchestrator `robot_engine/` | (superseded) | only `planning/grasp_*` → `modules/motion-grasp-library` | the rest (duplicates robot-algorithms, messier) is **dropped** |
| robot/gripper/scene config | `crates/workspace` validates robot-algorithms YAML | `RobotSystemDescription`/`WorldDescription` pydantic + the v2 YAML schema | reference `manifest.json`/`bin.json`/`scene.yaml` → migrated into robot-system/world YAML |
| `services/` (flat, mixed) | `services/` + `jobs/` | request/response shapes | folder split by interaction style |
| orchestrator UI sprawl | `ui/` single app + `crates/ui-host` | HTML/CSS/JS, Three.js/OCCT viewers, gizmo/studio logic | 125 REST endpoints + polling + per-page api.js → one `lib/bus` over Zenoh; standalone HTML pages → routed views |
| deployed tasks | workspace `processes/<id>/tasks/<id>.yaml` | the reference task *sequences* (acquire→…→place) as graph+FSM YAML | bespoke Python task runtimes (`_pick_runtime.py` etc.) → declarative graph driven by `crates/tasks` |
| robot/gripper/object assets | `catalog/` ← robot-algorithms `assets/` | manifests, URDFs, meshes, assemblies | path/loader → workspace/asset-get resolution |

## Tasks: engine vs. data vs. tutorials (answering point 3)

- `crates/tasks/` = the **engine** (Graph Compiler + Task Runtime + sequence FSM). Sealed. Knows no
  specific task.
- **Deployed tasks = workspace data**: `workspace/stations/<id>/processes/<id>/tasks/<task_id>.yaml`
  (graph + sequence + asset mappings + params). This is where *your* production tasks sit. Authored in
  **Task Composer** (writes via `task.save` service) or hand-edited; loaded/validated by the runtime
  (`task.validate` / `task.run`).
- `examples/` = starter **templates/tutorials** seeding new workspaces — not the deploy location.
- **UI linkage (functionally like reference, cleaner architecture):** Task Composer reads/writes
  `task.yaml`; Operator console starts/stops via `run.start`/`task.run`/`task.stop`; Run Monitor
  subscribes to `ctrl/runs/<id>/events`. Same user-facing flow as the reference's task JSON + operator
  run + monitor, but declarative graph over Zenoh instead of bespoke Python + REST polling.

## UI architecture (answering point 4)

One application, four rules that kill the current disconnection:

1. **One data layer** — `ui/lib/bus` is the *only* way views touch the system: `subscribe(keyexpr)`,
   `query(storage)`, `call(service)`, `job(name)`. No per-page REST client, no polling. Replaces the
   reference's 125 endpoints + `api.js` + status-LED polling (LEDs become topic subscriptions).
2. **Schema-driven** — `ui/lib/schemas` are generated from the same Protobuf as Rust/Py; panels and
   parameter **forms** auto-build from the Interface descriptor, so a new module/topic shows up in the UI
   with no UI code (README §13).
3. **Shared 3-D** — `ui/lib/viewer3d` is one Three.js+OCCT scene + URDF/mesh loaders + gizmo, reused by
   RobotViz, Grasp/Gripper Studio, Scene/Frame Editor, Gizmo. Replaces the duplicated `three-scene.js` /
   `cad-loader.js` / `urdf-loader.js` copies. 3-D state comes from `robot-digital-twin`'s
   `shapes_for()` serialization (single source of truth).
4. **Thin views** — each `ui/views/<surface>` only composes `lib/*`; it owns no logic and no transport.
   `crates/ui-host` serves the built bundle, runs `zenoh-bridge-ws`, and provides the native Tauri shell.

## Contract & schema layer (define first)

- **Protobuf IDL** (`crates/schemas`): README §6 messages + `Detections, Mask, Roi, PointCloud, Scalar,
  MotionCommand, GripperState/Command, IO, JointSolution, PoseTarget, Path, Trajectory, TfEdge` + per
  service/job request/response/progress/result. Codegen → Rust/Py/C++/**TS**.
- **Config schemas** (distinct from wire messages): robot-algorithms `descriptions` (robot_system v2,
  world v2) carried as the canonical pydantic config layer; `crates/workspace` validates them.
- **Interface descriptor** (README §7) `{name, direction, schema, qos_class, rate_hz?}` for HAL/module/
  service/job; QoS classes + keyexpr namespace per README §6.

## Build / phase order (each phase shippable)

0. **Scaffold + README edits.** Apply the §"README.md edits"; create `platform/` tree with per-folder
   `README.md`/`examples/` stubs + workspace manifests + entry-point wiring + CI doc rule.
1. **Substrate + contracts:** `crates/{core,bus,schemas,hal-contract,module-contract,service-contract}`;
   `hal/camera-realsense` + `hal/robot-mujoco-ur5e` exchanging real topics.
2. **Workspace + config:** `crates/workspace` (adopt robot-algorithms YAML) + `services/asset-get` +
   `jobs/run-task` skeleton; Station→Process→Task→Run loader, `trash/` backup, `catalog/` from
   robot-algorithms assets. (Rename reference `assets/asset-1` → `processes/process-1`.)
3. **Motion core + first perception:** port `robot-algorithms` → `modules/motion-core`; wrap as
   `motion-pinocchio/-coal/-ompl/-ruckig` + `spatial-*` with the Scene-fill Compute Runtime; first vision
   `perception-ppf-icp`, then `perception-megapose`; `motion-grasp-library` from orchestrator.
4. **Graph Compiler + Task Runtime** (`crates/tasks`) running `examples/bin-picking-ppf` end-to-end.
5. **Ops + CLI:** `crates/{supervisor,cli}` + introspection/Graph service; bag record/play; `imp doctor`.
6. **UI:** `ui/` scaffold + `lib/{bus,schemas,viewer3d,forms}` + `crates/ui-host`; then views in priority
   order RobotViz → Calibration Wizard → Task Composer → Grasp Studio → Run Monitor → Perception Debug →
   rest. (3-D fed by `robot-digital-twin` `shapes_for`.)
7. **Remaining** drivers/modules/services/jobs/templates: camera-basler/flir/uvc, robot-ur-rtde(new)/
   franka/xarm, grippers; perception-template/-sift/-opt-sift/-feature/-blob/-cuboid/-preview/-track(new)/
   -fusion(new); calibration + object-init + grasp/scene/pose/tf; example templates; `tools/train` port.
8. **Seal + package:** `sdk/{py,rs}`, Nuitka sealing, `crates/installer` (MSI + DEB/RPM/AppImage),
   `imp-router`; Win+Linux symmetry + `imp doctor` parity.

## Verification (per phase, end-to-end)

- **Schemas/codegen** build in all langs; subscriber rejects on `schema=` mismatch.
- **HAL:** `imp topic echo/hz` show live RealSense `Frame` + mujoco-ur5e state/command round-trip.
- **Motion:** run `robot-algorithms`' own pytest suite against `modules/motion-core` (it ships
  `tests/test_{fk,ik,collision,planning,trajectory,...}` — direct oracles); verify `Scene`-fill adapter
  produces identical results to direct library calls; FK/IK/collision examples reproduce.
- **Perception:** unit-test each ported pure fn vs reference module outputs; `imp bag record`→`bag play`
  into perception with hardware off → diff poses.
- **Task graph:** `imp task validate bin_picking_ppf` then `imp task run` in sim; events on
  `ctrl/runs/<id>/events`; populated `runs/<id>/`.
- **Services/jobs:** `imp calib hand-eye` job streams progress + writes `handeye.json`; `tf.lookup`
  returns a transform; `robot-digital-twin` returns a shape catalogue matching `shapes_for`.
- **UI:** load each view via `zenoh-bridge-ws`; panels render from topics + storage query with **no REST**;
  RobotViz renders robot+scene from `shapes_for` + live state; trigger a service from a button.
- **No-leak check:** grep `platform/` for any `reference/` path/import — must be empty.

## Open follow-ups (resolve in execution, not blockers)

- Multi-robot collision/planning: robot-algorithms documents the single-robot gate + the exact lift
  (architecture.md §17). imp can ship single-robot first; keep the composite-state signature so the
  planner is "born multi-robot."
- Object `.pt` model variants (`_real`/`_synthetic_v1`) + `*.lighting_lab.json`: carry under
  `objects/<id>/`, document in `tools/train` (not in README — additive).
- `poses_xarm/` per-robot pose sets → pose namespacing under the process.
- Default workspace `~/.imp/workspace`; `examples/*` are reorganized copies of `data-master` content.
