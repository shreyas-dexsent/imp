# DexSent VGR Vision Engine 2.5D

## Quick Start (Conda)

```bash
conda create -n vgr-vision python=3.11 -y
conda activate vgr-vision
pip install -r requirements.txt
python -m uvicorn app:app --host 127.0.0.1 --port 8000
```

A high-performance, multi-threaded **vision worker** that executes pluggable vision modules (template matching variants, blob detection, etc.) on frames provided by **DexSent Camera Core** via **Shared Memory (SHM)** + **FRAME_READY events**.

In the VIM-303 architecture, the Vision Engine is a **compute service** controlled by the **Orchestrator**, which owns:
- Task state machines (palletizing/sorting/tending)
- Recipes / operator UI
- Robot coordination
- Station data persistence (calibration, templates, pose library, run logs)

---

## Position in the VIM-303 Architecture

**Authority model**
> The **Orchestrator** is the only intended client in production. Vision Engine is an internal worker process.

**Topology (single station)**

```
Camera Core (SHM + FRAME_READY)  --->  Vision Engine (modules, sessions)  --->  Results (ZMQ + Result SHM)
                 ^                                  |
                 |                                  |
                 +----------- Orchestrator ----------+
                     (session control + result consume)
```

---

## Scope: What Vision Engine Does (and Does Not)

### ✅ Responsibilities
- Subscribe to **Camera Core** frame-ready events
- Read frames from **SHM**
- Execute **vision sessions** (one session = one module + parameters + FPS policy)
- Publish results via:
  - **ZMQ events** (for orchestrator subscribers)
  - **Result SHM** (latest result per session for low-latency consumers)
- Provide deterministic session lifecycle: start/stop, throttling, stats
- Provide **2.5D-ready outputs** (2D detection + optional depth-derived camera-frame 3D)

### ❌ Non-responsibilities
- No task logic (palletizing, sorting, tending)
- No recipes or UI
- No robot integration
- No base-frame transforms (no hand–eye application inside Vision Engine)
- No ownership of calibration files or object/template library (Orchestrator owns station data)

---

## Calibration: Ownership vs Consumption (Freeze This)

**Ownership (storage + versioning)**
- Orchestrator owns all station calibration files:
  - camera intrinsics
  - depth scale/alignment
  - hand–eye
  - color profiles

**Consumption (read-only use during processing)**
- Vision Engine may *consume* calibration parameters when provided at session start or via config:
  - ✅ camera intrinsics (K, D, image size) for 2D→3D
  - ✅ depth scale (if RGB-D)
  - ✅ optional color correction profile (for stable template matching)
  - ❌ never applies hand–eye (robot base transforms belong to Orchestrator)

---

## Inputs and Outputs

### Inputs (from Camera Core)

1) **Frames in SHM**  
- Segment: `<camera_id>_frame`
- Layout: header (timestamp/sequence/status) + payload
- Payload types:
  - RGB (mandatory)
  - Depth (optional, if camera provides)

2) **FRAME_READY events**  
- ZMQ SUB endpoint (default): `tcp://127.0.0.1:5555`
- Topic: `camera` (Camera Core event bus)

### Outputs (to Orchestrator)

1) **VISION_RESULT events** (via Camera Core event bus)  
- Vision Engine PUSHes results to the event bus PULL: `tcp://127.0.0.1:5556`
- Subscribers receive results on PUB: `tcp://127.0.0.1:5555`
- Topic: `vision`

2) **Per-session result SHM**  
- Segment: `vgr_result_<session_id>`
- Stores latest result JSON (+ header with timestamp/sequence/status)

---

## Control Plane (Orchestrator → Vision Engine)

Vision Engine exposes a **session control plane** for starting/stopping sessions. Bind to **localhost** by default.

### Recommended control endpoint
- ZMQ **REP** server in Vision Engine
- Orchestrator uses **REQ** client
- Endpoint default: `tcp://127.0.0.1:5561`

---

## Session Model

  A **session** is a long-running worker thread with:
  - `camera_id`
  - `module`
  - `fps_limit` (0 = unlimited)
  - `params` (module-specific)
  - `process_mode` (`continuous` or `trigger_only`)
  - optional calibration inputs used for processing

### Session Start (Orchestrator → Vision Engine)

```json
{
  "event": "VISION_SESSION_START",
  "session_id": "sess-001",
  "camera_id": "cam_realsense_0",
  "module": "conveyor_2p5d_pick",
  "fps_limit": 10.0,
  "process_mode": "continuous",
  "params": {
    "detector": {
      "type": "template_matching_v2",
      "template_id": "battery_A_front_v1",
      "threshold": 0.85,
      "roi": [120, 80, 300, 260]
    },
    "depth": {
      "enable": true,
      "method": "roi_median",
      "min_valid_ratio": 0.60,
      "max_std_m": 0.02
    },
    "stability": {
      "enable": true,
      "min_frames": 5,
      "ema_alpha": 0.4
    }
  },
  "calibration": {
    "camera_intrinsics": {
      "fx": 615.0, "fy": 615.0,
      "cx": 320.0, "cy": 240.0,
      "allow_distortion": true
    },
    "depth_scale": 0.001,
    "color_profile": "factory_daylight"
  }
}
```

**Process modes**
- `continuous` (default): keep processing frames until stop
- `trigger_only`: one-shot mode, processes a single frame and auto-stops

### Session Stop

```json
{
  "event": "VISION_SESSION_STOP",
  "session_id": "sess-001"
}
```

### Start/Stop Responses

```json
{
  "event": "VISION_SESSION_ACCEPTED",
  "session_id": "sess-001",
  "camera_id": "cam_realsense_0",
  "module": "conveyor_2p5d_pick",
  "fps_limit_effective": 10.0
}
```

```json
{
  "event": "VISION_SESSION_REJECTED",
  "session_id": "sess-001",
  "reason": "unknown_module"
}
```

---

## 2.5D Output Contract (Canonical)

A 2.5D session must return a **robot-safe, decision-ready** result format.

### Result Event Schema

```json
{
  "event": "VISION_RESULT",
  "session_id": "sess-001",
  "camera_id": "cam_realsense_0",
  "module": "conveyor_2p5d_pick",
  "sequence_id": 1234,
  "timestamp_ns": 1234567890123456789,
  "process_time_ms": 12.4,
  "result": {
    "valid": true,
    "reject_reason": null,

    "object_id": "battery_A_front",
    "confidence": 0.92,

    "bbox_xywh": [120, 80, 300, 260],
    "centroid_uv": [270, 210],

    "depth_m": 0.432,
    "depth_quality": {"valid_ratio": 0.87, "std_m": 0.008},

    "position_cam_m": [0.034, -0.012, 0.432],

    "yaw_rad": 1.57,
    "yaw_confidence": 0.78,

    "stability": {"stable": true, "frames": 7}
  }
}
```

### Rejection is a first-class outcome
Vision Engine must sometimes return:
- `valid=false`
- `reject_reason` populated (e.g., `low_confidence`, `depth_unstable`, `not_stable_yet`)

The Orchestrator decides recovery actions.

---

## Multi-Threading Model

- One main loop consumes FRAME_READY events.
- For each active session on a camera:
  - push frame metadata into that session’s queue (bounded; typically max 2)
- Each session thread:
  - applies FPS throttle
  - runs module on latest available frame
  - writes latest result into `vgr_result_<session_id>`
  - publishes `VISION_RESULT` event

**Key property:** slow sessions do not block fast sessions.

---

## Modules: Mandatory for 2.5D Conveyor Robotics

Your system can support many modules, but the **mandatory** 2.5D building blocks are:

1) **2D Detection**
- template matching variants (masked, color-gated, feature-based)
- blob detection (where applicable)

2) **Depth ROI Processing** (if RGB-D)
- robust depth statistic (median/trimmed mean)
- invalid pixel rejection
- quality score (valid_ratio, std)

3) **2D → 3D (Camera Frame)**
- back-project centroid (or grasp point) using intrinsics

4) **Yaw Estimation** (optional per task, but recommended)
- bbox/mask PCA, edge direction, or template orientation

5) **Temporal Stability**
- smoothing and “stable frame” gating

---

## Configuration

### `configs/engine.local.json` (example)

```json
{
  "camera_events": {
    "subscribe_endpoint": "tcp://127.0.0.1:5555",
    "subscribe_topic": "camera"
  },
  "results": {
    "publish_endpoint": "tcp://127.0.0.1:5556",
    "publish_topic": "vision"
  },
  "control_plane": {
    "mode": "reqrep",
    "endpoint": "tcp://127.0.0.1:5561",
    "bind_localhost_only": true
  },
  "vision": {
    "fps_limit": 5,
    "process_mode": "trigger_only",
    "modules": [
      { "name": "blob_detection", "params": { "min_area": 500, "max_area": 100000, "threshold": 128, "invert": false } },
      { "name": "template_matching", "params": { "templates_dir": "assets/templates", "threshold": 0.82 } }
    ]
  },
  "shm": {
    "frame_header_bytes": 64,
    "result_max_json_bytes": 4096
  },
  "logging": { "level": "INFO" }
}
```

Template matching supports optional 3D output when depth is available:
- `intrinsics`: `{fx, fy, cx, cy}`
- `depth_window_px`: median window size (default 5)
- `depth_min_m`, `depth_max_m`: depth filter range

---

## Adding a New Vision Module

1) Create a module package:

```
vision_engine/modules/<module_name>/
  module.py
  (optional) helper files
```

2) Implement the interface:

```python
class Module:
    def __init__(self, **params):
        ...

    def run(self, rgb, meta: dict, depth=None) -> dict:
        # rgb: np.ndarray HxWx3 uint8
        # depth: optional np.ndarray HxW (meters or raw units; define by meta)
        # meta: {camera_id, timestamp_ns, sequence_id, intrinsics?, depth_scale?, ...}
        return {"valid": True, "result": {...}}
```

3) Register it in `configs/engine.local.json` under `modules`.

**Guideline:** module instances must be thread-safe or instantiated per session.

---

## Running (Development)

### 1) Start Camera Core

```bash
python -m camera_core.main --config config/cam_realsense_d435i.yaml
```

### 2) Start Vision Engine

```bash
python app.py --config configs/engine.local.json
```

Or run directly with Uvicorn:

```bash
python -m uvicorn app:app --host 127.0.0.1 --port 8000
```

### 3) Drive via Orchestrator

In VIM-303 mode, do not use ad-hoc clients; orchestrator owns sessions.

---

## Testing

Recommended test categories:
- Module unit tests (pure image inputs)
- Session lifecycle tests (start/stop, fps throttling, shm write)
- Integration tests with Camera Core running locally

```bash
pytest -q
```

---

## Security / Deployment Notes

- Bind control plane to `127.0.0.1` by default.
- Do not expose session start/stop endpoints on the network in production.
- Orchestrator is the only external authority.

---

## Repository Structure (Reference)

```
dexsent-vgr-vision-engine-2.5d/
├── app.py
├── configs/
│   └── engine.local.json
├── vision_engine/
│   ├── core/
│   │   ├── engine.py
│   │   ├── session.py
│   │   └── registry.py
│   ├── io/
│   │   ├── camera_events/
│   │   │   └── subscriber.py
│   │   ├── control_plane/
│   │   │   └── server.py
│   │   ├── results/
│   │   │   └── publisher.py
│   │   └── shm/
│   │       ├── shm_reader.py
│   │       ├── shm_result_writer.py
│   │       └── shm_layout.py
│   ├── modules/
│   │   ├── conveyor_2p5d_pick/
│   │   │   └── module.py
│   │   ├── template_matching_v1/
│   │   │   └── module.py
│   │   ├── blob_detection/
│   │   │   └── module.py
│   │   └── ...
│   └── common/
│       ├── image_ops.py
│       └── time.py
└── README.md
```

---

## License

DexSent Robotics Pvt Ltd. Proprietary.
