# Vision Engine: Local vs Client-Server Mode

## Overview

Two modes exist, togglable at runtime via the Orchestrator UI **Vision Mode** toggle (local / blackwell) or by editing `orchestrator.local.yaml`. The rest of the stack — camera, orchestrator, robot — always runs on the **client (ASUS Windows)**. Only the vision engine process moves.

---

## Mode 1 — Local (default)

Everything runs on the client. Vision engine communicates with the orchestrator over ZMQ shared memory.

### Start order

```bash
# Terminal 1 — Vision Engine (client)
cd ~/imp/dexsent-vgr-vision-engine-2.5d
python app.py --config configs/engine.local.json   # binds 127.0.0.1:8000

# Terminal 2 — Orchestrator (client)
cd ~/imp/dexsent-vgr-application-orchestrator
python -m orchestrator --config configs/orchestrator.local.yaml

# Terminal 3 — Camera (client)
# Terminal 4 — Robot (client)
```

### Key config — `configs/orchestrator.local.yaml`

```yaml
vision_engine:
  transport: "zmq"          # <-- local mode
  control_push: "tcp://127.0.0.1:5556"
  results_sub:  "tcp://127.0.0.1:5555"
  results_push: "tcp://127.0.0.1:5556"
```

---

## Mode 2 — Client-Server (Blackwell GPU)

Vision engine runs on **dexblack** (`100.100.10.10`). Camera, orchestrator, and robot stay on **ASUS Windows** (`100.120.71.77`). Communication goes over Tailscale VPN via WebSocket.

### Network topology

```
ASUS Windows (100.120.71.77)     dexblack (100.100.10.10)
  camera  ──frames──>  orchestrator ──WS──> vision engine (port 8000)
  robot   <──poses───  orchestrator <──WS──
  data/runs/ (always local, artifacts relayed back over WS)
```

### Start order

```bash
# --- On dexblack (SSH in first) ---
cd ~/imp/dexsent-vgr-vision-engine-2.5d
python app.py \
  --config configs/engine.local.json \
  --host 0.0.0.0 \          # listen on all interfaces (Tailscale included)
  --port 8000

# --- On ASUS Windows (all 3 in separate terminals) ---
cd ~/imp/dexsent-vgr-application-orchestrator
python -m orchestrator --config configs/orchestrator.local.yaml
# (set transport to websocket first — see below, or toggle in UI)

# Camera and robot start as usual on ASUS Windows
```

### Key config — `configs/orchestrator.local.yaml`

```yaml
vision_engine:
  transport: "websocket"                        # <-- client-server mode
  websocket_url: "ws://100.100.10.10:8000/ws"  # dexblack Tailscale IP
  websocket_max_frame_fps: 0                    # 0 = unlimited; set e.g. 10 to cap bandwidth
  http_url: "http://100.100.10.10:8000"        # for prewarm HTTP calls
  # ZMQ fields below are ignored in websocket mode but kept for easy revert
  control_push: "tcp://127.0.0.1:5556"
  results_sub:  "tcp://127.0.0.1:5555"
  results_push: "tcp://127.0.0.1:5556"
```

---

## Switching modes at runtime

In the Orchestrator UI (operator panel) there is a **Vision Mode** toggle:
- **local** → sets `transport = zmq`, reconnects to local ZMQ sockets
- **blackwell** → sets `transport = websocket`, connects to `websocket_url`

No restart needed. The toggle calls `POST /vision/transport` on the orchestrator API.

---

## Artifact / debug file handling

Artifacts (`data/runs/run-<date>/vision/…`) always land on **ASUS Windows**, regardless of mode.

| Mode | How |
|------|-----|
| Local | Vision engine writes directly to the path passed in `output_root` |
| Client-server | Server writes to a temp dir; files are base64-relayed back over WS as `ARTIFACT_FILE` messages; orchestrator writes them to the local `output_root` before deleting the temp dir on dexblack |

No permanent files are stored on dexblack.

---

## Params to watch

### Vision engine server (`configs/engine.local.json`)

| Param | Default | Notes |
|-------|---------|-------|
| `bind_localhost_only` | `true` | **Must be `false` or omitted** when running on dexblack so it listens on the Tailscale interface |

### Orchestrator (`configs/orchestrator.local.yaml`)

| Param | Default | Notes |
|-------|---------|-------|
| `transport` | `zmq` | `zmq` = local, `websocket` = remote |
| `websocket_url` | `ws://100.100.10.10:8000/ws` | dexblack Tailscale IP and port |
| `websocket_max_frame_fps` | `0` | Cap frames/sec sent over WS to save bandwidth; `0` = unlimited |
| `http_url` | `http://100.100.10.10:8000` | Used for prewarm requests in websocket mode |

### Vision engine CLI (`app.py`)

| Flag | Local default | Server value |
|------|--------------|--------------|
| `--host` | `127.0.0.1` | `0.0.0.0` |
| `--port` | `8000` | `8000` (or any free port — update `websocket_url` to match) |
| `--config` | `configs/engine.local.json` | same file, just change `bind_localhost_only` |

### Per-recipe vision params (in `orchestrator/tasks/bin_picking.py`)

These are forwarded inside the `VISION_START` payload:

| Param | Effect |
|-------|--------|
| `save_outputs` | Whether any artifacts are saved at all |
| `save_debug_image` | Save raw detection debug image |
| `save_segmented_region` | Save cropped segmentation image |
| `save_pose_annotated_image` | Save pose overlay image |
| `output_root` | Set automatically from `data/runs/<run_id>/vision`; do not override |

---

## Reverting to local mode

1. Set `transport: "zmq"` in `orchestrator.local.yaml`, **or** toggle the UI to **local**.
2. Start vision engine on the client without `--host 0.0.0.0`.
3. Nothing else changes — ZMQ sockets, SHM, all existing behavior is untouched.

---

## Tailscale IPs (reference)

| Machine | IP | Role |
|---------|----|------|
| asus-windows | `100.120.71.77` | client (camera, orch, robot) |
| dexblack | `100.100.10.10` | vision engine server (Blackwell GPU) |
