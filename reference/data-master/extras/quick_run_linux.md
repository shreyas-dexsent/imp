# Quick Run — Linux

---

## Mode A — Local (all on client)

```bash
# Camera
conda activate vgr-camera; cd ~/imp/dexsent-vgr-camera-core-2.5d; python -m camera_core.main --config config/cam_realsense_d405.yaml

# Vision Engine
conda activate vgr-vision; cd ~/imp/dexsent-vgr-vision-engine-2.5d; python app.py --config configs/engine.local.json

# Robot
conda activate vgr-robot; cd ~/imp/dexsent-vgr-robot-engine; python app.py --config configs/robot.local.json

# Orchestrator
conda activate vgr-orch; cd ~/imp/dexsent-vgr-application-orchestrator; python app.py --config configs/orchestrator.local.yaml
```

---

## Mode B — Server (dexblack runs vision, client runs rest)

```bash
# --- on dexblack (SSH) ---
conda activate vgr-vision; cd ~/imp/dexsent-vgr-vision-engine-2.5d; python app.py --config configs/engine.server.json --host 0.0.0.0 --port 8000

# --- on client ---
# Camera
conda activate vgr-camera; cd ~/imp/dexsent-vgr-camera-core-2.5d; python -m camera_core.main --config config/cam_realsense_d405.yaml

# Robot
conda activate vgr-robot; cd ~/imp/dexsent-vgr-robot-engine; python app.py --config configs/robot.local.json

# Orchestrator
conda activate vgr-orch; cd ~/imp/dexsent-vgr-application-orchestrator; python app.py --config configs/orchestrator.server.yaml
```

---

## Robot variants

```bash
conda activate vgr-robot; cd ~/imp/dexsent-vgr-robot-engine; python app.py --config configs/robot.mujoco.json
conda activate vgr-robot; cd ~/imp/dexsent-vgr-robot-engine; python app.py --config configs/robot.xarm.json
```
