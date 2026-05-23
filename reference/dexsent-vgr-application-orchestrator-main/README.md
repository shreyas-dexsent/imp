# DexSent VGR Orchestrator 2.5D

API-first orchestrator for VIM-303. This service is the single authority
for run execution, vision session lifecycle, robot coordination, and station data.

## Operating Model (Station -> Process -> Task -> Run)

- Station: one physical cell with cameras/robots and calibration inventory.
- Process: one workflow under a station (exactly one task_type).
- Task: saved configuration under a process (mappings, parameters).
- Run: one execution instance of a task with its own timeline/state.

## Quick Start (Conda)

```bash
conda create -n vgr-orch python=3.11 -y
conda activate vgr-orch
pip install -r requirements.txt
python -m uvicorn app:app --host 127.0.0.1 --port 8100
```

## Start All Services (RealSense Default)

Camera Core (RealSense):
```bash
conda activate vgr-camera
cd C:\Users\asus\dexsent\vgr\dexsent-vgr-camera-core-2.5d
python -m camera_core.main --config config/cam_realsense_d435i.yaml
```

Vision Engine:
```bash
conda activate vgr-vision
cd C:\Users\asus\dexsent\vgr\dexsent-vgr-vision-engine-2.5d
python -m uvicorn app:app --host 127.0.0.1 --port 8000
```

Robot Controller (MuJoCo UR5e):
```bash
conda activate vgr-robot
cd C:\Users\asus\dexsent\vgr\dexsent-vgr-robot-controller-2.5d
python tools/setup_ur5e_mjcf.py
python app.py --config configs/robot.local.json
```

Orchestrator:
```bash
conda activate vgr-orch
cd C:\Users\asus\dexsent\vgr\dexsent-vgr-orchestrator-2.5d
python -m uvicorn app:app --host 127.0.0.1 --port 8100
```

Conda envs (separate):
- `vgr-camera` `C:\Users\asus\miniconda3\envs\vgr-camera`
- `vgr-vision` `C:\Users\asus\miniconda3\envs\vgr-vision`
- `vgr-orch` `C:\Users\asus\miniconda3\envs\vgr-orch`

## Runtime Defaults

- API host/port: `127.0.0.1:8100`
- Data root: `../data` (relative to repo root)
- Vision control: ZMQ PUSH to `tcp://127.0.0.1:5556` (topic `camera`)
- Vision results: ZMQ SUB from `tcp://127.0.0.1:5555` (topic `vision`)
- Robot adapter: `zmq` (command `tcp://127.0.0.1:5571`, state `tcp://127.0.0.1:5572`)

## API Skeleton

Endpoints:
- `GET /health`
- `GET /ready`
- `GET /ui`
- Station: `GET /stations`, `GET /stations/{station_id}`, `PATCH /stations/{station_id}`
- Process: `GET /stations/{station_id}/processes`, `POST /stations/{station_id}/processes`, `GET /processes/{process_id}`, `PATCH /processes/{process_id}`
- Task: `GET /processes/{process_id}/tasks`, `POST /processes/{process_id}/tasks`, `GET /tasks/{task_id}`, `PATCH /tasks/{task_id}`
- Run: `POST /tasks/{task_id}/runs/start`, `POST /runs/{run_id}/pause`, `POST /runs/{run_id}/stop`, `GET /runs/{run_id}`, `GET /runs/{run_id}/timeline`, `GET /tasks/{task_id}/runs`
- Objects (process-scoped): `/processes/{process_id}/objects/...`
- Poses (process-scoped): `/processes/{process_id}/poses/...`
- Calibration (station-scoped): `/stations/{station_id}/calibration/...`
- Vision control: `/vision/sessions/start`, `/vision/sessions/stop`, `/vision/capture`
- Robot: `/robot/*`

## Data Layout (Root = ../data)

```
data/
  stations/
    station-1/
      calibration/
      processes/
        process-1/
          objects/
          poses/
          tasks/
  runs/
  station/            # legacy (auto-imported)
  object_library/     # legacy (auto-imported)
  recipes/            # legacy (auto-imported)
```

## Vision Defaults

The debug UI defaults to `cam_rs_d435i` (RealSense) for vision sessions.

## Debug UI Flow

The `/ui` bench is sequential:
1) Station (select + calibration)
2) Process (objects + waypoints)
3) Task (parameters + mappings)
4) Run (start/stop + monitor)

## End-to-End Pick and Place Demo

1) Put templates in `data/object_library/obj1/templates`.
2) Use the sample legacy recipe at `data/recipes/pick_place_demo.yaml` (auto-imported into the default station/process as a Task).
3) Start a run for the task:

```bash
curl -X POST http://127.0.0.1:8100/tasks/<task_id>/runs/start ^
  -H "Content-Type: application/json" ^
  -d "{\"params\":{}}"
```

## Notes

- This is a minimal skeleton. The demo task runs a basic sequential lifecycle.
- Vision Engine control uses ZMQ events (`VISION_START/STOP`).
- Legacy folders (`data/recipes`, `data/object_library`, `data/station/poses`) are auto-imported into the default station/process.
