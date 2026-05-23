# dexsent-vgr-camera-core-2.5d

Multi-camera capture core with a PUSH-PULL-PUB event bus, shared memory triple buffering, and per-camera health endpoints. Supports RGB cameras and optional aligned depth (RealSense D435i / D405).

## Quick start

```bash
# create and activate environment
conda create --name vgr-camera python=3.9 -y
conda activate vgr-camera
pip install -r requirements.txt

# run a single camera
python -m camera_core.main --config config/cam_webcam.yaml
python -m camera_core.main --config config/cam_realsense_d435i.yaml
python -m camera_core.main --config config/cam_realsense_d405.yaml

# run multiple cameras
python -m camera_core.main \
  --config config/cam_webcam.yaml \
  --config config/cam_realsense_d435i.yaml \
  --config config/cam0_basler.yaml
```

## Output architecture

- Camera pipelines publish events to an internal PULL socket.
- The event bus forwards events to a PUB socket for subscribers.
- Shared memory triple buffers hold the latest frames for zero-copy access.

Default addresses:
- Internal PULL: `tcp://127.0.0.1:5556` (fixed in `camera_core/main.py`)
- PUB to subscribers: `ipc.zmq_pub` from your config (default `tcp://127.0.0.1:5555`)
- Topic: `ipc.topic` from your config (default `camera`)

## Events

Each published message is a single line:

```
<topic> <json>
```

Producers can override the topic by including `__topic__` in the JSON payload. The event bus strips `__topic__` before publishing.

Important events:
- `CAMERA_STARTED` when a camera pipeline starts
- `FRAME_READY` when a new frame is written to shared memory
- `CAMERA_ERROR` on capture/write failure

Example `FRAME_READY` payload:

```json
{
  "event":"FRAME_READY",
  "camera_id":"cam_rs_d435i",
  "sequence_id":123,
  "timestamp_ns":1700000000000000000,
  "calib_version":1,
  "status_flags":1,
  "rgb_shm":"cam_cam_rs_d435i_rgb_B",
  "rgb_shape":[480,640,3],
  "rgb_dtype":"uint8",
  "depth_shm":"cam_cam_rs_d435i_depth_C",
  "depth_shape":[480,640],
  "depth_dtype":"float32"
}
```

Depth fields are present only when `shared_memory.depth` is configured and the driver provides depth frames. For RealSense D435i and D405, depth is aligned to color and stored in meters as `float32`.

## Shared memory layout

Each shared memory buffer contains:

```
[64B header][image bytes...]
```

Header fields (see `camera_core/shm/header.py`):
- `timestamp_ns` (uint64)
- `sequence_id` (uint64)
- `calib_version` (uint32)
- `status_flags` (uint32)

Buffer names are `<name_prefix>_<suffix>`, for example:
- `cam_cam_webcam_rgb_A`
- `cam_cam_rs_d435i_depth_C`

The suffixes come from `shared_memory.rgb.triple_buffers` or `shared_memory.depth.triple_buffers`.

## External integration (any codebase)

Any process (Python, C++, Rust, etc.) can connect by:
1. Subscribing to the PUB socket (`ipc.zmq_pub`) and topic (`ipc.topic`).
2. Waiting for `FRAME_READY`.
3. Opening the shared memory name in the event payload.
4. Reading the header and image bytes.

Minimal Python example:

```python
import json
import zmq
from multiprocessing import shared_memory
import numpy as np

ctx = zmq.Context.instance()
sock = ctx.socket(zmq.SUB)
sock.connect("tcp://127.0.0.1:5555")
sock.setsockopt_string(zmq.SUBSCRIBE, "camera")

msg = sock.recv_string()
_, payload = msg.split(" ", 1)
event = json.loads(payload)

if event.get("event") == "FRAME_READY":
    name = event["rgb_shm"]
    shape = tuple(event["rgb_shape"])
    dtype = np.dtype(event["rgb_dtype"])
    shm = shared_memory.SharedMemory(name=name, create=False)
    img = np.ndarray(shape, dtype=dtype, buffer=shm.buf[64:])
    # use img.copy() if you need a stable snapshot
```

## Viewer / FPS test

Use the test subscriber to visualize RGB + depth and compute FPS:

```bash
python test/test_sub_rgb_depth.py --config config/cam_realsense_d435i.yaml
python test/test_sub_rgb_depth.py --config config/cam_realsense_d405.yaml
```

Press `q` to exit.
