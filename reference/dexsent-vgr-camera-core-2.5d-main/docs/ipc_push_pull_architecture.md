The Flow

┌─────────────────────────────────────────────────────────────┐
│                    Event Bus (Broker)                       │
│                                                             │
│  PULL Socket (5556) ──→ [Forward Logic] ──→ PUB Socket (5555) │
│      ▲                                            │         │
└──────┼────────────────────────────────────────────┼─────────┘
       │                                            │
       │ PUSH (cameras send)                        │ SUB (subscribers receive)
       │                                            ▼
┌──────┴────────┐                          ┌────────────────┐
│ cam_webcam    │                          │ Subscriber #1  │
│ (PUSH)        │                          └────────────────┘
└───────────────┘                          ┌────────────────┐
┌───────────────┐                          │ Subscriber #2  │
│ cam_realsense │                          └────────────────┘
│ (PUSH)        │
└───────────────┘
Port Breakdown
Port	Socket Type	Direction	Purpose
5556	PULL (bind)	← Cameras PUSH here	Internal - Cameras send events to bus
5555	PUB (bind)	→ Subscribers SUB here	External - Subscribers receive events
The Code That Does This
Event Bus (event_bus.py:37-44):

while True:
    # Receive from cameras (port 5556)
    msg_bytes = pull_sock.recv()
    event = json.loads(msg_bytes.decode('utf-8'))
    
    # Forward to subscribers (port 5555)
    payload = json.dumps(event, ...)
    pub_sock.send_string(f"{topic} {payload}")
Why This Design?
Decoupling: Cameras don't know about subscribers
Load Balancing: PULL automatically handles multiple PUSH sources
Fan-out: PUB sends to multiple SUB subscribers
Single Entry Point: All camera events flow through one broker
So yes - cameras push to 5556, event bus automatically pulls and republishes to 5555!


=====================


📡 Camera Core IPC Architecture
Push–Pull → Pub/Sub Event Pipeline

This document describes the inter-process communication (IPC) architecture used in the Camera Core service, specifically the Push–Pull → Pub/Sub event pipeline that enables safe, scalable, multi-camera operation.

1️⃣ Problem Statement

The Camera Core system must:

Support one or more cameras (webcam, RealSense, Basler, FLIR, etc.)

Allow multiple camera pipelines to run in parallel

Publish camera events (FRAME_READY, CAMERA_STARTED, CAMERA_ERROR)

Avoid ZeroMQ thread-safety issues

Avoid invalid socket topologies (e.g., PUB → PUB)

Scale without changing code when cameras are added or removed

2️⃣ Final IPC Topology (Frozen Design)
+------------------+       PUSH
| CameraPipeline  | ──────────────┐
| (Thread)        |               │
+------------------+               │
                                   ▼
+------------------+       PUSH   +---------------------+
| CameraPipeline  | ───────────▶ |  Event Bus (PULL)   |
| (Thread)        |               |                     |
+------------------+               |  - Serializes all  |
                                   |    camera events  |
+------------------+       PUSH   |  - Owns PUB socket |
| CameraPipeline  | ───────────▶ |                     |
| (Thread)        |               +----------┬----------+
+------------------+                          │
                                              │ PUB
                                              ▼
                                   +---------------------+
                                   |   ZMQ Subscribers   |
                                   |  (Vision Engine,    |
                                   |   test tools, etc.) |
                                   +---------------------+

3️⃣ Why Push–Pull is Used Internally
ZeroMQ constraint:

ZMQ sockets are not thread-safe

Therefore:

Multiple camera threads must NOT share a PUB socket

Multiple PUB sockets cannot feed into another PUB socket

A serialization stage is required

Solution:

Each camera pipeline uses a PUSH socket

A single Event Bus thread uses a PULL socket

The Event Bus is the only owner of the PUB socket

This ensures:

Thread safety

Deterministic ordering

No message loss due to invalid socket graphs

4️⃣ ZeroMQ Socket Roles
Component	Socket Type	Direction	Responsibility
CameraPipeline	PUSH	connect	Emit camera events
Event Bus	PULL	bind	Collect & serialize events
Event Bus	PUB	bind	Broadcast events
Vision Engine / Test	SUB	connect	Consume events
5️⃣ Message Flow (Step-by-Step)
1. Camera starts

Each CameraPipeline emits:

{
  "event": "CAMERA_STARTED",
  "camera_id": "cam_webcam",
  "timestamp_ns": 1766492341234567890
}


This is sent via PUSH → Event Queue.

2. Frame captured

When a frame is written to shared memory:

{
  "event": "FRAME_READY",
  "camera_id": "cam_webcam",
  "sequence_id": 42,
  "timestamp_ns": 1766492342345678901,
  "rgb_shm": "cam_cam_webcam_rgb_B",
  "rgb_shape": [480, 640, 3],
  "rgb_dtype": "uint8"
}

3. Event Bus publishes

The Event Bus:

Receives events via PULL

Publishes them via ONE PUB socket

Prefixes each message with a topic (e.g. camera)

camera { ...json payload... }

4. Subscriber receives

Subscribers connect once:

sock = ctx.socket(zmq.SUB)
sock.connect("tcp://127.0.0.1:5555")
sock.setsockopt_string(zmq.SUBSCRIBE, "camera")


They receive all camera events, interleaved and ordered.

6️⃣ Why Not PUB → PUB?
❌ Invalid topology
PUB → PUB → SUB   ❌


ZeroMQ does not forward messages between PUB sockets.

This was intentionally avoided by introducing Push–Pull as the internal fan-in mechanism.

7️⃣ Why Not Multiple PUB Sockets?

ZMQ PUB sockets are not thread-safe

Multiple PUB sockets lead to:

dropped messages

inconsistent ordering

undefined behavior

The single PUB owner rule is strictly enforced.

8️⃣ Single vs Multi-Camera Behavior
Single camera
python -m camera_core.main --config cam_webcam.yaml


Flow:

CameraPipeline → PUSH → Event Bus → PUB → SUB

Multiple cameras
python -m camera_core.main \
  --config cam_webcam.yaml \
  --config cam_realsense.yaml


Flow:

CameraPipeline(webcam)   ┐
CameraPipeline(realsense)├→ PUSH → Event Bus → PUB → SUB
CameraPipeline(basler)   ┘


No code changes required.

9️⃣ FPS Independence

Each camera pipeline runs its own loop

Different FPS values are supported (30 / 60 / 120 FPS)

Events are interleaved naturally

No blocking between cameras

🔒 Design Guarantees (Frozen)

Exactly one PUB socket per process

Unlimited camera pipelines

No ZMQ thread-safety violations

Deterministic event flow

Subscriber code remains unchanged

10️⃣ One-Line Architecture Statement

Camera pipelines push events into a serialized event bus (Push–Pull), which safely broadcasts them to downstream consumers via a single Pub/Sub channel.