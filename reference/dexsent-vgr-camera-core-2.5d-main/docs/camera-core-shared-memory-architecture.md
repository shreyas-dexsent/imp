Camera Core – Shared Memory, Buffer Rotation & Multi-Camera Architecture

Status: DESIGN FREEZE
Scope: Camera Core → Shared Memory → ZMQ → Vision Engine
Applies to: Webcam (UVC), Basler (GigE), FLIR, future cameras

1. Core Design Principle (Read First)

Binary sensor data is exchanged through shared memory (data plane), while semantic events flow through IPC (control plane).

Camera Core never sends image data via IPC

Camera Core never waits for Vision Engine

Vision Engine never guesses buffers

Events only tell which buffer is safe to read

2. Camera Core Execution Model
One Camera Core = One Physical Camera

For each camera, we run one Camera Core process:

Webcam  → CameraCore(webcam)
Basler  → CameraCore(basler)
FLIR    → CameraCore(flir)


All processes:

Use the same codebase

Differ only by config

3. Shared Memory Model (Per Camera)

Shared memory is NOT global.
It is per camera instance.

Example: Webcam
cam_cam_webcam_rgb_A
cam_cam_webcam_rgb_B
cam_cam_webcam_rgb_C

Example: Basler
cam_cam_basler_rgb_A
cam_cam_basler_rgb_B
cam_cam_basler_rgb_C


📌 No shared memory is ever shared across cameras.

4. What Is Inside Each Shared Memory Buffer

Each shared memory buffer contains exactly one frame:

┌──────────────────────────────────────────────┐
│ HEADER (64 bytes)                            │
│  - timestamp_ns                              │
│  - sequence_id                               │
│  - calib_version                             │
│  - status_flags                              │
│  - reserved                                 │
├──────────────────────────────────────────────┤
│ IMAGE DATA                                  │
│  - RGB: H × W × 3 (uint8)                   │
│  - Depth: H × W (float32 / uint16)          │
└──────────────────────────────────────────────┘


There are no queues, no multiple frames, no locks.

5. Triple Buffering (A / B / C)

Each stream (RGB, Depth) uses three buffers:

A, B, C


At any instant, buffers have roles:

WRITE    → Camera Core is writing
READ     → Safe for Vision Engine
STANDBY  → Unused / old

6. Triple Buffer Rotation (Exact Rules)
Initial State (example)
WRITE=A   READ=B   STANDBY=C

Camera Core writes one frame
1. Write image bytes to WRITE buffer
2. Write HEADER last (commit)
3. Rotate buffers
4. Publish FRAME_READY event

Rotation rule
WRITE    → STANDBY
STANDBY  → READ
READ     → WRITE

Example rotation
Before: WRITE=A, READ=B, STANDBY=C
After:  WRITE=C, READ=A, STANDBY=B


📌 Event always points to READ buffer, and is published after rotation.

7. Why There Is No Race Condition

Key invariant:

Camera Core never writes to the buffer it publishes.

Guarantees:

Camera writes → commits → rotates → notifies

Vision reads only the buffer named in the event

Camera will not touch that buffer for two more frames

Result:

No read/write overlap

No locks

No partial frames

Frame drops are allowed; corruption is not.

8. ZMQ Event Contract (FRAME_READY)

Example event:

{
  "event": "FRAME_READY",
  "camera_id": "cam_webcam",
  "sequence_id": 144605,
  "timestamp_ns": 1766490427236624800,
  "calib_version": 1,
  "status_flags": 1,

  "rgb_shm": "cam_cam_webcam_rgb_C",
  "rgb_shape": [480, 640, 3],
  "rgb_dtype": "uint8",

  "depth_shm": null
}


Rules:

Event never contains image data

Event explicitly names the safe buffer

Vision must not infer or guess buffers

9. Vision Engine Read Rules (MANDATORY)

Vision Engine must:

1. Receive FRAME_READY
2. Read ONLY buffer named in event
3. Read HEADER first
4. If header.VALID → read image
5. Otherwise → skip frame


Vision Engine must not:

Read A/B/C blindly

Track buffer indices

Assume latest by name

10. Depth Handling (When Present)

Depth is never mixed with RGB.

Additional shared memory per camera:
cam_<camera_id>_depth_A
cam_<camera_id>_depth_B
cam_<camera_id>_depth_C


Properties:

Same header schema

Independent triple buffering

Same sequence_id and timestamp_ns as RGB

ZMQ event includes both:

"rgb_shm":   "cam_cam0_rgb_B",
"depth_shm": "cam_cam0_depth_B"


or

"depth_shm": null

11. Multi-Camera Case: Basler + Webcam Together
What runs
# Webcam
python -m camera_core.main --config cam_webcam.yaml

# Basler
python -m camera_core.main --config cam_basler.yaml

What exists in memory
Webcam SHM:
  cam_cam_webcam_rgb_{A,B,C}

Basler SHM:
  cam_cam_basler_rgb_{A,B,C}

ZMQ stream (interleaved)
camera { camera_id: cam_webcam, rgb_shm: cam_cam_webcam_rgb_B }
camera { camera_id: cam_basler,  rgb_shm: cam_cam_basler_rgb_C }
camera { camera_id: cam_webcam, rgb_shm: cam_cam_webcam_rgb_A }

Vision Engine handling
latest_event_per_camera[camera_id] = event


Each camera:

processed independently

different FPS supported

failures isolated

12. Sequence ID Semantics

sequence_id is uint64

Incremented every frame

Wraparound is theoretical (~4.8 billion years at 120 FPS)

Correct handling:

diff = (new_seq - old_seq) & 0xFFFFFFFFFFFFFFFF


Uses:

Detect frame drops

Detect camera restart

Debug continuity

Not used for:

Cross-camera sync

Absolute time

13. Absolute Rules (Never Break)
- One Camera Core per physical camera
- Shared memory is per camera, never shared
- Header is written after image bytes
- Event is published after rotation
- Vision reads only buffer named in event
- RGB and Depth are separate memory planes

14. One-Line Mental Model (Archive This)

Each camera runs independently, writes frames into its own triple-buffered shared memory, and tells Vision exactly which buffer is safe to read via events.

15. Final Outcome

With this architecture you get:

Deterministic behavior

Zero-copy data flow

No blocking

No race conditions

Unlimited camera scalability

Hardware-agnostic design

This document represents the frozen, correct reference architecture for DexSent Camera Core.