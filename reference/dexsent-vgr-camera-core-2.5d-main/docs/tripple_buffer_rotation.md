
📌 Triple Buffer Rotation — ASCII Reference (ARCHIVE THIS)
1️⃣ Shared Memory Buffers (Per Camera)

For each camera stream (RGB shown):

cam_<camera_id>_rgb_A
cam_<camera_id>_rgb_B
cam_<camera_id>_rgb_C


Each buffer layout:

┌──────────────────────────────────────────────┐
│ HEADER (64 bytes)                            │
│  - timestamp_ns                              │
│  - sequence_id                               │
│  - calib_version                             │
│  - status_flags                              │
│  - reserved                                 │
├──────────────────────────────────────────────┤
│ IMAGE DATA (H × W × 3 bytes, uint8)          │
│  - One complete RGB frame                   │
└──────────────────────────────────────────────┘


Each buffer holds exactly one frame.

2️⃣ Buffer Roles (At Any Instant)

At any moment, the three buffers have roles, not fixed meanings:

WRITE    → Camera Core is writing here
READ     → Safe for Vision Engine to read
STANDBY  → Unused / old data


Only READ is meaningful to consumers.

3️⃣ Initial State (Example)
Buffers:   A        B        C
Roles:   WRITE     READ   STANDBY


Camera will write into A.

4️⃣ Camera Core Write Cycle (ONE FRAME)
Step 1 — Write Image Pixels (WRITE buffer)
WRITE = A

[A]  ← writing image bytes
[B]  (previous frame, safe)
[C]  (unused)


❗ Header is NOT written yet.

Step 2 — Write HEADER LAST (Commit)
[A]  ← header written (VALID flag set)


This marks the frame as complete and safe.

Step 3 — Rotate Buffer Roles

Rotation rule (fixed):

WRITE    → STANDBY
STANDBY  → READ
READ     → WRITE


Applied to our example:

Before rotation:
WRITE=A, READ=B, STANDBY=C

After rotation:
WRITE=C, READ=A, STANDBY=B

Step 4 — Publish FRAME_READY Event
ZMQ EVENT:
{
  rgb_shm = "cam_<camera_id>_rgb_A",
  sequence_id = N
}


📌 Event always points to READ buffer
📌 Event is sent after rotation

5️⃣ Next Frame Cycle (Continues)

Next frame:

WRITE = C
READ  = A
STANDBY = B


After next rotation:

WRITE = B
READ  = C
STANDBY = A


After next rotation:

WRITE = A
READ  = B
STANDBY = C


➡️ Full rotation cycle: A → C → B → A

6️⃣ Vision Engine Read Logic (MANDATORY)

Vision Engine must follow this exact rule:

1. Receive FRAME_READY event
2. Read ONLY the buffer named in the event
3. Read HEADER first
4. If header.VALID → read image
5. If not VALID → skip frame

Vision MUST NOT:
- Guess which buffer is latest
- Read A/B/C blindly
- Track buffer indices itself


The event is the authority.

7️⃣ Race Condition Proof (ASCII Timeline)
Time Axis →
Time →
Camera:  Write A → Rotate → Write C → Rotate → Write B → ...
Vision:              Read A          Read C


Key guarantee:

Camera never writes to READ buffer
Camera always writes to WRITE buffer


Therefore:

WRITE ≠ READ  (always true)


✅ No write/read overlap
✅ No locks required
✅ No partial frames

8️⃣ Slow Vision Engine Case (Frame Drops)

Example:

Camera FPS = 120
Vision FPS = 10


Timeline:

Camera: A → C → B → A → C → B → A ...
Vision:            Read A        Read C


Result:

Some frames skipped

Vision always reads a complete frame

No corruption

No blocking

📌 Frame drop ≠ race condition

9️⃣ What Happens If Vision Is Too Late

If Vision reads a buffer after it has been reused:

HEADER may show:
- OVERRUN flag
- Unexpected sequence_id


Vision behavior:

Skip frame
Log warning
Continue


Camera Core never waits.

10️⃣ One-Line Rule (Pin This)
Camera writes → commits → rotates → notifies
Vision reads → validates → processes

11️⃣ Absolute Invariants (NEVER BREAK)
- Header is written AFTER image bytes
- Event is published AFTER rotation
- Camera never writes to READ buffer
- Vision reads only buffer named in event


Breaking any of these introduces bugs.

12️⃣ Mental Model (Best Summary)
Three boxes.
One pen.
One reader.

The pen never writes in the box being read.
