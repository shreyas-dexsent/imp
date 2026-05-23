# Vision Engine Event Flow

## Trigger-Based Processing Event Flow

```
CLIENT                          VISION ENGINE                      THREAD POOL
  |                                   |                                 |
  |  VISION_TRIGGER                   |                                 |
  |  (request_id, camera_id, mode)    |                                 |
  |---------------------------------->|                                 |
  |                                   |                                 |
  |                           [Validate Request]                        |
  |                                   |                                 |
  |                           [Check Camera Available]                  |
  |                                   |                                 |
  |                                   +-- YES ----------------------->  |
  |                                   |                                 |
  |  VISION_REQUEST_STARTED           |                                 |
  |<----------------------------------|                                 |
  |                                   |                                 |
  |                           [Wait for Frame]                          |
  |                                   |                                 |
  |                           [Frame Received]                          |
  |                                   |                                 |
  |                                   |  Submit to Thread Pool          |
  |                                   |-------------------------------->|
  |                                   |                                 |
  |  VISION_REQUEST_IN_PROGRESS       |                                 |
  |<----------------------------------|                                 |
  |                                   |                                 |
  |                                   |         [Execute in Thread]     |
  |                                   |                                 |
  |                                   |  VISION_PROCESSING_STARTED      |
  |<--------------------------------------------------------------------|
  |                                   |                                 |
  |                                   |  VISION_MODULE_IN_PROGRESS      |
  |                                   |  (template_matching)            |
  |<--------------------------------------------------------------------|
  |                                   |                                 |
  |                                   |      [Run Template Matching]    |
  |                                   |                                 |
  |                                   |  VISION_MODULE_IN_PROGRESS      |
  |                                   |  (blob_detection)               |
  |<--------------------------------------------------------------------|
  |                                   |                                 |
  |                                   |      [Run Blob Detection]       |
  |                                   |                                 |
  |  VISION_TRIGGER_RESULT            |                                 |
  |<--------------------------------------------------------------------|
  |                                   |                                 |
  |  VISION_REQUEST_COMPLETED         |                                 |
  |<--------------------------------------------------------------------|
  |                                   |                                 |
```

## Error Flow: Camera Unavailable

```
CLIENT                          VISION ENGINE
  |                                   |
  |  VISION_TRIGGER                   |
  |  (request_id, camera_id, mode)    |
  |---------------------------------->|
  |                                   |
  |                           [Check Camera Available]
  |                                   |
  |                                   +-- NO
  |                                   |
  |  VISION_CAMERA_UNAVAILABLE        |
  |<----------------------------------|
  |                                   |
```

## Error Flow: Duplicate Request ID

```
CLIENT                          VISION ENGINE
  |                                   |
  |  VISION_TRIGGER                   |
  |  (request_id: "req-123")          |
  |---------------------------------->|
  |                                   |
  |                           [Request Active: req-123]
  |                                   |
  |  VISION_TRIGGER                   |
  |  (request_id: "req-123")          |
  |---------------------------------->|
  |                                   |
  |  VISION_TRIGGER_REJECTED          |
  |  (reason: duplicate_request_id)   |
  |<----------------------------------|
  |                                   |
```

## Concurrent Execution

When multiple triggers are sent:

```
REQUEST 1 (template)     REQUEST 2 (template)     REQUEST 3 (blob)
       |                        |                        |
       |                        |                        |
       +------------------------+------------------------+
                                |
                         [Thread Pool]
                                |
              +-----------------+-----------------+
              |                 |                 |
          Thread 1          Thread 2          Thread 3
              |                 |                 |
       [Template Match]  [Template Match]  [Blob Detect]
              |                 |                 |
          [Results 1]       [Results 2]       [Results 3]
              |                 |                 |
              +--------+--------+--------+--------+
                       |
                   [Publish]
                       |
                    CLIENT
```

## Event Types Summary

### Request Lifecycle Events

| Event | When | Contains |
|-------|------|----------|
| `VISION_REQUEST_STARTED` | Request accepted and queued | request_id, camera_id, mode |
| `VISION_REQUEST_IN_PROGRESS` | Frame received, submitting to thread pool | request_id, camera_id, mode |
| `VISION_PROCESSING_STARTED` | Thread starts processing | request_id, frame_id, timestamp_ns, mode |
| `VISION_MODULE_IN_PROGRESS` | Module execution begins | request_id, module, frame_id |
| `VISION_TRIGGER_RESULT` | Processing complete with results | request_id, results, frame_id |
| `VISION_REQUEST_COMPLETED` | Request finished | request_id, camera_id, mode |

### Error Events

| Event | When | Contains |
|-------|------|----------|
| `VISION_TRIGGER_REJECTED` | Validation failed | request_id, reason |
| `VISION_CAMERA_UNAVAILABLE` | Camera not detected | request_id, camera_id, reason |

### Rejection Reasons

| Reason | Description |
|--------|-------------|
| `missing_fields` | request_id or camera_id not provided |
| `duplicate_request_id` | request_id already exists in active requests |
| `invalid_mode` | mode is not "template", "blob", or "both" |

## Mode-Based Execution

### Mode: "template"
```
VISION_PROCESSING_STARTED
  → VISION_MODULE_IN_PROGRESS (template_matching)
    → Run template_matching module
  → VISION_TRIGGER_RESULT
    → results: { "template_matching": {...} }
  → VISION_REQUEST_COMPLETED
```

### Mode: "blob"
```
VISION_PROCESSING_STARTED
  → VISION_MODULE_IN_PROGRESS (blob_detection)
    → Run blob_detection module
  → VISION_TRIGGER_RESULT
    → results: { "blob_detection": {...} }
  → VISION_REQUEST_COMPLETED
```

### Mode: "both"
```
VISION_PROCESSING_STARTED
  → VISION_MODULE_IN_PROGRESS (template_matching)
    → Run template_matching module
  → VISION_MODULE_IN_PROGRESS (blob_detection)
    → Run blob_detection module
  → VISION_TRIGGER_RESULT
    → results: {
        "template_matching": {...},
        "blob_detection": {...}
      }
  → VISION_REQUEST_COMPLETED
```

## Camera Availability Tracking

The engine tracks camera availability based on frame reception:

```
[Engine Start]
    |
    └─ camera_available = {}  (empty set)
    |
[FRAME_READY event received from "cam_webcam"]
    |
    └─ camera_available.add("cam_webcam")
    |
[VISION_TRIGGER for "cam_webcam"]
    |
    └─ Check: "cam_webcam" in camera_available?
       └─ YES → Accept trigger
    |
[VISION_TRIGGER for "cam_unknown"]
    |
    └─ Check: "cam_unknown" in camera_available?
       └─ NO → Reject with VISION_CAMERA_UNAVAILABLE
```

## State Management

### Active Requests Tracking

```python
# Thread-safe storage
active_requests = {
    "req-123": {
        "request_id": "req-123",
        "camera_id": "cam_webcam",
        "mode": "template",
        "params": {}
    },
    "req-456": {
        "request_id": "req-456",
        "camera_id": "cam_webcam",
        "mode": "blob",
        "params": {}
    }
}

# Index by camera for quick lookup
active_by_camera = {
    "cam_webcam": {"req-123", "req-456"}
}
```

### Request Cleanup

When a request completes:
1. Publish `VISION_REQUEST_COMPLETED`
2. Remove from `active_requests`
3. Remove from `active_by_camera`
4. Thread terminates

All operations are protected by a `Lock` for thread safety.
