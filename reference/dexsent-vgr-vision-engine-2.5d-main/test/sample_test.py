"""Tests for `test.sample_test`."""

import json
import time

import zmq

print("=" * 80)
print("SIMPLE VISION ENGINE TEST")
print("=" * 80)

# Setup ZMQ
ctx = zmq.Context()

# PUSH socket to send triggers (to event bus PULL socket at 5556)
push = ctx.socket(zmq.PUSH)
push.connect("tcp://127.0.0.1:5556")
print("OK Connected PUSH socket to tcp://127.0.0.1:5556")

# SUB socket to listen for responses (from event bus PUB socket at 5555)
sub = ctx.socket(zmq.SUB)
sub.connect("tcp://127.0.0.1:5555")
sub.setsockopt_string(zmq.SUBSCRIBE, "")  # Subscribe to all topics
print("OK Connected SUB socket to tcp://127.0.0.1:5555")

# Allow sockets to connect
time.sleep(0.5)

print("\n" + "=" * 80)
print("Sending VISION_START command...")
print("=" * 80)

# Start template matching at 1 FPS
payload = {
    "event": "VISION_START",
    "request_id": "req-001",
    "camera_id": "cam_webcam",
    "module": "template_matching",
    "fps_limit": 1.0,
}
push.send_string(json.dumps(payload))  # Send raw JSON (event bus adds topic prefix)
print(f"OK Sent START request: {payload['request_id']}")
print(f"  Camera: {payload['camera_id']}")
print(f"  Module: {payload['module']}")
print(f"  FPS: {payload['fps_limit']}")

print("\n" + "=" * 80)
print("Listening for vision results for 10 seconds...")
print("=" * 80)

# Listen for events
poller = zmq.Poller()
poller.register(sub, zmq.POLLIN)

start_time = time.time()
result_count = 0

try:
    while time.time() - start_time < 10:
        events = dict(poller.poll(timeout=1000))

        if sub in events:
            raw = sub.recv_string()

            # Parse topic and payload
            if " " in raw:
                _, payload_str = raw.split(" ", 1)
            else:
                payload_str = raw

            try:
                msg = json.loads(payload_str)
                event_type = msg.get("event")
                req_id = msg.get("request_id")

                # Only show events related to our request
                if req_id == "req-001" or event_type in [
                    "VISION_START_ACCEPTED",
                    "VISION_START_REJECTED",
                ]:
                    if event_type == "VISION_START_ACCEPTED":
                        print("\nOK START ACCEPTED")
                        print(f"  Request: {req_id}")
                        print(f"  Module: {msg.get('module')}")
                        print(f"  FPS: {msg.get('fps_limit_effective')}")

                    elif event_type == "VISION_START_REJECTED":
                        print("\nERR START REJECTED")
                        print(f"  Reason: {msg.get('reason')}")

                    elif event_type == "VISION_REQUEST_STARTED":
                        print("\nOK REQUEST STARTED")
                        print(f"  Request: {req_id}")

                    elif event_type == "VISION_RESULT":
                        result_count += 1
                        print(f"\nOK RESULT #{result_count}")
                        print(f"  Frame: {msg.get('sequence_id')}")
                        print(f"  Process time: {msg.get('process_time_ms'):.2f}ms")
                        print(f"  Result: {msg.get('result')}")

                    elif event_type == "VISION_PROCESSING_ERROR":
                        print("\nERR ERROR")
                        print(f"  Error: {msg.get('error')}")

            except Exception:
                pass

except KeyboardInterrupt:
    print("\n\nInterrupted by user")

print("\n" + "=" * 80)
print("Sending VISION_STOP command...")
print("=" * 80)

# Stop
payload = {"event": "VISION_STOP", "request_id": "req-001"}
push.send_string(json.dumps(payload))  # Send raw JSON (event bus adds topic prefix)
print("OK Sent STOP request")

# Wait for stop confirmation
time.sleep(2)

print("\n" + "=" * 80)
print(f"Test completed! Received {result_count} vision results")
print("=" * 80)

# Cleanup
push.close()
sub.close()
ctx.term()
