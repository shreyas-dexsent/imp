"""Tests for `test.test_zmq_sub`."""

# import zmq

# ctx = zmq.Context()
# sock = ctx.socket(zmq.SUB)
# sock.connect("tcp://127.0.0.1:5555")

# # Subscribe to topic
# sock.setsockopt_string(zmq.SUBSCRIBE, "camera")

# print("Listening for ZMQ events...")

# while True:
#     msg = sock.recv_string()
#     print(msg)


import json
import time

import zmq

ctx = zmq.Context()
sock = ctx.socket(zmq.SUB)
sock.connect("tcp://127.0.0.1:5555")
sock.setsockopt_string(zmq.SUBSCRIBE, "camera")

# Give subscriber time to establish connection (slow joiner protection)
print("Connecting to ZMQ publisher...")
time.sleep(1.0)

# Add timeout so we don't block forever if no publisher is running
sock.setsockopt(zmq.RCVTIMEO, 10000)  # 10 second timeout

print("Listening for ZMQ events...")

while True:
    try:
        msg = sock.recv_string()
        topic, payload = msg.split(" ", 1)
        event = json.loads(payload)

        # print(
        #     f"[{event['camera_id']}] "
        #     f"seq={event['sequence_id']} "
        #     f"shm={event.get('rgb_shm')}"
        # )

        print(msg)
    except zmq.Again:
        print("No message received (timeout). Is camera_core running?")
        print(
            "Start it with: python -m camera_core.main --config config/cam_webcam.yaml"
        )
        break
    except KeyboardInterrupt:
        print("\nStopping subscriber...")
        break
