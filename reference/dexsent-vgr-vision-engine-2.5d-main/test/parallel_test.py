"""Tests for `test.parallel_test`."""

import json
import threading
import time
from collections import defaultdict
from datetime import datetime

import zmq

print("=" * 80)
print("PARALLEL VISION ENGINE TEST - 5 CONCURRENT THREADS")
print("=" * 80)

# Shared context
ctx = zmq.Context()

# Shared data structures for result collection
results_lock = threading.Lock()
thread_results = defaultdict(
    lambda: {
        "request_id": None,
        "start_time": None,
        "accepted": False,
        "started": False,
        "result_count": 0,
        "results": [],
        "errors": [],
        "process_times": [],
        "stopped": False,
    }
)

# Event to signal all threads to stop
stop_event = threading.Event()


def vision_thread(
    thread_id,
    camera_id="cam_webcam",
    module="template_matching",
    fps_limit=1.0,
    duration=10,
):
    """
    Each thread creates its own sockets and sends/receives vision requests
    """
    request_id = f"req-thread-{thread_id:02d}"

    # Create thread-local ZMQ sockets
    push = ctx.socket(zmq.PUSH)
    push.connect("tcp://127.0.0.1:5556")

    sub = ctx.socket(zmq.SUB)
    sub.connect("tcp://127.0.0.1:5555")
    sub.setsockopt_string(zmq.SUBSCRIBE, "")

    # Allow sockets to connect
    time.sleep(0.5)

    print(f"[Thread {thread_id}] Starting with request_id: {request_id}")

    # Initialize thread results
    with results_lock:
        thread_results[thread_id]["request_id"] = request_id
        thread_results[thread_id]["start_time"] = time.time()

    # Send VISION_START command
    payload = {
        "event": "VISION_START",
        "request_id": request_id,
        "camera_id": camera_id,
        "module": module,
        "fps_limit": fps_limit,
    }
    push.send_string(json.dumps(payload))
    print(f"[Thread {thread_id}] Sent START request")

    # Listen for events
    poller = zmq.Poller()
    poller.register(sub, zmq.POLLIN)

    start_time = time.time()

    try:
        while not stop_event.is_set() and (time.time() - start_time < duration):
            events = dict(poller.poll(timeout=500))

            if sub in events:
                raw = sub.recv_string()

                # Parse topic and payload
                if " " in raw:
                    topic, payload_str = raw.split(" ", 1)
                else:
                    topic, payload_str = "", raw

                try:
                    msg = json.loads(payload_str)
                    event_type = msg.get("event")
                    req_id = msg.get("request_id")

                    # Only process events for this thread's request
                    if req_id == request_id:
                        with results_lock:
                            if event_type == "VISION_START_ACCEPTED":
                                thread_results[thread_id]["accepted"] = True
                                print(
                                    f"[Thread {thread_id}] START ACCEPTED - FPS: {msg.get('fps_limit_effective')}"
                                )

                            elif event_type == "VISION_START_REJECTED":
                                print(
                                    f"[Thread {thread_id}] START REJECTED - Reason: {msg.get('reason')}"
                                )

                            elif event_type == "VISION_REQUEST_STARTED":
                                thread_results[thread_id]["started"] = True
                                print(f"[Thread {thread_id}] REQUEST STARTED")

                            elif event_type == "VISION_RESULT":
                                thread_results[thread_id]["result_count"] += 1
                                result_data = {
                                    "sequence_id": msg.get("sequence_id"),
                                    "process_time_ms": msg.get("process_time_ms"),
                                    "result": msg.get("result"),
                                    "timestamp": time.time(),
                                }
                                thread_results[thread_id]["results"].append(result_data)
                                thread_results[thread_id]["process_times"].append(
                                    msg.get("process_time_ms", 0)
                                )
                                print(
                                    f"[Thread {thread_id}] RESULT #{thread_results[thread_id]['result_count']} - "
                                    f"Frame: {msg.get('sequence_id')}, "
                                    f"Time: {msg.get('process_time_ms'):.2f}ms"
                                )

                            elif event_type == "VISION_PROCESSING_ERROR":
                                error_data = {
                                    "error": msg.get("error"),
                                    "timestamp": time.time(),
                                }
                                thread_results[thread_id]["errors"].append(error_data)
                                print(
                                    f"[Thread {thread_id}] ERROR - {msg.get('error')}"
                                )

                            elif event_type == "VISION_STOP_ACCEPTED":
                                thread_results[thread_id]["stopped"] = True
                                print(f"[Thread {thread_id}] STOP ACCEPTED")

                except Exception as e:
                    print(f"[Thread {thread_id}] Parse error: {e}")

    except KeyboardInterrupt:
        print(f"[Thread {thread_id}] Interrupted")

    # Send VISION_STOP command
    print(f"[Thread {thread_id}] Sending STOP request")
    payload = {"event": "VISION_STOP", "request_id": request_id}
    push.send_string(json.dumps(payload))

    # Wait briefly for stop confirmation
    time.sleep(1)

    # Cleanup
    push.close()
    sub.close()

    print(
        f"[Thread {thread_id}] Completed - Received {thread_results[thread_id]['result_count']} results"
    )


# Configuration for parallel tests
NUM_THREADS = 50
TEST_DURATION = 10  # seconds
CAMERA_ID = "cam_webcam"
MODULE = "template_matching"
FPS_LIMIT = 1.0

print(f"\nStarting {NUM_THREADS} parallel threads...")
print(f"Configuration:")
print(f"  Camera: {CAMERA_ID}")
print(f"  Module: {MODULE}")
print(f"  FPS Limit: {FPS_LIMIT}")
print(f"  Duration: {TEST_DURATION} seconds")
print("=" * 80)
print()

# Create and start threads
threads = []
test_start_time = time.time()

for i in range(1, NUM_THREADS + 1):
    thread = threading.Thread(
        target=vision_thread,
        args=(i, CAMERA_ID, MODULE, FPS_LIMIT, TEST_DURATION),
        daemon=True,
    )
    threads.append(thread)
    thread.start()
    time.sleep(0.1)  # Small delay to stagger thread starts

print(f"\nAll {NUM_THREADS} threads started!")
print("=" * 80)

# Wait for all threads to complete
try:
    for thread in threads:
        thread.join()
except KeyboardInterrupt:
    print("\n\nInterrupted by user - stopping all threads...")
    stop_event.set()
    for thread in threads:
        thread.join(timeout=2)

test_end_time = time.time()
total_test_time = test_end_time - test_start_time

# Analysis and reporting
print("\n" + "=" * 80)
print("TEST RESULTS ANALYSIS")
print("=" * 80)

print(f"\nTotal test duration: {total_test_time:.2f} seconds")
print(f"Threads executed: {NUM_THREADS}")

# Per-thread summary
print("\n" + "-" * 80)
print("PER-THREAD SUMMARY")
print("-" * 80)

total_results = 0
total_errors = 0
all_process_times = []

for thread_id in sorted(thread_results.keys()):
    data = thread_results[thread_id]
    print(f"\nThread {thread_id} ({data['request_id']}):")
    print(f"  Status: {'✓ Accepted' if data['accepted'] else '✗ Not accepted'}")
    print(f"  Started: {'✓ Yes' if data['started'] else '✗ No'}")
    print(f"  Results received: {data['result_count']}")
    print(f"  Errors: {len(data['errors'])}")

    if data["process_times"]:
        avg_time = sum(data["process_times"]) / len(data["process_times"])
        min_time = min(data["process_times"])
        max_time = max(data["process_times"])
        print(
            f"  Processing time - Avg: {avg_time:.2f}ms, Min: {min_time:.2f}ms, Max: {max_time:.2f}ms"
        )
        all_process_times.extend(data["process_times"])

    total_results += data["result_count"]
    total_errors += len(data["errors"])

# Overall statistics
print("\n" + "-" * 80)
print("OVERALL STATISTICS")
print("-" * 80)

print(f"\nTotal results across all threads: {total_results}")
print(f"Total errors across all threads: {total_errors}")
print(f"Average results per thread: {total_results / NUM_THREADS:.2f}")

if all_process_times:
    avg_all = sum(all_process_times) / len(all_process_times)
    min_all = min(all_process_times)
    max_all = max(all_process_times)
    print(f"\nProcessing time statistics (all threads):")
    print(f"  Average: {avg_all:.2f}ms")
    print(f"  Minimum: {min_all:.2f}ms")
    print(f"  Maximum: {max_all:.2f}ms")
    print(f"  Total samples: {len(all_process_times)}")

# Expected vs actual results
expected_results_per_thread = TEST_DURATION * FPS_LIMIT
expected_total = expected_results_per_thread * NUM_THREADS
print(
    f"\nExpected results (based on {FPS_LIMIT} FPS × {TEST_DURATION}s × {NUM_THREADS} threads): ~{expected_total:.0f}"
)
print(f"Actual results: {total_results}")
print(
    f"Success rate: {(total_results / expected_total * 100) if expected_total > 0 else 0:.1f}%"
)

# Performance insights
print("\n" + "-" * 80)
print("PERFORMANCE INSIGHTS")
print("-" * 80)

accepted_threads = sum(1 for data in thread_results.values() if data["accepted"])
started_threads = sum(1 for data in thread_results.values() if data["started"])

print(
    f"\nThreads accepted: {accepted_threads}/{NUM_THREADS} ({accepted_threads/NUM_THREADS*100:.1f}%)"
)
print(
    f"Threads started: {started_threads}/{NUM_THREADS} ({started_threads/NUM_THREADS*100:.1f}%)"
)

if all_process_times and total_results > 0:
    throughput = total_results / total_test_time
    print(f"Overall throughput: {throughput:.2f} results/second")
    print(f"Per-thread throughput: {throughput/NUM_THREADS:.2f} results/second")

print("\n" + "=" * 80)
print("PARALLEL TEST COMPLETED")
print("=" * 80)

# Cleanup
ctx.term()
