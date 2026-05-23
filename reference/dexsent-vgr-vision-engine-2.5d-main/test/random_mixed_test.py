"""Tests for `test.random_mixed_test`."""

import json
import random
import threading
import time
from collections import defaultdict
from datetime import datetime

import zmq

print("=" * 80)
print("RANDOM MIXED PARALLEL VISION ENGINE TEST")
print("10 THREADS WITH RANDOM START TIMES & DURATIONS")
print("=" * 80)

# Shared context
ctx = zmq.Context()

# Shared data structures for result collection
results_lock = threading.Lock()
thread_results = defaultdict(
    lambda: {
        "request_id": None,
        "module": None,
        "start_delay": 0,
        "duration": 0,
        "actual_start_time": None,
        "actual_end_time": None,
        "accepted": False,
        "started": False,
        "result_count": 0,
        "results": [],
        "errors": [],
        "process_times": [],
        "stopped": False,
    }
)

# Event to signal all threads to stop immediately (for Ctrl+C)
emergency_stop_event = threading.Event()


def vision_thread(
    thread_id, module_name, start_delay, duration, camera_id="cam_webcam", fps_limit=1.0
):
    """
    Each thread waits for start_delay seconds, then runs for duration seconds
    """
    request_id = f"req-{module_name[:4]}-{thread_id:02d}"

    # Wait for the start delay
    if start_delay > 0:
        print(
            f"[Thread {thread_id}] ({module_name}) Waiting {start_delay:.1f}s before starting..."
        )
        time.sleep(start_delay)

    if emergency_stop_event.is_set():
        print(f"[Thread {thread_id}] ({module_name}) Cancelled before start")
        return

    # Create thread-local ZMQ sockets
    push = ctx.socket(zmq.PUSH)
    push.connect("tcp://127.0.0.1:5556")

    sub = ctx.socket(zmq.SUB)
    sub.connect("tcp://127.0.0.1:5555")
    sub.setsockopt_string(zmq.SUBSCRIBE, "")

    # Allow sockets to connect
    time.sleep(0.3)

    actual_start_time = time.time()
    print(
        f"[Thread {thread_id}] ({module_name}) STARTING NOW (after {start_delay:.1f}s delay) - Will run for {duration:.1f}s"
    )

    # Initialize thread results
    with results_lock:
        thread_results[thread_id]["request_id"] = request_id
        thread_results[thread_id]["module"] = module_name
        thread_results[thread_id]["start_delay"] = start_delay
        thread_results[thread_id]["duration"] = duration
        thread_results[thread_id]["actual_start_time"] = actual_start_time

    # Send VISION_START command
    payload = {
        "event": "VISION_START",
        "request_id": request_id,
        "camera_id": camera_id,
        "module": module_name,
        "fps_limit": fps_limit,
    }
    push.send_string(json.dumps(payload))
    print(f"[Thread {thread_id}] ({module_name}) Sent START request")

    # Listen for events
    poller = zmq.Poller()
    poller.register(sub, zmq.POLLIN)

    thread_start_time = time.time()

    try:
        while not emergency_stop_event.is_set() and (
            time.time() - thread_start_time < duration
        ):
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
                                    f"[Thread {thread_id}] ({module_name}) ✓ START ACCEPTED - FPS: {msg.get('fps_limit_effective')}"
                                )

                            elif event_type == "VISION_START_REJECTED":
                                print(
                                    f"[Thread {thread_id}] ({module_name}) ✗ START REJECTED - Reason: {msg.get('reason')}"
                                )

                            elif event_type == "VISION_REQUEST_STARTED":
                                thread_results[thread_id]["started"] = True
                                print(
                                    f"[Thread {thread_id}] ({module_name}) ▶ REQUEST STARTED"
                                )

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
                                    f"[Thread {thread_id}] ({module_name}) ◆ RESULT #{thread_results[thread_id]['result_count']} - "
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
                                    f"[Thread {thread_id}] ({module_name}) ⚠ ERROR - {msg.get('error')}"
                                )

                            elif event_type == "VISION_STOP_ACCEPTED":
                                thread_results[thread_id]["stopped"] = True
                                print(
                                    f"[Thread {thread_id}] ({module_name}) ⏹ STOP ACCEPTED"
                                )

                except Exception as e:
                    print(f"[Thread {thread_id}] ({module_name}) Parse error: {e}")

    except KeyboardInterrupt:
        print(f"[Thread {thread_id}] ({module_name}) Interrupted")

    actual_end_time = time.time()
    actual_duration = actual_end_time - actual_start_time

    # Send VISION_STOP command
    print(f"[Thread {thread_id}] ({module_name}) STOPPING after {actual_duration:.1f}s")
    payload = {"event": "VISION_STOP", "request_id": request_id}
    push.send_string(json.dumps(payload))

    # Wait briefly for stop confirmation
    time.sleep(1)

    # Record end time
    with results_lock:
        thread_results[thread_id]["actual_end_time"] = actual_end_time

    # Cleanup
    push.close()
    sub.close()

    print(
        f"[Thread {thread_id}] ({module_name}) COMPLETED - Received {thread_results[thread_id]['result_count']} results in {actual_duration:.1f}s"
    )


# Configuration for random mixed parallel tests
NUM_BLOB_THREADS = 50
NUM_TEMPLATE_THREADS = 50
TOTAL_THREADS = NUM_BLOB_THREADS + NUM_TEMPLATE_THREADS
CAMERA_ID = "cam_webcam"
FPS_LIMIT = 1.0

# Random timing configuration
MIN_START_DELAY = 0  # Minimum seconds before thread starts
MAX_START_DELAY = 10  # Maximum seconds before thread starts
MIN_DURATION = 5  # Minimum seconds thread runs
MAX_DURATION = 15  # Maximum seconds thread runs

# Set random seed for reproducibility (comment out for true randomness)
random.seed(42)

print(f"\nConfiguration:")
print(f"  Total Threads: {TOTAL_THREADS}")
print(f"  Blob Detection: {NUM_BLOB_THREADS}")
print(f"  Template Matching: {NUM_TEMPLATE_THREADS}")
print(f"  Camera: {CAMERA_ID}")
print(f"  FPS Limit: {FPS_LIMIT}")
print(f"  Start Delay Range: {MIN_START_DELAY}-{MAX_START_DELAY}s")
print(f"  Duration Range: {MIN_DURATION}-{MAX_DURATION}s")
print("=" * 80)

# Generate random thread configurations
thread_configs = []
thread_id = 1

# Generate blob_detection configs
for i in range(NUM_BLOB_THREADS):
    config = {
        "thread_id": thread_id,
        "module": "blob_detection",
        "start_delay": random.uniform(MIN_START_DELAY, MAX_START_DELAY),
        "duration": random.uniform(MIN_DURATION, MAX_DURATION),
    }
    thread_configs.append(config)
    thread_id += 1

# Generate template_matching configs
for i in range(NUM_TEMPLATE_THREADS):
    config = {
        "thread_id": thread_id,
        "module": "template_matching",
        "start_delay": random.uniform(MIN_START_DELAY, MAX_START_DELAY),
        "duration": random.uniform(MIN_DURATION, MAX_DURATION),
    }
    thread_configs.append(config)
    thread_id += 1

# Sort by start delay to show timeline
sorted_configs = sorted(thread_configs, key=lambda x: x["start_delay"])

print("\n" + "-" * 80)
print("PLANNED THREAD TIMELINE")
print("-" * 80)
print(f"{'Thread':<8} {'Module':<20} {'Start @':<10} {'Duration':<10} {'End @':<10}")
print("-" * 80)

for config in sorted_configs:
    start_at = config["start_delay"]
    end_at = start_at + config["duration"]
    print(
        f"T{config['thread_id']:<7} {config['module']:<20} {start_at:>8.1f}s  {config['duration']:>8.1f}s  {end_at:>8.1f}s"
    )

print("-" * 80)

# Calculate overlaps
print("\nAnalyzing concurrent threads...")
max_end_time = max(c["start_delay"] + c["duration"] for c in thread_configs)
time_slots = [0] * int(max_end_time + 1)

for config in thread_configs:
    start = int(config["start_delay"])
    end = int(config["start_delay"] + config["duration"])
    for t in range(start, min(end + 1, len(time_slots))):
        time_slots[t] += 1

max_concurrent = max(time_slots)
avg_concurrent = sum(time_slots) / len(time_slots) if time_slots else 0

print(f"Maximum concurrent threads: {max_concurrent}")
print(f"Average concurrent threads: {avg_concurrent:.1f}")
print(f"Total test duration: ~{max_end_time:.1f}s")

print("\n" + "=" * 80)
print("STARTING THREADS WITH RANDOM TIMING")
print("=" * 80)
print()

# Create and start all threads
threads = []
test_start_time = time.time()

for config in thread_configs:
    thread = threading.Thread(
        target=vision_thread,
        args=(
            config["thread_id"],
            config["module"],
            config["start_delay"],
            config["duration"],
            CAMERA_ID,
            FPS_LIMIT,
        ),
        daemon=True,
    )
    threads.append(thread)
    thread.start()
    time.sleep(0.05)  # Tiny delay between thread creation

print(f"All {TOTAL_THREADS} threads created and will start at their scheduled times!")
print("=" * 80)

# Wait for all threads to complete
try:
    for thread in threads:
        thread.join()
except KeyboardInterrupt:
    print("\n\nInterrupted by user - stopping all threads...")
    emergency_stop_event.set()
    for thread in threads:
        thread.join(timeout=3)

test_end_time = time.time()
total_test_time = test_end_time - test_start_time

# Analysis and reporting
print("\n" + "=" * 80)
print("TEST RESULTS ANALYSIS")
print("=" * 80)

print(f"\nTotal wall clock time: {total_test_time:.2f} seconds")
print(f"Total threads executed: {TOTAL_THREADS}")

# Separate results by module
blob_results = {}
template_results = {}

for tid, data in thread_results.items():
    if data["module"] == "blob_detection":
        blob_results[tid] = data
    elif data["module"] == "template_matching":
        template_results[tid] = data

# Per-thread detailed summary
print("\n" + "-" * 80)
print("DETAILED THREAD EXECUTION LOG")
print("-" * 80)
print(
    f"{'Thread':<8} {'Module':<20} {'Delay':<8} {'Duration':<10} {'Started':<8} {'Results':<8} {'Errors':<8}"
)
print("-" * 80)

for tid in sorted(thread_results.keys()):
    data = thread_results[tid]
    started_str = "✓" if data["started"] else "✗"
    print(
        f"T{tid:<7} {data['module']:<20} {data['start_delay']:>6.1f}s  {data['duration']:>8.1f}s  {started_str:<8} {data['result_count']:<8} {len(data['errors']):<8}"
    )

# Per-module summary
print("\n" + "-" * 80)
print("BLOB DETECTION SUMMARY")
print("-" * 80)

blob_total_results = 0
blob_total_errors = 0
blob_process_times = []
blob_total_active_time = 0

for thread_id in sorted(blob_results.keys()):
    data = blob_results[thread_id]
    if data["actual_start_time"] and data["actual_end_time"]:
        active_time = data["actual_end_time"] - data["actual_start_time"]
        blob_total_active_time += active_time

    print(f"\nThread {thread_id} ({data['request_id']}):")
    print(
        f"  Scheduled: Start @ {data['start_delay']:.1f}s, Duration: {data['duration']:.1f}s"
    )
    print(f"  Status: {'✓ Accepted' if data['accepted'] else '✗ Not accepted'}")
    print(f"  Results received: {data['result_count']}")
    print(f"  Errors: {len(data['errors'])}")

    if data["process_times"]:
        avg_time = sum(data["process_times"]) / len(data["process_times"])
        min_time = min(data["process_times"])
        max_time = max(data["process_times"])
        print(
            f"  Processing time - Avg: {avg_time:.2f}ms, Min: {min_time:.2f}ms, Max: {max_time:.2f}ms"
        )
        blob_process_times.extend(data["process_times"])

    blob_total_results += data["result_count"]
    blob_total_errors += len(data["errors"])

print(f"\nBlob Detection Totals:")
print(f"  Total results: {blob_total_results}")
print(f"  Total errors: {blob_total_errors}")
print(f"  Average per thread: {blob_total_results / NUM_BLOB_THREADS:.2f}")
print(f"  Total active time: {blob_total_active_time:.2f}s")

if blob_process_times:
    avg_blob = sum(blob_process_times) / len(blob_process_times)
    min_blob = min(blob_process_times)
    max_blob = max(blob_process_times)
    print(f"  Avg processing time: {avg_blob:.2f}ms")
    print(f"  Min processing time: {min_blob:.2f}ms")
    print(f"  Max processing time: {max_blob:.2f}ms")

print("\n" + "-" * 80)
print("TEMPLATE MATCHING SUMMARY")
print("-" * 80)

template_total_results = 0
template_total_errors = 0
template_process_times = []
template_total_active_time = 0

for thread_id in sorted(template_results.keys()):
    data = template_results[thread_id]
    if data["actual_start_time"] and data["actual_end_time"]:
        active_time = data["actual_end_time"] - data["actual_start_time"]
        template_total_active_time += active_time

    print(f"\nThread {thread_id} ({data['request_id']}):")
    print(
        f"  Scheduled: Start @ {data['start_delay']:.1f}s, Duration: {data['duration']:.1f}s"
    )
    print(f"  Status: {'✓ Accepted' if data['accepted'] else '✗ Not accepted'}")
    print(f"  Results received: {data['result_count']}")
    print(f"  Errors: {len(data['errors'])}")

    if data["process_times"]:
        avg_time = sum(data["process_times"]) / len(data["process_times"])
        min_time = min(data["process_times"])
        max_time = max(data["process_times"])
        print(
            f"  Processing time - Avg: {avg_time:.2f}ms, Min: {min_time:.2f}ms, Max: {max_time:.2f}ms"
        )
        template_process_times.extend(data["process_times"])

    template_total_results += data["result_count"]
    template_total_errors += len(data["errors"])

print(f"\nTemplate Matching Totals:")
print(f"  Total results: {template_total_results}")
print(f"  Total errors: {template_total_errors}")
print(f"  Average per thread: {template_total_results / NUM_TEMPLATE_THREADS:.2f}")
print(f"  Total active time: {template_total_active_time:.2f}s")

if template_process_times:
    avg_template = sum(template_process_times) / len(template_process_times)
    min_template = min(template_process_times)
    max_template = max(template_process_times)
    print(f"  Avg processing time: {avg_template:.2f}ms")
    print(f"  Min processing time: {min_template:.2f}ms")
    print(f"  Max processing time: {max_template:.2f}ms")

# Overall statistics
print("\n" + "-" * 80)
print("OVERALL STATISTICS")
print("-" * 80)

total_results = blob_total_results + template_total_results
total_errors = blob_total_errors + template_total_errors
all_process_times = blob_process_times + template_process_times

print(f"\nTotal results across all threads: {total_results}")
print(
    f"  Blob Detection: {blob_total_results} ({blob_total_results/total_results*100 if total_results > 0 else 0:.1f}%)"
)
print(
    f"  Template Matching: {template_total_results} ({template_total_results/total_results*100 if total_results > 0 else 0:.1f}%)"
)
print(f"\nTotal errors across all threads: {total_errors}")
print(f"Average results per thread: {total_results / TOTAL_THREADS:.2f}")

if all_process_times:
    avg_all = sum(all_process_times) / len(all_process_times)
    min_all = min(all_process_times)
    max_all = max(all_process_times)
    print(f"\nProcessing time statistics (all threads):")
    print(f"  Average: {avg_all:.2f}ms")
    print(f"  Minimum: {min_all:.2f}ms")
    print(f"  Maximum: {max_all:.2f}ms")
    print(f"  Total samples: {len(all_process_times)}")

# Performance insights
print("\n" + "-" * 80)
print("PERFORMANCE INSIGHTS")
print("-" * 80)

accepted_threads = sum(1 for data in thread_results.values() if data["accepted"])
started_threads = sum(1 for data in thread_results.values() if data["started"])

print(
    f"\nThreads accepted: {accepted_threads}/{TOTAL_THREADS} ({accepted_threads/TOTAL_THREADS*100:.1f}%)"
)
print(
    f"Threads started: {started_threads}/{TOTAL_THREADS} ({started_threads/TOTAL_THREADS*100:.1f}%)"
)

if all_process_times and total_results > 0:
    throughput = total_results / total_test_time
    print(f"\nOverall throughput: {throughput:.2f} results/second")

# Module performance comparison
if blob_process_times and template_process_times:
    avg_blob = sum(blob_process_times) / len(blob_process_times)
    avg_template = sum(template_process_times) / len(template_process_times)
    print(f"\nModule Performance Comparison:")
    print(f"  Blob Detection avg: {avg_blob:.2f}ms")
    print(f"  Template Matching avg: {avg_template:.2f}ms")
    if avg_blob > avg_template:
        diff_pct = ((avg_blob - avg_template) / avg_template) * 100
        print(f"  Blob Detection is {diff_pct:.1f}% slower than Template Matching")
    elif avg_template > avg_blob:
        diff_pct = ((avg_template - avg_blob) / avg_blob) * 100
        print(f"  Template Matching is {diff_pct:.1f}% slower than Blob Detection")
    else:
        print(f"  Both modules have similar performance")

print("\n" + "=" * 80)
print("RANDOM MIXED PARALLEL TEST COMPLETED")
print("=" * 80)

# Cleanup
ctx.term()
