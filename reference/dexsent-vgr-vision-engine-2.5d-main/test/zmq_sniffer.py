"""Tests for `test.zmq_sniffer`."""

import json

import zmq

ctx = zmq.Context()
sock = ctx.socket(zmq.SUB)
sock.connect("tcp://127.0.0.1:5555")
sock.setsockopt_string(zmq.SUBSCRIBE, "")  # subscribe to all topics

print("ZMQ sniffer started...")

while True:
    raw = sock.recv_string()
    topic, payload = raw.split(" ", 1)
    print(f"\n[topic={topic}]")
    print(json.dumps(json.loads(payload), indent=2))
