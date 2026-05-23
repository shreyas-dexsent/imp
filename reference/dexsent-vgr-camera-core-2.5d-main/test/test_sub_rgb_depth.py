"""Tests for `test.test_sub_rgb_depth`."""

import argparse
import json
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import yaml
import zmq

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from camera_core.shm.header import HEADER_SIZE, unpack_header
from camera_core.shm.shm_buffer import ShmImageBuffer


def load_depth_config(path: str):
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    depth_cfg = (data.get("shared_memory") or {}).get("depth") or {}
    if not depth_cfg:
        return None
    name_prefix = depth_cfg.get("name_prefix")
    width = depth_cfg.get("width")
    height = depth_cfg.get("height")
    dtype = depth_cfg.get("dtype", "float32")
    suffixes = depth_cfg.get("triple_buffers", ["A", "B", "C"])
    if not name_prefix or not width or not height:
        return None
    return {
        "names": [f"{name_prefix}_{s}" for s in suffixes],
        "shape": (int(height), int(width)),
        "dtype": np.dtype(dtype),
    }


def get_shm_buffer(cache, name, shape, dtype):
    buf = cache.get(name)
    if buf is not None:
        return buf
    try:
        buf = ShmImageBuffer(name=name, shape=shape, dtype=dtype, create=False)
    except FileNotFoundError:
        return None
    cache[name] = buf
    return buf


def best_depth_buffer(depth_cfg, cache):
    best = None
    best_seq = -1
    for name in depth_cfg["names"]:
        buf = get_shm_buffer(cache, name, depth_cfg["shape"], depth_cfg["dtype"])
        if buf is None:
            continue
        ts, seq, _, _ = unpack_header(bytes(buf.buf[:HEADER_SIZE]))
        if ts == 0 and seq == 0:
            continue
        if seq > best_seq:
            best_seq = seq
            best = buf
    return best


def colorize_depth(depth_m, max_m):
    clipped = np.clip(depth_m, 0.0, max_m)
    scaled = (clipped / max_m * 255.0).astype(np.uint8)
    return cv2.applyColorMap(scaled, cv2.COLORMAP_JET)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pub", default="tcp://127.0.0.1:5555")
    ap.add_argument("--topic", default="camera")
    ap.add_argument("--config", help="Optional camera config to infer depth SHM info")
    ap.add_argument("--depth-max-m", type=float, default=5.0)
    ap.add_argument("--fps-window", type=float, default=1.0)
    args = ap.parse_args()

    depth_cfg = load_depth_config(args.config) if args.config else None
    if args.config and depth_cfg is None:
        print("Depth config not found in config file; depth will be optional.")

    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.SUB)
    sock.connect(args.pub)
    sock.setsockopt_string(zmq.SUBSCRIBE, args.topic)

    poller = zmq.Poller()
    poller.register(sock, zmq.POLLIN)

    rgb_cache = {}
    depth_cache = {}
    ts_window = deque()

    print(f"Listening on {args.pub} topic={args.topic}")
    print("Press 'q' to quit.")

    while True:
        events = dict(poller.poll(50))
        if sock in events and events[sock] == zmq.POLLIN:
            msg = sock.recv_string()
            _, payload = msg.split(" ", 1)
            event = json.loads(payload)

            if event.get("event") != "FRAME_READY":
                continue

            rgb_name = event.get("rgb_shm")
            rgb_shape = tuple(event.get("rgb_shape") or [])
            rgb_dtype = np.dtype(event.get("rgb_dtype") or "uint8")
            if not rgb_name or not rgb_shape:
                continue

            rgb_buf = get_shm_buffer(rgb_cache, rgb_name, rgb_shape, rgb_dtype)
            if rgb_buf is None:
                continue

            rgb = rgb_buf.img.copy()

            now = time.perf_counter()
            ts_window.append(now)
            while ts_window and (now - ts_window[0]) > args.fps_window:
                ts_window.popleft()
            fps = 0.0
            if len(ts_window) > 1:
                fps = (len(ts_window) - 1) / (ts_window[-1] - ts_window[0])

            overlay = f"FPS {fps:.1f}  seq {event.get('sequence_id', 0)}"
            cv2.putText(
                rgb, overlay, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
            )
            cv2.imshow("RGB", rgb)

            depth = None
            depth_name = event.get("depth_shm")
            depth_shape = event.get("depth_shape")
            depth_dtype = event.get("depth_dtype")

            if depth_name and depth_shape and depth_dtype:
                depth_buf = get_shm_buffer(
                    depth_cache,
                    depth_name,
                    tuple(depth_shape),
                    np.dtype(depth_dtype),
                )
                if depth_buf is not None:
                    depth = depth_buf.img.copy()
            elif depth_cfg:
                depth_buf = best_depth_buffer(depth_cfg, depth_cache)
                if depth_buf is not None:
                    depth = depth_buf.img.copy()

            if depth is not None:
                depth_vis = colorize_depth(depth, args.depth_max_m)
                cv2.putText(
                    depth_vis,
                    overlay,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2,
                )
                cv2.imshow("Depth", depth_vis)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    for buf in rgb_cache.values():
        buf.close()
    for buf in depth_cache.values():
        buf.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
