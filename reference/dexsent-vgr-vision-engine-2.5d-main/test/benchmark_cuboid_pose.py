#!/usr/bin/env python3
"""
Benchmark script for cuboid 6D pose estimation.

Generates synthetic cuboid data and measures performance.
Usage:
    python benchmark_cuboid_pose.py --num-frames 100 --visualize
"""

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
from vision_engine.io.data_plane.frame_bundle import FrameBundle
from vision_engine.modules.cuboid_pose_6d.module import CuboidPose6DModule
from vision_engine.modules.cuboid_pose_6d.pose_estimator import CuboidPoseEstimator


def generate_synthetic_cuboid_scene(
    frame_idx: int,
    image_shape: tuple = (480, 640, 3),
    depth_shape: tuple = (480, 640),
    center: np.ndarray = None,
    rotation_angles: np.ndarray = None,
    cuboid_dims: Dict[str, float] = None,
) -> Dict[str, Any]:
    """
    Generate synthetic RGB-D scene with a cuboid.

    Args:
        frame_idx: Frame number (for animation)
        image_shape: (H, W, 3) for RGB
        depth_shape: (H, W) for depth
        center: (3,) center position
        rotation_angles: (3,) Euler angles in degrees
        cuboid_dims: {"L": mm, "B": mm, "H": mm}

    Returns:
        Dict with "rgb" and "depth" arrays
    """
    h, w = depth_shape

    # Default parameters
    if center is None:
        center = np.array([0.0, 0.0, 1.0])
    if rotation_angles is None:
        rotation_angles = np.array([frame_idx * 2, frame_idx * 1.5, frame_idx])
    if cuboid_dims is None:
        cuboid_dims = {"L": 100, "B": 80, "H": 60}

    # Create background depth (simple wall at 1.5m)
    depth = np.full(depth_shape, 1.5, dtype=np.float32)

    # Add some noise to depth
    depth += np.random.normal(0, 0.01, depth_shape)

    # Simple cuboid depth rendering (very basic)
    # In real scenario, would render 3D model
    h_center, w_center = h // 2, w // 2
    cuboid_h, cuboid_w = 120, 140  # pixels

    depth[
        h_center - cuboid_h // 2 : h_center + cuboid_h // 2,
        w_center - cuboid_w // 2 : w_center + cuboid_w // 2,
    ] = 1.0

    # Add motion to depth
    motion_offset = int(frame_idx * 2) % 50 - 25
    h1 = max(0, h_center - cuboid_h // 2 + motion_offset)
    h2 = min(h, h_center + cuboid_h // 2 + motion_offset)
    w1 = max(0, w_center - cuboid_w // 2)
    w2 = min(w, w_center + cuboid_w // 2)

    depth[h1:h2, w1:w2] = 1.0 + (frame_idx % 5) * 0.02

    # Create RGB (simple gradient)
    rgb = np.zeros(image_shape, dtype=np.uint8)
    rgb[:, :, 0] = 100  # R channel
    rgb[:, :, 1] = 80  # G channel
    rgb[:, :, 2] = 120  # B channel

    # Cuboid region in RGB
    rgb[h1:h2, w1:w2, 0] = 255
    rgb[h1:h2, w1:w2, 1] = 150
    rgb[h1:h2, w1:w2, 2] = 100

    return {
        "rgb": rgb,
        "depth": depth,
        "center": center,
        "rotation_angles": rotation_angles,
        "cuboid_dims": cuboid_dims,
    }


def run_benchmark(num_frames: int = 100, visualize: bool = False):
    """Run performance benchmark."""

    print(f"Cuboid 6D Pose Estimation Benchmark")
    print(f"{'='*50}")
    print(f"Frames: {num_frames}")
    print()

    # Initialize module
    module = CuboidPose6DModule(
        "cuboid_pose_6d",
        {
            "intrinsics": {"fx": 634.5, "fy": 634.5, "cx": 320, "cy": 240},
            "depth_min_m": 0.5,
            "depth_max_m": 2.0,
            "depth_roi_margin_px": 20,
            "ransac_iterations": 100,
            "ransac_thresh_m": 0.01,
            "min_inliers": 50,
            "alpha_trans": 0.3,
            "alpha_rot": 0.3,
            "max_jump_m": 0.05,
            "max_rot_deg": 15.0,
            "lost_grace_s": 0.8,
            "template_score_threshold": 0.6,
            "require_template_match": False,
            "cuboid_dims": {"L": 100, "B": 80, "H": 60},
            "enable_debug_dumps": False,
        },
    )

    # Generate frames and measure
    times = []
    results_list = []

    print(f"{'Frame':<6} {'Time (ms)':<12} {'FPS':<10} {'State':<10} {'Conf':<8}")
    print(f"{'-'*50}")

    for frame_idx in range(num_frames):
        # Generate synthetic frame
        scene = generate_synthetic_cuboid_scene(frame_idx)

        # Create frame bundle
        frame_data = {
            "rgb": scene["rgb"],
            "depth": scene["depth"],
            "intrinsics": {"fx": 634.5, "fy": 634.5, "cx": 320, "cy": 240},
            "template_results": [
                {
                    "object_id": "obj1",
                    "score": 0.8,
                    "roi": (240, 180, 160, 140),  # x, y, w, h
                }
            ],
            "cuboid_objects": [{"object_id": "obj1", "geometry": scene["cuboid_dims"]}],
        }
        frame = FrameBundle(frame_data)

        # Process frame
        t0 = time.time()
        result = module.run(frame)
        t1 = time.time()

        elapsed_ms = (t1 - t0) * 1000
        times.append(elapsed_ms)

        # Extract status
        if result.get("cuboid_poses"):
            pose_result = result["cuboid_poses"][0]
            state = pose_result.get("state", "UNKNOWN")
            confidence = pose_result.get("confidence", 0.0)
        else:
            state = "ERROR"
            confidence = 0.0

        fps = 1000 / elapsed_ms if elapsed_ms > 0 else 0

        print(
            f"{frame_idx:<6} {elapsed_ms:<12.2f} {fps:<10.1f} {state:<10} {confidence:<8.3f}"
        )

        results_list.append(
            {
                "frame": frame_idx,
                "time_ms": elapsed_ms,
                "fps": fps,
                "state": state,
                "confidence": confidence,
            }
        )

    # Summary statistics
    times_arr = np.array(times)
    print(f"{'-'*50}")
    print()
    print(f"Performance Summary:")
    print(f"  Mean: {np.mean(times_arr):.2f} ms ({1000/np.mean(times_arr):.1f} FPS)")
    print(f"  Std:  {np.std(times_arr):.2f} ms")
    print(f"  Min:  {np.min(times_arr):.2f} ms ({1000/np.max(times_arr):.1f} FPS)")
    print(f"  Max:  {np.max(times_arr):.2f} ms ({1000/np.min(times_arr):.1f} FPS)")
    print(f"  P50:  {np.percentile(times_arr, 50):.2f} ms")
    print(f"  P95:  {np.percentile(times_arr, 95):.2f} ms")
    print(f"  P99:  {np.percentile(times_arr, 99):.2f} ms")
    print()

    # Target rate check
    target_fps = 23
    target_ms = 1000 / target_fps
    target_achieved = np.mean(times_arr) <= target_ms

    if target_achieved:
        print(
            f"✓ Target {target_fps} FPS achieved ({np.mean(times_arr):.2f} ms < {target_ms:.2f} ms)"
        )
    else:
        print(
            f"✗ Target {target_fps} FPS NOT achieved ({np.mean(times_arr):.2f} ms > {target_ms:.2f} ms)"
        )

    # Save results
    results_file = Path("cuboid_pose_benchmark_results.json")
    with open(results_file, "w") as f:
        json.dump(
            {
                "num_frames": num_frames,
                "statistics": {
                    "mean_ms": float(np.mean(times_arr)),
                    "std_ms": float(np.std(times_arr)),
                    "min_ms": float(np.min(times_arr)),
                    "max_ms": float(np.max(times_arr)),
                    "p50_ms": float(np.percentile(times_arr, 50)),
                    "p95_ms": float(np.percentile(times_arr, 95)),
                    "p99_ms": float(np.percentile(times_arr, 99)),
                    "mean_fps": float(1000 / np.mean(times_arr)),
                    "target_fps": target_fps,
                    "target_achieved": target_achieved,
                },
                "frames": results_list,
            },
            f,
            indent=2,
        )

    print(f"\nResults saved to: {results_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark cuboid 6D pose estimation")
    parser.add_argument(
        "--num-frames", type=int, default=100, help="Number of frames to process"
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Visualize results (requires matplotlib)",
    )

    args = parser.parse_args()

    run_benchmark(num_frames=args.num_frames, visualize=args.visualize)
