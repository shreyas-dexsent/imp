# Cuboid 6D Pose Tracking Pipeline

## Overview

This document describes the **Cuboid 6D Pose Tracking** system, which estimates and tracks the 3D position and 6D orientation (SE(3)) of cuboid (box-shaped) objects using RGB-D data from Intel RealSense cameras.

The system is designed to:
- Run smoothly at ~23 FPS
- Fuse template matching (for identity) with depth-based pose estimation
- Provide stable, temporally smooth tracking suitable for robotic follow/servo tasks
- Handle partial occlusions and gracefully degrade during brief vision dropouts
- Require only manual geometry input (L, B, H dimensions) — no CAD models needed

## Architecture

```
Frame Input (RGB + Depth)
        ↓
   Template Matching (Identity + ROI)
        ↓
   Depth ROI Extraction
        ↓
   Plane + PCA Pose Estimation
        ↓
   Temporal Filtering (SLERP + Exponential)
        ↓
   Gating + LOST Handling
        ↓
   Output (Pose + Confidence + Quality)
```

### Key Components

#### 1. **Pose Estimator** (`pose_estimator.py`)

Converts depth data into a 6D pose estimate using:

- **RANSAC Plane Fitting**: Identifies the dominant planar surface (typically the top face)
- **PCA on Plane**: Extracts two orthogonal axes on the plane
- **Axis Disambiguation**: Determines which axis is L vs B using:
  - Extent comparison (longer extent → L)
  - Template hints (face_id) if available
  - Temporal consistency
- **Box Refinement**: Optional refinement to minimize point-to-box distance

**Input:**
- Depth ROI (meters, float32)
- ROI bounding box (x, y, w, h)
- Cuboid dimensions (L, B, H in mm)
- Optional previous pose for axis disambiguation

**Output:**
- Pose in camera frame (position + quaternion)
- Confidence score (0-1)
- Quality metrics (inlier count, residual, plane fit ratio)

#### 2. **Temporal Filter** (`temporal_filter.py`)

Smooths noisy pose estimates and handles tracking state:

- **Translation Smoothing**: Exponential smoothing with configurable alpha
- **Rotation Smoothing**: SLERP (spherical linear interpolation) on quaternions
- **Gating**: Rejects outliers based on:
  - Maximum translation jump per frame
  - Maximum rotation jump per frame
  - Confidence threshold
- **LOST Handling**:
  - Grace period: continues prediction for `lost_grace_s` seconds
  - Velocity-based forward prediction
  - Decays confidence during LOST state

**States:**
- `INIT`: Initializing, waiting for first valid measurement
- `TARGET`: Successfully tracking
- `LOST`: Lost target, using prediction

#### 3. **Vision Module** (`module.py`)

Orchestrates the entire pipeline:
- Receives frames from vision engine
- Manages per-object temporal filters
- Integrates template matching results
- Produces output in standardized format

## Configuration

### Engine Config (`vision_engine/configs/engine.local.json`)

Add the module to the vision pipeline:

```json
{
  "vision": {
    "modules": [
      {
        "name": "cuboid_pose_6d",
        "params": {
          "intrinsics": {
            "fx": 634.5,
            "fy": 634.5,
            "cx": 320.0,
            "cy": 240.0
          },
          "depth_min_m": 0.05,
          "depth_max_m": 5.0,
          "depth_downsample_stride": 1,
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
          "require_template_match": false,
          "cuboid_dims": {
            "L": 100.0,
            "B": 80.0,
            "H": 60.0
          },
          "enable_debug_dumps": false,
          "debug_dump_dir": "/tmp/cuboid_debug",
          "debug_frame_interval": 30
        }
      }
    ]
  }
}
```

### Per-Object Metadata

Object geometry is stored in `data/stations/{station}/processes/{process}/objects/{object}/metadata.json`:

```json
{
  "geometry": {
    "type": "cuboid",
    "L_mm": 100.0,
    "B_mm": 80.0,
    "H_mm": 60.0,
    "axis_convention": {
      "L_axis": "x",
      "B_axis": "y",
      "H_axis": "z"
    }
  },
  "pose_filter": {
    "alpha_trans": 0.3,
    "alpha_rot": 0.3,
    "max_jump_m": 0.05,
    "max_rot_deg": 15.0,
    "lost_grace_s": 0.8
  }
}
```

## Output Format

The module outputs structured JSON for each object:

```json
{
  "object_id": "obj1",
  "frame_ts": 1234.567,
  "state": "TARGET",
  "pose_cam": {
    "t_m": [0.123, -0.045, 0.654],
    "q_xyzw": [0.0, 0.0, 0.707, 0.707]
  },
  "pose_cam_raw": {
    "t_m": [0.125, -0.043, 0.652],
    "q_xyzw": [0.0, 0.0, 0.710, 0.705]
  },
  "confidence": 0.87,
  "quality": {
    "template_score": 0.87,
    "depth_inliers": 342,
    "depth_residual_m": 0.0082,
    "plane_inlier_ratio": 0.65,
    "axis_flip_flag": false
  },
  "debug": {
    "gate_reason": null,
    "roi": [150, 100, 200, 180]
  }
}
```

### Fields Explained

- **state**: `TARGET` (tracking) or `LOST` (using prediction)
- **pose_cam**: Filtered pose in camera frame (smoothed)
- **pose_cam_raw**: Raw depth-estimated pose (before temporal filtering)
- **confidence**: Combined confidence (0-1) from template score and inliers
- **quality.template_score**: Score from template matching (0-1)
- **quality.depth_inliers**: Number of points within 20mm of box surface
- **quality.depth_residual_m**: Mean point-to-box distance
- **quality.plane_inlier_ratio**: Fraction of points on dominant plane
- **quality.axis_flip_flag**: Whether axes flipped (indicates ambiguity)

## Tuning Guide

### Performance vs Smoothing Trade-off

**More responsive tracking** (lower latency):
- `alpha_trans`: 0.5-0.8 (higher = more responsive)
- `alpha_rot`: 0.5-0.8
- `max_jump_m`: 0.08-0.15
- `max_rot_deg`: 20-30

**Smoother tracking** (less jitter, more lag):
- `alpha_trans`: 0.1-0.3
- `alpha_rot`: 0.1-0.3
- `max_jump_m`: 0.02-0.05
- `max_rot_deg`: 5-15

### Robustness Tuning

**For well-textured objects with good depth**:
- `ransac_iterations`: 50 (faster)
- `ransac_thresh_m`: 0.015 (stricter)
- `template_score_threshold`: 0.75 (stricter)

**For low-contrast or partially occluded objects**:
- `ransac_iterations`: 200 (slower but more robust)
- `ransac_thresh_m`: 0.02 (more lenient)
- `template_score_threshold`: 0.4 (more lenient)

### LOST Handling

**For manipulation tasks** (accept frequent LOST):
- `lost_grace_s`: 0.3-0.5
- Allow brief dropouts without stopping

**For following tasks** (minimize LOST):
- `lost_grace_s`: 1.0-2.0
- Smooth prediction over longer periods

## API Endpoints

### Get Object Geometry

```
GET /processes/{process_id}/objects/{object_id}/geometry
```

Response:
```json
{
  "object_id": "obj1",
  "geometry": {
    "type": "cuboid",
    "L_mm": 100.0,
    "B_mm": 80.0,
    "H_mm": 60.0
  }
}
```

### Set Object Geometry

```
PUT /processes/{process_id}/objects/{object_id}/geometry
Content-Type: application/json

{
  "L_mm": 100.0,
  "B_mm": 80.0,
  "H_mm": 60.0,
  "axis_convention": {
    "L_axis": "x",
    "B_axis": "y",
    "H_axis": "z"
  }
}
```

## Performance

### Target Metrics

- **Throughput**: 23 FPS on RealSense D435i
- **Latency**: <50ms per frame
- **Static jitter**: <3mm RMS translation, <2° RMS rotation
- **Graceful LOST**: Prediction for 0.8-1.0s before complete loss

### Performance Breakdown (estimated)

On Intel i7 with RealSense D435i (640x480 depth):

| Component | Time (ms) | % |
|-----------|-----------|---|
| Depth to points | 2-3 | 10% |
| Plane fitting | 5-8 | 30% |
| PCA | 1-2 | 5% |
| Refinement | 3-5 | 20% |
| Temporal filter | <1 | 2% |
| **Total** | **15-20** | **100%** |

**23 FPS = 43.5ms per frame**, so 15-20ms is well within budget.

Optimization options:
- Reduce `ransac_iterations` from 100 to 50
- Skip refinement step if real-time feedback not critical
- Downsample depth (set `depth_downsample_stride: 2`)

## Testing

### Unit Tests

```bash
cd vgr-vision-engine-2.5d
python -m pytest test/test_cuboid_pose_6d.py -v
```

Tests cover:
- Plane fitting on synthetic planes
- Depth to 3D conversion
- Perfect cuboid pose recovery
- Temporal filtering and smoothing
- Gating behavior
- LOST grace period prediction

### Benchmark Script

Create `benchmark_cuboid_pose.py`:

```python
import time
import numpy as np
from vision_engine.modules.cuboid_pose_6d.module import CuboidPose6DModule
from vision_engine.io.data_plane.frame_bundle import FrameBundle

# Load a recorded run
rgb = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
depth = np.random.uniform(0.5, 2.0, (480, 640)).astype(np.float32)

module = CuboidPose6DModule("cuboid_pose_6d", {
    "intrinsics": {"fx": 634.5, "fy": 634.5, "cx": 320, "cy": 240}
})

# Benchmark
times = []
for _ in range(100):
    frame = FrameBundle({"rgb": rgb, "depth": depth})
    t0 = time.time()
    result = module.run(frame)
    t1 = time.time()
    times.append((t1 - t0) * 1000)

print(f"Mean: {np.mean(times):.2f}ms")
print(f"Std: {np.std(times):.2f}ms")
print(f"FPS: {1000 / np.mean(times):.1f}")
```

## Integration with Robot Follow

The orchestrator's follow task should consume `pose_filtered`:

```python
# In robot follow controller
pose = vision_result["pose_cam"]  # Use filtered pose

if vision_result["state"] == "LOST":
    # Gracefully handle loss
    if elapsed_since_lost < lost_grace_s:
        # Continue moving towards last pose
        continue_to_target(pose)
    else:
        # Stop and wait
        stop_motion()
else:
    # Normal follow
    update_target(pose)
```

## Debugging

### Enable Debug Dumps

Set in config:

```json
{
  "enable_debug_dumps": true,
  "debug_dump_dir": "/tmp/cuboid_debug",
  "debug_frame_interval": 30
}
```

Outputs per 30 frames:
- `depth_XXXXXX.npy`: Depth image
- `results_XXXXXX.json`: Pose results

### Analyze a Run

```python
import json
import numpy as np
from pathlib import Path

debug_dir = Path("/tmp/cuboid_debug")

for result_file in sorted(debug_dir.glob("results_*.json")):
    with open(result_file) as f:
        results = json.load(f)
    
    for obj_result in results:
        print(f"Frame {result_file.stem}: "
              f"State={obj_result['state']}, "
              f"Confidence={obj_result['confidence']:.2f}, "
              f"Inliers={obj_result['quality'].get('depth_inliers', 0)}")
```

## Known Limitations

1. **Axis Ambiguity**: If L ≈ B, the system may flip axes. Use template hints or temporal consistency to disambiguate.

2. **Partial Occlusion**: Works with up to ~50% occlusion. Beyond that, depth-only estimation becomes unreliable.

3. **Reflective Surfaces**: Shiny metallic cuboids may have depth artifacts from RealSense. Apply spatial filtering in preprocessing.

4. **Very Small Objects**: Cuboids < 5cm on any dimension are unreliable due to limited depth resolution.

5. **No Sliding Motion**: Assumes cuboid remains roughly stationary or moves smoothly. Rapid rotations may exceed gating thresholds.

## References

- **RANSAC Plane Fitting**: Fischler & Bolles (1981), "Random Sample Consensus"
- **PCA**: Pearson (1901), "On Lines and Planes of Closest Fit"
- **SLERP**: Shoemake (1985), "Animating rotation with quaternion curves"
- **Temporal Smoothing**: Exponential moving average (standard)

## Future Enhancements

- [ ] Multi-face fusion (use multiple templates)
- [ ] ICP refinement for higher precision
- [ ] Learned pose estimation (neural network backbone)
- [ ] Support for non-cuboid primitives (cylinders, etc.)
- [ ] CAD model-based rendering for overlay visualization
- [ ] Integration with SLAM for improved drift compensation
