"""
Bridge from multi-view fusion -> vision-engine RGBD consumers.

Vision-engine modules (megapose_bin_picking, FoundationPose, any future pose
estimator or 6D detector that wants RGB+depth+K) all share the same shape of
input:
    bgr    : HxWx3 uint8 (BGR, OpenCV order)
    depth  : HxW float32 in meters, or HxW uint16 with depth_scale.
             Must be pixel-aligned with `bgr`.
    K      : 3x3 pinhole intrinsics for the same frame.

A single live D405 (or D435/Zivid/etc.) frame is often noisy or holey. This
bridge replaces that single depth map with a DENSER, CLEANER one assembled by
multi-view fusion:

    1. Run multi-view fusion -> fused cloud in BASE frame.
    2. Pick a "primary" capture whose RGB will be sent downstream
       (typically the center / nadir view).
    3. Transform the fused cloud back into that primary camera's frame:
           p_cam_primary = inv(T_base_cam_primary) @ p_base
    4. Project to the primary camera's image plane using its intrinsics.
       Z-buffer per pixel keeps the nearest hit.
    5. Return (bgr, dense_depth_m, K) ready to feed any vision-engine module.

Megapose is one consumer; the same output shape works for any module that
takes pixel-aligned RGBD + intrinsics. Keep this file vision-engine-agnostic
- module-specific param dict shaping should happen in the orchestrator.
"""

from __future__ import annotations

import logging
from typing import Tuple

import cv2
import numpy as np
import open3d as o3d

from .multiview_fusion import Capture


log = logging.getLogger("multiview_fusion.bridge")


def render_cloud_to_depth(
    cloud_base: o3d.geometry.PointCloud,
    T_base_cam: np.ndarray,
    intrinsics: o3d.camera.PinholeCameraIntrinsic,
    depth_min: float = 0.05,
    depth_max: float = 0.6,
    splat_radius_px: int = 1,
) -> np.ndarray:
    """Project a base-frame point cloud into a camera image, return HxW float32 m.

    Z-buffer: each pixel keeps the nearest valid depth. `splat_radius_px>0`
    dilates each projected point into a small square so the rendered depth is
    less holey when the source cloud is sparser than the image grid.
    """
    W, H = intrinsics.width, intrinsics.height
    fx, fy = intrinsics.get_focal_length()
    cx, cy = intrinsics.get_principal_point()

    pts_base = np.asarray(cloud_base.points)
    if pts_base.size == 0:
        return np.zeros((H, W), dtype=np.float32)

    T_cam_base = np.linalg.inv(np.asarray(T_base_cam))
    R = T_cam_base[:3, :3]
    t = T_cam_base[:3, 3]
    pts_cam = pts_base @ R.T + t                # Nx3

    z = pts_cam[:, 2]
    mask = (z > depth_min) & (z < depth_max)
    pts_cam = pts_cam[mask]
    if pts_cam.shape[0] == 0:
        return np.zeros((H, W), dtype=np.float32)

    z = pts_cam[:, 2]
    u = (pts_cam[:, 0] * fx) / z + cx
    v = (pts_cam[:, 1] * fy) / z + cy
    ui = np.round(u).astype(np.int32)
    vi = np.round(v).astype(np.int32)

    in_img = (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
    ui, vi, z = ui[in_img], vi[in_img], z[in_img]

    depth = np.full(H * W, np.inf, dtype=np.float32)
    flat_idx = vi.astype(np.int64) * int(W) + ui.astype(np.int64)
    # O(N) z-buffer update: repeated pixels keep the nearest depth.
    np.minimum.at(depth, flat_idx, z.astype(np.float32, copy=False))
    depth = depth.reshape(H, W)

    if splat_radius_px > 0:
        # Greyscale erosion = nearest-neighbor min over a window.
        # OpenCV has no "min filter" directly; emulate via -dilate(-x).
        k = 2 * splat_radius_px + 1
        kernel = np.ones((k, k), np.uint8)
        neg = -depth
        neg = cv2.dilate(neg, kernel)
        depth = -neg

    depth[~np.isfinite(depth)] = 0.0
    return depth


def build_dense_rgbd(
    fused_cloud_base: o3d.geometry.PointCloud,
    primary_capture: Capture,
    refined_T_base_cam_primary: np.ndarray,
    splat_radius_px: int = 1,
    depth_min: float = 0.05,
    depth_max: float = 0.6,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Build a dense, fused RGBD packet aligned to one chosen view.

    Returns (bgr, dense_depth_m_float32, camera_dict). The caller maps
    `camera_dict` onto whatever vision-engine module's param schema expects
    (megapose, FoundationPose, ...). `dense_depth_m` is already in meters,
    so any consumer should pass `depth_scale=1.0`.
    """
    bgr = cv2.cvtColor(primary_capture.color, cv2.COLOR_RGB2BGR)

    intr = primary_capture.intrinsics
    dense_depth = render_cloud_to_depth(
        fused_cloud_base,
        refined_T_base_cam_primary,
        intr,
        depth_min=depth_min,
        depth_max=depth_max,
        splat_radius_px=splat_radius_px,
    )

    fx, fy = intr.get_focal_length()
    cx, cy = intr.get_principal_point()
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)

    camera = {
        "K": K.tolist(),
        "width": intr.width,
        "height": intr.height,
        "depth_scale": 1.0,
    }

    log.info("Dense RGBD ready: bgr=%s depth_valid=%d/%d",
             bgr.shape,
             int(np.count_nonzero(dense_depth > 0)),
             dense_depth.size)
    return bgr, dense_depth, camera
