"""
Multi-view point-cloud fusion for any depth camera mounted on a robot arm.

The fusion pipeline itself is camera-agnostic: it consumes `Capture` objects
that bundle (color, depth_in_meters, pinhole_intrinsics, robot pose, hand-eye
transform). Any depth source - RealSense D405/D435/D455, Zivid, Photoneo,
Basler blaze, OAK-D, Azure Kinect, or pre-recorded data - can be used as long
as it produces those four things.

A pyrealsense2-based acquisition helper is included as a convenience
(`RealSenseCamera`) with a `D405_PRESET` for close-range stereo. Other cameras
should provide their own equivalent that yields aligned RGBD + intrinsics.

Frame conventions (right-handed, meters):
    - cam   : optical frame of the color sensor.
              +x right, +y down, +z forward (out of the lens).
              All raw point clouds from depth_to_cloud() live here.
    - grip  : robot tool/gripper flange frame as reported by the robot driver.
              Hand-eye calibration provides T_grip_cam (constant).
    - base  : robot base / world frame. Per capture the robot reports
              T_base_grip_i. Then:
                  T_base_cam_i = T_base_grip_i @ T_grip_cam
              and a point p_cam in the camera frame becomes
                  p_base = T_base_cam_i @ [p_cam; 1]

The pipeline:
    1. Acquire synchronized color+depth frames (depth aligned to color frame).
    2. Convert each aligned depth frame to a point cloud in cam frame.
    3. Lift to base frame using T_base_cam_i (initial guess from robot+hand-eye).
    4. ROI crop, voxel downsample, normal estimation.
    5. Pairwise point-to-plane ICP refinement between neighboring views,
       with the robot-pose-derived relative transform as the initial guess.
    6. Build an Open3D PoseGraph and run global multiway optimization.
    7. Concatenate optimized clouds, run statistical + radius outlier removal,
       optional plane removal (bin floor).
    8. Optionally TSDF-integrate all RGBD frames using the optimized
       extrinsics and extract a mesh + dense cloud.

Tuning by working distance:
    Close-range bin scanning (D405 7-50 cm, Photoneo M, etc.):
        voxel 1.5-3 mm, ROI ~0.3 m cube, TSDF voxel 1.5 mm, sdf_trunc 6 mm.
    Mid-range (D435/D455 0.3-2 m):
        voxel 5-10 mm, ROI ~1 m cube, TSDF voxel 5 mm, sdf_trunc 20 mm.
    Long-range (>2 m):
        voxel 15-30 mm, larger ROI, TSDF voxel 15 mm.

ICP coarse/fine distance generally ~5x and ~1.5x voxel size regardless of
range. Reject edges with low fitness or large RMSE before pose-graph opt.
"""

from __future__ import annotations

import copy
import json
import logging
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import open3d as o3d

log = logging.getLogger("multiview_fusion")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class FusionConfig:
    """Pipeline-level config. Camera/stream config lives in `CameraConfig`.

    Defaults are tuned for RealSense D4xx (D435/D435i/D455) at ~0.3-0.7 m
    standoff, which is where the typical bin-picking capture pose sits.
    D4xx depth has ~2-3 mm stereo noise at that range with frequent flyers,
    so voxel/ICP/outlier params are sized to suppress noise without
    carving into real geometry.
    """
    # Depth gating in meters (applied AFTER acquisition, regardless of sensor).
    depth_min_m: float = 0.10
    depth_max_m: float = 0.80
    view_max_distance_m: float = 1.2

    # Workspace ROI. Interpretation depends on `roi_mode`:
    #   "anchor_cam"  (default) - box in the ANCHOR camera frame: +x right,
    #                  +y down, +z forward (looking into the scene). The
    #                  crop is applied in base frame via an oriented
    #                  bounding box derived from the anchor extrinsic.
    #   "base"       - legacy absolute box in the robot base frame.
    # For "anchor_cam", a natural work volume is a ~40 cm wide box starting
    # just in front of the near clip and extending out to depth_max_m.
    roi_mode: str = "anchor_cam"
    roi_min: Tuple[float, float, float] = (-0.20, -0.20, 0.05)
    roi_max: Tuple[float, float, float] = ( 0.20,  0.20, 0.80)

    # Preprocessing. Voxel ~= D435 single-shot noise so we average out
    # stereo jitter instead of preserving it as distinct points.
    voxel_size: float = 0.003          # 3 mm
    normal_radius_mult: float = 3.0    # normals search radius = mult * voxel
    normal_max_nn: int = 30

    # ICP. Distances scale with voxel so retuning voxel alone is enough.
    icp_coarse_dist_mult: float = 8.0  # 24 mm at 3 mm voxel - survives coarse misalignment
    icp_fine_dist_mult: float = 2.0    # 6 mm at 3 mm voxel - >= D435 noise floor
    icp_max_iter_coarse: int = 50
    icp_max_iter_fine: int = 30
    icp_fitness_min: float = 0.25      # dithered views have modest overlap fraction
    icp_rmse_max: float = 0.015        # 15 mm; D435 inlier rmse realistically ~5-10 mm

    # Pose graph
    pose_graph_pref_loop_closure: float = 0.1
    pose_graph_edge_prune: float = 0.25
    pose_graph_reference_node: int = 0

    # Post-fusion cleanup. More aggressive than defaults because D435 has
    # many flyers (edge/parallax pixels) that survive single-view filters.
    sor_nb_neighbors: int = 30
    sor_std_ratio: float = 1.5
    ror_nb_points: int = 8             # require 8 neighbours inside radius
    ror_radius_mult: float = 2.5       # radius = mult * voxel (7.5 mm @ 3 mm voxel)
    remove_plane: bool = False
    plane_dist_thresh: float = 0.004
    plane_ransac_n: int = 3
    plane_iters: int = 1000

    # TSDF. Voxel matches cloud voxel; sdf_trunc = 4x voxel is the standard
    # Open3D recipe. depth_trunc must match depth_max_m or TSDF integration
    # silently drops far geometry.
    tsdf_enable: bool = True
    tsdf_voxel_length: float = 0.003   # 3 mm
    tsdf_sdf_trunc: float = 0.012      # 4x voxel
    tsdf_depth_trunc: float = 0.80     # match depth_max_m

    # Output
    out_dir: str = "fusion_out"


@dataclass
class Capture:
    """One synchronized RGBD capture taken at a known robot pose."""
    color: np.ndarray                  # HxWx3 uint8, RGB
    depth_m: np.ndarray                # HxW float32, meters, 0 = invalid
    intrinsics: o3d.camera.PinholeCameraIntrinsic
    T_base_grip: np.ndarray            # 4x4, robot reported gripper pose
    T_grip_cam: np.ndarray             # 4x4, hand-eye (constant across captures)

    @property
    def T_base_cam(self) -> np.ndarray:
        return self.T_base_grip @ self.T_grip_cam


def _normalize_fusion_config(cfg: FusionConfig) -> FusionConfig:
    cfg = copy.deepcopy(cfg)
    if cfg.voxel_size <= 0.0:
        cfg.voxel_size = 0.003
    if cfg.tsdf_voxel_length <= 0.0 or (
        np.isclose(cfg.tsdf_voxel_length, 0.003, atol=1e-9) and cfg.voxel_size < cfg.tsdf_voxel_length
    ):
        cfg.tsdf_voxel_length = float(cfg.voxel_size)
    min_sdf_trunc = max(cfg.tsdf_voxel_length * 4.0, cfg.tsdf_voxel_length)
    if cfg.tsdf_sdf_trunc <= 0.0 or np.isclose(cfg.tsdf_sdf_trunc, 0.012, atol=1e-9):
        cfg.tsdf_sdf_trunc = float(min_sdf_trunc)
    else:
        cfg.tsdf_sdf_trunc = float(max(cfg.tsdf_sdf_trunc, min_sdf_trunc))
    if cfg.tsdf_depth_trunc <= 0.0 or np.isclose(cfg.tsdf_depth_trunc, 0.80, atol=1e-9):
        cfg.tsdf_depth_trunc = float(max(cfg.depth_max_m, 0.05))
    return cfg


# ---------------------------------------------------------------------------
# Frame acquisition
# ---------------------------------------------------------------------------
#
# This module is camera-agnostic. It does NOT open any device directly.
#
# Frames are pulled from a running camera-core process (started separately,
# e.g. `python -m camera_core.main --config config/cam_realsense_d405.yaml`)
# via the camera-core IPC: ZMQ topic 'camera' on tcp://127.0.0.1:5555 carrying
# FRAME_READY events, with image bytes in POSIX shared memory. The same fusion
# code therefore works for any camera that camera-core can serve (D405, D435,
# Basler, FLIR, webcam, future drivers).
#
# See `camera_core.fusion.camera_core_client.CameraCoreClient` for the pull
# side. To use a non-camera-core source (recorded data, custom driver), build
# `Capture` objects yourself and feed them directly to `run_pipeline`.


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def depth_to_cloud(
    color: np.ndarray,
    depth_m: np.ndarray,
    intrinsics: o3d.camera.PinholeCameraIntrinsic,
    depth_trunc: float,
) -> o3d.geometry.PointCloud:
    """Aligned RGBD -> coloured point cloud in the CAMERA frame."""
    color_o3d = o3d.geometry.Image(np.ascontiguousarray(color))
    depth_o3d = o3d.geometry.Image(np.ascontiguousarray(depth_m.astype(np.float32)))
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_o3d, depth_o3d,
        depth_scale=1.0,            # depth is already in meters
        depth_trunc=depth_trunc,
        convert_rgb_to_intensity=False,
    )
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsics)
    return pcd


def filter_cloud_by_camera_distance(
    pcd: o3d.geometry.PointCloud,
    max_distance_m: float,
) -> o3d.geometry.PointCloud:
    if max_distance_m <= 0.0 or len(pcd.points) == 0:
        return pcd
    pts = np.asarray(pcd.points)
    keep = np.linalg.norm(pts, axis=1) <= float(max_distance_m)
    if not np.any(keep):
        return o3d.geometry.PointCloud()
    indices = np.flatnonzero(keep).astype(np.int32)
    return pcd.select_by_index(indices.tolist())


def transform_cloud(pcd: o3d.geometry.PointCloud, T: np.ndarray) -> o3d.geometry.PointCloud:
    """Return a copy of `pcd` transformed by 4x4 `T`. Pure (no in-place)."""
    out = copy.deepcopy(pcd)
    out.transform(T)
    return out


def crop_to_roi(
    pcd: o3d.geometry.PointCloud,
    roi_min: Sequence[float],
    roi_max: Sequence[float],
    T_base_anchor: Optional[np.ndarray] = None,
) -> o3d.geometry.PointCloud:
    """Crop `pcd` (in base frame) to an axis-aligned box.

    If `T_base_anchor` is None, the box is interpreted in base frame (legacy).
    Otherwise the box is defined in the anchor camera frame and transformed
    into base as an oriented bounding box, so the ROI follows wherever the
    operator pointed the camera.
    """
    mn = np.asarray(roi_min, dtype=np.float64)
    mx = np.asarray(roi_max, dtype=np.float64)
    if T_base_anchor is None:
        bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=mn, max_bound=mx)
        return pcd.crop(bbox)

    center_cam = 0.5 * (mn + mx)
    extent = np.maximum(mx - mn, 1e-6)
    T = np.asarray(T_base_anchor, dtype=np.float64)
    center_base = (T[:3, :3] @ center_cam) + T[:3, 3]
    obb = o3d.geometry.OrientedBoundingBox(
        center=center_base,
        R=T[:3, :3],
        extent=extent,
    )
    return pcd.crop(obb)


def preprocess_cloud(
    pcd: o3d.geometry.PointCloud,
    voxel_size: float,
    normal_radius: float,
    normal_max_nn: int,
) -> o3d.geometry.PointCloud:
    """Clean + voxel downsample + filter + normal estimation. Returns a new cloud."""
    down = pcd.remove_non_finite_points()
    if isinstance(down, tuple):
        down = down[0]
    if len(down.points) == 0:
        return down

    down = down.voxel_down_sample(voxel_size)
    if len(down.points) == 0:
        return down

    if len(down.points) >= 8:
        nb_neighbors = min(max(8, normal_max_nn), len(down.points))
        down, _ = down.remove_statistical_outlier(
            nb_neighbors=nb_neighbors,
            std_ratio=2.0,
        )
    if len(down.points) == 0:
        return down

    if len(down.points) >= 3:
        # Try with the configured radius first, then progressively expand
        # until we get normals or exhaust attempts. On D405 + 1 mm voxel,
        # the configured 3 mm radius is often too small for sparse views.
        for radius_mult in (1.0, 2.0, 4.0, 8.0):
            try:
                down.estimate_normals(
                    search_param=o3d.geometry.KDTreeSearchParamHybrid(
                        radius=float(normal_radius) * radius_mult,
                        max_nn=min(max(8, normal_max_nn), len(down.points)),
                    ),
                )
            except Exception:
                continue
            if down.has_normals():
                break
        if down.has_normals():
            try:
                down.orient_normals_towards_camera_location(
                    camera_location=np.zeros(3)
                )
            except Exception:
                log.warning("normal_orientation_skipped", exc_info=True)
    # Even if normals failed, KEEP the cloud's points so fuse_clouds and
    # TSDF integration still receive geometry. The pose-graph edge layer
    # (optimize_pose_graph) is responsible for skipping pairs whose normals
    # are missing — returning empty here was breaking the fused cloud
    # downstream when even one view degenerated.
    if not down.has_normals() and len(down.points) > 0:
        log.warning(
            "preprocess_cloud_no_normals_kept_points input=%d kept=%d voxel=%.4f radius=%.4f",
            len(pcd.points), len(down.points), voxel_size, normal_radius,
        )
    return down


# ---------------------------------------------------------------------------
# Pairwise registration
# ---------------------------------------------------------------------------


def register_pair_icp(
    src: o3d.geometry.PointCloud,
    tgt: o3d.geometry.PointCloud,
    init: np.ndarray,
    voxel_size: float,
    cfg: FusionConfig,
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """Coarse + fine point-to-plane ICP refining `init` to align src->tgt.

    Returns: (T_refined, information_matrix, fitness, rmse).
    Both clouds are assumed to live in the SAME (base) frame already, so the
    initial guess is typically np.eye(4) when poses are good.
    """
    coarse_dist = voxel_size * cfg.icp_coarse_dist_mult
    fine_dist = voxel_size * cfg.icp_fine_dist_mult

    icp_coarse = o3d.pipelines.registration.registration_icp(
        src, tgt, coarse_dist, init,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(
            max_iteration=cfg.icp_max_iter_coarse),
    )
    icp_fine = o3d.pipelines.registration.registration_icp(
        src, tgt, fine_dist, icp_coarse.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(
            max_iteration=cfg.icp_max_iter_fine),
    )

    info = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
        src, tgt, fine_dist, icp_fine.transformation,
    )
    return (
        np.asarray(icp_fine.transformation),
        np.asarray(info),
        float(icp_fine.fitness),
        float(icp_fine.inlier_rmse),
    )


# ---------------------------------------------------------------------------
# Pose graph multiway registration
# ---------------------------------------------------------------------------


def optimize_pose_graph(
    clouds_base: List[o3d.geometry.PointCloud],
    T_base_cams: List[np.ndarray],
    cfg: FusionConfig,
) -> Tuple[o3d.pipelines.registration.PoseGraph, List[np.ndarray]]:
    """Build and optimize a PoseGraph over all views.

    Conventions:
        - Node i absolute pose is the world-from-camera-i transform.
          We seed it with T_base_cam_i (robot+hand-eye).
        - Edge (i,j) transformation is the camera-j-from-camera-i refinement
          discovered by ICP, applied AFTER seeding the nodes with the robot
          guesses. Because we feed ICP the clouds already lifted to base,
          the relative refinement is applied in base frame and we convert
          back to a camera-frame edge.

    Returns (pose_graph, refined_T_base_cam_list).
    """
    pose_graph = o3d.pipelines.registration.PoseGraph()
    for T in T_base_cams:
        pose_graph.nodes.append(
            o3d.pipelines.registration.PoseGraphNode(np.asarray(T)))

    n = len(clouds_base)
    for i in range(n):
        for j in range(i + 1, n):
            uncertain = (j != i + 1)  # only neighbors are "certain"

            ci = clouds_base[i]
            cj = clouds_base[j]
            if (
                len(ci.points) < 3
                or len(cj.points) < 3
                or not ci.has_normals()
                or not cj.has_normals()
            ):
                log.info(
                    "skip edge %d-%d  reason=missing_points_or_normals "
                    "pts_i=%d pts_j=%d normals_i=%s normals_j=%s",
                    i, j, len(ci.points), len(cj.points),
                    ci.has_normals(), cj.has_normals(),
                )
                continue

            T_refine_base, info_base, fitness, rmse = register_pair_icp(
                ci, cj,
                init=np.eye(4),
                voxel_size=cfg.voxel_size,
                cfg=cfg,
            )

            if fitness < cfg.icp_fitness_min or rmse > cfg.icp_rmse_max:
                log.info("skip edge %d-%d  fit=%.3f rmse=%.4f", i, j, fitness, rmse)
                continue
            log.info("edge %d-%d  fit=%.3f rmse=%.4f", i, j, fitness, rmse)

            # T_refine_base aligns cloud_i (in base) onto cloud_j (in base).
            # In camera-frame edge convention used by Open3D's PoseGraph:
            #   T_edge = T_j^{-1} @ T_refine_base @ T_i
            T_i = np.asarray(T_base_cams[i])
            T_j = np.asarray(T_base_cams[j])
            T_edge = np.linalg.inv(T_j) @ T_refine_base @ T_i

            pose_graph.edges.append(
                o3d.pipelines.registration.PoseGraphEdge(
                    i, j, T_edge, info_base, uncertain=uncertain),
            )

    option = o3d.pipelines.registration.GlobalOptimizationOption(
        max_correspondence_distance=cfg.voxel_size * cfg.icp_fine_dist_mult,
        edge_prune_threshold=cfg.pose_graph_edge_prune,
        preference_loop_closure=cfg.pose_graph_pref_loop_closure,
        reference_node=cfg.pose_graph_reference_node,
    )
    o3d.pipelines.registration.global_optimization(
        pose_graph,
        o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
        o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
        option,
    )

    refined = [np.asarray(node.pose) for node in pose_graph.nodes]
    return pose_graph, refined


# ---------------------------------------------------------------------------
# Fusion + cleanup
# ---------------------------------------------------------------------------


def fuse_clouds(
    clouds_cam: List[o3d.geometry.PointCloud],
    T_base_cams: List[np.ndarray],
    cfg: FusionConfig,
) -> o3d.geometry.PointCloud:
    """Lift each cam-frame cloud to base, downsample together."""
    fused = o3d.geometry.PointCloud()
    for pcd, T in zip(clouds_cam, T_base_cams):
        fused += transform_cloud(pcd, T)
    fused = fused.voxel_down_sample(cfg.voxel_size)
    return fused


def postprocess_fused(
    fused: o3d.geometry.PointCloud,
    cfg: FusionConfig,
) -> o3d.geometry.PointCloud:
    out = fused
    if len(out.points) == 0:
        return out

    out, _ = out.remove_statistical_outlier(
        nb_neighbors=cfg.sor_nb_neighbors, std_ratio=cfg.sor_std_ratio)
    out, _ = out.remove_radius_outlier(
        nb_points=cfg.ror_nb_points,
        radius=cfg.voxel_size * cfg.ror_radius_mult,
    )

    if cfg.remove_plane and len(out.points) > 100:
        plane_model, inliers = out.segment_plane(
            distance_threshold=cfg.plane_dist_thresh,
            ransac_n=cfg.plane_ransac_n,
            num_iterations=cfg.plane_iters,
        )
        log.info("plane removed: %d / %d points", len(inliers), len(out.points))
        out = out.select_by_index(inliers, invert=True)
    return out


# ---------------------------------------------------------------------------
# TSDF integration
# ---------------------------------------------------------------------------


def tsdf_integrate(
    captures: Sequence[Capture],
    refined_T_base_cams: Sequence[np.ndarray],
    cfg: FusionConfig,
) -> Tuple[o3d.geometry.TriangleMesh, o3d.geometry.PointCloud]:
    """Fuse all RGBD frames into a TSDF volume using optimized extrinsics.

    Open3D's ScalableTSDFVolume expects `extrinsic` as world-from-camera
    INVERSE, i.e. camera_from_world. So we pass np.linalg.inv(T_base_cam).
    """
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=cfg.tsdf_voxel_length,
        sdf_trunc=cfg.tsdf_sdf_trunc,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )
    for cap, T_base_cam in zip(captures, refined_T_base_cams):
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(np.ascontiguousarray(cap.color)),
            o3d.geometry.Image(np.ascontiguousarray(cap.depth_m.astype(np.float32))),
            depth_scale=1.0,
            depth_trunc=cfg.tsdf_depth_trunc,
            convert_rgb_to_intensity=False,
        )
        volume.integrate(rgbd, cap.intrinsics, np.linalg.inv(np.asarray(T_base_cam)))

    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    pcd = volume.extract_point_cloud()
    return mesh, pcd


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------


def _frame_at(T: np.ndarray, size: float = 0.03) -> o3d.geometry.TriangleMesh:
    f = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)
    f.transform(T)
    return f


def visualize_poses(T_base_cams: Sequence[np.ndarray], extra=None) -> None:
    geoms = [o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)]
    for T in T_base_cams:
        geoms.append(_frame_at(np.asarray(T)))
    if extra:
        geoms.extend(extra)
    o3d.visualization.draw_geometries(geoms)


def visualize_clouds(clouds: Sequence[o3d.geometry.PointCloud]) -> None:
    o3d.visualization.draw_geometries(list(clouds))


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------


def save_results(
    out_dir: str,
    per_view_clouds_base: Sequence[o3d.geometry.PointCloud],
    refined_T_base_cams: Sequence[np.ndarray],
    fused_cloud: o3d.geometry.PointCloud,
    tsdf_mesh: Optional[o3d.geometry.TriangleMesh] = None,
    tsdf_cloud: Optional[o3d.geometry.PointCloud] = None,
    cfg: Optional[FusionConfig] = None,
) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "views").mkdir(exist_ok=True)

    for i, pcd in enumerate(per_view_clouds_base):
        o3d.io.write_point_cloud(str(out / "views" / f"view_{i:03d}.ply"), pcd)

    np.save(out / "T_base_cam_optimized.npy",
            np.stack([np.asarray(T) for T in refined_T_base_cams], axis=0))

    o3d.io.write_point_cloud(str(out / "fused.ply"), fused_cloud)
    o3d.io.write_point_cloud(str(out / "fused.pcd"), fused_cloud)

    # Anchor-camera-frame copies for the 3D viewer. The viewer displays the
    # raw scene cloud in camera frame, so saving fused/TSDF in the same frame
    # lets the user A/B them against raw without juggling extrinsics.
    if len(refined_T_base_cams) > 0:
        T_anchor_base = np.linalg.inv(np.asarray(refined_T_base_cams[0]))
        fused_cam = o3d.geometry.PointCloud(fused_cloud)
        fused_cam.transform(T_anchor_base)
        o3d.io.write_point_cloud(str(out / "fused_cam.ply"), fused_cam)
        if tsdf_cloud is not None:
            tsdf_cam = o3d.geometry.PointCloud(tsdf_cloud)
            tsdf_cam.transform(T_anchor_base)
            o3d.io.write_point_cloud(str(out / "tsdf_cloud_cam.ply"), tsdf_cam)

    if tsdf_mesh is not None:
        o3d.io.write_triangle_mesh(str(out / "tsdf_mesh.ply"), tsdf_mesh)
    if tsdf_cloud is not None:
        o3d.io.write_point_cloud(str(out / "tsdf_cloud.ply"), tsdf_cloud)

    if cfg is not None:
        with open(out / "fusion_config.json", "w") as f:
            json.dump(asdict(cfg), f, indent=2)
    log.info("Saved outputs to %s", out.resolve())


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def run_pipeline(captures: List[Capture], cfg: FusionConfig):
    """End-to-end fusion. Returns dict of artifacts."""
    assert len(captures) >= 2, "Need at least 2 captures for multi-view fusion"
    cfg = _normalize_fusion_config(cfg)
    log.info("Running fusion on %d captures", len(captures))

    # 1. cam-frame clouds + initial base-frame seeds
    clouds_cam: List[o3d.geometry.PointCloud] = []
    T_base_cams: List[np.ndarray] = []
    for cap in captures:
        pcd_cam = depth_to_cloud(cap.color, cap.depth_m, cap.intrinsics,
                                 depth_trunc=cfg.depth_max_m)
        pcd_cam = filter_cloud_by_camera_distance(
            pcd_cam,
            cfg.view_max_distance_m,
        )
        clouds_cam.append(pcd_cam)
        T_base_cams.append(cap.T_base_cam)

    # 2. lift -> downsample/filter -> normals (per view, in base frame)
    # Keep the full view for registration; do not ROI-crop before ICP.
    normal_radius = cfg.voxel_size * cfg.normal_radius_mult
    clouds_base_proc: List[o3d.geometry.PointCloud] = []
    for pcd_cam, T in zip(clouds_cam, T_base_cams):
        pcd_base = transform_cloud(pcd_cam, T)
        pcd_base = preprocess_cloud(
            pcd_base, cfg.voxel_size, normal_radius, cfg.normal_max_nn,
        )
        clouds_base_proc.append(pcd_base)

    # 3. pose-graph multiway optimization
    pose_graph, refined_T_base_cams = optimize_pose_graph(
        clouds_base_proc, T_base_cams, cfg,
    )

    # 4. fuse using REFINED extrinsics (re-lift the original cam clouds)
    fused = fuse_clouds(clouds_cam, refined_T_base_cams, cfg)
    if str(cfg.roi_mode).lower() == "anchor_cam":
        # Anchor = first capture pose (the recorded capture pose in the
        # capture_pose_fusion flow; the same convention dither callers use).
        T_anchor = np.asarray(refined_T_base_cams[0])
        fused = crop_to_roi(fused, cfg.roi_min, cfg.roi_max, T_base_anchor=T_anchor)
    else:
        fused = crop_to_roi(fused, cfg.roi_min, cfg.roi_max)
    fused = postprocess_fused(fused, cfg)

    # 5. optional TSDF
    tsdf_mesh = tsdf_cloud = None
    if cfg.tsdf_enable:
        tsdf_mesh, tsdf_cloud = tsdf_integrate(captures, refined_T_base_cams, cfg)
        # TSDF volume is populated from the raw camera frustums, so the
        # extracted cloud contains everything camera-core saw inside
        # depth_trunc - including walls, the conveyor, operator hands, etc.
        # Apply the same ROI crop as `fused` so downstream consumers (and
        # the dense-RGBD renderer) see only the work volume.
        if tsdf_cloud is not None and len(tsdf_cloud.points) > 0:
            if str(cfg.roi_mode).lower() == "anchor_cam":
                T_anchor = np.asarray(refined_T_base_cams[0])
                tsdf_cloud = crop_to_roi(
                    tsdf_cloud, cfg.roi_min, cfg.roi_max, T_base_anchor=T_anchor,
                )
            else:
                tsdf_cloud = crop_to_roi(tsdf_cloud, cfg.roi_min, cfg.roi_max)

    # 6. save
    per_view_clouds_base_refined = [
        transform_cloud(c, T) for c, T in zip(clouds_cam, refined_T_base_cams)
    ]
    save_results(
        cfg.out_dir,
        per_view_clouds_base_refined,
        refined_T_base_cams,
        fused,
        tsdf_mesh=tsdf_mesh,
        tsdf_cloud=tsdf_cloud,
        cfg=cfg,
    )

    return {
        "pose_graph": pose_graph,
        "refined_T_base_cams": refined_T_base_cams,
        "per_view_clouds_base": per_view_clouds_base_refined,
        "fused": fused,
        "tsdf_mesh": tsdf_mesh,
        "tsdf_cloud": tsdf_cloud,
    }


# ---------------------------------------------------------------------------
# Example main
# ---------------------------------------------------------------------------


def _placeholder_hand_eye() -> np.ndarray:
    """Placeholder T_grip_cam: camera mounted ~5 cm in front of the flange,
    looking along +z of the gripper. REPLACE with real calibration."""
    T = np.eye(4)
    T[:3, 3] = [0.00, 0.00, 0.05]
    return T


def _placeholder_grip_poses(n: int) -> List[np.ndarray]:
    """Generate n poses on a small arc above the workspace origin (base = world).
    Used only as a smoke test when no real robot poses are provided."""
    poses = []
    for k in range(n):
        angle = (k / max(1, n - 1) - 0.5) * np.deg2rad(40.0)
        T = np.eye(4)
        c, s = np.cos(angle), np.sin(angle)
        # rotate around base x axis a little
        T[1, 1], T[1, 2] = c, -s
        T[2, 1], T[2, 2] = s, c
        T[:3, 3] = [0.0, 0.05 * np.sin(angle), 0.30]   # 30 cm above bin
        poses.append(T)
    return poses


def main() -> None:
    """Smoke test: pull frames from a running camera-core process.

    Prereqs (in another terminal):
        conda activate vgr-camera
        cd ~/imp/dexsent-vgr-camera-core-2.5d
        python -m camera_core.main --config config/cam_realsense_d405.yaml

    The camera type is whatever camera-core is serving; this script doesn't
    care. Robot poses are placeholders here - the orchestrator should provide
    real T_base_grip from the robot engine.
    """
    from .camera_core_client import CameraCoreClient

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    cfg = FusionConfig()

    n_views = 6
    T_grip_cam = _placeholder_hand_eye()
    T_base_grips = _placeholder_grip_poses(n_views)

    captures: List[Capture] = []
    with CameraCoreClient() as cam:
        for i, T_bg in enumerate(T_base_grips):
            log.info("Capture %d/%d  (move robot to pose, then press Enter)",
                     i + 1, n_views)
            try:
                input()
            except EOFError:
                pass
            color, depth_m, intrinsics = cam.grab_rgbd()
            captures.append(Capture(
                color=color,
                depth_m=depth_m,
                intrinsics=intrinsics,
                T_base_grip=T_bg,
                T_grip_cam=T_grip_cam,
            ))

    artifacts = run_pipeline(captures, cfg)
    log.info("Done. Fused cloud: %d points", len(artifacts["fused"].points))


if __name__ == "__main__":
    main()
