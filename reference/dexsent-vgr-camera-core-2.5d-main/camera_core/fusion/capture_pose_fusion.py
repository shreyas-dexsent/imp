"""
Capture-pose-centered multi-view fusion.

The orchestrator already records a "capture_pose" per process (see
`orchestrator.tasks._pick_runtime`): a dict like
    {
        "name": "capture_pose",
        "tcp_pose": {
            "position_m":   [x, y, z],
            "quat_xyzw":    [qx, qy, qz, qw],
            "frame":        "base",
        },
        "profile": "normal",
        # optionally: "joints": [...]
    }

For megapose-style 6D pose estimation the robot normally moves to that pose,
grabs ONE RGBD frame, and feeds it to the model. That single frame is noisy
and holey on a D405. This module replaces it with a denser, fused RGBD by:

    1. Dithering the robot a few millimeters / a few degrees around the
       recorded capture pose (translation grid + small tilts).
    2. At each dithered pose, grabbing one synchronized RGBD frame from
       camera-core (whatever camera-core is currently serving).
    3. Running the camera-agnostic fusion pipeline (`run_pipeline`) using
       the robot poses at each capture as the initial extrinsic guess.
    4. Re-projecting the optimized fused cloud back into the CAMERA FRAME
       OF THE ORIGINAL CAPTURE POSE, so the output is a dense RGBD that
       drops in as a 1:1 replacement for the single-shot RGBD megapose
       would have used at that pose.

This module is intentionally robot-agnostic: it depends on a `RobotMover`
Protocol with a single `move_tcp(...)` method. The orchestrator wires that
to `ctx.robot.movel`. Fusion math has no idea what robot it is.

Frame conventions are inherited from `multiview_fusion`:
    cam   - color sensor optical frame (+x right, +y down, +z forward)
    grip  - robot tool/flange frame
    base  - robot base / world frame
    T_base_cam_i = T_base_grip_i @ T_grip_cam     (hand-eye constant)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Sequence, Tuple

import numpy as np
import open3d as o3d

from .camera_core_client import CameraCoreClient
from .multiview_fusion import (
    Capture,
    FusionConfig,
    run_pipeline,
)
from .vision_engine_bridge import build_dense_rgbd


log = logging.getLogger("multiview_fusion.capture_pose")


# ---------------------------------------------------------------------------
# Robot mover Protocol (decouples this file from the orchestrator)
# ---------------------------------------------------------------------------


class RobotMover(Protocol):
    """Minimum surface the fusion helper needs from a robot driver.

    The orchestrator wires this to e.g. ctx.robot.movel and ctx.robot.get_tcp.
    """

    def move_tcp(
        self,
        position_m: Sequence[float],
        quat_xyzw: Sequence[float],
        frame: str = "base",
        profile: str = "normal",
    ) -> None: ...

    def get_tcp(self) -> Tuple[List[float], List[float]]:
        """Return (position_m, quat_xyzw) of the current TCP in base frame."""
        ...


# ---------------------------------------------------------------------------
# Quaternion / SE(3) helpers (kept local so this file is self-contained)
# ---------------------------------------------------------------------------


def _quat_xyzw_to_R(q: Sequence[float]) -> np.ndarray:
    x, y, z, w = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    n = x * x + y * y + z * z + w * w
    if n < 1e-12:
        return np.eye(3)
    s = 2.0 / n
    xx, yy, zz = x * x * s, y * y * s, z * z * s
    xy, xz, yz = x * y * s, x * z * s, y * z * s
    wx, wy, wz = w * x * s, w * y * s, w * z * s
    return np.array([
        [1 - (yy + zz),     xy - wz,     xz + wy],
        [    xy + wz, 1 - (xx + zz),     yz - wx],
        [    xz - wy,     yz + wx, 1 - (xx + yy)],
    ], dtype=np.float64)


def _R_to_quat_xyzw(R: np.ndarray) -> List[float]:
    m = np.asarray(R, dtype=np.float64)
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0.0:
        s = np.sqrt(tr + 1.0) * 2.0
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif (m[0, 0] > m[1, 1]) and (m[0, 0] > m[2, 2]):
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    return [float(x), float(y), float(z), float(w)]


def pose_dict_to_T(pose: Dict[str, Any]) -> np.ndarray:
    """Convert an orchestrator capture-pose dict to a 4x4 T_base_grip.

    Accepts the same shapes `_move_to_pose` accepts: nested under `tcp_pose`
    or `tcp`, or top-level `position_m`/`quat_xyzw`.
    """
    target = pose.get("tcp_pose") or pose.get("tcp") or pose
    pos = target.get("position_m")
    quat = target.get("quat_xyzw") or [0.0, 0.0, 0.0, 1.0]
    if pos is None:
        raise ValueError("capture_pose has no position_m")
    T = np.eye(4)
    T[:3, :3] = _quat_xyzw_to_R(quat)
    T[:3, 3] = np.asarray(pos, dtype=np.float64)
    return T


def T_to_pose_dict(
    T: np.ndarray,
    *,
    frame: str = "base",
    profile: str = "normal",
) -> Dict[str, Any]:
    """Inverse of `pose_dict_to_T`. Returns a dict shaped like the orchestrator's."""
    return {
        "tcp_pose": {
            "position_m": [float(v) for v in T[:3, 3]],
            "quat_xyzw": _R_to_quat_xyzw(T[:3, :3]),
            "frame": frame,
        },
        "profile": profile,
    }


# ---------------------------------------------------------------------------
# Dither pattern around a capture pose
# ---------------------------------------------------------------------------


@dataclass
class CapturePoseFusionConfig:
    """Where to move and how to fuse around a recorded capture pose.

    Camera target: Intel RealSense D405 (close-range stereo, 7-50 cm sweet
    spot, ~0.1% range noise). The dither pattern is sized for D405 geometry
    and would need to be loosened for longer-range sensors.

    Translations are applied in the GRIPPER frame so the camera moves
    laterally relative to the part, not along base axes. Tilts are applied
    in the same local frame BEFORE the translation, so each off-center view
    points back at the same world point the anchor was looking at ("toe-in"
    to the scene center).

    Default pattern: 9 views total: the recorded capture pose first, then an
    8-view ring at 45 degree spacing around it in clockwise order, with each
    off-center view toeing 10 deg back toward the center. This gives the D405
    more view diversity than the earlier 5-view cross while keeping the motion
    predictable and symmetric around the anchor pose.
    """
    # Simple UI-exposed ring controls. The generated pattern is:
    # center, then 8 views around a circle in clockwise order.
    xy_offset_m: float = 0.10
    toe_in_deg: float = 10.0
    # Ordered translation offsets in gripper frame (meters).
    # 9-point ring: center, then 8 points at radius 10 cm.
    translations_grip_m: List[Tuple[float, float, float]] = field(default_factory=lambda: [
        ( 0.000, 0.000, 0.000),   # the recorded capture pose itself (anchor)
        ( 0.100, 0.000, 0.000),
        ( 0.0707106781,-0.0707106781, 0.000),
        ( 0.000,-0.100, 0.000),
        (-0.0707106781,-0.0707106781, 0.000),
        (-0.100, 0.000, 0.000),
        (-0.0707106781, 0.0707106781, 0.000),
        ( 0.000, 0.100, 0.000),
        ( 0.0707106781, 0.0707106781, 0.000),
    ])
    # Per-view tilts (ax about gripper X, ay about gripper Y), radians.
    # 10 deg toe-in toward scene center for the 8 off-center views. Each
    # entry stores (ax about gripper X, ay about gripper Y).
    tilts_grip_rad: List[Tuple[float, float]] = field(default_factory=lambda: [
        ( 0.000,            0.000),            # anchor
        ( 0.000,           -0.1745329252),     # +X view: ay = -10 deg
        (-0.1234134149,    -0.1234134149),     # +X,-Y diagonal
        (-0.1745329252,     0.000),            # -Y view: ax = -10 deg
        (-0.1234134149,     0.1234134149),     # -X,-Y diagonal
        ( 0.000,            0.1745329252),     # -X view: ay = +10 deg
        ( 0.1234134149,     0.1234134149),     # -X,+Y diagonal
        ( 0.1745329252,     0.000),            # +Y view: ax = +10 deg
        ( 0.1234134149,    -0.1234134149),     # +X,+Y diagonal
    ])
    # Settle time after each robot move before grabbing a frame.
    settle_s: float = 0.35
    # Move profile name to pass to RobotMover.move_tcp.
    move_profile: str = "normal"
    # Frame for movel commands (matches orchestrator default).
    frame: str = "base"
    # Return to the original capture pose at the end.
    return_to_capture: bool = True
    # If True, the dense output RGBD is rendered in the ORIGINAL capture-pose
    # camera frame (drop-in replacement for single-shot megapose input).
    # If False, it is rendered in whichever pose ended up at index 0
    # (the anchor pose - usually the same thing).
    render_at_recorded_capture: bool = True


def _resolve_capture_pattern(
    cfg: CapturePoseFusionConfig,
) -> Tuple[List[Tuple[float, float, float]], List[Tuple[float, float]]]:
    offset = max(0.0, float(getattr(cfg, "xy_offset_m", 0.10) or 0.10))
    toe_in_rad = np.deg2rad(float(getattr(cfg, "toe_in_deg", 10.0) or 0.0))
    ring_angles_deg = [0.0, -45.0, -90.0, -135.0, 180.0, 135.0, 90.0, 45.0]
    translations = [(0.0, 0.0, 0.0)]
    tilts = [(0.0, 0.0)]
    for angle_deg in ring_angles_deg:
        angle_rad = np.deg2rad(angle_deg)
        tx = offset * np.cos(angle_rad)
        ty = offset * np.sin(angle_rad)
        # Toe back toward the origin: ay controls x convergence, ax controls y.
        ay = -toe_in_rad * np.cos(angle_rad)
        ax = toe_in_rad * np.sin(angle_rad)
        translations.append((float(tx), float(ty), 0.0))
        tilts.append((float(ax), float(ay)))
    return translations, tilts


def _se3(R: np.ndarray, t: Sequence[float]) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(t, dtype=np.float64)
    return T


def _rotx(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)


def _roty(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)


def generate_dither_poses(
    T_base_grip_capture: np.ndarray,
    cfg: CapturePoseFusionConfig,
) -> List[np.ndarray]:
    """Return a list of T_base_grip targets to visit around the capture pose.

    Each translation entry corresponds to one capture. Tilts are applied
    per-entry if provided; otherwise the view is captured with no tilt.
    """
    out: List[np.ndarray] = []
    Tc = np.asarray(T_base_grip_capture, dtype=np.float64)
    translations, default_tilts = _resolve_capture_pattern(cfg)
    configured_translations = list(cfg.translations_grip_m or [])
    configured_tilts = list(cfg.tilts_grip_rad or [])
    # Keep the simple UI controls authoritative for the common default ring
    # pattern. If someone passes a custom number of views, honor that instead.
    if configured_translations and len(configured_translations) != 9:
        translations = configured_translations
    if configured_tilts and len(configured_tilts) != 9:
        tilts = configured_tilts
    else:
        tilts = default_tilts
    for i, (tx, ty, tz) in enumerate(translations):
        ax, ay = tilts[i] if i < len(tilts) else (0.0, 0.0)
        R_local = _roty(ay) @ _rotx(ax)
        T_local = _se3(R_local, (tx, ty, tz))   # in gripper frame
        T_target = Tc @ T_local                  # right-multiply: local frame
        out.append(T_target)
    return out


# ---------------------------------------------------------------------------
# Capture orchestration
# ---------------------------------------------------------------------------


def capture_around_pose(
    robot: RobotMover,
    cam: CameraCoreClient,
    capture_pose: Dict[str, Any],
    T_grip_cam: np.ndarray,
    cfg: CapturePoseFusionConfig,
) -> Tuple[List[Capture], int]:
    """Move the robot through the dither pattern and collect Captures.

    Returns (captures, anchor_index) where anchor_index is the position in
    the captures list corresponding to the original recorded capture pose
    (translation = 0, tilt = 0). The orchestrator should re-visit that pose
    at the end if `return_to_capture` is True.
    """
    T_capture = pose_dict_to_T(capture_pose)
    targets = generate_dither_poses(T_capture, cfg)

    # Find the "anchor" pose (zero offset, zero tilt) so we can re-render
    # the fused cloud into its camera frame later.
    anchor_idx = 0
    for i, T in enumerate(targets):
        delta = np.linalg.inv(T_capture) @ T
        if (np.allclose(delta[:3, 3], 0.0, atol=1e-9)
                and np.allclose(delta[:3, :3], np.eye(3), atol=1e-9)):
            anchor_idx = i
            break

    captures: List[Capture] = []
    for i, T_bg in enumerate(targets):
        target_pose = T_to_pose_dict(T_bg, frame=cfg.frame, profile=cfg.move_profile)
        tcp = target_pose["tcp_pose"]
        log.info("[%d/%d] move_tcp pos=%s",
                 i + 1, len(targets), [round(v, 4) for v in tcp["position_m"]])
        robot.move_tcp(
            position_m=tcp["position_m"],
            quat_xyzw=tcp["quat_xyzw"],
            frame=tcp["frame"],
            profile=cfg.move_profile,
        )
        if cfg.settle_s > 0:
            time.sleep(cfg.settle_s)
        rgb, depth_m, intrinsics = cam.grab_rgbd()
        captures.append(Capture(
            color=rgb,
            depth_m=depth_m,
            intrinsics=intrinsics,
            T_base_grip=T_bg,
            T_grip_cam=np.asarray(T_grip_cam, dtype=np.float64),
        ))

    if cfg.return_to_capture:
        tcp = capture_pose.get("tcp_pose") or capture_pose.get("tcp") or capture_pose
        log.info("returning to recorded capture pose")
        robot.move_tcp(
            position_m=tcp["position_m"],
            quat_xyzw=tcp.get("quat_xyzw", [0.0, 0.0, 0.0, 1.0]),
            frame=tcp.get("frame", cfg.frame),
            profile=cfg.move_profile,
        )

    return captures, anchor_idx


# ---------------------------------------------------------------------------
# End-to-end: capture-pose -> dense fused RGBD ready for vision-engine
# ---------------------------------------------------------------------------


def fuse_at_capture_pose(
    robot: RobotMover,
    capture_pose: Dict[str, Any],
    T_grip_cam: np.ndarray,
    fusion_cfg: Optional[FusionConfig] = None,
    capture_cfg: Optional[CapturePoseFusionConfig] = None,
    cam_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Top-level: dither around a capture pose, fuse, return dense RGBD.

    Returns:
        {
            "bgr":          HxWx3 uint8     (BGR for OpenCV / megapose),
            "depth_m":      HxW  float32    (meters, dense, fused),
            "camera":       {K, width, height, depth_scale=1.0},
            "fused_cloud":  o3d.geometry.PointCloud  (in BASE frame),
            "captures":     list[Capture],
            "anchor_index": int,
            "T_base_cam_anchor": 4x4 ndarray  (cam frame for the recorded capture pose,
                                               post pose-graph optimization),
            "artifacts":    full dict returned by run_pipeline(...),
        }
    """
    fusion_cfg = fusion_cfg or FusionConfig()
    capture_cfg = capture_cfg or CapturePoseFusionConfig()
    cam_kwargs = cam_kwargs or {}

    with CameraCoreClient(**cam_kwargs) as cam:
        captures, anchor_idx = capture_around_pose(
            robot, cam, capture_pose, T_grip_cam, capture_cfg,
        )

    artifacts = run_pipeline(captures, fusion_cfg)
    refined_T_base_cams = artifacts["refined_T_base_cams"]

    # Render the fused cloud into the camera frame of the anchor pose
    # (= the recorded capture pose). That gives megapose a dense RGBD that
    # is pixel-aligned with the RGB it would normally see at that pose.
    if capture_cfg.render_at_recorded_capture:
        # Use the OPTIMIZED extrinsic for the anchor pose, not the raw
        # robot+hand-eye guess - it's strictly better.
        T_base_cam_anchor = np.asarray(refined_T_base_cams[anchor_idx])
    else:
        T_base_cam_anchor = np.asarray(refined_T_base_cams[0])

    # Prefer the TSDF-extracted cloud over the raw concat `fused` cloud when
    # available: TSDF does a true weighted SDF average across all views, which
    # is what actually cancels D405 range noise on small objects. The
    # concat+voxel-downsample `fused` cloud only averages *within* a voxel and
    # bakes single-view jitter in between voxels. Fall back to `fused` if TSDF
    # was disabled or produced nothing.
    tsdf_cloud = artifacts.get("tsdf_cloud")
    dense_source_cloud = artifacts["fused"]
    dense_source_name = "fused"
    if tsdf_cloud is not None and len(tsdf_cloud.points) > 0:
        dense_source_cloud = tsdf_cloud
        dense_source_name = "tsdf_cloud"
    else:
        tsdf_points = len(tsdf_cloud.points) if tsdf_cloud is not None else 0
        log.warning(
            "TSDF render source unavailable; falling back to fused cloud "
            "(tsdf_present=%s tsdf_points=%d)",
            tsdf_cloud is not None,
            tsdf_points,
        )

    bgr, dense_depth, camera = build_dense_rgbd(
        dense_source_cloud,
        captures[anchor_idx],
        T_base_cam_anchor,
    )
    log.info("dense RGBD rendered from %s cloud (%d points)",
             dense_source_name, len(dense_source_cloud.points))

    log.info("Fused %d views around capture pose; dense RGBD %s, %d valid depth px",
             len(captures), bgr.shape, int(np.count_nonzero(dense_depth > 0)))

    return {
        "bgr": bgr,
        "depth_m": dense_depth,
        "camera": camera,
        "fused_cloud": artifacts["fused"],
        "render_source": dense_source_name,
        "render_source_points": int(len(dense_source_cloud.points)),
        "captures": captures,
        "anchor_index": anchor_idx,
        "T_base_cam_anchor": T_base_cam_anchor,
        "artifacts": artifacts,
    }
