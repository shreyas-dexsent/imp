"""Implementation for `vision_engine.modules.tamplate_matching_sift.module`."""

import base64
import math
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from vision_engine.common.image_ops import to_gray
from vision_engine.core.module_base import VisionModule
from vision_engine.io.data_plane.frame_bundle import FrameBundle


class TamplateMatchingSiftModule(VisionModule):
    """SIFT-based template matching with geometric and temporal validation.

    Processing stages:
    1) Detect SIFT features in scene and template.
    2) Match descriptors + ratio test.
    3) Estimate homography and validate geometry.
    4) Compute pose-ish outputs (center, yaw, scale, depth, optional normal).
    5) Apply temporal stability gates/smoothing before publishing.
    """

    def __init__(self, name: str, params: Dict[str, Any]):
        super().__init__(name, params)

        templates_dir = params.get("templates_dir")
        if not templates_dir:
            raise RuntimeError("templates_dir is required for tamplate_matching_sift")
        if not os.path.isdir(templates_dir):
            raise RuntimeError(f"Template directory not found: {templates_dir}")

        self.threshold = float(params.get("threshold", 0.8))
        self.max_results = int(params.get("max_results", 0))
        self.roi = params.get("roi")
        self.enable_geometric_filters = bool(
            params.get("enable_geometric_filters", True)
        )
        self.allow_non_convex_quad = bool(params.get("allow_non_convex_quad", False))
        self.enable_temporal_filter = bool(params.get("enable_temporal_filter", True))
        self.enable_pose_smoothing = bool(params.get("enable_pose_smoothing", True))
        self.enable_corner_smoothing = bool(params.get("enable_corner_smoothing", True))

        self.match_ratio = float(params.get("match_ratio", 0.75))
        self.min_score = float(params.get("min_score", 0.0))
        self.min_good_matches = int(params.get("min_good_matches", 8))
        if self.min_good_matches < 4:
            self.min_good_matches = 4
        self.max_good_matches = int(params.get("max_good_matches", 0))
        self.min_inlier_ratio = float(params.get("min_inlier_ratio", 0.35))
        self.max_reproj_rmse_px = float(params.get("max_reproj_rmse_px", 6.0))
        self.min_projected_area_px = float(params.get("min_projected_area_px", 600.0))
        self.projected_bounds_margin_px = float(
            params.get("projected_bounds_margin_px", 0.0)
        )
        self.min_scale = float(params.get("min_scale", 0.45))
        self.max_scale = float(params.get("max_scale", 2.4))
        self.max_scale_anisotropy = float(params.get("max_scale_anisotropy", 2.2))
        self.min_area_ratio = float(params.get("min_area_ratio", 0.2))
        self.max_area_ratio = float(params.get("max_area_ratio", 5.0))
        self.max_scale_jump_ratio = float(params.get("max_scale_jump_ratio", 0.45))
        self.max_yaw_jump_deg = float(params.get("max_yaw_jump_deg", 85.0))
        self.hold_last_valid_frames = max(
            0, int(params.get("hold_last_valid_frames", 0))
        )
        self.allow_partial_visibility = bool(
            params.get("allow_partial_visibility", False)
        )
        self.min_visible_corners = max(
            1, min(4, int(params.get("min_visible_corners", 2)))
        )
        self.min_visible_area_ratio = float(params.get("min_visible_area_ratio", 0.2))
        self.min_visible_area_ratio = min(1.0, max(0.0, self.min_visible_area_ratio))
        self.clip_bbox_to_image = bool(params.get("clip_bbox_to_image", True))
        self.ransac_thresh = float(params.get("ransac_thresh", 4.0))
        self.scene_scale = float(params.get("scene_scale", 1.0))
        if self.scene_scale <= 0:
            self.scene_scale = 1.0
        self.scene_scale = min(1.0, max(0.2, self.scene_scale))
        self.motion_blur_adaptive = bool(params.get("motion_blur_adaptive", True))
        self.blur_lap_var_threshold = float(params.get("blur_lap_var_threshold", 60.0))
        self.blur_adapt_strength = float(params.get("blur_adapt_strength", 1.0))
        self.blur_adapt_strength = min(1.0, max(0.0, self.blur_adapt_strength))

        min_inliers = params.get("min_inliers")
        if min_inliers is None:
            if self.threshold >= 1.0:
                self.min_inliers = int(self.threshold)
            else:
                self.min_inliers = max(6, int(round(self.threshold * 15)))
        else:
            self.min_inliers = int(min_inliers)

        self.use_clahe = bool(params.get("use_clahe", False))
        self.clahe = (
            cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            if self.use_clahe
            else None
        )

        self.nfeatures = int(params.get("nfeatures", 0))
        self.n_octave_layers = int(params.get("n_octave_layers", 3))
        self.contrast_threshold = float(params.get("contrast_threshold", 0.04))
        self.edge_threshold = float(
            params.get("sift_edge_threshold", params.get("edge_threshold", 10))
        )
        self.sigma = float(params.get("sigma", 1.6))

        intrinsics = params.get("intrinsics") or {}
        self._fx_cfg = float(intrinsics.get("fx", 0.0))
        self._fy_cfg = float(intrinsics.get("fy", 0.0))
        self._cx_cfg = float(intrinsics.get("cx", 0.0))
        self._cy_cfg = float(intrinsics.get("cy", 0.0))
        self.fx = self._fx_cfg
        self.fy = self._fy_cfg
        self.cx = self._cx_cfg
        self.cy = self._cy_cfg
        src_w, src_h = self._parse_intrinsics_resolution(params, intrinsics)
        self._intrinsics_src_w = src_w
        self._intrinsics_src_h = src_h
        self._intrinsics_warned = False
        self.depth_window_px = int(params.get("depth_window_px", 5))
        self.depth_min_m = float(params.get("depth_min_m", 0.05))
        self.depth_max_m = float(params.get("depth_max_m", 5.0))
        self.object_bbox_trim_px = max(0, int(params.get("object_bbox_trim_px", 5)))
        self.object_min_points = max(8, int(params.get("object_min_points", 40)))
        self.depth_mad_scale = float(params.get("depth_mad_scale", 3.0))
        self.depth_mad_min_tol_m = float(params.get("depth_mad_min_tol_m", 0.005))
        self.compute_surface_normal = bool(params.get("compute_surface_normal", False))

        self.include_image = bool(params.get("include_image", False))
        self.debug = bool(params.get("debug", False))
        self.image_format = str(params.get("image_format", "jpg")).lower().strip(".")
        if self.image_format not in ("jpg", "jpeg", "png"):
            self.image_format = "jpg"
        self.image_quality = int(params.get("image_quality", 80))
        self.image_fps_limit = float(params.get("image_fps_limit", 0.0))
        if self.image_fps_limit < 0:
            self.image_fps_limit = 0.0
        self._image_min_interval_s = (
            1.0 / self.image_fps_limit if self.image_fps_limit > 0 else 0.0
        )
        self._last_image_ts = 0.0
        self.require_consecutive = max(1, int(params.get("require_consecutive", 1)))
        self.stable_center_px_threshold = float(
            params.get("stable_center_px_threshold", 10.0)
        )
        self.temporal_alpha = float(params.get("temporal_alpha", 0.3))
        self.temporal_alpha = min(1.0, max(0.0, self.temporal_alpha))
        self._stable_count = 0
        self._last_center_uv: Optional[np.ndarray] = None
        self._last_yaw_deg: Optional[float] = None
        self._last_center_xyz_m: Optional[np.ndarray] = None
        self._smooth_center_uv: Optional[np.ndarray] = None
        self._smooth_yaw_deg: Optional[float] = None
        self._smooth_center_xyz_m: Optional[np.ndarray] = None
        self._last_template_name: Optional[str] = None
        self._last_scale_mean: Optional[float] = None
        self._smooth_corners: Optional[np.ndarray] = None
        self._last_published_match: Optional[Dict[str, Any]] = None
        self._hold_count = 0

        if not hasattr(cv2, "SIFT_create"):
            raise RuntimeError(
                "OpenCV SIFT not available. Install opencv-contrib-python."
            )

        self.sift = cv2.SIFT_create(
            nfeatures=self.nfeatures,
            nOctaveLayers=self.n_octave_layers,
            contrastThreshold=self.contrast_threshold,
            edgeThreshold=self.edge_threshold,
            sigma=self.sigma,
        )
        self.matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

        self.templates: List[Dict[str, Any]] = []
        self._load_templates(templates_dir)

        print(
            f"[tamplate_matching_sift] Loaded {len(self.templates)} templates from {templates_dir}"
        )

    def _preprocess(self, img: np.ndarray) -> np.ndarray:
        gray = to_gray(img)
        if self.clahe:
            return self.clahe.apply(gray)
        return gray

    def _parse_intrinsics_resolution(
        self, params: Dict[str, Any], intrinsics: Dict[str, Any]
    ) -> Tuple[float, float]:
        src = (
            params.get("intrinsics_resolution")
            or params.get("intrinsics_image_size")
            or intrinsics.get("resolution")
            or {}
        )
        if isinstance(src, dict):
            width = float(src.get("width", 0.0) or 0.0)
            height = float(src.get("height", 0.0) or 0.0)
            if width > 0 and height > 0:
                return width, height
        if isinstance(src, (list, tuple)) and len(src) >= 2:
            try:
                width = float(src[0])
                height = float(src[1])
                if width > 0 and height > 0:
                    return width, height
            except (TypeError, ValueError):
                pass
        if self._cx_cfg > 0 and self._cy_cfg > 0:
            return 2.0 * self._cx_cfg, 2.0 * self._cy_cfg
        return 0.0, 0.0

    @staticmethod
    def _parse_camera_resolution(camera_data: Dict[str, Any]) -> Tuple[float, float]:
        intr = camera_data.get("intrinsics") if isinstance(camera_data, dict) else {}
        if isinstance(intr, dict):
            res = intr.get("resolution")
            if isinstance(res, dict):
                try:
                    width = float(res.get("width", 0.0) or 0.0)
                    height = float(res.get("height", 0.0) or 0.0)
                    if width > 0 and height > 0:
                        return width, height
                except (TypeError, ValueError):
                    pass
        src = camera_data.get("resolution") if isinstance(camera_data, dict) else None
        if isinstance(src, (list, tuple)) and len(src) >= 2:
            try:
                # Camera-core publishes [height, width].
                height = float(src[0])
                width = float(src[1])
                if width > 0 and height > 0:
                    return width, height
            except (TypeError, ValueError):
                pass
        if isinstance(src, dict):
            try:
                width = float(src.get("width", 0.0) or 0.0)
                height = float(src.get("height", 0.0) or 0.0)
                if width > 0 and height > 0:
                    return width, height
            except (TypeError, ValueError):
                pass
        return 0.0, 0.0

    def _apply_frame_camera_data(self, frame_bundle: FrameBundle) -> None:
        meta = frame_bundle.meta if isinstance(frame_bundle.meta, dict) else {}
        camera_data = meta.get("camera_data") if isinstance(meta, dict) else {}
        if not isinstance(camera_data, dict) or not camera_data:
            return

        intr = camera_data.get("intrinsics") if isinstance(camera_data, dict) else {}
        fx = fy = cx = cy = None
        if isinstance(intr, dict):
            try:
                fx = float(intr.get("fx")) if intr.get("fx") is not None else None
                fy = float(intr.get("fy")) if intr.get("fy") is not None else None
                cx = float(intr.get("cx")) if intr.get("cx") is not None else None
                cy = float(intr.get("cy")) if intr.get("cy") is not None else None
            except (TypeError, ValueError):
                fx = fy = cx = cy = None
        if fx is None or fy is None or cx is None or cy is None:
            K = camera_data.get("K")
            if isinstance(K, (list, tuple)) and len(K) >= 3:
                try:
                    fx = float(K[0][0])
                    fy = float(K[1][1])
                    cx = float(K[0][2])
                    cy = float(K[1][2])
                except (TypeError, ValueError, IndexError):
                    fx = fy = cx = cy = None
        if fx is None or fy is None or cx is None or cy is None:
            return

        width, height = self._parse_camera_resolution(camera_data)
        self._fx_cfg = fx
        self._fy_cfg = fy
        self._cx_cfg = cx
        self._cy_cfg = cy
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        if width > 0 and height > 0:
            self._intrinsics_src_w = width
            self._intrinsics_src_h = height
        self._intrinsics_warned = False

    def _update_intrinsics_for_frame(self, width: int, height: int) -> None:
        self.fx = self._fx_cfg
        self.fy = self._fy_cfg
        self.cx = self._cx_cfg
        self.cy = self._cy_cfg
        if width <= 0 or height <= 0:
            return
        if self._fx_cfg <= 0 or self._fy_cfg <= 0:
            return
        src_w = self._intrinsics_src_w
        src_h = self._intrinsics_src_h
        if src_w <= 0 or src_h <= 0:
            return
        sx = float(width) / float(src_w)
        sy = float(height) / float(src_h)
        if abs(sx - 1.0) < 1e-6 and abs(sy - 1.0) < 1e-6:
            return
        self.fx = self._fx_cfg * sx
        self.fy = self._fy_cfg * sy
        self.cx = self._cx_cfg * sx
        self.cy = self._cy_cfg * sy
        if not self._intrinsics_warned:
            print(
                f"[tamplate_matching_sift] Intrinsics scaled from {int(src_w)}x{int(src_h)} "
                f"to {int(width)}x{int(height)} (sx={sx:.4f}, sy={sy:.4f})"
            )
            self._intrinsics_warned = True

    def _load_templates(self, templates_dir: str) -> None:
        for fname in sorted(os.listdir(templates_dir)):
            path = os.path.join(templates_dir, fname)
            if not os.path.isfile(path):
                continue
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            if img is None:
                continue
            gray = self._preprocess(img)
            keypoints, desc = self.sift.detectAndCompute(gray, None)
            if desc is None or len(keypoints) < 4:
                continue
            h, w = gray.shape[:2]
            self.templates.append(
                {
                    "name": fname,
                    "kp": keypoints,
                    "desc": desc,
                    "width": w,
                    "height": h,
                }
            )

    def _apply_roi(self, gray: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int]]:
        if (
            not self.roi
            or not isinstance(self.roi, (list, tuple))
            or len(self.roi) != 4
        ):
            return gray, (0, 0)

        x, y, w, h = [int(v) for v in self.roi]
        x = max(0, x)
        y = max(0, y)
        w = max(1, w)
        h = max(1, h)

        max_x = min(gray.shape[1], x + w)
        max_y = min(gray.shape[0], y + h)

        if x >= max_x or y >= max_y:
            return gray, (0, 0)

        return gray[y:max_y, x:max_x], (x, y)

    def _encode_image(self, image: np.ndarray) -> Dict[str, Any]:
        params = []
        ext = ".jpg"
        fmt = self.image_format
        if fmt == "png":
            ext = ".png"
            params = [int(cv2.IMWRITE_PNG_COMPRESSION), 3]
        else:
            ext = ".jpg"
            quality = max(20, min(self.image_quality, 95))
            params = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        ok, buf = cv2.imencode(ext, image, params)
        if not ok:
            return {"ok": False}
        payload = base64.b64encode(buf).decode("ascii")
        return {
            "ok": True,
            "format": "jpg" if fmt == "jpeg" else fmt,
            "width": int(image.shape[1]),
            "height": int(image.shape[0]),
            "image_b64": payload,
        }

    def _ratio_test(
        self, matches: List[List[cv2.DMatch]], ratio: Optional[float] = None
    ) -> List[cv2.DMatch]:
        ratio_thr = float(self.match_ratio if ratio is None else ratio)
        good = []
        for pair in matches:
            if len(pair) < 2:
                continue
            m, n = pair
            if m.distance < ratio_thr * n.distance:
                good.append(m)
        if self.max_good_matches > 0 and len(good) > self.max_good_matches:
            good = sorted(good, key=lambda d: d.distance)[: self.max_good_matches]
        return good

    def _align_corners_to_reference(
        self, corners: np.ndarray, ref_corners: np.ndarray
    ) -> np.ndarray:
        if corners.shape != (4, 2) or ref_corners.shape != (4, 2):
            return corners
        candidates = []
        base = corners.copy()
        for shift in range(4):
            candidates.append(np.roll(base, shift=shift, axis=0))
        rev = corners[::-1].copy()
        for shift in range(4):
            candidates.append(np.roll(rev, shift=shift, axis=0))
        best = corners
        best_err = float("inf")
        for c in candidates:
            err = float(np.mean(np.linalg.norm(c - ref_corners, axis=1)))
            if err < best_err:
                best_err = err
                best = c
        return best

    def _depth_at(self, depth: np.ndarray, u: int, v: int) -> Tuple[bool, float]:
        if depth is None or depth.size == 0:
            return False, 0.0
        h, w = depth.shape[:2]
        if u < 0 or v < 0 or u >= w or v >= h:
            return False, 0.0
        window = max(1, int(self.depth_window_px))
        half = window // 2
        x0 = max(0, u - half)
        x1 = min(w, u + half + 1)
        y0 = max(0, v - half)
        y1 = min(h, v + half + 1)
        patch = depth[y0:y1, x0:x1]
        valid = patch[np.isfinite(patch)]
        valid = valid[(valid > self.depth_min_m) & (valid < self.depth_max_m)]
        if valid.size == 0:
            return False, 0.0
        return True, float(np.median(valid))

    def _project(
        self, u: int, v: int, depth_m: float
    ) -> Tuple[bool, Tuple[float, float, float]]:
        if self.fx <= 0 or self.fy <= 0:
            return False, (0.0, 0.0, 0.0)
        x = (u - self.cx) * depth_m / self.fx
        y = (v - self.cy) * depth_m / self.fy
        z = depth_m
        return True, (float(x), float(y), float(z))

    def _yaw_from_corners(self, corners: np.ndarray) -> float:
        edge = corners[1] - corners[0]
        yaw = math.degrees(math.atan2(edge[1], edge[0]))
        return ((yaw + 180.0) % 360.0) - 180.0

    def _wrap_angle_deg(self, angle_deg: float) -> float:
        return ((float(angle_deg) + 180.0) % 360.0) - 180.0

    def _ema_yaw_deg(self, prev_deg: float, current_deg: float, alpha: float) -> float:
        delta = self._wrap_angle_deg(current_deg - prev_deg)
        return self._wrap_angle_deg(prev_deg + alpha * delta)

    def _centroid_from_corners_depth(
        self, depth: np.ndarray, corners: np.ndarray
    ) -> Tuple[bool, float, Tuple[int, int]]:
        """
        Compute centroid of depth points within the region defined by corner points.

        Args:
            depth: Depth frame (HxW)
            corners: 4 corner points [[x0,y0], [x1,y1], [x2,y2], [x3,y3]]

        Returns:
            (success, depth_m, (centroid_u, centroid_v))
        """
        if depth is None or depth.size == 0 or corners is None or len(corners) < 4:
            return False, 0.0, (0, 0)

        h, w = depth.shape[:2]

        # Get bounding box from corners
        corner_x = corners[:, 0]
        corner_y = corners[:, 1]
        x_min, x_max = int(np.floor(np.min(corner_x))), int(np.ceil(np.max(corner_x)))
        y_min, y_max = int(np.floor(np.min(corner_y))), int(np.ceil(np.max(corner_y)))

        # Clamp to image bounds
        x_min = max(0, x_min)
        x_max = min(w, x_max)
        y_min = max(0, y_min)
        y_max = min(h, y_max)

        if x_max <= x_min or y_max <= y_min:
            return False, 0.0, (0, 0)

        # Create mask for the polygonal region
        mask = np.zeros((h, w), dtype=np.uint8)
        corner_int = np.array(corners, dtype=np.int32)
        cv2.fillPoly(mask, [corner_int], 1)

        # Extract depth in ROI
        roi_depth = depth[y_min:y_max, x_min:x_max]
        roi_mask = mask[y_min:y_max, x_min:x_max]

        # Filter valid depth points
        valid_mask = (
            (roi_mask > 0)
            & np.isfinite(roi_depth)
            & (roi_depth > self.depth_min_m)
            & (roi_depth < self.depth_max_m)
        )

        if not np.any(valid_mask):
            return False, 0.0, (0, 0)

        # Get coordinates of valid points
        valid_y, valid_x = np.where(valid_mask)
        valid_depth = roi_depth[valid_mask]

        # Compute centroid
        centroid_x = int(np.mean(valid_x) + x_min)
        centroid_y = int(np.mean(valid_y) + y_min)
        centroid_depth = float(np.median(valid_depth))

        return True, centroid_depth, (centroid_x, centroid_y)

    def _surface_normal_from_corners(
        self, corners: np.ndarray, depth: np.ndarray
    ) -> Optional[Tuple[float, float, float]]:
        """
        Compute surface normal from depth patch at centroid using plane fitting.
        Uses PCA to fit a plane to 3D points extracted from depth frame.

        Args:
            corners: 4 corner points in image space
            depth: Depth frame

        Returns:
            (nx, ny, nz) normalized surface normal pointing towards camera
        """
        if corners is None or len(corners) < 4 or depth is None:
            return None

        try:
            # Check if intrinsics are available
            if self.fx <= 0 or self.fy <= 0:
                return None

            # Get centroid in image space
            center = corners.mean(axis=0)
            u_c, v_c = int(round(center[0])), int(round(center[1]))

            # Extract depth patch around centroid
            h, w = depth.shape[:2]
            radius = 15  # Window radius in pixels
            x0 = max(0, u_c - radius)
            x1 = min(w - 1, u_c + radius)
            y0 = max(0, v_c - radius)
            y1 = min(h - 1, v_c + radius)

            patch = depth[y0 : y1 + 1, x0 : x1 + 1]
            if patch.size == 0:
                return None

            # Find valid depth points in patch
            valid = np.isfinite(patch)
            valid &= (patch > self.depth_min_m) & (patch < self.depth_max_m)

            if np.count_nonzero(valid) < 3:
                return None

            # Extract 3D points from valid depth pixels
            ys, xs = np.where(valid)
            zs = patch[valid].astype(np.float32)
            us = xs + x0
            vs = ys + y0

            # Project to 3D using camera intrinsics
            xs_m = (us - self.cx) * zs / self.fx
            ys_m = (vs - self.cy) * zs / self.fy
            pts = np.stack([xs_m, ys_m, zs], axis=1)

            # Fit plane using PCA
            centroid = pts.mean(axis=0, keepdims=True)
            centered = pts - centroid
            cov = centered.T @ centered / max(1, pts.shape[0])

            # Eigendecomposition
            eigvals, eigvecs = np.linalg.eigh(cov)

            # Normal is eigenvector with smallest eigenvalue
            normal = eigvecs[:, int(np.argmin(eigvals))]
            norm = float(np.linalg.norm(normal))

            if norm < 1e-6:
                return None

            normal = normal / norm

            # Ensure normal points towards camera (positive z)
            if normal[2] < 0:
                normal = -normal

            return tuple(normal)
        except Exception:
            return None

    def _fit_plane_normal(self, pts: np.ndarray) -> Optional[np.ndarray]:
        if pts.shape[0] < 3:
            return None
        centroid = pts.mean(axis=0, keepdims=True)
        centered = pts - centroid
        cov = centered.T @ centered / max(1, pts.shape[0])
        try:
            eigvals, eigvecs = np.linalg.eigh(cov)
        except np.linalg.LinAlgError:
            return None
        normal = eigvecs[:, int(np.argmin(eigvals))]
        norm = float(np.linalg.norm(normal))
        if norm < 1e-6:
            return None
        return normal / norm

    def _object_pointcloud_stats(
        self,
        depth: np.ndarray,
        bbox_xywh: List[int],
    ) -> Optional[Dict[str, Any]]:
        if depth is None or depth.size == 0:
            return None
        if self.fx <= 0 or self.fy <= 0:
            return None
        if not isinstance(bbox_xywh, list) or len(bbox_xywh) < 4:
            return None

        h, w = depth.shape[:2]
        bx = int(round(float(bbox_xywh[0])))
        by = int(round(float(bbox_xywh[1])))
        bw = int(round(float(bbox_xywh[2])))
        bh = int(round(float(bbox_xywh[3])))
        if bw <= 0 or bh <= 0:
            return None

        trim = int(self.object_bbox_trim_px)
        x0 = max(0, bx + trim)
        y0 = max(0, by + trim)
        x1 = min(w, bx + bw - trim)
        y1 = min(h, by + bh - trim)
        # If trim erases ROI, fallback to untrimmed bbox.
        if x1 <= x0 or y1 <= y0:
            x0 = max(0, bx)
            y0 = max(0, by)
            x1 = min(w, bx + bw)
            y1 = min(h, by + bh)
        if x1 <= x0 or y1 <= y0:
            return None

        patch = depth[y0:y1, x0:x1]
        if patch.size == 0:
            return None

        valid = np.isfinite(patch)
        valid &= (patch > self.depth_min_m) & (patch < self.depth_max_m)
        if np.count_nonzero(valid) < self.object_min_points:
            return None

        ys, xs = np.where(valid)
        zs = patch[valid].astype(np.float32)
        us = xs.astype(np.float32) + float(x0)
        vs = ys.astype(np.float32) + float(y0)

        # Robustly reject depth outliers before centroid/normal.
        if zs.size >= self.object_min_points:
            med_z = float(np.median(zs))
            abs_dev = np.abs(zs - med_z)
            mad = float(np.median(abs_dev))
            tol = max(self.depth_mad_min_tol_m, self.depth_mad_scale * mad)
            if tol > 0.0:
                inlier = abs_dev <= tol
                if int(np.count_nonzero(inlier)) >= self.object_min_points:
                    zs = zs[inlier]
                    us = us[inlier]
                    vs = vs[inlier]

        if zs.size < self.object_min_points:
            return None

        xs_m = (us - self.cx) * zs / self.fx
        ys_m = (vs - self.cy) * zs / self.fy
        pts = np.stack([xs_m, ys_m, zs], axis=1)
        centroid = pts.mean(axis=0)
        depth_m = float(np.median(zs))

        result: Dict[str, Any] = {
            "depth_m": depth_m,
            "center_uv_depth": [float(np.mean(us)), float(np.mean(vs))],
            "center_xyz_m": [
                float(centroid[0]),
                float(centroid[1]),
                float(centroid[2]),
            ],
        }

        if self.compute_surface_normal:
            normal = self._fit_plane_normal(pts)
            if normal is not None:
                if normal[2] < 0:
                    normal = -normal
                result["surface_normal_cam"] = [
                    float(normal[0]),
                    float(normal[1]),
                    float(normal[2]),
                ]
        return result

    def _make_match(
        self,
        name: str,
        corners: np.ndarray,
        w: int,
        h: int,
        score: float,
        inliers: Optional[int] = None,
        good_matches: Optional[int] = None,
        min_inliers_req: Optional[int] = None,
        area_px: Optional[float] = None,
        depth: Optional[np.ndarray] = None,
        img_w: Optional[int] = None,
        img_h: Optional[int] = None,
    ) -> Dict[str, Any]:
        # Keep geometric center from full projected template (do not shrink on partial visibility).
        center_geom = corners.mean(axis=0)
        depth_sample_uv = [float(center_geom[0]), float(center_geom[1])]
        x_min = float(np.min(corners[:, 0]))
        x_max = float(np.max(corners[:, 0]))
        y_min = float(np.min(corners[:, 1]))
        y_max = float(np.max(corners[:, 1]))
        if (
            self.clip_bbox_to_image
            and img_w is not None
            and img_h is not None
            and img_w > 0
            and img_h > 0
        ):
            cx_min = max(0.0, min(float(img_w - 1), x_min))
            cx_max = max(0.0, min(float(img_w - 1), x_max))
            cy_min = max(0.0, min(float(img_h - 1), y_min))
            cy_max = max(0.0, min(float(img_h - 1), y_max))
            x_min, x_max, y_min, y_max = cx_min, cx_max, cy_min, cy_max
            if self.allow_partial_visibility:
                clipped = corners.copy()
                clipped[:, 0] = np.clip(clipped[:, 0], 0.0, float(img_w - 1))
                clipped[:, 1] = np.clip(clipped[:, 1], 0.0, float(img_h - 1))
                # Depth lookup point should be in-frame even when true center is outside view.
                center_vis = clipped.mean(axis=0)
                depth_sample_uv = [float(center_vis[0]), float(center_vis[1])]
        bbox = [
            int(round(x_min)),
            int(round(y_min)),
            int(round(x_max - x_min)),
            int(round(y_max - y_min)),
        ]
        center = center_geom
        edge_x = float(np.linalg.norm(corners[1] - corners[0]))
        edge_y = float(np.linalg.norm(corners[2] - corners[1]))
        yaw = self._yaw_from_corners(corners)

        # Convert corners to list for JSON serialization
        corner_list = [[float(c[0]), float(c[1])] for c in corners]

        # Compute centroid from average of 4 measured corners
        centroid_uv = [float(center[0]), float(center[1])]
        centroid_depth_m = None
        surface_normal = None

        if depth is not None:
            # Get robust depth at the centroid point using a small patch
            u_int = int(round(centroid_uv[0]))
            v_int = int(round(centroid_uv[1]))
            h, w = depth.shape[:2]

            # Use a small window to extract multiple depth values for robustness
            patch_radius = 3  # 3 pixel radius = 7x7 patch
            x0 = max(0, u_int - patch_radius)
            x1 = min(w - 1, u_int + patch_radius)
            y0 = max(0, v_int - patch_radius)
            y1 = min(h - 1, v_int + patch_radius)

            patch = depth[y0 : y1 + 1, x0 : x1 + 1]
            if patch.size > 0:
                # Extract valid depth values from patch
                valid = (
                    np.isfinite(patch)
                    & (patch > self.depth_min_m)
                    & (patch < self.depth_max_m)
                )
                if np.count_nonzero(valid) > 0:
                    # Use median depth for robustness
                    centroid_depth_m = float(np.median(patch[valid]))

            # Surface normal via local plane-fit is expensive; keep optional for low-latency mode.
            if self.compute_surface_normal:
                normal = self._surface_normal_from_corners(corners, depth)
                if normal is not None:
                    surface_normal = [float(n) for n in normal]

        match_dict = {
            "template": name,
            "score": float(score),
            "bbox_xywh": bbox,
            "center_uv": centroid_uv,
            "center_uv_depth": depth_sample_uv,
            "yaw_deg": float(yaw),
            "x_scale": float(edge_x / max(1.0, float(w))),
            "y_scale": float(edge_y / max(1.0, float(h))),
            "template_width": int(w),
            "template_height": int(h),
            "detected_width": float(edge_x),
            "detected_height": float(edge_y),
            "obb_points": corner_list,
        }
        if inliers is not None:
            match_dict["inliers"] = int(inliers)
        if good_matches is not None:
            match_dict["good_matches"] = int(good_matches)
        if min_inliers_req is not None:
            match_dict["min_inliers_req"] = int(min_inliers_req)
        if area_px is not None:
            match_dict["area_px"] = float(area_px)

        if centroid_depth_m is not None:
            match_dict["centroid_depth_m"] = centroid_depth_m

        if surface_normal is not None:
            # Keep both keys for compatibility with orchestrator consumers.
            match_dict["surface_normal"] = surface_normal
            match_dict["surface_normal_cam"] = list(surface_normal)

        return match_dict

    def _score(self, inliers: int, total: int) -> float:
        if total <= 0:
            return 0.0
        return float(inliers) / float(total)

    def run(self, frame_bundle: FrameBundle) -> Dict[str, Any]:
        """Run one inference step on the latest RGB(+optional depth) frame."""
        rgb = frame_bundle.rgb
        if rgb is None:
            return {
                "valid": False,
                "reject_reason": "missing_rgb",
                "matches": [],
            }
        self._apply_frame_camera_data(frame_bundle)
        depth = frame_bundle.depth
        if depth is not None and depth.size > 0:
            h, w = depth.shape[:2]
            self._update_intrinsics_for_frame(w, h)
        else:
            self._update_intrinsics_for_frame(int(rgb.shape[1]), int(rgb.shape[0]))

        if not self.templates:
            return {
                "valid": False,
                "reject_reason": "no_templates",
                "matches": [],
            }

        gray = self._preprocess(rgb)
        img_h, img_w = gray.shape[:2]
        search_img, offset = self._apply_roi(gray)
        scale = self.scene_scale
        if abs(scale - 1.0) > 1e-6:
            # Downscaling scene can improve throughput and matching stability on noise.
            search_img = cv2.resize(
                search_img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA
            )

        kp_scene, desc_scene = self.sift.detectAndCompute(search_img, None)
        if desc_scene is None or len(kp_scene) < 4:
            return {
                "valid": False,
                "reject_reason": "no_match",
                "match_count": 0,
                "matches": [],
            }

        # Motion blur handling: relax SIFT gating on blurred frames to reduce dropouts.
        eff_match_ratio = float(self.match_ratio)
        eff_min_good_matches = int(self.min_good_matches)
        eff_min_inliers = int(self.min_inliers)
        eff_min_score = float(self.min_score)
        blur_lap_var = 0.0
        blur_factor = 0.0
        if self.motion_blur_adaptive:
            blur_lap_var = float(cv2.Laplacian(search_img, cv2.CV_64F).var())
            denom = max(1e-6, self.blur_lap_var_threshold)
            blur_factor = max(
                0.0, min(1.0, (self.blur_lap_var_threshold - blur_lap_var) / denom)
            )
            blur_factor *= self.blur_adapt_strength
            if blur_factor > 0.0:
                eff_match_ratio = min(0.86, self.match_ratio + 0.08 * blur_factor)
                eff_min_good_matches = max(
                    4, int(round(self.min_good_matches - 3.0 * blur_factor))
                )
                eff_min_inliers = max(
                    6, int(round(self.min_inliers - 3.0 * blur_factor))
                )
                eff_min_score = max(0.0, self.min_score - 0.18 * blur_factor)

        matches: List[Dict[str, Any]] = []
        reject_stats: Dict[str, int] = {}
        total_templates = 0

        def _rej(reason: str) -> None:
            reject_stats[reason] = reject_stats.get(reason, 0) + 1

        for tmpl in self.templates:
            total_templates += 1
            kp_t = tmpl["kp"]
            desc_t = tmpl["desc"]
            raw_matches = self.matcher.knnMatch(desc_t, desc_scene, k=2)
            good = self._ratio_test(raw_matches, ratio=eff_match_ratio)
            if len(good) < eff_min_good_matches:
                _rej("no_good_matches")
                continue

            src = np.float32([kp_t[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst = np.float32([kp_scene[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            homography, mask = cv2.findHomography(
                src, dst, cv2.RANSAC, self.ransac_thresh
            )
            if homography is None or mask is None:
                _rej("homography_failed")
                continue
            inlier_mask = mask.ravel().astype(bool)
            inliers = int(inlier_mask.sum())
            if inliers < eff_min_inliers:
                _rej("inliers_low")
                continue
            inlier_ratio = float(inliers) / float(len(good))
            if inlier_ratio < self.min_inlier_ratio:
                _rej("inlier_ratio_low")
                continue

            reprojected = cv2.perspectiveTransform(src, homography)
            errors = np.linalg.norm((reprojected - dst).reshape(-1, 2), axis=1)
            if inliers > 0:
                inlier_errors = errors[inlier_mask]
                rmse = float(np.sqrt(np.mean(inlier_errors * inlier_errors)))
                if rmse > self.max_reproj_rmse_px:
                    _rej("rmse_high")
                    continue

            h = tmpl["height"]
            w = tmpl["width"]
            # Project template corners into scene image to build oriented bbox.
            corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
            projected = cv2.perspectiveTransform(corners, homography).reshape(-1, 2)
            if abs(scale - 1.0) > 1e-6:
                projected = projected * (1.0 / scale)
            projected[:, 0] += offset[0]
            projected[:, 1] += offset[1]
            poly = projected.reshape(-1, 1, 2).astype(np.float32)
            if not cv2.isContourConvex(poly):
                if self.allow_non_convex_quad:
                    rect = cv2.minAreaRect(projected.astype(np.float32))
                    projected = cv2.boxPoints(rect).astype(np.float32)
                    poly = projected.reshape(-1, 1, 2).astype(np.float32)
                    reject_reason = "quad_non_convex_rectified"
                else:
                    _rej("quad_not_convex")
                    continue
            if not self.allow_partial_visibility:
                if np.any(projected[:, 0] < -self.projected_bounds_margin_px) or np.any(
                    projected[:, 0] > (img_w - 1 + self.projected_bounds_margin_px)
                ):
                    _rej("quad_out_of_bounds_x")
                    continue
                if np.any(projected[:, 1] < -self.projected_bounds_margin_px) or np.any(
                    projected[:, 1] > (img_h - 1 + self.projected_bounds_margin_px)
                ):
                    _rej("quad_out_of_bounds_y")
                    continue
            else:
                in_bounds = (
                    (projected[:, 0] >= 0.0)
                    & (projected[:, 0] <= float(img_w - 1))
                    & (projected[:, 1] >= 0.0)
                    & (projected[:, 1] <= float(img_h - 1))
                )
                if int(np.count_nonzero(in_bounds)) < self.min_visible_corners:
                    _rej("quad_not_visible_enough")
                    continue
                clipped = projected.copy()
                clipped[:, 0] = np.clip(clipped[:, 0], 0.0, float(img_w - 1))
                clipped[:, 1] = np.clip(clipped[:, 1], 0.0, float(img_h - 1))
                visible_area_px = float(
                    abs(cv2.contourArea(clipped.reshape(-1, 1, 2).astype(np.float32)))
                )
                if visible_area_px < (
                    self.min_projected_area_px * self.min_visible_area_ratio
                ):
                    _rej("visible_area_low")
                    continue
            area_px = float(abs(cv2.contourArea(poly)))
            if area_px < self.min_projected_area_px:
                _rej("projected_area_low")
                continue
            score = self._score(inliers, len(good))
            if score < eff_min_score:
                _rej("score_low")
                continue
            # Build match object. Depth is passed so centroid/depth is computed from
            # object region rather than a single pixel when possible.
            match = self._make_match(
                tmpl["name"],
                projected,
                w,
                h,
                score,
                inliers=inliers,
                good_matches=len(good),
                min_inliers_req=eff_min_inliers,
                area_px=area_px,
                depth=depth,
                img_w=img_w,
                img_h=img_h,
            )
            if self.enable_geometric_filters:
                sx = float(match.get("x_scale", 0.0))
                sy = float(match.get("y_scale", 0.0))
                if (
                    sx < self.min_scale
                    or sy < self.min_scale
                    or sx > self.max_scale
                    or sy > self.max_scale
                ):
                    _rej("scale_out_of_range")
                    continue
                denom = max(1e-6, min(sx, sy))
                anisotropy = max(sx, sy) / denom
                if anisotropy > self.max_scale_anisotropy:
                    _rej("scale_anisotropy_high")
                    continue
                area_ratio = sx * sy
                if area_ratio < self.min_area_ratio or area_ratio > self.max_area_ratio:
                    _rej("area_ratio_out_of_range")
                    continue
            match["_inliers"] = inliers
            matches.append(match)

        if matches:
            # Deterministic ordering: highest quality first.
            matches.sort(
                key=lambda m: (m.get("score", 0.0), m.get("_inliers", 0)), reverse=True
            )
            if self.max_results > 0:
                matches = matches[: self.max_results]

        if matches and depth is not None:
            for match in matches:
                # Preferred depth path: robust ROI stats from object point cloud crop.
                roi_stats = self._object_pointcloud_stats(
                    depth, match.get("bbox_xywh", [])
                )
                if roi_stats is not None:
                    depth_m = float(roi_stats["depth_m"])
                    # Keep XY from the geometric center; only stabilize depth from ROI cloud.
                    u, v = match.get("center_uv_depth", match["center_uv"])
                    proj_ok, xyz = self._project(int(round(u)), int(round(v)), depth_m)
                    if proj_ok:
                        match["depth_m"] = depth_m
                        match["center_xyz_m"] = [xyz[0], xyz[1], xyz[2]]
                    else:
                        match["depth_m"] = depth_m
                        match["center_xyz_m"] = roi_stats["center_xyz_m"]
                    match["center_uv_depth_roi"] = roi_stats["center_uv_depth"]
                    normal_cam = roi_stats.get("surface_normal_cam")
                    if normal_cam is not None:
                        match["surface_normal"] = list(normal_cam)
                        match["surface_normal_cam"] = list(normal_cam)
                    continue

                # Fallback depth path: local patch around center pixel.
                if "centroid_depth_m" in match:
                    depth_m = float(match["centroid_depth_m"])
                else:
                    u, v = match.get("center_uv_depth", match["center_uv"])
                    ok, depth_m = self._depth_at(depth, int(round(u)), int(round(v)))
                    if not ok:
                        continue

                u, v = match.get("center_uv_depth", match["center_uv"])
                proj_ok, xyz = self._project(int(round(u)), int(round(v)), depth_m)
                if proj_ok:
                    match["depth_m"] = depth_m
                    match["center_xyz_m"] = [xyz[0], xyz[1], xyz[2]]

        for match in matches:
            match.pop("_inliers", None)

        reject_reason = None if matches else "no_match"
        if matches and self.enable_temporal_filter:
            # Temporal gate: suppress jitter/outliers before downstream consumers
            # (tracking/pick pipelines) see this as a valid detection.
            best = matches[0]
            curr_center = np.array(best.get("center_uv", [0.0, 0.0]), dtype=np.float32)
            curr_yaw = float(best.get("yaw_deg", 0.0))
            curr_xyz = best.get("center_xyz_m")
            curr_template = str(best.get("template", ""))
            curr_sx = float(best.get("x_scale", 0.0))
            curr_sy = float(best.get("y_scale", 0.0))
            curr_scale_mean = 0.5 * (curr_sx + curr_sy)
            curr_corners_raw = best.get("obb_points") or []
            curr_corners = None
            if isinstance(curr_corners_raw, list) and len(curr_corners_raw) >= 4:
                curr_corners = np.array(curr_corners_raw[:4], dtype=np.float32)
                if self._smooth_corners is not None and self._smooth_corners.shape == (
                    4,
                    2,
                ):
                    curr_corners = self._align_corners_to_reference(
                        curr_corners, self._smooth_corners
                    )
            if self._last_center_uv is None:
                self._stable_count = 1
            else:
                ref_center = (
                    self._smooth_center_uv
                    if self._smooth_center_uv is not None
                    else self._last_center_uv
                )
                movement_px = float(np.linalg.norm(curr_center - ref_center))
                scale_jump = 0.0
                if self._last_scale_mean is not None and self._last_scale_mean > 1e-6:
                    scale_jump = (
                        abs(curr_scale_mean - self._last_scale_mean)
                        / self._last_scale_mean
                    )
                yaw_jump = 0.0
                if self._smooth_yaw_deg is not None:
                    yaw_jump = abs(
                        self._wrap_angle_deg(curr_yaw - self._smooth_yaw_deg)
                    )
                if (
                    movement_px > self.stable_center_px_threshold
                    or scale_jump > self.max_scale_jump_ratio
                    or yaw_jump > self.max_yaw_jump_deg
                    or (
                        self._last_template_name is not None
                        and curr_template != self._last_template_name
                    )
                ):
                    _rej("temporal_reset")
                    self._stable_count = 0
                    self._smooth_center_uv = None
                    self._smooth_yaw_deg = None
                    self._smooth_center_xyz_m = None
                    self._smooth_corners = None
                else:
                    self._stable_count += 1
            self._last_center_uv = curr_center
            self._last_yaw_deg = curr_yaw
            self._last_template_name = curr_template
            self._last_scale_mean = curr_scale_mean
            if isinstance(curr_xyz, (list, tuple)) and len(curr_xyz) >= 3:
                self._last_center_xyz_m = np.array(curr_xyz[:3], dtype=np.float32)

            if self._stable_count >= self.require_consecutive:
                if self.enable_pose_smoothing:
                    # Exponential smoothing for center/yaw/XYZ to reduce frame-to-frame
                    # discontinuities without introducing long-memory lag.
                    if self._smooth_center_uv is None:
                        self._smooth_center_uv = curr_center
                    else:
                        self._smooth_center_uv = (
                            (1.0 - self.temporal_alpha) * self._smooth_center_uv
                            + self.temporal_alpha * curr_center
                        )

                    if self._smooth_yaw_deg is None:
                        self._smooth_yaw_deg = self._wrap_angle_deg(curr_yaw)
                    else:
                        self._smooth_yaw_deg = self._ema_yaw_deg(
                            self._smooth_yaw_deg, curr_yaw, self.temporal_alpha
                        )

                    best["center_uv"] = [
                        float(self._smooth_center_uv[0]),
                        float(self._smooth_center_uv[1]),
                    ]
                    best["yaw_deg"] = float(self._smooth_yaw_deg)
                if curr_corners is not None and self.enable_corner_smoothing:
                    if self._smooth_corners is None:
                        self._smooth_corners = curr_corners
                    else:
                        self._smooth_corners = (
                            (1.0 - self.temporal_alpha) * self._smooth_corners
                            + self.temporal_alpha * curr_corners
                        )
                    sc = self._smooth_corners
                    best["obb_points"] = [[float(p[0]), float(p[1])] for p in sc]
                    x_min = float(np.min(sc[:, 0]))
                    x_max = float(np.max(sc[:, 0]))
                    y_min = float(np.min(sc[:, 1]))
                    y_max = float(np.max(sc[:, 1]))
                    best["bbox_xywh"] = [
                        int(round(x_min)),
                        int(round(y_min)),
                        int(round(x_max - x_min)),
                        int(round(y_max - y_min)),
                    ]
                    c = sc.mean(axis=0)
                    best["center_uv"] = [float(c[0]), float(c[1])]
                    best["yaw_deg"] = float(self._yaw_from_corners(sc))
                    edge_x = float(np.linalg.norm(sc[1] - sc[0]))
                    edge_y = float(np.linalg.norm(sc[2] - sc[1]))
                    tw = float(best.get("template_width", 1.0))
                    th = float(best.get("template_height", 1.0))
                    best["detected_width"] = edge_x
                    best["detected_height"] = edge_y
                    best["x_scale"] = edge_x / max(1.0, tw)
                    best["y_scale"] = edge_y / max(1.0, th)

                if isinstance(curr_xyz, (list, tuple)) and len(curr_xyz) >= 3:
                    curr_xyz_np = np.array(curr_xyz[:3], dtype=np.float32)
                    if self._smooth_center_xyz_m is None:
                        self._smooth_center_xyz_m = curr_xyz_np
                    else:
                        self._smooth_center_xyz_m = (
                            (1.0 - self.temporal_alpha) * self._smooth_center_xyz_m
                            + self.temporal_alpha * curr_xyz_np
                        )
                    best["center_xyz_m"] = [
                        float(self._smooth_center_xyz_m[0]),
                        float(self._smooth_center_xyz_m[1]),
                        float(self._smooth_center_xyz_m[2]),
                    ]
                self._last_published_match = dict(best)
                self._hold_count = self.hold_last_valid_frames
            else:
                if self._last_published_match is not None and self._hold_count > 0:
                    # Optional hold-last-good to bridge tiny dropout bursts.
                    matches = [dict(self._last_published_match)]
                    self._hold_count -= 1
                    reject_reason = "holding_last_stable"
                else:
                    matches = []
                    reject_reason = "not_stable_yet"
        elif not matches:
            self._stable_count = 0
            self._smooth_corners = None
            if self._last_published_match is not None and self._hold_count > 0:
                matches = [dict(self._last_published_match)]
                self._hold_count -= 1
                reject_reason = "holding_last_stable"

        result: Dict[str, Any] = {
            "valid": len(matches) > 0,
            "reject_reason": reject_reason,
            "match_count": len(matches),
            "matches": matches,
        }
        if self.debug:
            result["debug"] = {
                "templates_total": total_templates,
                "reject_stats": reject_stats,
                "stable_count": self._stable_count,
                "require_consecutive": self.require_consecutive,
                "blur_lap_var": blur_lap_var,
                "blur_factor": blur_factor,
                "eff_match_ratio": eff_match_ratio,
                "eff_min_good_matches": eff_min_good_matches,
                "eff_min_inliers": eff_min_inliers,
                "eff_min_score": eff_min_score,
            }
        if self.include_image:
            now = time.monotonic()
            if (
                self._image_min_interval_s <= 0
                or (now - self._last_image_ts) >= self._image_min_interval_s
            ):
                encoded = self._encode_image(rgb)
                if encoded.get("ok"):
                    result.update(
                        {
                            "format": encoded["format"],
                            "width": encoded["width"],
                            "height": encoded["height"],
                            "image_b64": encoded["image_b64"],
                        }
                    )
                    self._last_image_ts = now
        return result
