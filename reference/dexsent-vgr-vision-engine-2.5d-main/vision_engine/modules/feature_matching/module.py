"""Implementation for `vision_engine.modules.feature_matching.module`."""

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


class FeatureMatchingModule(VisionModule):
    def __init__(self, name: str, params: Dict[str, Any]):
        super().__init__(name, params)

        templates_dir = params.get("templates_dir")
        if not templates_dir:
            raise RuntimeError("templates_dir is required for feature_matching")
        if not os.path.isdir(templates_dir):
            raise RuntimeError(f"Template directory not found: {templates_dir}")

        self.nfeatures = int(params.get("nfeatures", 1200))
        if self.nfeatures <= 0:
            # ORB requires a positive target feature count; 0 produces no keypoints.
            self.nfeatures = 1200
        self.scale_factor = float(params.get("scale_factor", 1.2))
        self.nlevels = int(params.get("nlevels", 8))
        self.orb_edge_threshold = int(params.get("orb_edge_threshold", 31))
        self.fast_threshold = int(params.get("fast_threshold", 20))
        self.match_ratio = float(params.get("match_ratio", 0.75))
        self.max_matches = int(params.get("max_matches", 200))
        self.min_inliers = int(params.get("min_inliers", 12))
        self.min_inliers_mode = str(params.get("min_inliers_mode", "dynamic")).lower()
        self.min_inliers_ratio = float(params.get("min_inliers_ratio", 0.12))
        self.min_inliers_min = int(params.get("min_inliers_min", 8))
        self.min_inliers_max = int(params.get("min_inliers_max", 40))
        self.ransac_thresh = float(params.get("ransac_thresh", 4.0))
        self.max_results = int(params.get("max_results", 1))
        self.use_clahe = bool(params.get("use_clahe", True))
        self.edge_fallback = bool(params.get("edge_fallback", True))
        self.edge_match_threshold = float(params.get("edge_threshold", 0.45))
        self.edge_low = int(params.get("edge_low", 60))
        self.edge_high = int(params.get("edge_high", 180))
        self.edge_blur = int(params.get("edge_blur", 3))
        self.edge_rotations_deg = self._parse_list(
            params.get("edge_rotations_deg"),
            default=[-90, -60, -30, 0, 30, 60, 90, 120, 150, 180],
        )
        self.edge_scales = self._parse_list(
            params.get("edge_scales"),
            default=[0.8, 0.9, 1.0, 1.1, 1.2],
        )
        self.scene_scale = float(params.get("scene_scale", 1.0))
        if self.scene_scale <= 0:
            self.scene_scale = 1.0
        self.scene_scale = min(1.0, max(0.2, self.scene_scale))

        self.include_image = bool(params.get("include_image", False))
        self.image_format = str(params.get("image_format", "jpg")).lower().strip(".")
        if self.image_format not in ("jpg", "jpeg", "png"):
            self.image_format = "jpg"
        self.image_quality = int(params.get("image_quality", 75))
        self.image_fps_limit = float(params.get("image_fps_limit", 0.0))
        if self.image_fps_limit < 0:
            self.image_fps_limit = 0.0
        self._image_min_interval_s = (
            1.0 / self.image_fps_limit if self.image_fps_limit > 0 else 0.0
        )
        self._last_image_ts = 0.0
        self.image_on_match = bool(params.get("image_on_match", True))
        self.image_on_match_max = int(params.get("image_on_match_max", 2))
        self.image_on_match_cooldown_ms = int(
            params.get("image_on_match_cooldown_ms", 0)
        )
        if self.image_on_match_cooldown_ms < 0:
            self.image_on_match_cooldown_ms = 0
        self._match_images_emitted = 0
        self._match_image_ts = 0.0

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
        self.compute_surface_normal = bool(params.get("compute_surface_normal", False))
        self.object_bbox_trim_px = max(0, int(params.get("object_bbox_trim_px", 5)))
        self.object_min_points = max(8, int(params.get("object_min_points", 40)))
        self.normal_window_px = int(params.get("normal_window_px", 5))
        self.normal_min_points = int(params.get("normal_min_points", 12))
        self.normal_depth_tol_m = float(params.get("normal_depth_tol_m", 0.01))
        self.normal_mad_scale = float(params.get("normal_mad_scale", 3.0))
        self.normal_use_mad = bool(params.get("normal_use_mad", True))
        self.normal_ransac_enable = bool(params.get("normal_ransac_enable", True))
        self.normal_ransac_iters = int(params.get("normal_ransac_iters", 120))
        self.normal_ransac_tol_m = float(params.get("normal_ransac_tol_m", 0.003))
        self.normal_ransac_min_inliers = int(
            params.get("normal_ransac_min_inliers", 20)
        )
        self.normal_ema_alpha = float(params.get("normal_ema_alpha", 0.0))
        self._normal_ema: Optional[np.ndarray] = None

        self.orb = cv2.ORB_create(
            nfeatures=self.nfeatures,
            scaleFactor=self.scale_factor,
            nlevels=self.nlevels,
            edgeThreshold=self.orb_edge_threshold,
            fastThreshold=self.fast_threshold,
        )
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        self.clahe = (
            cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            if self.use_clahe
            else None
        )

        self.templates: List[Dict[str, Any]] = []
        self._load_templates(templates_dir)

        print(
            f"[feature_matching] Loaded {len(self.templates)} templates from {templates_dir} "
            f"(edge_fallback={self.edge_fallback})"
        )

    def _parse_list(self, value: Any, default: list) -> list:
        if value is None:
            return list(default)
        if isinstance(value, (int, float)):
            return [float(value)]
        if isinstance(value, (list, tuple)):
            out = []
            for item in value:
                try:
                    out.append(float(item))
                except (TypeError, ValueError):
                    continue
            return out or list(default)
        return list(default)

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
                f"[feature_matching] Intrinsics scaled from {int(src_w)}x{int(src_h)} "
                f"to {int(width)}x{int(height)} (sx={sx:.4f}, sy={sy:.4f})"
            )
            self._intrinsics_warned = True

    def _min_inliers_for(self, good_count: int) -> int:
        if self.min_inliers_mode != "dynamic":
            return self.min_inliers
        base = int(round(max(1, good_count) * self.min_inliers_ratio))
        return max(self.min_inliers_min, min(self.min_inliers_max, base))

    def _preprocess(self, img: np.ndarray) -> np.ndarray:
        gray = to_gray(img)
        if self.clahe:
            return self.clahe.apply(gray)
        return gray

    def _load_templates(self, templates_dir: str) -> None:
        for fname in sorted(os.listdir(templates_dir)):
            path = os.path.join(templates_dir, fname)
            if not os.path.isfile(path):
                continue
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            if img is None:
                continue
            gray = self._preprocess(img)
            keypoints, desc = self.orb.detectAndCompute(gray, None)
            if desc is None or len(keypoints) < 4:
                continue
            h, w = gray.shape[:2]
            edge = self._edge_map(gray)
            self.templates.append(
                {
                    "name": fname,
                    "kp": keypoints,
                    "desc": desc,
                    "width": w,
                    "height": h,
                    "edge": edge,
                }
            )

    def _edge_map(self, gray: np.ndarray) -> np.ndarray:
        if self.edge_blur and self.edge_blur > 1:
            k = int(self.edge_blur)
            if k % 2 == 0:
                k += 1
            gray = cv2.GaussianBlur(gray, (k, k), 0)
        return cv2.Canny(gray, self.edge_low, self.edge_high)

    def _rotate_image(self, image: np.ndarray, angle_deg: float) -> np.ndarray:
        if abs(angle_deg) < 1e-6:
            return image
        h, w = image.shape[:2]
        c_x, c_y = (w / 2.0), (h / 2.0)
        mat = cv2.getRotationMatrix2D((c_x, c_y), angle_deg, 1.0)
        cos = abs(mat[0, 0])
        sin = abs(mat[0, 1])
        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))
        mat[0, 2] += (new_w / 2.0) - c_x
        mat[1, 2] += (new_h / 2.0) - c_y
        return cv2.warpAffine(
            image, mat, (new_w, new_h), flags=cv2.INTER_NEAREST, borderValue=0
        )

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

    def _ratio_test(self, matches: List[List[cv2.DMatch]]) -> List[cv2.DMatch]:
        good = []
        for pair in matches:
            if len(pair) < 2:
                continue
            m, n = pair
            if m.distance < self.match_ratio * n.distance:
                good.append(m)
        if self.max_matches > 0:
            good = sorted(good, key=lambda m: m.distance)[: self.max_matches]
        return good

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

    def _normal_at(self, depth: np.ndarray, u: int, v: int) -> Optional[List[float]]:
        if self.fx <= 0 or self.fy <= 0:
            return None
        h, w = depth.shape[:2]
        radius = max(2, int(self.normal_window_px))
        x0 = max(0, u - radius)
        x1 = min(w - 1, u + radius)
        y0 = max(0, v - radius)
        y1 = min(h - 1, v + radius)
        patch = depth[y0 : y1 + 1, x0 : x1 + 1]
        if patch.size == 0:
            return None
        valid = np.isfinite(patch)
        valid &= (patch > self.depth_min_m) & (patch < self.depth_max_m)
        if np.count_nonzero(valid) < self.normal_min_points:
            return None
        if self.normal_use_mad:
            zs_all = patch[valid].astype(np.float32)
            if zs_all.size >= self.normal_min_points:
                median_z = float(np.median(zs_all))
                abs_dev = np.abs(zs_all - median_z)
                mad = float(np.median(abs_dev))
                tol = max(self.normal_depth_tol_m, self.normal_mad_scale * mad)
                if tol > 0:
                    valid &= np.abs(patch - median_z) <= tol
                    if np.count_nonzero(valid) < self.normal_min_points:
                        return None
        ys, xs = np.where(valid)
        zs = patch[valid].astype(np.float32)
        us = xs + x0
        vs = ys + y0
        xs_m = (us - self.cx) * zs / self.fx
        ys_m = (vs - self.cy) * zs / self.fy
        pts = np.stack([xs_m, ys_m, zs], axis=1)
        normal = None
        if self.normal_ransac_enable:
            normal = self._ransac_normal(pts)
        if normal is None:
            normal = self._fit_plane_normal(pts)
        if normal is None:
            return None
        if normal[2] < 0:
            normal = -normal
        if self.normal_ema_alpha > 0:
            alpha = min(1.0, max(0.0, self.normal_ema_alpha))
            if self._normal_ema is None:
                self._normal_ema = normal
            else:
                blended = (1.0 - alpha) * self._normal_ema + alpha * normal
                norm = float(np.linalg.norm(blended))
                if norm >= 1e-6:
                    self._normal_ema = blended / norm
            normal = self._normal_ema if self._normal_ema is not None else normal
        return [float(normal[0]), float(normal[1]), float(normal[2])]

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
        xs_m = (us - self.cx) * zs / self.fx
        ys_m = (vs - self.cy) * zs / self.fy
        pts = np.stack([xs_m, ys_m, zs], axis=1)

        # Reject depth outliers before centroid/normal to stabilize Z.
        if self.normal_use_mad and pts.shape[0] >= self.normal_min_points:
            z_vals = pts[:, 2]
            med = float(np.median(z_vals))
            mad = float(np.median(np.abs(z_vals - med)))
            tol = max(self.normal_depth_tol_m, self.normal_mad_scale * mad)
            if tol > 0:
                inlier = np.abs(z_vals - med) <= tol
                if int(np.count_nonzero(inlier)) >= self.normal_min_points:
                    pts = pts[inlier]
                    us = us[inlier]
                    vs = vs[inlier]

        if pts.shape[0] < self.normal_min_points:
            return None

        centroid = pts.mean(axis=0)
        center_uv = [float(np.mean(us)), float(np.mean(vs))]
        normal = None
        if self.compute_surface_normal:
            normal = self._ransac_normal(pts) if self.normal_ransac_enable else None
            if normal is None:
                normal = self._fit_plane_normal(pts)
            if normal is not None and normal[2] < 0:
                normal = -normal
        return {
            "center_xyz_m": [
                float(centroid[0]),
                float(centroid[1]),
                float(centroid[2]),
            ],
            "depth_m": float(centroid[2]),
            "center_uv_depth": center_uv,
            "surface_normal_cam": (
                [float(normal[0]), float(normal[1]), float(normal[2])]
                if normal is not None
                else None
            ),
        }

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

    def _ransac_normal(self, pts: np.ndarray) -> Optional[np.ndarray]:
        count = pts.shape[0]
        if count < 3:
            return None
        best_inliers = None
        best_count = 0
        tol = max(1e-6, float(self.normal_ransac_tol_m))
        rng = np.random.default_rng()
        for _ in range(max(1, self.normal_ransac_iters)):
            try:
                idx = rng.choice(count, size=3, replace=False)
            except ValueError:
                return None
            a, b, c = pts[idx]
            normal = np.cross(b - a, c - a)
            norm = float(np.linalg.norm(normal))
            if norm < 1e-8:
                continue
            normal = normal / norm
            dists = np.abs((pts - a) @ normal)
            inliers = dists <= tol
            inlier_count = int(np.count_nonzero(inliers))
            if inlier_count > best_count:
                best_count = inlier_count
                best_inliers = inliers
                if best_count >= count * 0.8:
                    break
        if best_inliers is None or best_count < self.normal_ransac_min_inliers:
            return None
        return self._fit_plane_normal(pts[best_inliers])

    def _yaw_from_corners(self, corners: np.ndarray) -> float:
        edge = corners[1] - corners[0]
        yaw = math.degrees(math.atan2(edge[1], edge[0]))
        return ((yaw + 180.0) % 360.0) - 180.0

    def _make_match(
        self, name: str, corners: np.ndarray, inliers: int, w: int, h: int
    ) -> Dict[str, Any]:
        x_min = float(np.min(corners[:, 0]))
        x_max = float(np.max(corners[:, 0]))
        y_min = float(np.min(corners[:, 1]))
        y_max = float(np.max(corners[:, 1]))
        bbox = [
            int(round(x_min)),
            int(round(y_min)),
            int(round(x_max - x_min)),
            int(round(y_max - y_min)),
        ]
        center = corners.mean(axis=0)
        edge_x = np.linalg.norm(corners[1] - corners[0])
        edge_y = np.linalg.norm(corners[2] - corners[1])
        yaw = self._yaw_from_corners(corners)
        area = float(cv2.contourArea(corners.astype(np.float32)))
        return {
            "template": name,
            "score": float(inliers),
            "inliers": int(inliers),
            "bbox_xywh": bbox,
            "center_uv": [float(center[0]), float(center[1])],
            "yaw_deg": float(yaw),
            "x_scale": float(edge_x / max(1.0, float(w))),
            "y_scale": float(edge_y / max(1.0, float(h))),
            "area_px": area,
            "obb_points": corners.round().astype(int).tolist(),
            "obb_size": [float(edge_x), float(edge_y)],
        }

    def _obb_from_center(
        self, center: Tuple[float, float], w: float, h: float, yaw_deg: float
    ) -> List[List[int]]:
        theta = math.radians(yaw_deg)
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)
        half_w = w / 2.0
        half_h = h / 2.0
        corners = [
            (-half_w, -half_h),
            (half_w, -half_h),
            (half_w, half_h),
            (-half_w, half_h),
        ]
        pts = []
        for x, y in corners:
            rx = x * cos_t - y * sin_t + center[0]
            ry = x * sin_t + y * cos_t + center[1]
            pts.append([int(round(rx)), int(round(ry))])
        return pts

    def _edge_fallback_match(
        self, gray: np.ndarray, scene_scale: float = 1.0
    ) -> List[Dict[str, Any]]:
        scene_edges = self._edge_map(gray)
        scene_h, scene_w = scene_edges.shape[:2]
        best: Optional[Dict[str, Any]] = None
        inv_scale = 1.0
        if scene_scale and abs(scene_scale - 1.0) > 1e-6:
            inv_scale = 1.0 / scene_scale
        for tmpl in self.templates:
            t_edge = tmpl["edge"]
            t_h = tmpl["height"]
            t_w = tmpl["width"]
            for scale in self.edge_scales:
                if scale <= 0:
                    continue
                sw = max(2, int(t_w * scale))
                sh = max(2, int(t_h * scale))
                scaled = cv2.resize(t_edge, (sw, sh), interpolation=cv2.INTER_AREA)
                for rot in self.edge_rotations_deg:
                    rotated = self._rotate_image(scaled, rot)
                    r_h, r_w = rotated.shape[:2]
                    if r_h >= scene_h or r_w >= scene_w:
                        continue
                    res = cv2.matchTemplate(scene_edges, rotated, cv2.TM_CCOEFF_NORMED)
                    _, maxv, _, maxl = cv2.minMaxLoc(res)
                    if maxv < self.edge_match_threshold:
                        continue
                    center = (maxl[0] + r_w / 2.0, maxl[1] + r_h / 2.0)
                    obb = self._obb_from_center(center, sw, sh, float(rot))
                    bbox = [int(maxl[0]), int(maxl[1]), int(r_w), int(r_h)]
                    size_w = float(sw)
                    size_h = float(sh)
                    if inv_scale != 1.0:
                        center = (center[0] * inv_scale, center[1] * inv_scale)
                        obb = [
                            [
                                int(round(pt[0] * inv_scale)),
                                int(round(pt[1] * inv_scale)),
                            ]
                            for pt in obb
                        ]
                        bbox = [
                            int(round(bbox[0] * inv_scale)),
                            int(round(bbox[1] * inv_scale)),
                            int(round(bbox[2] * inv_scale)),
                            int(round(bbox[3] * inv_scale)),
                        ]
                        size_w *= inv_scale
                        size_h *= inv_scale
                    match = {
                        "template": tmpl["name"],
                        "score": float(maxv),
                        "inliers": 0,
                        "bbox_xywh": bbox,
                        "center_uv": [float(center[0]), float(center[1])],
                        "yaw_deg": float(rot),
                        "x_scale": float(size_w / max(1.0, float(t_w))),
                        "y_scale": float(size_h / max(1.0, float(t_h))),
                        "area_px": float(size_w * size_h),
                        "obb_points": obb,
                        "obb_size": [float(size_w), float(size_h)],
                        "method": "edge_template",
                    }
                    if not best or match["score"] > best["score"]:
                        best = match
        if best:
            return [best]
        return []

    def run(self, frame_bundle: FrameBundle) -> Dict[str, Any]:
        rgb = frame_bundle.rgb
        if rgb is None:
            return {"valid": False, "reject_reason": "missing_rgb", "matches": []}
        depth = frame_bundle.depth
        if depth is not None and depth.size > 0:
            h, w = depth.shape[:2]
            self._update_intrinsics_for_frame(w, h)
        else:
            self._update_intrinsics_for_frame(int(rgb.shape[1]), int(rgb.shape[0]))

        if not self.templates:
            return {"valid": False, "reject_reason": "no_templates", "matches": []}

        scale = self.scene_scale
        rgb_proc = rgb
        if abs(scale - 1.0) > 1e-6:
            rgb_proc = cv2.resize(
                rgb, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA
            )
        gray = self._preprocess(rgb_proc)
        kp_scene, desc_scene = self.orb.detectAndCompute(gray, None)
        matches: List[Dict[str, Any]] = []
        if desc_scene is not None and len(kp_scene) >= 4:
            for tmpl in self.templates:
                kp_t = tmpl["kp"]
                desc_t = tmpl["desc"]
                raw_matches = self.matcher.knnMatch(desc_t, desc_scene, k=2)
                good = self._ratio_test(raw_matches)
                min_req = self._min_inliers_for(len(good))
                if len(good) < min_req:
                    continue
                src = np.float32([kp_t[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                dst = np.float32([kp_scene[m.trainIdx].pt for m in good]).reshape(
                    -1, 1, 2
                )
                homography, mask = cv2.findHomography(
                    src, dst, cv2.RANSAC, self.ransac_thresh
                )
                if homography is None or mask is None:
                    continue
                inliers = int(mask.ravel().sum())
                if inliers < min_req:
                    continue
                h = tmpl["height"]
                w = tmpl["width"]
                corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
                projected = cv2.perspectiveTransform(corners, homography).reshape(-1, 2)
                if abs(scale - 1.0) > 1e-6:
                    projected = projected * (1.0 / scale)
                match = self._make_match(tmpl["name"], projected, inliers, w, h)
                match["min_inliers_req"] = min_req
                matches.append(match)

        if not matches and self.edge_fallback:
            matches = self._edge_fallback_match(gray, scale)

        if matches:
            matches.sort(key=lambda m: m.get("score", 0.0), reverse=True)
            if self.max_results > 0:
                matches = matches[: self.max_results]

        if matches and depth is not None:
            for match in matches:
                cloud_stats = self._object_pointcloud_stats(
                    depth, match.get("bbox_xywh", [])
                )
                if cloud_stats:
                    match["depth_m"] = cloud_stats["depth_m"]
                    match["center_xyz_m"] = cloud_stats["center_xyz_m"]
                    match["center_uv_depth"] = cloud_stats["center_uv_depth"]
                    normal = cloud_stats.get("surface_normal_cam")
                    if normal:
                        match["surface_normal_cam"] = normal
                    continue

                # Fallback: center-pixel projection if bbox cloud is invalid.
                u, v = match["center_uv"]
                ok, depth_m = self._depth_at(depth, int(round(u)), int(round(v)))
                if not ok:
                    continue
                proj_ok, xyz = self._project(int(round(u)), int(round(v)), depth_m)
                if proj_ok:
                    match["depth_m"] = depth_m
                    match["center_xyz_m"] = [xyz[0], xyz[1], xyz[2]]
                    if self.compute_surface_normal:
                        normal = self._normal_at(depth, int(round(u)), int(round(v)))
                        if normal:
                            match["surface_normal_cam"] = normal

        result: Dict[str, Any] = {
            "valid": len(matches) > 0,
            "reject_reason": None if matches else "no_match",
            "match_count": len(matches),
            "matches": matches,
        }

        if self.include_image:
            now = time.monotonic()
            force_image = False
            if (
                matches
                and self.image_on_match
                and self._match_images_emitted < self.image_on_match_max
            ):
                if self.image_on_match_cooldown_ms == 0 or (
                    (now - self._match_image_ts) * 1000.0
                    >= self.image_on_match_cooldown_ms
                ):
                    force_image = True
            if (
                force_image
                or self._image_min_interval_s <= 0
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
                    if force_image:
                        self._match_images_emitted += 1
                        self._match_image_ts = now

        return result
