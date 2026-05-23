"""Implementation for `vision_engine.modules.opt_sift.module`."""

import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from vision_engine.io.data_plane.frame_bundle import FrameBundle
from vision_engine.modules.tamplate_matching_sift.module import (
    TamplateMatchingSiftModule,
)


class OptSiftModule(TamplateMatchingSiftModule):
    """
    Hybrid tracker:
    - periodic full SIFT re-detection (same behavior as parent module),
    - optical-flow tracking between SIFT refreshes for lower per-frame cost.

    Output schema is kept compatible with `tamplate_matching_sift`.
    """

    def __init__(self, name: str, params: Dict[str, Any]):
        super().__init__(name, params)

        self.optical_flow_enabled = bool(params.get("optical_flow_enabled", True))
        self.optical_flow_refresh_interval = max(
            1, int(params.get("optical_flow_refresh_interval", 4))
        )
        self.optical_flow_max_track_age = max(
            self.optical_flow_refresh_interval,
            int(params.get("optical_flow_max_track_age", 12)),
        )
        self.optical_flow_min_points = max(
            6, int(params.get("optical_flow_min_points", self.min_good_matches))
        )
        self.optical_flow_max_points = max(
            self.optical_flow_min_points,
            int(params.get("optical_flow_max_points", 96)),
        )
        self.optical_flow_quality = max(
            1e-4, float(params.get("optical_flow_quality", 0.01))
        )
        self.optical_flow_min_distance_px = max(
            1.0, float(params.get("optical_flow_min_distance_px", 6.0))
        )
        self.optical_flow_fb_thresh_px = max(
            0.1, float(params.get("optical_flow_fb_thresh_px", 1.5))
        )
        self.optical_flow_max_err = max(
            0.1, float(params.get("optical_flow_max_err", 24.0))
        )
        self.optical_flow_win_size = max(
            9, int(params.get("optical_flow_win_size", 21))
        )
        if self.optical_flow_win_size % 2 == 0:
            self.optical_flow_win_size += 1
        self.optical_flow_max_level = max(
            0, int(params.get("optical_flow_max_level", 3))
        )
        self.optical_flow_reseed_min_points = max(
            self.optical_flow_min_points,
            int(
                params.get(
                    "optical_flow_reseed_min_points", self.optical_flow_min_points * 2
                )
            ),
        )
        # To avoid behavior changes for tasks relying on temporal gates/smoothing in parent.
        self.optical_flow_safe_mode_only = bool(
            params.get("optical_flow_safe_mode_only", True)
        )

        self._flow_prev_gray: Optional[np.ndarray] = None
        self._flow_prev_scene_pts: Optional[np.ndarray] = None
        self._flow_prev_template_pts: Optional[np.ndarray] = None
        self._flow_template_name: Optional[str] = None
        self._flow_template_w: int = 0
        self._flow_template_h: int = 0
        self._flow_frames_since_sift: int = self.optical_flow_refresh_interval

        print(
            "[opt_sift] initialized "
            f"(enabled={self.optical_flow_enabled}, refresh={self.optical_flow_refresh_interval})"
        )

    def _reset_flow_state(self, keep_prev_gray: bool = False) -> None:
        if not keep_prev_gray:
            self._flow_prev_gray = None
        self._flow_prev_scene_pts = None
        self._flow_prev_template_pts = None
        self._flow_template_name = None
        self._flow_template_w = 0
        self._flow_template_h = 0

    def _find_template(self, name: str) -> Optional[Dict[str, Any]]:
        for tmpl in self.templates:
            if str(tmpl.get("name", "")) == name:
                return tmpl
        return None

    def _can_use_flow(self) -> bool:
        if not self.optical_flow_enabled:
            return False
        if self.optical_flow_safe_mode_only and (
            self.enable_temporal_filter
            or self.enable_pose_smoothing
            or self.enable_corner_smoothing
        ):
            return False
        if self._flow_prev_gray is None:
            return False
        if self._flow_prev_scene_pts is None or self._flow_prev_template_pts is None:
            return False
        if self._flow_frames_since_sift >= self.optical_flow_refresh_interval:
            return False
        if self._flow_frames_since_sift >= self.optical_flow_max_track_age:
            return False
        return True

    def _validate_projected_quad(
        self, projected: np.ndarray, img_w: int, img_h: int
    ) -> Tuple[bool, Optional[str], float]:
        poly = projected.reshape(-1, 1, 2).astype(np.float32)
        if not cv2.isContourConvex(poly):
            return False, "quad_not_convex", 0.0

        if not self.allow_partial_visibility:
            if np.any(projected[:, 0] < -self.projected_bounds_margin_px) or np.any(
                projected[:, 0] > (img_w - 1 + self.projected_bounds_margin_px)
            ):
                return False, "quad_out_of_bounds_x", 0.0
            if np.any(projected[:, 1] < -self.projected_bounds_margin_px) or np.any(
                projected[:, 1] > (img_h - 1 + self.projected_bounds_margin_px)
            ):
                return False, "quad_out_of_bounds_y", 0.0
        else:
            in_bounds = (
                (projected[:, 0] >= 0.0)
                & (projected[:, 0] <= float(img_w - 1))
                & (projected[:, 1] >= 0.0)
                & (projected[:, 1] <= float(img_h - 1))
            )
            if int(np.count_nonzero(in_bounds)) < self.min_visible_corners:
                return False, "quad_not_visible_enough", 0.0
            clipped = projected.copy()
            clipped[:, 0] = np.clip(clipped[:, 0], 0.0, float(img_w - 1))
            clipped[:, 1] = np.clip(clipped[:, 1], 0.0, float(img_h - 1))
            visible_area_px = float(
                abs(cv2.contourArea(clipped.reshape(-1, 1, 2).astype(np.float32)))
            )
            if visible_area_px < (
                self.min_projected_area_px * self.min_visible_area_ratio
            ):
                return False, "visible_area_low", 0.0

        area_px = float(abs(cv2.contourArea(poly)))
        if area_px < self.min_projected_area_px:
            return False, "projected_area_low", 0.0
        return True, None, area_px

    def _apply_depth_to_matches(
        self, matches: List[Dict[str, Any]], depth: Optional[np.ndarray]
    ) -> None:
        if not matches or depth is None:
            return
        for match in matches:
            roi_stats = self._object_pointcloud_stats(depth, match.get("bbox_xywh", []))
            if roi_stats is not None:
                depth_m = float(roi_stats["depth_m"])
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

    def _seed_flow_from_match(self, gray: np.ndarray, match: Dict[str, Any]) -> bool:
        obb = match.get("obb_points")
        if not isinstance(obb, list) or len(obb) < 4:
            return False

        corners_scene = np.array(obb[:4], dtype=np.float32)
        if corners_scene.shape != (4, 2):
            return False

        template_name = str(match.get("template", "")).strip()
        if not template_name:
            return False

        t_w = int(match.get("template_width", 0))
        t_h = int(match.get("template_height", 0))
        if t_w <= 0 or t_h <= 0:
            tmpl = self._find_template(template_name)
            if tmpl is None:
                return False
            t_w = int(tmpl.get("width", 0))
            t_h = int(tmpl.get("height", 0))
        if t_w <= 0 or t_h <= 0:
            return False

        corners_template = np.array(
            [[0.0, 0.0], [float(t_w), 0.0], [float(t_w), float(t_h)], [0.0, float(t_h)]],
            dtype=np.float32,
        )
        h_t2s = cv2.getPerspectiveTransform(corners_template, corners_scene)
        try:
            h_s2t = np.linalg.inv(h_t2s)
        except np.linalg.LinAlgError:
            return False

        mask = np.zeros(gray.shape[:2], dtype=np.uint8)
        cv2.fillConvexPoly(mask, corners_scene.astype(np.int32), 255)
        scene_pts = cv2.goodFeaturesToTrack(
            gray,
            maxCorners=self.optical_flow_max_points,
            qualityLevel=self.optical_flow_quality,
            minDistance=self.optical_flow_min_distance_px,
            mask=mask,
            blockSize=5,
            useHarrisDetector=False,
        )
        if scene_pts is None or len(scene_pts) < self.optical_flow_min_points:
            scene_pts = corners_scene.reshape(-1, 1, 2)

        template_pts = cv2.perspectiveTransform(scene_pts, h_s2t)
        if template_pts is None:
            return False

        scene_flat = scene_pts.reshape(-1, 2)
        templ_flat = template_pts.reshape(-1, 2)
        valid = np.isfinite(scene_flat).all(axis=1) & np.isfinite(templ_flat).all(
            axis=1
        )
        valid &= (templ_flat[:, 0] >= -2.0) & (templ_flat[:, 0] <= float(t_w + 2))
        valid &= (templ_flat[:, 1] >= -2.0) & (templ_flat[:, 1] <= float(t_h + 2))
        if int(np.count_nonzero(valid)) < 4:
            return False

        self._flow_prev_gray = gray.copy()
        self._flow_prev_scene_pts = scene_flat[valid].astype(np.float32).reshape(
            -1, 1, 2
        )
        self._flow_prev_template_pts = templ_flat[valid].astype(np.float32).reshape(
            -1, 1, 2
        )
        self._flow_template_name = template_name
        self._flow_template_w = t_w
        self._flow_template_h = t_h
        return True

    def _try_flow_track(
        self, gray: np.ndarray
    ) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
        lk_params = {
            "winSize": (self.optical_flow_win_size, self.optical_flow_win_size),
            "maxLevel": self.optical_flow_max_level,
            "criteria": (
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                30,
                0.01,
            ),
        }
        prev_scene_pts = self._flow_prev_scene_pts
        prev_template_pts = self._flow_prev_template_pts
        if prev_scene_pts is None or prev_template_pts is None:
            return None, {"flow_reject": "flow_state_empty"}
        if len(prev_scene_pts) != len(prev_template_pts):
            return None, {"flow_reject": "flow_state_mismatch"}
        if len(prev_scene_pts) < 4:
            return None, {"flow_reject": "flow_state_points_low"}

        next_pts, st, err = cv2.calcOpticalFlowPyrLK(
            self._flow_prev_gray, gray, prev_scene_pts, None, **lk_params
        )
        if next_pts is None or st is None:
            return None, {"flow_reject": "flow_forward_failed"}

        back_pts, st_back, _ = cv2.calcOpticalFlowPyrLK(
            gray, self._flow_prev_gray, next_pts, None, **lk_params
        )
        if back_pts is None or st_back is None:
            return None, {"flow_reject": "flow_backward_failed"}

        prev_flat = prev_scene_pts.reshape(-1, 2)
        next_flat = next_pts.reshape(-1, 2)
        templ_flat = prev_template_pts.reshape(-1, 2)
        back_flat = back_pts.reshape(-1, 2)
        st_flat = st.reshape(-1) > 0
        st_back_flat = st_back.reshape(-1) > 0
        err_flat = (
            err.reshape(-1)
            if err is not None
            else np.zeros((next_flat.shape[0],), dtype=np.float32)
        )
        fb_err = np.linalg.norm(back_flat - prev_flat, axis=1)

        img_h, img_w = gray.shape[:2]
        valid = st_flat & st_back_flat
        valid &= np.isfinite(next_flat).all(axis=1)
        valid &= np.isfinite(templ_flat).all(axis=1)
        valid &= err_flat <= self.optical_flow_max_err
        valid &= fb_err <= self.optical_flow_fb_thresh_px
        valid &= (next_flat[:, 0] >= 0.0) & (next_flat[:, 0] <= float(img_w - 1))
        valid &= (next_flat[:, 1] >= 0.0) & (next_flat[:, 1] <= float(img_h - 1))

        if int(np.count_nonzero(valid)) < self.optical_flow_min_points:
            return None, {"flow_reject": "flow_points_low"}

        src_t = templ_flat[valid].astype(np.float32).reshape(-1, 1, 2)
        dst_s = next_flat[valid].astype(np.float32).reshape(-1, 1, 2)
        homography, mask = cv2.findHomography(src_t, dst_s, cv2.RANSAC, self.ransac_thresh)
        if homography is None or mask is None:
            return None, {"flow_reject": "flow_homography_failed"}

        inlier_mask = mask.ravel().astype(bool)
        inliers = int(inlier_mask.sum())
        if inliers < max(4, min(self.min_inliers, len(dst_s))):
            return None, {"flow_reject": "flow_inliers_low"}
        inlier_ratio = float(inliers) / float(len(dst_s))
        if inlier_ratio < self.min_inlier_ratio:
            return None, {"flow_reject": "flow_inlier_ratio_low"}

        t_w = int(self._flow_template_w)
        t_h = int(self._flow_template_h)
        template_name = str(self._flow_template_name or "")
        if t_w <= 0 or t_h <= 0 or not template_name:
            return None, {"flow_reject": "flow_template_missing"}

        corners = np.float32([[0, 0], [t_w, 0], [t_w, t_h], [0, t_h]]).reshape(-1, 1, 2)
        projected = cv2.perspectiveTransform(corners, homography).reshape(-1, 2)
        ok_quad, reason, area_px = self._validate_projected_quad(projected, img_w, img_h)
        if not ok_quad:
            return None, {"flow_reject": str(reason or "flow_quad_invalid")}

        score = self._score(inliers, len(dst_s))
        if score < self.min_score:
            return None, {"flow_reject": "flow_score_low"}

        match = self._make_match(
            template_name,
            projected,
            t_w,
            t_h,
            score,
            inliers=inliers,
            good_matches=len(dst_s),
            min_inliers_req=self.min_inliers,
            area_px=area_px,
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
                return None, {"flow_reject": "flow_scale_out_of_range"}
            denom = max(1e-6, min(sx, sy))
            anisotropy = max(sx, sy) / denom
            if anisotropy > self.max_scale_anisotropy:
                return None, {"flow_reject": "flow_scale_anisotropy_high"}
            area_ratio = sx * sy
            if area_ratio < self.min_area_ratio or area_ratio > self.max_area_ratio:
                return None, {"flow_reject": "flow_area_ratio_out_of_range"}

        stats = {
            "flow_tracks": int(len(dst_s)),
            "flow_inliers": int(inliers),
            "flow_inlier_ratio": float(inlier_ratio),
            "flow_scene_inliers": dst_s[inlier_mask].reshape(-1, 1, 2),
            "flow_template_inliers": src_t[inlier_mask].reshape(-1, 1, 2),
        }
        return match, stats

    def _update_flow_state_after_success(
        self,
        gray: np.ndarray,
        match: Dict[str, Any],
        scene_inliers: Optional[np.ndarray],
        template_inliers: Optional[np.ndarray],
    ) -> None:
        if (
            scene_inliers is not None
            and template_inliers is not None
            and len(scene_inliers) >= self.optical_flow_reseed_min_points
            and len(scene_inliers) == len(template_inliers)
        ):
            self._flow_prev_gray = gray.copy()
            self._flow_prev_scene_pts = scene_inliers.astype(np.float32).reshape(
                -1, 1, 2
            )
            self._flow_prev_template_pts = template_inliers.astype(np.float32).reshape(
                -1, 1, 2
            )
            self._flow_template_name = str(match.get("template", "")).strip() or None
            self._flow_template_w = int(match.get("template_width", 0))
            self._flow_template_h = int(match.get("template_height", 0))
            return

        if not self._seed_flow_from_match(gray, match):
            self._reset_flow_state()

    def run(self, frame_bundle: FrameBundle) -> Dict[str, Any]:
        rgb = frame_bundle.rgb
        if rgb is None:
            self._reset_flow_state()
            return super().run(frame_bundle)

        depth = frame_bundle.depth
        if depth is not None and depth.size > 0:
            h, w = depth.shape[:2]
            self._update_intrinsics_for_frame(w, h)
        else:
            self._update_intrinsics_for_frame(int(rgb.shape[1]), int(rgb.shape[0]))

        gray = self._preprocess(rgb)

        if self._can_use_flow():
            flow_match, flow_stats = self._try_flow_track(gray)
            if flow_match is not None:
                matches = [flow_match]
                self._apply_depth_to_matches(matches, depth)
                result: Dict[str, Any] = {
                    "valid": True,
                    "reject_reason": None,
                    "match_count": 1,
                    "matches": matches,
                }
                if self.debug:
                    result["debug"] = {
                        "templates_total": 1,
                        "reject_stats": {},
                        "stable_count": self._stable_count,
                        "require_consecutive": self.require_consecutive,
                        "flow_used": True,
                        "flow_tracks": int(flow_stats.get("flow_tracks", 0)),
                        "flow_inliers": int(flow_stats.get("flow_inliers", 0)),
                        "flow_inlier_ratio": float(
                            flow_stats.get("flow_inlier_ratio", 0.0)
                        ),
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

                self._update_flow_state_after_success(
                    gray,
                    flow_match,
                    flow_stats.get("flow_scene_inliers"),
                    flow_stats.get("flow_template_inliers"),
                )
                self._flow_frames_since_sift += 1
                return result

        # Fallback/refresh path: original SIFT module behavior.
        result = super().run(frame_bundle)
        if result.get("valid") and result.get("matches"):
            best = result["matches"][0]
            if self._seed_flow_from_match(gray, best):
                self._flow_frames_since_sift = 0
            else:
                self._reset_flow_state()
                self._flow_frames_since_sift = self.optical_flow_refresh_interval
        else:
            self._reset_flow_state()
            self._flow_frames_since_sift = self.optical_flow_refresh_interval
        return result

