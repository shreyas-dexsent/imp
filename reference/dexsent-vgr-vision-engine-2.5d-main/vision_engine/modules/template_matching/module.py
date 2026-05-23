"""Implementation for `vision_engine.modules.template_matching.module`."""

import base64
import os
import time
from typing import Any, Dict, Tuple

import cv2
import numpy as np
from vision_engine.common.image_ops import to_gray
from vision_engine.core.module_base import VisionModule
from vision_engine.io.data_plane.frame_bundle import FrameBundle


class TemplateMatchingModule(VisionModule):
    def __init__(self, name: str, params: Dict[str, Any]):
        super().__init__(name, params)

        templates_dir = params.get("templates_dir")
        if not templates_dir:
            raise RuntimeError("templates_dir is required for template_matching")
        if not os.path.isdir(templates_dir):
            raise RuntimeError(f"Template directory not found: {templates_dir}")

        self.threshold = float(params.get("threshold", 0.8))
        self.max_results = int(params.get("max_results", 0))
        self.roi = params.get("roi")
        self.method_name = params.get("method", "TM_CCOEFF_NORMED")
        self.method = getattr(cv2, self.method_name, cv2.TM_CCOEFF_NORMED)
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

        self.rotations_deg = self._parse_list(
            params.get("rotations_deg"), default=[0.0]
        )
        self.scales = self._parse_list(params.get("scales"), default=[1.0])

        self.include_image = bool(params.get("include_image", False))
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

        self.templates = []
        self.variants = []
        self._load_templates(templates_dir)
        self._build_variants()

        print(
            f"[template_matching] Loaded {len(self.templates)} templates from {templates_dir}"
        )
        print(
            f"[template_matching] Variants: {len(self.variants)} "
            f"(rotations={len(self.rotations_deg)}, scales={len(self.scales)})"
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
                f"[template_matching] Intrinsics scaled from {int(src_w)}x{int(src_h)} "
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
            tmpl_gray = to_gray(img)
            h, w = tmpl_gray.shape[:2]
            self.templates.append(
                {
                    "name": fname,
                    "gray": tmpl_gray,
                    "width": w,
                    "height": h,
                }
            )

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
            image, mat, (new_w, new_h), flags=cv2.INTER_LINEAR, borderValue=0
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

    def _build_variants(self) -> None:
        self.variants = []
        scales = [s for s in self.scales if s > 0]
        if not scales:
            scales = [1.0]
        rotations = self.rotations_deg or [0.0]
        for tmpl in self.templates:
            base = tmpl["gray"]
            for scale in scales:
                if abs(scale - 1.0) < 1e-6:
                    scaled = base
                else:
                    new_w = max(1, int(tmpl["width"] * scale))
                    new_h = max(1, int(tmpl["height"] * scale))
                    scaled = cv2.resize(
                        base, (new_w, new_h), interpolation=cv2.INTER_AREA
                    )
                for rot in rotations:
                    rotated = self._rotate_image(scaled, rot)
                    h, w = rotated.shape[:2]
                    if h < 2 or w < 2:
                        continue
                    self.variants.append(
                        {
                            "name": tmpl["name"],
                            "gray": rotated,
                            "width": w,
                            "height": h,
                            "yaw_deg": float(rot),
                            "x_scale": float(scale),
                            "y_scale": float(scale),
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

    def _is_sqdiff(self) -> bool:
        return self.method in (cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED)

    def _match_ok(self, minv: float, maxv: float) -> Tuple[bool, float]:
        if self._is_sqdiff():
            raw = minv
            if self.method == cv2.TM_SQDIFF_NORMED:
                score = 1.0 - raw
                return score >= self.threshold, score
            return raw <= self.threshold, -raw
        return maxv >= self.threshold, maxv

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

    def run(self, frame_bundle: FrameBundle) -> Dict[str, Any]:
        rgb = frame_bundle.rgb
        if rgb is None:
            return {
                "valid": False,
                "reject_reason": "missing_rgb",
                "matches": [],
            }

        if not self.templates:
            return {
                "valid": False,
                "reject_reason": "no_templates",
                "matches": [],
            }

        gray = to_gray(rgb)
        search_img, offset = self._apply_roi(gray)
        depth = frame_bundle.depth
        if depth is not None and depth.size > 0:
            h, w = depth.shape[:2]
            self._update_intrinsics_for_frame(w, h)
        else:
            self._update_intrinsics_for_frame(int(rgb.shape[1]), int(rgb.shape[0]))

        matches = []

        for tmpl in self.variants:
            t_gray = tmpl["gray"]
            t_h = tmpl["height"]
            t_w = tmpl["width"]

            if t_h > search_img.shape[0] or t_w > search_img.shape[1]:
                continue

            res = cv2.matchTemplate(search_img, t_gray, self.method)
            minv, maxv, minl, maxl = cv2.minMaxLoc(res)
            ok, score = self._match_ok(minv, maxv)

            if not ok:
                continue

            if self._is_sqdiff():
                top_left = minl
            else:
                top_left = maxl

            x = int(top_left[0] + offset[0])
            y = int(top_left[1] + offset[1])
            cx = int(x + (t_w / 2))
            cy = int(y + (t_h / 2))

            matches.append(
                {
                    "template": tmpl["name"],
                    "score": float(score),
                    "bbox_xywh": [x, y, int(t_w), int(t_h)],
                    "center_uv": [cx, cy],
                    "yaw_deg": float(tmpl.get("yaw_deg", 0.0)),
                    "x_scale": float(tmpl.get("x_scale", 1.0)),
                    "y_scale": float(tmpl.get("y_scale", 1.0)),
                }
            )

        if matches:
            matches.sort(key=lambda m: m["score"], reverse=True)
            if self.max_results > 0:
                matches = matches[: self.max_results]

        if matches and depth is not None:
            for match in matches:
                u, v = match["center_uv"]
                ok, depth_m = self._depth_at(depth, int(u), int(v))
                if not ok:
                    continue
                proj_ok, xyz = self._project(int(u), int(v), depth_m)
                if proj_ok:
                    match["depth_m"] = depth_m
                    match["center_xyz_m"] = [xyz[0], xyz[1], xyz[2]]

        result = {
            "valid": len(matches) > 0,
            "reject_reason": None if matches else "no_match",
            "match_count": len(matches),
            "matches": matches,
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
