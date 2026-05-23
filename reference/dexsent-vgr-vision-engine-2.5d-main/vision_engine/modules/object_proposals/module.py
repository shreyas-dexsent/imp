"""Implementation for `vision_engine.modules.object_proposals.module`."""

import base64
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from vision_engine.core.module_base import VisionModule
from vision_engine.io.data_plane.frame_bundle import FrameBundle


class ObjectProposalsModule(VisionModule):
    def __init__(self, name: str, params: Dict[str, Any]):
        super().__init__(name, params)
        self.method = str(params.get("method", "background")).lower()
        self.color_method = str(params.get("color_method", "background")).lower()
        self.combine_mode = str(params.get("combine_mode", "depth")).lower()
        self.min_area_px = int(params.get("min_area_px", 600))
        self.max_area_px = int(params.get("max_area_px", 300000))
        self.max_proposals = int(params.get("max_proposals", 6))
        self.blur_ksize = int(params.get("blur_ksize", 5))
        self.morph_kernel = int(params.get("morph_kernel", 5))
        self.morph_iters = int(params.get("morph_iters", 2))
        self.depth_min_m = float(params.get("depth_min_m", 0.05))
        self.depth_max_m = float(params.get("depth_max_m", 3.5))
        self.depth_plane_quantile = float(params.get("depth_plane_quantile", 0.7))
        self.depth_plane_source = str(
            params.get("depth_plane_source", "border")
        ).lower()
        self.depth_plane_fit = bool(params.get("depth_plane_fit", True))
        self.depth_plane_bins = int(params.get("depth_plane_bins", 120))
        self.depth_plane_border_px = int(params.get("depth_plane_border_px", 24))
        self.depth_smooth_ksize = int(params.get("depth_smooth_ksize", 3))
        self.object_min_height_m = float(params.get("depth_object_min_height_m", 0.008))
        self.object_max_height_m = float(params.get("depth_object_max_height_m", 0.25))
        self.bg_delta = float(params.get("bg_delta", 18.0))
        self.bg_border_px = int(params.get("bg_border_px", 24))
        self.bg_color_bgr = self._parse_color(params.get("background_color_bgr"))
        self.bg_delta_scale = float(params.get("bg_delta_scale", 2.5))
        self.bg_delta_min = float(params.get("bg_delta_min", self.bg_delta))
        self.fill_holes = bool(params.get("fill_holes", True))
        self.min_extent = float(params.get("min_extent", 0.25))
        self.min_solidity = float(params.get("min_solidity", 0.6))
        self.nms_iou_threshold = float(params.get("nms_iou_threshold", 0.4))
        self.include_image = bool(params.get("include_image", True))
        self.format = self._normalize_format(params.get("format", "jpg"))
        self.quality = int(params.get("quality", 80))

    def _normalize_format(self, fmt: Any) -> str:
        value = str(fmt or "jpg").lower().strip(".")
        if value in ("jpeg", "jpg"):
            return "jpg"
        if value == "png":
            return "png"
        return "jpg"

    def _encode_image(self, image: np.ndarray) -> Dict[str, Any]:
        params: List[int] = []
        ext = ".jpg"
        if self.format == "png":
            ext = ".png"
            params = [int(cv2.IMWRITE_PNG_COMPRESSION), 3]
        else:
            ext = ".jpg"
            quality = max(20, min(self.quality, 95))
            params = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        ok, buf = cv2.imencode(ext, image, params)
        if not ok:
            return {"ok": False}
        payload = base64.b64encode(buf).decode("ascii")
        return {
            "ok": True,
            "format": self.format,
            "width": int(image.shape[1]),
            "height": int(image.shape[0]),
            "image_b64": payload,
        }

    def _parse_color(self, value: Any) -> Optional[Tuple[int, int, int]]:
        if value is None:
            return None
        if isinstance(value, (list, tuple)) and len(value) >= 3:
            try:
                b = int(value[0])
                g = int(value[1])
                r = int(value[2])
                return (b, g, r)
            except (TypeError, ValueError):
                return None
        if isinstance(value, str):
            raw = value.strip().lstrip("#")
            if len(raw) == 6:
                try:
                    r = int(raw[0:2], 16)
                    g = int(raw[2:4], 16)
                    b = int(raw[4:6], 16)
                    return (b, g, r)
                except ValueError:
                    return None
        return None

    def _estimate_background_bgr(self, rgb: np.ndarray) -> Tuple[int, int, int]:
        h, w = rgb.shape[:2]
        border = max(4, min(self.bg_border_px, min(h, w) // 2))
        samples = []
        samples.append(rgb[:border, :, :])
        samples.append(rgb[-border:, :, :])
        samples.append(rgb[:, :border, :])
        samples.append(rgb[:, -border:, :])
        stacked = np.concatenate([s.reshape(-1, 3) for s in samples], axis=0)
        if stacked.size == 0:
            return (255, 255, 255)
        med = np.median(stacked, axis=0)
        return (int(med[0]), int(med[1]), int(med[2]))

    def _border_mask(self, h: int, w: int, border: int) -> np.ndarray:
        border = max(1, min(border, min(h, w) // 2))
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[:border, :] = 1
        mask[-border:, :] = 1
        mask[:, :border] = 1
        mask[:, -border:] = 1
        return mask.astype(bool)

    def _dominant_depth(self, depth: np.ndarray) -> float:
        if depth.size == 0:
            return float("nan")
        bins = max(10, int(self.depth_plane_bins))
        counts, edges = np.histogram(depth, bins=bins)
        if counts.size == 0:
            return float(np.median(depth))
        idx = int(np.argmax(counts))
        lo = edges[idx]
        hi = edges[min(idx + 1, len(edges) - 1)]
        band = depth[(depth >= lo) & (depth <= hi)]
        if band.size == 0:
            return float(np.median(depth))
        return float(np.median(band))

    def _fit_plane(
        self, xs: np.ndarray, ys: np.ndarray, zs: np.ndarray
    ) -> Optional[Tuple[float, float, float]]:
        if xs.size < 50:
            return None
        a = np.stack([xs, ys, np.ones_like(xs)], axis=1)
        try:
            coeffs, _, _, _ = np.linalg.lstsq(a, zs, rcond=None)
        except np.linalg.LinAlgError:
            return None
        return float(coeffs[0]), float(coeffs[1]), float(coeffs[2])

    def _segment_background(self, rgb: np.ndarray) -> np.ndarray:
        if self.bg_color_bgr:
            bg = np.array(self.bg_color_bgr, dtype=np.uint8)
        else:
            bg = np.array(self._estimate_background_bgr(rgb), dtype=np.uint8)
        lab = cv2.cvtColor(rgb, cv2.COLOR_BGR2LAB).astype(np.float32)
        bg_lab = cv2.cvtColor(bg.reshape(1, 1, 3), cv2.COLOR_BGR2LAB).astype(
            np.float32
        )[0, 0]
        diff = lab - bg_lab
        dist = np.sqrt(np.sum(diff * diff, axis=2))
        h, w = rgb.shape[:2]
        border = max(4, min(self.bg_border_px, min(h, w) // 2))
        border_mask = self._border_mask(h, w, border)
        border_dist = dist[border_mask]
        if border_dist.size:
            med = float(np.median(border_dist))
            mad = float(np.median(np.abs(border_dist - med)))
            threshold = max(self.bg_delta_min, med + self.bg_delta_scale * mad)
        else:
            threshold = self.bg_delta
        mask = (dist >= threshold).astype(np.uint8) * 255
        return mask

    def _segment_depth(self, depth: np.ndarray) -> Optional[np.ndarray]:
        valid = np.isfinite(depth)
        valid &= (depth > self.depth_min_m) & (depth < self.depth_max_m)
        if np.count_nonzero(valid) < 500:
            return None
        depth_f = depth.astype(np.float32)
        depth_f = depth_f.copy()
        depth_f[~valid] = 0.0
        if self.depth_smooth_ksize > 1:
            k = self.depth_smooth_ksize
            if k % 2 == 0:
                k += 1
            depth_f = cv2.medianBlur(depth_f, k)

        h, w = depth.shape[:2]
        if self.depth_plane_source == "quantile":
            plane_depth = float(np.quantile(depth_f[valid], self.depth_plane_quantile))
            plane = None
        else:
            if self.depth_plane_source == "border":
                border_mask = self._border_mask(h, w, self.depth_plane_border_px)
                sample_mask = valid & border_mask
                if np.count_nonzero(sample_mask) < 100:
                    sample_mask = valid
            else:
                sample_mask = valid

            ys, xs = np.where(sample_mask)
            zs = depth_f[sample_mask]
            plane = None
            if self.depth_plane_fit and xs.size > 0:
                plane = self._fit_plane(
                    xs.astype(np.float32), ys.astype(np.float32), zs.astype(np.float32)
                )

        if plane:
            a, b, c = plane
            grid_y, grid_x = np.indices((h, w), dtype=np.float32)
            plane_depth = a * grid_x + b * grid_y + c
        else:
            plane_depth = self._dominant_depth(zs)

        if isinstance(plane_depth, float) and not np.isfinite(plane_depth):
            return None

        height = plane_depth - depth_f
        mask = valid & (height > self.object_min_height_m)
        if self.object_max_height_m > 0:
            mask &= height < self.object_max_height_m
        return mask.astype(np.uint8) * 255

    def _segment_rgb(self, rgb: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
        k = max(3, self.blur_ksize)
        if k % 2 == 0:
            k += 1
        blur = cv2.GaussianBlur(gray, (k, k), 0)
        _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if np.mean(mask) > 128:
            mask = cv2.bitwise_not(mask)
        return mask

    def _fill_holes(self, mask: np.ndarray) -> np.ndarray:
        if not self.fill_holes:
            return mask
        h, w = mask.shape[:2]
        filled = mask.copy()
        inv = cv2.bitwise_not(mask)
        flood = inv.copy()
        cv2.floodFill(flood, None, (0, 0), 255)
        holes = cv2.bitwise_not(flood)
        filled = cv2.bitwise_or(filled, holes)
        return filled

    def _clean_mask(self, mask: np.ndarray) -> np.ndarray:
        k = max(3, self.morph_kernel)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        cleaned = cv2.morphologyEx(
            mask, cv2.MORPH_OPEN, kernel, iterations=self.morph_iters
        )
        cleaned = cv2.morphologyEx(
            cleaned, cv2.MORPH_CLOSE, kernel, iterations=max(1, self.morph_iters - 1)
        )
        return self._fill_holes(cleaned)

    def _combine_masks(
        self, depth_mask: Optional[np.ndarray], color_mask: Optional[np.ndarray]
    ) -> Optional[np.ndarray]:
        if depth_mask is None and color_mask is None:
            return None
        if depth_mask is None:
            return color_mask
        if color_mask is None:
            return depth_mask
        mode = self.combine_mode
        if mode == "and":
            return cv2.bitwise_and(depth_mask, color_mask)
        if mode == "or":
            return cv2.bitwise_or(depth_mask, color_mask)
        if mode == "color":
            return color_mask
        return depth_mask

    def _iou(self, a: List[int], b: List[int]) -> float:
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        inter_x0 = max(ax, bx)
        inter_y0 = max(ay, by)
        inter_x1 = min(ax + aw, bx + bw)
        inter_y1 = min(ay + ah, by + bh)
        if inter_x1 <= inter_x0 or inter_y1 <= inter_y0:
            return 0.0
        inter = (inter_x1 - inter_x0) * (inter_y1 - inter_y0)
        area_a = aw * ah
        area_b = bw * bh
        union = area_a + area_b - inter
        return float(inter / union) if union > 0 else 0.0

    def _nms(self, proposals: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if self.nms_iou_threshold <= 0:
            return proposals
        kept: List[Dict[str, Any]] = []
        for prop in proposals:
            keep = True
            for other in kept:
                if (
                    self._iou(prop["bbox_xywh"], other["bbox_xywh"])
                    >= self.nms_iou_threshold
                ):
                    keep = False
                    break
            if keep:
                kept.append(prop)
        return kept

    def _extract_proposals(self, mask: np.ndarray) -> List[Dict[str, Any]]:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        proposals: List[Dict[str, Any]] = []
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < self.min_area_px or area > self.max_area_px:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            if w <= 0 or h <= 0:
                continue
            extent = area / float(max(1, w * h))
            if extent < self.min_extent:
                continue
            hull = cv2.convexHull(contour)
            hull_area = float(cv2.contourArea(hull))
            solidity = area / hull_area if hull_area > 0 else 0.0
            if solidity < self.min_solidity:
                continue
            rect = cv2.minAreaRect(contour)
            (c_x, c_y), (r_w, r_h), angle = rect
            yaw = float(angle)
            if r_w < r_h:
                yaw += 90.0
                r_w, r_h = r_h, r_w
            box = cv2.boxPoints(rect)
            box = box.astype(int).tolist()
            proposals.append(
                {
                    "bbox_xywh": [int(x), int(y), int(w), int(h)],
                    "center_uv": [float(c_x), float(c_y)],
                    "area_px": float(area),
                    "extent": float(extent),
                    "solidity": float(solidity),
                    "yaw_deg": yaw,
                    "x_scale": 1.0,
                    "y_scale": 1.0,
                    "score": float(area),
                    "obb_points": box,
                    "obb_size": [float(r_w), float(r_h)],
                }
            )
        proposals.sort(key=lambda p: p.get("score", 0.0), reverse=True)
        proposals = self._nms(proposals)
        if self.max_proposals > 0:
            proposals = proposals[: self.max_proposals]
        return proposals

    def run(self, frame_bundle: FrameBundle) -> Dict[str, Any]:
        rgb = frame_bundle.rgb
        if rgb is None:
            return {"valid": False, "reject_reason": "missing_rgb", "proposals": []}

        depth = frame_bundle.depth
        mask = None
        method = self.method
        if method == "hybrid":
            method = "auto"

        depth_mask = None
        color_mask = None
        if method in ("depth", "auto") and depth is not None:
            depth_mask = self._segment_depth(depth)
            if depth_mask is None and method == "depth":
                return {
                    "valid": False,
                    "reject_reason": "depth_invalid",
                    "proposals": [],
                }

        if method in ("background", "auto"):
            if self.color_method == "rgb":
                color_mask = self._segment_rgb(rgb)
            else:
                color_mask = self._segment_background(rgb)
        elif method == "rgb":
            color_mask = self._segment_rgb(rgb)

        if method == "depth":
            mask = depth_mask
        else:
            mask = self._combine_masks(depth_mask, color_mask)

        if mask is None:
            return {"valid": False, "reject_reason": "no_mask", "proposals": []}

        cleaned = self._clean_mask(mask)
        proposals = self._extract_proposals(cleaned)

        result: Dict[str, Any] = {
            "valid": len(proposals) > 0,
            "reject_reason": None if proposals else "no_proposals",
            "proposal_count": len(proposals),
            "proposals": proposals,
        }

        if self.include_image:
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

        return result
