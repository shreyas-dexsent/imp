"""Implementation for `vision_engine.modules.blob_detection.module`."""

from typing import Any, Dict, List

import cv2
import numpy as np
from vision_engine.common.image_ops import to_gray
from vision_engine.core.module_base import VisionModule
from vision_engine.io.data_plane.frame_bundle import FrameBundle


class BlobDetectionModule(VisionModule):
    """
    Generic blob detection using contour analysis.
    """

    def __init__(self, name: str, params: Dict[str, Any]):
        super().__init__(name, params)

        self.min_area = int(params.get("min_area", 500))
        self.max_area = int(params.get("max_area", 100000))
        self.threshold = int(params.get("threshold", 128))
        self.invert = bool(params.get("invert", False))
        self.blur_ksize = int(params.get("blur_ksize", 5))
        self.morph_kernel = int(params.get("morph_kernel", 0))

        if self.blur_ksize % 2 == 0:
            self.blur_ksize += 1
        if self.min_area < 0:
            self.min_area = 0
        if self.max_area < self.min_area:
            self.max_area = self.min_area

        print(
            f"[blob_detection] init min_area={self.min_area} "
            f"max_area={self.max_area} threshold={self.threshold}"
        )

    def run(self, frame_bundle: FrameBundle) -> Dict[str, Any]:
        """
        Run blob detection on an RGB frame.
        """
        rgb = frame_bundle.rgb
        if rgb is None:
            return {
                "valid": False,
                "reject_reason": "missing_rgb",
                "blobs": [],
            }

        gray = to_gray(rgb)
        if self.blur_ksize > 1:
            gray = cv2.GaussianBlur(gray, (self.blur_ksize, self.blur_ksize), 0)

        thresh_type = cv2.THRESH_BINARY_INV if self.invert else cv2.THRESH_BINARY
        _, bin_img = cv2.threshold(gray, self.threshold, 255, thresh_type)

        if self.morph_kernel > 0:
            ksize = max(1, self.morph_kernel)
            kernel = np.ones((ksize, ksize), np.uint8)
            bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(
            bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        blobs = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_area or area > self.max_area:
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            cx = int(x + w / 2)
            cy = int(y + h / 2)

            blobs.append(
                {
                    "bbox_xywh": [int(x), int(y), int(w), int(h)],
                    "centroid_uv": [cx, cy],
                    "area": float(area),
                }
            )

        return {
            "valid": len(blobs) > 0,
            "reject_reason": None if blobs else "no_blobs",
            "blob_count": len(blobs),
            "blobs": blobs,
        }
