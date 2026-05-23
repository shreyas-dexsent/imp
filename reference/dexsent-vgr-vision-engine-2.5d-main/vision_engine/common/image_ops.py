"""Implementation for `vision_engine.common.image_ops`."""

from typing import Tuple

import cv2
import numpy as np

# -----------------------------
# Basic image ops
# -----------------------------


def to_gray(img: np.ndarray) -> np.ndarray:
    if len(img.shape) == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def resize_keep_aspect(img: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    h, w = img.shape[:2]
    scale = min(target_w / w, target_h / h)
    nw, nh = int(w * scale), int(h * scale)
    return cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)


def normalize(img: np.ndarray) -> np.ndarray:
    img = img.astype(np.float32)
    return cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


# -----------------------------
# Edge & feature helpers
# -----------------------------


def edge_map(img: np.ndarray, low: int = 50, high: int = 150) -> np.ndarray:
    gray = to_gray(img)
    return cv2.Canny(gray, low, high)


# -----------------------------
# Drawing / debug overlays
# -----------------------------


def draw_bbox(
    img: np.ndarray,
    bbox: Tuple[int, int, int, int],
    label: str | None = None,
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> np.ndarray:
    x, y, w, h = bbox
    cv2.rectangle(img, (x, y), (x + w, y + h), color, thickness)

    if label:
        cv2.putText(
            img,
            label,
            (x, max(0, y - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )
    return img


def draw_crosshair(
    img: np.ndarray,
    center: Tuple[int, int],
    size: int = 10,
    color: Tuple[int, int, int] = (0, 0, 255),
) -> np.ndarray:
    cx, cy = center
    cv2.line(img, (cx - size, cy), (cx + size, cy), color, 2)
    cv2.line(img, (cx, cy - size), (cx, cy + size), color, 2)
    return img
