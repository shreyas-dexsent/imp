"""Implementation for `vision_engine.common.geometry`."""

import math
from typing import Tuple

import numpy as np

# -----------------------------
# Bounding box utilities
# bbox format: (x, y, w, h)
# -----------------------------


def bbox_area(bbox: Tuple[int, int, int, int]) -> int:
    _, _, w, h = bbox
    return max(0, w) * max(0, h)


def bbox_intersection(b1, b2) -> int:
    x1, y1, w1, h1 = b1
    x2, y2, w2, h2 = b2

    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)

    iw = max(0, xi2 - xi1)
    ih = max(0, yi2 - yi1)

    return iw * ih


def bbox_iou(b1, b2) -> float:
    inter = bbox_intersection(b1, b2)
    union = bbox_area(b1) + bbox_area(b2) - inter
    if union <= 0:
        return 0.0
    return inter / union


# -----------------------------
# Coordinate transforms
# -----------------------------


def pixel_to_camera(u: float, v: float, depth: float, K: np.ndarray) -> np.ndarray:
    """
    Convert pixel (u,v,depth) to camera-frame XYZ
    K = camera intrinsics (3x3)
    """
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    X = (u - cx) * depth / fx
    Y = (v - cy) * depth / fy
    Z = depth

    return np.array([X, Y, Z], dtype=float)


def transform_point(pt: np.ndarray, T: np.ndarray) -> np.ndarray:
    """
    Apply homogeneous transform T (4x4) to 3D point
    """
    pt_h = np.append(pt, 1.0)
    out = T @ pt_h
    return out[:3]


# -----------------------------
# Angle helpers
# -----------------------------


def normalize_angle(theta: float) -> float:
    """
    Normalize angle to [-pi, pi]
    """
    return (theta + math.pi) % (2 * math.pi) - math.pi


def yaw_from_rotation(R: np.ndarray) -> float:
    """
    Extract yaw from rotation matrix
    """
    return math.atan2(R[1, 0], R[0, 0])
