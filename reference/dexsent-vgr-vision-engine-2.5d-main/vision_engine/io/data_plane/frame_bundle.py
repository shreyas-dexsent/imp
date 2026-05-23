"""Implementation for `vision_engine.io.data_plane.frame_bundle`."""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np


@dataclass
class FrameBundle:
    frame_id: str
    camera_id: str
    sequence_id: int
    timestamp_ns: int
    rgb: np.ndarray
    depth: Optional[np.ndarray] = None
    meta: Dict[str, Any] = field(default_factory=dict)
