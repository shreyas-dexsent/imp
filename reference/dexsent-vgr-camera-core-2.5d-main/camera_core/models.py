"""Implementation for `camera_core.models`."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class FrameReadyEvent(BaseModel):
    event: str = "FRAME_READY"
    camera_id: str
    sequence_id: int
    timestamp_ns: int
    calib_version: int
    status_flags: int
    rgb_shm: str
    rgb_shape: List[int]
    rgb_dtype: str
    depth_shm: Optional[str] = None
    depth_shape: Optional[List[int]] = None
    depth_dtype: Optional[str] = None
    note: Optional[str] = None
    extra: Dict[str, Any] = {}
