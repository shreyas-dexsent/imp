from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from robot_engine.interfaces.schemas import Transform3D


FrameType = Literal["base", "flange", "tcp", "gripper", "object", "grasp", "bin", "camera", "fixture", "custom"]


@dataclass(frozen=True)
class FrameModel:
    frame_id: str
    parent_frame_id: str
    transform: Transform3D
    frame_type: FrameType = "custom"

