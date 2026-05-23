from __future__ import annotations

from pydantic import BaseModel


class ToleranceConfig(BaseModel):
    numerical: float = 1e-9
    transform: float = 1e-8
    rotation_orthonormal: float = 1e-7
    rotation_determinant: float = 1e-7
    ik_position: float = 1e-4
    ik_orientation: float = 1e-3
    collision_clearance: float = 1e-4
    trajectory: float = 1e-8


DEFAULT_TOLERANCES = ToleranceConfig()
