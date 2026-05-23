from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class JointLimitConstraint:
    lower: np.ndarray
    upper: np.ndarray

    def satisfied(self, q) -> bool:
        q = np.asarray(q, dtype=float)
        return bool(np.all(q >= self.lower) and np.all(q <= self.upper))


@dataclass(frozen=True)
class PositionToleranceConstraint:
    tolerance: float


@dataclass(frozen=True)
class OrientationToleranceConstraint:
    tolerance: float


@dataclass(frozen=True)
class PoseToleranceConstraint:
    position_tolerance: float
    orientation_tolerance: float


@dataclass(frozen=True)
class SingularityConstraint:
    condition_number_threshold: float


@dataclass(frozen=True)
class CollisionFreeConstraint:
    checker: Callable | None = None


@dataclass(frozen=True)
class PreferredPostureConstraint:
    q_nominal: np.ndarray
    weight: float = 1.0


@dataclass(frozen=True)
class MinimumJointMotionConstraint:
    q_current: np.ndarray
    weight: float = 1.0

