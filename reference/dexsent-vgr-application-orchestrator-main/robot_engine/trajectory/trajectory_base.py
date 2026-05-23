from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np


@dataclass
class JointTrajectoryPoint:
    time: float
    q: list[float]
    q_dot: list[float]
    q_ddot: list[float]
    q_jerk: list[float] | None = None


@dataclass
class JointTrajectory:
    points: List[JointTrajectoryPoint]
    generation_method: str = ""
    success: bool = True
    rejection_reason: str = "OK"

    @property
    def duration(self) -> float:
        return self.points[-1].time if self.points else 0.0

    @property
    def dof(self) -> int:
        return len(self.points[0].q) if self.points else 0

    def evaluate(self, t: float) -> JointTrajectoryPoint:
        if not self.points:
            raise ValueError("empty trajectory")
        if t <= self.points[0].time:
            return self.points[0]
        if t >= self.points[-1].time:
            return self.points[-1]
        times = np.asarray([p.time for p in self.points])
        i = int(np.searchsorted(times, t) - 1)
        a = (t - times[i]) / (times[i + 1] - times[i])
        def blend(field):
            return ((1.0 - a) * np.asarray(getattr(self.points[i], field)) + a * np.asarray(getattr(self.points[i + 1], field))).tolist()
        return JointTrajectoryPoint(t, blend("q"), blend("q_dot"), blend("q_ddot"))

    def sample(self, dt: float) -> list[JointTrajectoryPoint]:
        return [self.evaluate(float(t)) for t in np.arange(0.0, self.duration + 0.5 * dt, dt)]


@dataclass
class TrajectoryResult:
    success: bool
    trajectory: JointTrajectory | None = None
    duration: float = 0.0
    max_velocity_ratio: float = 0.0
    max_acceleration_ratio: float = 0.0
    max_jerk_ratio: float = 0.0
    generation_method: str = ""
    rejection_reason: str = "OK"
    debug_info: dict = field(default_factory=dict)


TrajectoryBase = JointTrajectory

