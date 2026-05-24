# Copyright (c) 2026 DexSent Robotics Pvt Ltd.
# All Rights Reserved.
#
# SPDX-License-Identifier: LicenseRef-Proprietary
# See LICENSE file for details.
#
# Author: Shreyas <shreyas.kumar@dexsentrobotics.com>
# Proprietary and confidential.

"""Trajectory data type.

A Trajectory is a function over time: for every t in [0, duration]
it returns the kinematic state (q(t), qd(t), qdd(t)). It is the only
artifact the controller layer can execute.

The trajectory is stored as a dense-sample representation (one sample
per `dt`, taken at construction time). `at(t)` is linear interpolation
between bracketing samples; `sample(dt_new)` resamples to any tick rate.
At construction `dt` (default 1 ms) the linear interpolation is
indistinguishable from the underlying polynomial / Ruckig curve.

See docs/plan.md §6.0 Glossary for the locked vocabulary.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict

import numpy as np


@dataclass(frozen=True)
class Trajectory:
    """Time-parameterized motion. Function from t to (q, qd, qdd).

    Invariants
    ----------
    1. `at(0)` equals the source path's first waypoint.
    2. `at(duration)` equals the source path's last waypoint.
    3. `q(t)` is continuous. For Ruckig + polynomial backends qd(t) is
       also continuous (C^1). The polynomial backend is C^2 by
       construction; Ruckig respects the jerk limit but allows jerk
       discontinuities at sub-segment boundaries.
    4. The trajectory's geometric path equals the path it was built
       from. Time parameterization re-times, doesn't re-route.

    Attributes
    ----------
    times : np.ndarray, shape (M,)
        Monotone-increasing timestamps, in seconds, starting at 0.
    positions, velocities, accelerations : np.ndarray, shape (M, dof)
        Per-sample kinematic state. Accelerations may be zero on
        backends that don't model them.
    joint_names : tuple[str, ...]
        Active-joint order; matches the source path's joint_names.
    backend_used : str
        Name of the backend that produced this trajectory ("polynomial",
        "ruckig", ...). Used by diagnostics.
    metadata : dict
        Backend / construction notes (dt, scale factors, ...). Free-form.
    """

    times: np.ndarray
    positions: np.ndarray
    velocities: np.ndarray
    accelerations: np.ndarray
    joint_names: tuple[str, ...]
    backend_used: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.times.ndim != 1:
            raise ValueError(f"Trajectory.times must be 1D; got shape {self.times.shape}")
        m = int(self.times.shape[0])
        if m < 2:
            raise ValueError(f"Trajectory needs at least 2 samples; got {m}")
        for name in ("positions", "velocities", "accelerations"):
            arr = getattr(self, name)
            if arr.ndim != 2 or arr.shape[0] != m:
                raise ValueError(
                    f"Trajectory.{name} must have shape ({m}, dof); got {arr.shape}"
                )
        dof = self.positions.shape[1]
        if dof != len(self.joint_names):
            raise ValueError(
                f"Trajectory.joint_names has {len(self.joint_names)} entries but "
                f"positions has dof={dof}"
            )
        if not np.all(np.diff(self.times) >= 0.0):
            raise ValueError("Trajectory.times must be monotone non-decreasing")

    @property
    def duration(self) -> float:
        return float(self.times[-1] - self.times[0])

    @property
    def dof(self) -> int:
        return int(self.positions.shape[1])

    @property
    def num_samples(self) -> int:
        return int(self.times.shape[0])

    def at(self, t: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (q, qd, qdd) at time t. Linear interpolation between
        the bracketing stored samples. Clamps to the endpoints when
        t is outside [0, duration]."""
        t = float(t)
        if t <= self.times[0]:
            return (
                self.positions[0].copy(),
                self.velocities[0].copy(),
                self.accelerations[0].copy(),
            )
        if t >= self.times[-1]:
            return (
                self.positions[-1].copy(),
                self.velocities[-1].copy(),
                self.accelerations[-1].copy(),
            )
        idx = int(np.searchsorted(self.times, t, side="right") - 1)
        idx = max(0, min(idx, self.num_samples - 2))
        dt = self.times[idx + 1] - self.times[idx]
        if dt < 1e-12:
            return (
                self.positions[idx].copy(),
                self.velocities[idx].copy(),
                self.accelerations[idx].copy(),
            )
        alpha = (t - self.times[idx]) / dt
        q = (1.0 - alpha) * self.positions[idx] + alpha * self.positions[idx + 1]
        qd = (1.0 - alpha) * self.velocities[idx] + alpha * self.velocities[idx + 1]
        qdd = (1.0 - alpha) * self.accelerations[idx] + alpha * self.accelerations[idx + 1]
        return q, qd, qdd

    def sample(self, dt: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return (times, q, qd, qdd) sampled at uniform `dt`.

        Useful for streaming controllers that consume at a fixed rate.
        Internally calls `at()` at each tick.
        """
        if dt <= 0:
            raise ValueError("sample dt must be > 0")
        n = max(2, int(np.floor(self.duration / dt)) + 1)
        times_out = np.linspace(0.0, self.duration, n)
        q_out = np.zeros((n, self.dof), dtype=float)
        qd_out = np.zeros((n, self.dof), dtype=float)
        qdd_out = np.zeros((n, self.dof), dtype=float)
        for i, t in enumerate(times_out):
            q_out[i], qd_out[i], qdd_out[i] = self.at(float(t))
        return times_out, q_out, qd_out, qdd_out
