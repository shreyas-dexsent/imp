# Copyright (c) 2026 DexSent Robotics Pvt Ltd.
# All Rights Reserved.
#
# SPDX-License-Identifier: LicenseRef-Proprietary
# See LICENSE file for details.
#
# Author: Shreyas <shreyas.kumar@dexsentrobotics.com>
# Proprietary and confidential.

"""Time-parameterization backend protocol + raw result type."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Protocol

import numpy as np

from algorithms.planning.path import Path
from algorithms.trajectory.options import TimeParameterizationOptions
from algorithms.trajectory.result import TrajectoryStatus


@dataclass(frozen=True)
class RawTrajectoryResult:
    """Backend-side result. `time_parameterize` wraps this into a
    `TrajectoryResult` with timing metadata."""

    status: TrajectoryStatus
    times: np.ndarray | None
    positions: np.ndarray | None
    velocities: np.ndarray | None
    accelerations: np.ndarray | None
    message: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)


class TimeParameterizationBackend(Protocol):
    """Every time-parameterization backend implements this signature.

    All backends are responsible for producing **pass-through** motion
    by default: continuous velocity across interior waypoints, no rest
    pauses, unless `options.rest_to_rest` is True. Limits are taken from
    the supplied arrays directly (no model access inside the backend).
    """

    name: str

    def parameterize(
        self,
        path: Path,
        v_limits: np.ndarray,
        a_limits: np.ndarray,
        j_limits: np.ndarray | None,
        options: TimeParameterizationOptions,
    ) -> RawTrajectoryResult:
        ...
