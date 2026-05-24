# Copyright (c) 2026 DexSent Robotics Pvt Ltd.
# All Rights Reserved.
#
# SPDX-License-Identifier: LicenseRef-Proprietary
# See LICENSE file for details.
#
# Author: Shreyas <shreyas.kumar@dexsentrobotics.com>
# Proprietary and confidential.

"""Path planner backend protocol + raw result type."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Protocol, Tuple

import numpy as np

from algorithms.planning.options import PlanOptions
from algorithms.planning.path import PathStatus


@dataclass(frozen=True)
class RawPlanResult:
    """Backend-side result. The user-facing `PathPlanResult` wraps this
    with elapsed-time metadata and a `Path` object."""

    status: PathStatus
    waypoints: np.ndarray | None    # (N, dof) or None on failure
    iterations: int = 0
    message: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)


class PathPlannerBackend(Protocol):
    """Protocol implemented by every path planner backend.

    The validity function is a single closure: `q -> bool`. Backends
    must not assume anything about its internals; in single-robot use
    it's a `make_state_validity_fn` closure, in multi-robot composite
    use it's adapted by the caller to accept a flat composite q.
    """

    name: str

    def plan(
        self,
        q_start: np.ndarray,
        q_goal: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
        state_validity_fn: Callable[[np.ndarray], bool],
        options: PlanOptions,
    ) -> RawPlanResult:
        ...
