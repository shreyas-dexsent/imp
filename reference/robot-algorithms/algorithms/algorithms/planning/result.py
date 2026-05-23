# Copyright (c) 2026 DexSent Robotics Pvt Ltd.
# All Rights Reserved.
#
# SPDX-License-Identifier: LicenseRef-Proprietary
# See LICENSE file for details.
#
# Author: Shreyas <shreyas.kumar@dexsentrobotics.com>
# Proprietary and confidential.

"""Result + diagnostic types for path planning and validation."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Tuple

import numpy as np

from algorithms.planning.path import Path, PathStatus


@dataclass(frozen=True)
class PathDiagnostics:
    """Debug payload returned alongside every PathPlanResult.

    Carries information about the solve attempt the planner just made,
    plus the per-stage messages an integrator needs to debug failures.
    Successful plans also populate this so monitoring / regression code
    can record planner behaviour over time.
    """

    message: str = ""
    seed_reports: Tuple[Dict[str, Any], ...] = ()
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PathPlanResult:
    """Final result of a path-planning call.

    `path` is populated only when `status == PathStatus.SUCCESS`. The
    successful path satisfies the planner's discrete-state validity
    contract; the post-plan validator (`validate_path`) runs additional
    advisory checks.
    """

    status: PathStatus
    path: Path | None
    planner_used: str
    iterations: int
    elapsed_ms: float
    diagnostics: PathDiagnostics

    @property
    def success(self) -> bool:
        return self.status is PathStatus.SUCCESS


@dataclass(frozen=True)
class CheckResult:
    """One row in a validator report."""

    name: str
    ok: bool
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PathValidationReport:
    """Result of `validate_path(...)`.

    `passed` is the AND of every check. `first_failure` is `(waypoint_index, reason)`
    if a check failed, else `None`. `checks` carries the per-check pass / fail
    for every check run.
    """

    passed: bool
    checks: Tuple[CheckResult, ...]
    first_failure: Tuple[int, str] | None = None
