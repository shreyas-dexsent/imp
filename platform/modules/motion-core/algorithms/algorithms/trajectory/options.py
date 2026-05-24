# Copyright (c) 2026 DexSent Robotics Pvt Ltd.
# All Rights Reserved.
#
# SPDX-License-Identifier: LicenseRef-Proprietary
# See LICENSE file for details.
#
# Author: Shreyas <shreyas.kumar@dexsentrobotics.com>
# Proprietary and confidential.

"""Options for time parameterization and trajectory validation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class TimeParameterizationOptions:
    """Knobs for `time_parameterize`.

    Defaults produce **pass-through** motion (the robot does NOT pause
    at interior waypoints). Set `rest_to_rest=True` for the legacy
    stop-at-each-waypoint behaviour if needed for debugging.
    """

    backend: Literal["auto", "ruckig", "polynomial"] = "auto"
    """Which backend to use. `"auto"` picks Ruckig if installed, else
    polynomial."""

    dt: float = 0.001
    """Output sample period in seconds. Smaller = denser, larger memory,
    smoother linear-interp evaluation. 1 ms matches a 1 kHz controller
    tick."""

    v_scale: float = 1.0
    """Multiplier applied to `model.active_velocity_limits()`. Use < 1
    for conservative motion."""

    a_scale: float = 1.0
    """Multiplier applied to `model.active_acceleration_limits()`."""

    j_scale: float = 1.0
    """Multiplier applied to `model.active_jerk_limits()`. Only Ruckig
    uses jerk."""

    start_velocity: float | None = None
    end_velocity: float | None = None
    start_acceleration: float | None = None
    end_acceleration: float | None = None
    """Optional non-zero boundary conditions, in joint-space norm
    units. None = at rest. Set when chaining trajectory segments in
    Phase 7 primitives."""

    rest_to_rest: bool = False
    """If True, FORCE the robot to come to rest at every interior
    waypoint. Default False — interior waypoints are pass-through with
    Catmull-Rom velocities."""

    interior_velocity_scale: float = 1.0
    """Scales the Catmull-Rom interior-waypoint velocity. 1.0 uses the
    full chord-length finite difference; smaller values are more
    conservative (closer to rest-to-rest)."""


@dataclass(frozen=True)
class TrajectoryValidationOptions:
    """Knobs for `validate_trajectory`."""

    validation_dt: float = 0.01
    """Sampling step the validator walks the trajectory at. Smaller =
    more sample points, slower. Should be <= the controller tick."""

    joint_margin: float = 1e-3
    """Joint-limit margin."""

    v_scale: float = 1.0
    """Multiplier on the model's velocity limits when checking."""

    a_scale: float = 1.0
    """Multiplier on the model's acceleration limits when checking."""

    j_scale: float = 1.0
    """Multiplier on the model's jerk limits when checking."""

    check_collision: bool = True
    """Run dense-time collision check at every validation sample."""

    tcp_v_max: float | None = None
    """If set, validator computes TCP linear speed via FK + finite
    differences and flags points where it exceeds this value (m/s)."""

    tcp_omega_max: float | None = None
    """Same for TCP angular speed (rad/s)."""

    tcp_frame_id: str | None = None
    """Frame at which to evaluate TCP speed. Required if either
    `tcp_v_max` or `tcp_omega_max` is set."""

    controller_dt: float | None = None
    """If set, validator confirms the trajectory's stored dt is fine
    enough to be streamed at this controller tick (output dt <=
    controller_dt)."""

    numerical_slack: float = 1e-6
    """Tolerance added to limit checks to absorb floating-point noise.
    Same role as IK's joint_margin in the validator."""
