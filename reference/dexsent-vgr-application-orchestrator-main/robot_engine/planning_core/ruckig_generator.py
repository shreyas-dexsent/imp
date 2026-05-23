from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class TimedTrajectory:
    """Timed joint trajectory output from RuckigTrajectoryGenerator."""

    # Each element: (time_s, q, dq, ddq)  – all (n,) arrays
    times: List[float] = field(default_factory=list)
    positions: List[np.ndarray] = field(default_factory=list)
    velocities: List[np.ndarray] = field(default_factory=list)
    accelerations: List[np.ndarray] = field(default_factory=list)
    duration: float = 0.0
    generator_used: str = ""
    planner_used: str = ""
    path_waypoints: List[np.ndarray] = field(default_factory=list)
    # Sparse shortcutted keyframes before interpolation — use these for franky execution
    sparse_waypoints: List[np.ndarray] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.times)

    def is_empty(self) -> bool:
        return len(self.times) == 0

    def at(self, t: float):
        """Linear interpolation of q at time t."""
        if not self.times:
            raise ValueError("Empty trajectory.")
        if t <= self.times[0]:
            return self.positions[0].copy()
        if t >= self.times[-1]:
            return self.positions[-1].copy()
        # Find segment
        idx = np.searchsorted(self.times, t) - 1
        idx = int(np.clip(idx, 0, len(self.times) - 2))
        dt = self.times[idx + 1] - self.times[idx]
        if dt < 1e-12:
            return self.positions[idx].copy()
        alpha = (t - self.times[idx]) / dt
        return (1 - alpha) * self.positions[idx] + alpha * self.positions[idx + 1]


class RuckigTrajectoryGenerator:
    """
    Convert a joint-space waypoint path to a timed trajectory.

    Uses Ruckig for jerk-limited trajectory generation. The planning-core stack
    uses Ruckig directly so the production backend is explicit and auditable.

    Parameters
    ----------
    velocity_limits : (n,) array
    acceleration_limits : (n,) array
    jerk_limits : (n,) array   (used only by Ruckig)
    dt : float
        Sampling period for the output trajectory.
    """

    def __init__(
        self,
        velocity_limits: np.ndarray,
        acceleration_limits: np.ndarray,
        jerk_limits: Optional[np.ndarray] = None,
        dt: float = 0.01,
    ) -> None:
        self._vel = np.asarray(velocity_limits, dtype=float)
        self._acc = np.asarray(acceleration_limits, dtype=float)
        self._jrk = (
            np.asarray(jerk_limits, dtype=float)
            if jerk_limits is not None
            else 10.0 * self._acc
        )
        self.dt = dt

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self, waypoints: List[np.ndarray]) -> Optional[TimedTrajectory]:
        """
        Generate a timed trajectory through all waypoints.

        Returns None if generation fails.
        """
        if len(waypoints) < 2:
            return None
        return self._generate_ruckig(waypoints)

    def generate_smooth(self, waypoints: List[np.ndarray]) -> Optional[TimedTrajectory]:
        """
        Fit a single continuous quintic spline through all waypoints.

        Uses chord-length parameterization for timing and Catmull-Rom
        finite differences for interior-waypoint velocities, so the robot
        never decelerates to zero at intermediate poses.  Velocities and
        accelerations are clipped to the configured limits.

        Returns None if generation fails.
        """
        if len(waypoints) < 2:
            return None
        qs = [np.asarray(q, dtype=float) for q in waypoints]
        if len(qs) == 2:
            return self._generate_ruckig(qs)
        try:
            return self._generate_smooth_spline(qs)
        except Exception:
            return self._generate_ruckig(qs)

    # ------------------------------------------------------------------
    # Ruckig (segment-by-segment)
    # ------------------------------------------------------------------

    def _generate_ruckig(self, waypoints: List[np.ndarray]) -> TimedTrajectory:
        from ruckig import InputParameter, OutputParameter, Result, Ruckig  # type: ignore

        n = len(waypoints[0])
        otg = Ruckig(n, self.dt)
        inp = InputParameter(n)
        out = OutputParameter(n)

        inp.max_velocity = self._vel.tolist()
        inp.max_acceleration = self._acc.tolist()
        inp.max_jerk = self._jrk.tolist()

        times: List[float] = []
        positions: List[np.ndarray] = []
        velocities: List[np.ndarray] = []
        accelerations: List[np.ndarray] = []

        t_global = 0.0

        for seg_idx in range(len(waypoints) - 1):
            qa = np.asarray(waypoints[seg_idx], dtype=float)
            qb = np.asarray(waypoints[seg_idx + 1], dtype=float)

            inp.current_position = qa.tolist()
            inp.current_velocity = [0.0] * n
            inp.current_acceleration = [0.0] * n
            inp.target_position = qb.tolist()
            inp.target_velocity = [0.0] * n
            inp.target_acceleration = [0.0] * n

            while True:
                res = otg.update(inp, out)
                times.append(t_global)
                positions.append(np.asarray(out.new_position, dtype=float))
                velocities.append(np.asarray(out.new_velocity, dtype=float))
                accelerations.append(np.asarray(out.new_acceleration, dtype=float))
                t_global += self.dt
                if res == Result.Finished:
                    break
                if res == Result.Error:
                    raise RuntimeError(f"Ruckig error on segment {seg_idx}")
                inp.current_position = out.new_position
                inp.current_velocity = out.new_velocity
                inp.current_acceleration = out.new_acceleration

        return TimedTrajectory(
            times=times,
            positions=positions,
            velocities=velocities,
            accelerations=accelerations,
            duration=t_global,
            generator_used="Ruckig",
        )

    def _generate_smooth_spline(self, qs: List[np.ndarray]) -> TimedTrajectory:
        """
        Quintic spline through all waypoints with continuous velocity.

        Algorithm:
          1. Chord-length parameterization → segment durations scaled by vel limit
          2. Catmull-Rom finite differences for interior velocities (zero-accel BC)
          3. Quintic polynomial per segment sampled at self.dt
        """
        from robot_engine.trajectory.polynomial import (
            solve_quintic_coefficients,
            evaluate_polynomial,
            evaluate_polynomial_derivative,
        )

        n = len(qs[0])
        # --- 1. Segment durations via chord-length + velocity limit ---
        seg_durations = _chord_length_durations(qs, self._vel)

        # --- 2. Catmull-Rom velocities at each waypoint ---
        vels = _catmull_rom_velocities(qs, seg_durations)

        # --- 3. Build quintic per segment and sample ---
        times: List[float] = []
        positions: List[np.ndarray] = []
        velocities: List[np.ndarray] = []
        accelerations: List[np.ndarray] = []

        zeros = np.zeros(n, dtype=float)
        t_global = 0.0
        for i in range(len(qs) - 1):
            T = seg_durations[i]
            q0, q1 = qs[i], qs[i + 1]
            v0, v1 = vels[i], vels[i + 1]
            coeffs = solve_quintic_coefficients(q0, q1, v0, v1, zeros, zeros, T)
            n_steps = max(2, int(round(T / self.dt)))
            ts = np.linspace(0.0, T, n_steps, endpoint=(i == len(qs) - 2))
            for t_local in ts:
                times.append(t_global + float(t_local))
                positions.append(np.clip(evaluate_polynomial(coeffs, t_local), None, None))
                velocities.append(
                    np.clip(
                        evaluate_polynomial_derivative(coeffs, t_local, 1),
                        -self._vel, self._vel,
                    )
                )
                accelerations.append(
                    np.clip(
                        evaluate_polynomial_derivative(coeffs, t_local, 2),
                        -self._acc, self._acc,
                    )
                )
            t_global += T

        return TimedTrajectory(
            times=times,
            positions=positions,
            velocities=velocities,
            accelerations=accelerations,
            duration=t_global,
            generator_used="QuinticSpline",
        )


# ---------------------------------------------------------------------------
# Helpers for smooth spline generation
# ---------------------------------------------------------------------------

def _chord_length_durations(
    qs: List[np.ndarray],
    vel_limits: np.ndarray,
    min_duration: float = 0.05,
) -> List[float]:
    """Assign segment durations proportional to max-joint-range / velocity_limit."""
    durations = []
    for i in range(len(qs) - 1):
        delta = np.abs(qs[i + 1] - qs[i])
        # Time needed per joint: delta / vel_limit; take the max (slowest joint drives)
        t_needed = float(np.max(delta / np.maximum(vel_limits, 1e-6)))
        durations.append(max(t_needed, min_duration))
    return durations


def _catmull_rom_velocities(
    qs: List[np.ndarray],
    durations: List[float],
) -> List[np.ndarray]:
    """
    Estimate velocity at each waypoint using Catmull-Rom finite differences.

    v_i = (q_{i+1} - q_{i-1}) / (t_{i+1} - t_{i-1})

    Endpoints are set to zero (start and finish at rest).
    """
    n_pts = len(qs)
    vels = [np.zeros_like(qs[0]) for _ in range(n_pts)]
    # cumulative times
    cum = [0.0]
    for d in durations:
        cum.append(cum[-1] + d)
    for i in range(1, n_pts - 1):
        dt = cum[i + 1] - cum[i - 1]
        if dt > 1e-9:
            vels[i] = (qs[i + 1] - qs[i - 1]) / dt
    return vels
