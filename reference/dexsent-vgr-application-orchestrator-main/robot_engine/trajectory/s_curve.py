from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from robot_engine.interfaces.error_codes import ErrorCode
from robot_engine.interfaces.result_types import APIResult
from robot_engine.trajectory.trajectory_base import JointTrajectory, JointTrajectoryPoint


_SMOOTHSTEP_V_MAX = 1.875
_SMOOTHSTEP_A_MAX = 5.773502691896258
_SMOOTHSTEP_J_MAX = 60.0


@dataclass
class SCurveProfile:
    q0: float
    q1: float
    v_max: float
    a_max: float
    j_max: float
    duration: float
    phase_durations: dict

    def evaluate(self, t: float):
        if self.duration <= 0.0:
            return self.q1, 0.0, 0.0, 0.0
        u = min(1.0, max(0.0, float(t) / self.duration))
        D = self.q1 - self.q0
        s = 10*u**3 - 15*u**4 + 6*u**5
        ds = 30*u**2 - 60*u**3 + 30*u**4
        dds = 60*u - 180*u**2 + 120*u**3
        ddds = 60 - 360*u + 360*u**2
        return (
            self.q0 + D * s,
            D * ds / self.duration,
            D * dds / self.duration**2,
            D * ddds / self.duration**3,
        )

    def sample(self, count: int = 101):
        return [self.evaluate(t) for t in np.linspace(0.0, self.duration, max(2, count))]

    def plan(self, *args, **kwargs):
        return self


def s_curve_profile_1d(q0, q1, v_max, a_max, j_max, v0=0.0, v1=0.0, a0=0.0, a1=0.0):
    if any(abs(float(x)) > 1e-12 for x in (v0, v1, a0, a1)):
        return APIResult.fail(ErrorCode.NOT_IMPLEMENTED, "S-curve currently supports rest-to-rest zero boundary velocity/acceleration only.")
    v_max = abs(float(v_max))
    a_max = abs(float(a_max))
    j_max = abs(float(j_max))
    if v_max <= 0 or a_max <= 0 or j_max <= 0:
        raise ValueError("velocity, acceleration, and jerk limits must be positive")
    D = abs(float(q1) - float(q0))
    if D <= 1e-15:
        duration = 0.0
    else:
        duration = max(
            _SMOOTHSTEP_V_MAX * D / v_max,
            np.sqrt(_SMOOTHSTEP_A_MAX * D / a_max),
            (_SMOOTHSTEP_J_MAX * D / j_max) ** (1.0 / 3.0),
        )
    return SCurveProfile(float(q0), float(q1), v_max, a_max, j_max, float(duration), {"smoothstep": float(duration)})


def synchronized_multi_joint_s_curve(q0, q1, v_limits, a_limits, j_limits, samples: int = 101):
    profiles = [s_curve_profile_1d(a, b, v, acc, jerk) for a, b, v, acc, jerk in zip(q0, q1, v_limits, a_limits, j_limits)]
    unsupported = [p for p in profiles if isinstance(p, APIResult)]
    if unsupported:
        return unsupported[0]
    duration = max((p.duration for p in profiles), default=0.0)
    for p in profiles:
        p.duration = duration
        p.phase_durations = {"smoothstep": duration}
    points = []
    for t in np.linspace(0.0, duration, max(2, samples)):
        values = [p.evaluate(t) for p in profiles]
        q, qd, qdd, qj = zip(*values)
        points.append(JointTrajectoryPoint(float(t), list(map(float, q)), list(map(float, qd)), list(map(float, qdd)), list(map(float, qj))))
    return JointTrajectory(points, generation_method="s_curve", success=True)


def validate_s_curve_limits(profile: SCurveProfile, samples: int = 1001):
    data = np.asarray(profile.sample(samples), dtype=float)
    return {
        "max_velocity": float(np.max(np.abs(data[:, 1]))) if data.size else 0.0,
        "max_acceleration": float(np.max(np.abs(data[:, 2]))) if data.size else 0.0,
        "max_jerk": float(np.max(np.abs(data[:, 3]))) if data.size else 0.0,
        "velocity_ok": bool(np.max(np.abs(data[:, 1])) <= profile.v_max + 1e-9) if data.size else True,
        "acceleration_ok": bool(np.max(np.abs(data[:, 2])) <= profile.a_max + 1e-9) if data.size else True,
        "jerk_ok": bool(np.max(np.abs(data[:, 3])) <= profile.j_max + 1e-9) if data.size else True,
    }

