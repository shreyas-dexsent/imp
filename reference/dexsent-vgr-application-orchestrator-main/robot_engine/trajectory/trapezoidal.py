from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class TrapezoidalProfile:
    q0: float
    q1: float
    v_max: float
    a_max: float
    t_acc: float
    t_cruise: float
    t_dec: float
    duration: float
    triangular: bool = False

    def evaluate(self, t: float):
        sign = 1.0 if self.q1 >= self.q0 else -1.0
        t = min(max(float(t), 0.0), self.duration)
        if t <= self.t_acc:
            q = self.q0 + sign * 0.5 * self.a_max * t**2
            v = sign * self.a_max * t
            a = sign * self.a_max
        elif t <= self.t_acc + self.t_cruise:
            dt = t - self.t_acc
            q_acc = 0.5 * self.a_max * self.t_acc**2
            q = self.q0 + sign * (q_acc + self.v_max * dt)
            v = sign * self.v_max
            a = 0.0
        else:
            dt = t - self.t_acc - self.t_cruise
            q_dec_start = abs(self.q1 - self.q0) - 0.5 * self.a_max * self.t_dec**2
            q = self.q0 + sign * (q_dec_start + self.v_max * dt - 0.5 * self.a_max * dt**2)
            v = sign * (self.v_max - self.a_max * dt)
            a = -sign * self.a_max
        return q, v, a


def trapezoidal_profile_1d(q0, q1, v_max, a_max):
    distance = abs(float(q1) - float(q0))
    v_max = abs(float(v_max))
    a_max = abs(float(a_max))
    if v_max <= 0 or a_max <= 0:
        raise ValueError("velocity and acceleration limits must be positive")
    t_acc = v_max / a_max
    d_acc = 0.5 * a_max * t_acc**2
    if 2.0 * d_acc >= distance:
        return triangular_profile_1d(q0, q1, v_max, a_max)
    t_cruise = (distance - 2.0 * d_acc) / v_max
    return TrapezoidalProfile(float(q0), float(q1), v_max, a_max, t_acc, t_cruise, t_acc, 2*t_acc + t_cruise)


def triangular_profile_1d(q0, q1, v_max, a_max):
    distance = abs(float(q1) - float(q0))
    t_acc = np.sqrt(distance / abs(a_max)) if distance > 0 else 0.0
    peak_v = abs(a_max) * t_acc
    return TrapezoidalProfile(float(q0), float(q1), float(peak_v), abs(float(a_max)), float(t_acc), 0.0, float(t_acc), float(2*t_acc), triangular=True)


def synchronized_multi_joint_trapezoidal(q0, q1, v_limits, a_limits):
    profiles = [trapezoidal_profile_1d(a, b, v, acc) for a, b, v, acc in zip(q0, q1, v_limits, a_limits)]
    duration = max((p.duration for p in profiles), default=0.0)
    for p in profiles:
        p.duration = duration
    return profiles

