"""Implementation for `vision_engine.common.time`."""

import time

# -----------------------------
# Time helpers
# -----------------------------


def now_ns() -> int:
    return time.time_ns()


def now_ms() -> float:
    return time.time() * 1000.0


def sleep_ms(ms: float):
    time.sleep(ms / 1000.0)


# -----------------------------
# Rate / FPS helpers
# -----------------------------


class RateLimiter:
    """
    Generic rate limiter using timestamps (ns)
    """

    def __init__(self, fps: float):
        self.period_ns = int(1e9 / fps) if fps > 0 else 0
        self.last_ts = 0

    def allow(self, ts_ns: int) -> bool:
        if self.period_ns <= 0:
            return True

        if self.last_ts == 0:
            self.last_ts = ts_ns
            return True

        if ts_ns - self.last_ts >= self.period_ns:
            self.last_ts = ts_ns
            return True

        return False


class FPSCounter:
    """
    Simple rolling FPS estimator
    """

    def __init__(self, window: int = 30):
        self.window = window
        self.timestamps = []

    def tick(self, ts_ns: int):
        self.timestamps.append(ts_ns)
        if len(self.timestamps) > self.window:
            self.timestamps.pop(0)

    def fps(self) -> float:
        if len(self.timestamps) < 2:
            return 0.0
        dt = self.timestamps[-1] - self.timestamps[0]
        if dt <= 0:
            return 0.0
        return (len(self.timestamps) - 1) * 1e9 / dt
