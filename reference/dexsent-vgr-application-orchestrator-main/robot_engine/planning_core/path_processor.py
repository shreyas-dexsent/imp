from __future__ import annotations

from typing import Callable, List

import numpy as np


class PathProcessor:
    """
    Post-processes a raw joint-space path:
      1. Shortcut / simplify
      2. Interpolate to a target number of waypoints
      3. Validate every waypoint against is_state_valid
    """

    def __init__(
        self,
        is_state_valid: Callable[[np.ndarray], bool],
        interpolation_waypoints: int = 100,
        shortcut_iterations: int = 100,
    ) -> None:
        self._valid = is_state_valid
        self.interpolation_waypoints = interpolation_waypoints
        self.shortcut_iterations = shortcut_iterations

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(self, waypoints: List[np.ndarray]) -> List[np.ndarray]:
        """Shortcut → interpolate → validate.  Returns processed waypoints."""
        if len(waypoints) < 2:
            return waypoints
        path = self.shortcut(waypoints)
        path = self.interpolate(path, self.interpolation_waypoints)
        return path

    # ------------------------------------------------------------------
    # Shortcut
    # ------------------------------------------------------------------

    def shortcut(self, waypoints: List[np.ndarray]) -> List[np.ndarray]:
        """Remove redundant intermediate waypoints using random shortcutting."""
        if len(waypoints) <= 2:
            return list(waypoints)

        path = [np.asarray(w, dtype=float) for w in waypoints]
        rng = np.random.default_rng(42)

        for _ in range(self.shortcut_iterations):
            if len(path) <= 2:
                break
            i = int(rng.integers(0, len(path) - 2))
            j = int(rng.integers(i + 2, min(i + 10, len(path))))
            if j >= len(path):
                continue
            # Try to connect path[i] directly to path[j]
            if self._segment_valid(path[i], path[j]):
                path = path[: i + 1] + path[j:]

        return path

    def _segment_valid(self, qa: np.ndarray, qb: np.ndarray, steps: int = 10) -> bool:
        for t in np.linspace(0, 1, steps + 1):
            q = qa + t * (qb - qa)
            if not self._valid(q):
                return False
        return True

    # ------------------------------------------------------------------
    # Interpolation
    # ------------------------------------------------------------------

    @staticmethod
    def interpolate(
        waypoints: List[np.ndarray],
        target_count: int,
    ) -> List[np.ndarray]:
        """Linearly interpolate a path to exactly target_count waypoints."""
        if len(waypoints) < 2 or target_count <= len(waypoints):
            return list(waypoints)

        # Build cumulative arc-length parameterisation
        segments = [np.linalg.norm(b - a) for a, b in zip(waypoints[:-1], waypoints[1:])]
        total = sum(segments)
        if total < 1e-12:
            return [waypoints[0].copy() for _ in range(target_count)]

        cumulative = [0.0]
        for s in segments:
            cumulative.append(cumulative[-1] + s)

        # Sample uniformly along arc length
        ts = np.linspace(0.0, total, target_count)
        result = []
        seg_idx = 0
        for t in ts:
            while seg_idx < len(segments) - 1 and cumulative[seg_idx + 1] < t:
                seg_idx += 1
            t0 = cumulative[seg_idx]
            t1 = cumulative[seg_idx + 1]
            frac = (t - t0) / max(t1 - t0, 1e-12)
            frac = float(np.clip(frac, 0.0, 1.0))
            result.append(waypoints[seg_idx] + frac * (waypoints[seg_idx + 1] - waypoints[seg_idx]))
        return result

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self, waypoints: List[np.ndarray]) -> bool:
        """Return True if every waypoint passes the validity check."""
        return all(self._valid(q) for q in waypoints)
