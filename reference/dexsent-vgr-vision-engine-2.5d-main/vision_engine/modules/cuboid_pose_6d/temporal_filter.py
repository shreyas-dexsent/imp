"""
Temporal filtering for smooth 6D pose tracking.
Combines exponential smoothing for translation with quaternion slerp for rotation.
"""

import time
from typing import Any, Dict, Optional, Tuple

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


class TemporalPoseFilter:
    """
    Smooth 6D pose tracking with:
    - Exponential smoothing for translation
    - SLERP for rotation
    - Gating for outliers
    - LOST handling with prediction
    """

    def __init__(
        self,
        alpha_trans: float = 0.3,
        alpha_rot: float = 0.3,
        max_jump_m: float = 0.05,
        max_rot_deg: float = 15.0,
        lost_grace_s: float = 0.8,
        lost_hold_frames: Optional[int] = None,
    ):
        """
        Args:
            alpha_trans: Smoothing factor for translation (0-1, higher = more responsive)
            alpha_rot: Smoothing factor for rotation (0-1)
            max_jump_m: Maximum allowed translation jump per frame (gating threshold)
            max_rot_deg: Maximum allowed rotation jump per frame in degrees
            lost_grace_s: Grace period to maintain prediction when target is lost
            lost_hold_frames: Alternative: max frames to hold prediction (overrides lost_grace_s if set)
        """
        self.alpha_trans = alpha_trans
        self.alpha_rot = alpha_rot
        self.max_jump_m = max_jump_m
        self.max_rot_deg = max_rot_deg
        self.lost_grace_s = lost_grace_s
        self.lost_hold_frames = lost_hold_frames or 0

        # State
        self.position_filtered = None
        self.rotation_filtered = None
        self.velocity = None
        self.last_timestamp = None
        self.state = "INIT"  # INIT, TARGET, LOST
        self.lost_start_time = None
        self.lost_start_frame = None
        self.frame_count = 0
        self.last_gate_reason = None

    def update(
        self, pose_cam: Optional[Dict[str, Any]], confidence: float, frame_time: float
    ) -> Tuple[Dict[str, Any], str, str]:
        """
        Update filter with new pose measurement.

        Args:
            pose_cam: {"t_m": [x,y,z], "q_xyzw": [x,y,z,w]} or None if lost
            confidence: 0-1 confidence score
            frame_time: Absolute timestamp (seconds)

        Returns:
            (filtered_pose, state, gate_reason)
            - filtered_pose: smoothed pose dict or None
            - state: "TARGET" or "LOST"
            - gate_reason: reason if pose was gated (e.g. "HIGH_JUMP", "LOW_CONFIDENCE")
        """
        self.frame_count += 1
        dt = (
            (frame_time - self.last_timestamp)
            if self.last_timestamp is not None
            else 0.0
        )
        self.last_timestamp = frame_time

        gate_reason = None

        # Handle initialization
        if self.state == "INIT":
            if pose_cam is not None and confidence > 0.3:
                self._initialize(pose_cam)
                self.state = "TARGET"
                return self._get_filtered_pose(), "TARGET", None
            else:
                return None, "LOST", "INIT_NO_POSE"

        # Check if measurement should be gated
        should_gate = False
        if pose_cam is None:
            should_gate = True
            gate_reason = "NO_DETECTION"
        elif confidence < 0.2:
            should_gate = True
            gate_reason = "LOW_CONFIDENCE"
        elif not self._passes_gate(pose_cam):
            should_gate = True
            gate_reason = self.last_gate_reason or "HIGH_JUMP"

        if should_gate:
            # Handle LOST state
            if self.state == "TARGET":
                self.state = "LOST"
                self.lost_start_time = frame_time
                self.lost_start_frame = self.frame_count

            # Check if grace period has expired
            if self._is_grace_expired(frame_time):
                return None, "LOST", f"{gate_reason}:GRACE_EXPIRED"

            # Predict forward while in grace period
            if dt > 0 and self.velocity is not None:
                predicted_pos = self.position_filtered + self.velocity * dt
                return (
                    {
                        "t_m": predicted_pos.tolist(),
                        "q_xyzw": self.rotation_filtered.as_quat().tolist(),
                        "predicted": True,
                    },
                    "LOST",
                    f"{gate_reason}:PREDICTING",
                )

            return self._get_filtered_pose(), "LOST", gate_reason

        # Apply smoothing
        self._apply_smoothing(pose_cam, dt)
        self.state = "TARGET"

        return self._get_filtered_pose(), "TARGET", None

    def _initialize(self, pose_cam: Dict[str, Any]):
        """Initialize filter state with first measurement."""
        self.position_filtered = np.array(pose_cam["t_m"], dtype=np.float32)
        q = np.array(pose_cam["q_xyzw"], dtype=np.float32)
        self.rotation_filtered = Rotation.from_quat(q)
        self.velocity = np.zeros(3, dtype=np.float32)

    def _passes_gate(self, pose_cam: Dict[str, Any]) -> bool:
        """Check if pose passes gating thresholds."""
        pos_new = np.array(pose_cam["t_m"], dtype=np.float32)
        q_new = np.array(pose_cam["q_xyzw"], dtype=np.float32)

        # Translation gate
        delta_pos = np.linalg.norm(pos_new - self.position_filtered)
        if delta_pos > self.max_jump_m:
            self.last_gate_reason = "HIGH_JUMP"
            return False

        # Rotation gate
        rot_new = Rotation.from_quat(q_new)
        delta_rot = self.rotation_filtered.inv() * rot_new
        angle_rad = np.linalg.norm(delta_rot.as_rotvec())
        angle_deg = np.degrees(angle_rad)
        if angle_deg > self.max_rot_deg:
            self.last_gate_reason = "HIGH_ROT_JUMP"
            return False

        return True

    def _apply_smoothing(self, pose_cam: Dict[str, Any], dt: float):
        """Apply temporal smoothing to pose."""
        pos_new = np.array(pose_cam["t_m"], dtype=np.float32)
        q_new = np.array(pose_cam["q_xyzw"], dtype=np.float32)

        # Translation: exponential smoothing
        delta_pos = pos_new - self.position_filtered
        self.position_filtered = self.position_filtered + self.alpha_trans * delta_pos

        # Update velocity estimate
        if dt > 0:
            self.velocity = delta_pos / dt

        # Rotation: SLERP
        rot_new = Rotation.from_quat(q_new)
        times = [0, 1]
        rotations = [self.rotation_filtered, rot_new]
        slerp = Slerp(times, Rotation.concatenate(rotations))
        t_interp = self.alpha_rot
        self.rotation_filtered = slerp([t_interp])[0]

    def _is_grace_expired(self, frame_time: float) -> bool:
        """Check if LOST grace period has expired."""
        if self.state != "LOST":
            return False

        if self.lost_start_time is None:
            return True

        # Check time-based grace
        elapsed_s = frame_time - self.lost_start_time
        if elapsed_s > self.lost_grace_s:
            return True

        # Check frame-based grace (if enabled)
        if self.lost_hold_frames > 0:
            elapsed_frames = self.frame_count - self.lost_start_frame
            if elapsed_frames > self.lost_hold_frames:
                return True

        return False

    def _get_filtered_pose(self) -> Optional[Dict[str, Any]]:
        """Get current filtered pose."""
        if self.position_filtered is None:
            return None

        return {
            "t_m": self.position_filtered.tolist(),
            "q_xyzw": self.rotation_filtered.as_quat().tolist(),
        }

    def reset(self):
        """Reset filter to initial state."""
        self.position_filtered = None
        self.rotation_filtered = None
        self.velocity = None
        self.last_timestamp = None
        self.state = "INIT"
        self.lost_start_time = None
        self.lost_start_frame = None
        self.frame_count = 0
