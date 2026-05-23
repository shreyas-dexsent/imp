from __future__ import annotations

from abc import ABC, abstractmethod

from robot_engine.motion.motion_result import JointTrajectory


class MotionExecutorInterface(ABC):
    """Boundary for sending generated trajectories to a real robot backend."""

    @abstractmethod
    def execute_joint_trajectory(self, trajectory: JointTrajectory):
        raise NotImplementedError
