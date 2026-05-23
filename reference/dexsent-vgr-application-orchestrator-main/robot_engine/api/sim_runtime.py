from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional

import numpy as np
from pydantic import BaseModel, Field

from robot_engine.interfaces.schemas import FKRequest, KinematicChainConfig
from robot_engine.kinematics.fk_solver import compute_fk


PublishFn = Callable[[str, Dict[str, Any]], Awaitable[None]]


class SimRobotConfig(BaseModel):
    robot_id: str
    joint_names: List[str]
    initial_positions: List[float] = Field(default_factory=list)
    chain: Optional[KinematicChainConfig] = None
    publish_rate_hz: float = 30.0


class SimJointState(BaseModel):
    joint_names: List[str] = Field(default_factory=list)
    positions: List[float] = Field(default_factory=list)
    velocities: List[float] = Field(default_factory=list)
    timestamp: float = Field(default_factory=time.time)


class SimTrajectory(BaseModel):
    trajectory_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    joint_names: List[str]
    timestamps: List[float]
    q: List[List[float]]
    q_dot: Optional[List[List[float]]] = None


@dataclass
class SimRuntime:
    mode: str = "IDLE"
    robot_id: Optional[str] = None
    joint_names: List[str] = field(default_factory=list)
    q: List[float] = field(default_factory=list)
    dq: List[float] = field(default_factory=list)
    tcp_pose: Dict[str, Any] = field(default_factory=lambda: {
        "position_m": [0.0, 0.0, 0.0],
        "quat_xyzw": [0.0, 0.0, 0.0, 1.0],
        "frame": "base",
    })
    chain: Optional[KinematicChainConfig] = None
    active_trajectory_id: Optional[str] = None
    publish_rate_hz: float = 30.0
    _task: Optional[asyncio.Task] = None
    _pause: asyncio.Event = field(default_factory=asyncio.Event)
    _stop_requested: bool = False

    def __post_init__(self) -> None:
        self._pause.set()

    def load_robot(self, config: SimRobotConfig) -> Dict[str, Any]:
        self.stop()
        self.robot_id = config.robot_id
        self.joint_names = list(config.joint_names)
        self.q = list(config.initial_positions or [0.0] * len(self.joint_names))
        self.dq = [0.0] * len(self.joint_names)
        self.chain = config.chain
        self.publish_rate_hz = max(float(config.publish_rate_hz), 1.0)
        self.mode = "IDLE"
        self.active_trajectory_id = None
        self._update_tcp_from_fk()
        return self.state()

    def set_joint_state(self, joint_state: SimJointState | Dict[str, Any]) -> Dict[str, Any]:
        data = joint_state if isinstance(joint_state, SimJointState) else SimJointState(**joint_state)
        if data.joint_names:
            self.joint_names = list(data.joint_names)
        self.q = list(data.positions)
        self.dq = list(data.velocities or [0.0] * len(self.q))
        self.mode = "IDLE" if self.mode != "PLAYING" else self.mode
        self._update_tcp_from_fk()
        return self.state()

    async def play(self, trajectory: SimTrajectory, publish: PublishFn) -> Dict[str, Any]:
        validation_error = self._validate_trajectory(trajectory)
        if validation_error:
            self.mode = "ERROR"
            return {"success": False, "error_code": "INVALID_REQUEST", "error_message": validation_error}
        self.stop()
        self._stop_requested = False
        self._pause.set()
        self.active_trajectory_id = trajectory.trajectory_id
        self.mode = "PLAYING"
        self._task = asyncio.create_task(self._play_loop(trajectory, publish))
        return {"success": True, "trajectory_id": trajectory.trajectory_id, "state": self.state()}

    def pause(self) -> Dict[str, Any]:
        if self.mode == "PLAYING":
            self.mode = "PAUSED"
            self._pause.clear()
        return self.state()

    def resume(self) -> Dict[str, Any]:
        if self.mode == "PAUSED":
            self.mode = "PLAYING"
            self._pause.set()
        return self.state()

    def stop(self) -> Dict[str, Any]:
        self._stop_requested = True
        self._pause.set()
        if self._task and not self._task.done():
            self._task.cancel()
        self._task = None
        self.active_trajectory_id = None
        if self.mode in {"PLAYING", "PAUSED"}:
            self.mode = "STOPPED"
        return self.state()

    def state(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "robot_id": self.robot_id,
            "joint_state": {
                "joint_names": self.joint_names,
                "positions": self.q,
                "velocities": self.dq,
                "timestamp": time.time(),
            },
            "q": self.q,
            "dq": self.dq,
            "tcp_pose": self.tcp_pose,
            "active_trajectory_id": self.active_trajectory_id,
            "publish_rate_hz": self.publish_rate_hz,
        }

    async def _play_loop(self, trajectory: SimTrajectory, publish: PublishFn) -> None:
        try:
            await publish("trajectory.started", {"trajectory_id": trajectory.trajectory_id})
            start_monotonic = time.monotonic()
            duration = float(trajectory.timestamps[-1])
            dt = 1.0 / max(self.publish_rate_hz, 1.0)
            while True:
                await self._pause.wait()
                if self._stop_requested:
                    await publish("trajectory.failed", {"trajectory_id": trajectory.trajectory_id, "reason": "STOPPED"})
                    return
                elapsed = min(time.monotonic() - start_monotonic, duration)
                self.q, self.dq = sample_trajectory(trajectory, elapsed)
                self._update_tcp_from_fk()
                payload = {
                    "trajectory_id": trajectory.trajectory_id,
                    "joint_names": self.joint_names,
                    "positions": self.q,
                    "velocities": self.dq,
                    "timestamp": time.time(),
                }
                await publish("robot.joint_state", payload)
                await publish("robot.tcp_pose", {"tcp_pose": self.tcp_pose, "trajectory_id": trajectory.trajectory_id})
                await publish("trajectory.sample", {"trajectory_id": trajectory.trajectory_id, "t": elapsed, "q": self.q})
                if elapsed >= duration:
                    self.mode = "IDLE"
                    self.active_trajectory_id = None
                    await publish("trajectory.finished", {"trajectory_id": trajectory.trajectory_id, "final_q": self.q})
                    return
                await asyncio.sleep(dt)
        except asyncio.CancelledError:
            await publish("trajectory.failed", {"trajectory_id": trajectory.trajectory_id, "reason": "CANCELLED"})
        except Exception as exc:
            self.mode = "ERROR"
            await publish("trajectory.failed", {"trajectory_id": trajectory.trajectory_id, "reason": str(exc)})

    def _validate_trajectory(self, trajectory: SimTrajectory) -> Optional[str]:
        if not trajectory.joint_names:
            return "joint_names are required"
        if len(trajectory.timestamps) != len(trajectory.q):
            return "timestamps and q must have the same length"
        if len(trajectory.q) < 2:
            return "trajectory requires at least two waypoints"
        if any(b < a for a, b in zip(trajectory.timestamps[:-1], trajectory.timestamps[1:])):
            return "timestamps must be monotonically increasing"
        dof = len(trajectory.joint_names)
        if any(len(q) != dof for q in trajectory.q):
            return "every q waypoint must match joint_names length"
        return None

    def _update_tcp_from_fk(self) -> None:
        if not self.chain or not self.joint_names or len(self.q) != len(self.joint_names):
            return
        joint_positions = dict(zip(self.joint_names, self.q))
        result = compute_fk(FKRequest(chain=self.chain, joint_positions=joint_positions, target_frame=self.chain.tcp.tcp_id if self.chain.tcp else None))
        if not result.ok or not result.transforms:
            return
        transform = next(iter(result.transforms.values()))
        matrix = np.asarray(transform.matrix, dtype=float)
        self.tcp_pose = {
            "position_m": matrix[:3, 3].tolist(),
            "quat_xyzw": [0.0, 0.0, 0.0, 1.0],
            "frame": transform.parent_frame,
        }


def sample_trajectory(trajectory: SimTrajectory, t: float) -> tuple[List[float], List[float]]:
    times = np.asarray(trajectory.timestamps, dtype=float)
    q = np.asarray(trajectory.q, dtype=float)
    if t <= times[0]:
        return q[0].tolist(), _velocity_at(trajectory, 0, default_velocity=np.zeros(q.shape[1])).tolist()
    if t >= times[-1]:
        return q[-1].tolist(), _velocity_at(trajectory, len(times) - 1, default_velocity=np.zeros(q.shape[1])).tolist()
    i = int(np.searchsorted(times, t) - 1)
    span = max(times[i + 1] - times[i], 1e-9)
    alpha = (t - times[i]) / span
    pos = ((1.0 - alpha) * q[i] + alpha * q[i + 1])
    vel = (q[i + 1] - q[i]) / span
    return pos.tolist(), vel.tolist()


def _velocity_at(trajectory: SimTrajectory, index: int, default_velocity: np.ndarray) -> np.ndarray:
    if trajectory.q_dot and index < len(trajectory.q_dot):
        return np.asarray(trajectory.q_dot[index], dtype=float)
    return default_velocity
