"""MuJoCo UR5e simulation robot, exposed as an imp HAL device.

Publishes ``RobotState`` and subscribes ``MotionCommand`` (joint target or
trajectory). Kinematic sim: motion smoothly interpolates joint positions and
recomputes forward kinematics — matching the VGR reference adapter, minus the
transport and the IK (IK is a motion module, not the HAL's job; spec §8/§9).
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import List, Optional

import mujoco
import numpy as np

from imp_sdk import HalDevice, Pub, Sub, QosClass
from imp_sdk.schemas import imp_pb2

_JOINTS = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]
_TCP_CANDIDATES = ["tool0", "tcp", "ee_link", "wrist_3_link", "wrist_3", "flange"]
_HOME_Q = [0.0, -1.5708, 1.5708, -1.5708, -1.5708, 0.0]


def default_model_path() -> str:
    return str(Path(__file__).resolve().parent.parent / "assets" / "ur5e_mjcf" / "scene.xml")


class MujocoUR5e(HalDevice):
    kind = "robot"

    def __init__(
        self,
        model_path: Optional[str] = None,
        home_q: Optional[List[float]] = None,
        state_hz: float = 125.0,
        joint_speed_rad_s: float = 1.5,
        step_dt: float = 0.01,
    ):
        self.model_path = Path(model_path or default_model_path()).resolve()
        self.home_q = list(home_q) if home_q else list(_HOME_Q)
        self.state_hz = state_hz
        self.joint_speed_rad_s = joint_speed_rad_s
        self.step_dt = step_dt

        self._lock = threading.RLock()
        self._mode = "disconnected"
        self._active_motion_id = ""
        self._seq = 0
        self._motion_thread: Optional[threading.Thread] = None
        self._motion_stop = threading.Event()
        self.model = None
        self.data = None

    # ---- lifecycle -------------------------------------------------------

    def configure(self) -> None:
        if not self.model_path.exists():
            raise FileNotFoundError(self.model_path)
        self.model = mujoco.MjModel.from_xml_path(str(self.model_path))
        self.data = mujoco.MjData(self.model)
        self._qpos_adr = [int(self.model.jnt_qposadr[self._jid(n)]) for n in _JOINTS]
        self._dof_adr = [int(self.model.jnt_dofadr[self._jid(n)]) for n in _JOINTS]
        self._tcp_body = self._resolve_tcp_body()
        with self._lock:
            for adr, q in zip(self._qpos_adr, self.home_q):
                self.data.qpos[adr] = q
            mujoco.mj_forward(self.model, self.data)

    def activate(self) -> None:
        self._mode = "idle"

    def deactivate(self) -> None:
        self._motion_stop.set()
        self._mode = "disconnected"

    # ---- interfaces ------------------------------------------------------

    def publishes(self) -> List[Pub]:
        return [Pub("state", imp_pb2.RobotState, QosClass.STATE, rate_hz=self.state_hz)]

    def subscribes(self) -> List[Sub]:
        return [Sub("command", imp_pb2.MotionCommand, QosClass.COMMAND)]

    # ---- io --------------------------------------------------------------

    def read(self, signal: str):
        if signal != "state":
            return None
        with self._lock:
            q = [float(self.data.qpos[a]) for a in self._qpos_adr]
            dq = [float(self.data.qvel[a]) for a in self._dof_adr]
            tcp = self._tcp_pose()
        self._seq += 1
        return imp_pb2.RobotState(
            header=imp_pb2.Header(seq=self._seq, stamp_ns=time.time_ns(), schema="imp.RobotState/1"),
            q=q,
            dq=dq,
            tcp_pose=tcp,
            mode=self._mode,
            active_motion_id=self._active_motion_id,
        )

    def on_command(self, signal: str, msg) -> None:
        if signal != "command":
            return
        if msg.kind == "joint":
            targets = [list(msg.q_target)]
        elif msg.kind == "trajectory":
            targets = self._trajectory_waypoints(msg.trajectory)
        else:
            return  # tcp/IK targets are resolved by a motion module, not the HAL
        if not targets or len(targets[0]) != len(_JOINTS):
            return
        self._start_motion(targets, msg.motion_id or "cmd")

    # ---- motion ----------------------------------------------------------

    def _start_motion(self, targets: List[List[float]], motion_id: str) -> None:
        # Preempt any in-flight motion.
        self._motion_stop.set()
        if self._motion_thread and self._motion_thread.is_alive():
            self._motion_thread.join(timeout=1.0)
        self._motion_stop = threading.Event()
        self._active_motion_id = motion_id
        self._mode = "moving"
        self._motion_thread = threading.Thread(
            target=self._run_motion, args=(targets, self._motion_stop), daemon=True
        )
        self._motion_thread.start()

    def _run_motion(self, targets: List[List[float]], stop: threading.Event) -> None:
        try:
            for target in targets:
                self._move_smooth(np.array(target, dtype=float), stop)
                if stop.is_set():
                    break
        finally:
            if not stop.is_set():
                self._mode = "idle"
                self._active_motion_id = ""

    def _move_smooth(self, target: np.ndarray, stop: threading.Event) -> None:
        with self._lock:
            start = np.array([self.data.qpos[a] for a in self._qpos_adr], dtype=float)
        delta = target - start
        max_delta = float(np.max(np.abs(delta))) if delta.size else 0.0
        duration = max_delta / max(1e-3, self.joint_speed_rad_s)
        steps = max(1, int(duration / self.step_dt)) if duration > 0 else 1
        for i in range(1, steps + 1):
            if stop.is_set():
                return
            q = start + delta * (i / steps)
            with self._lock:
                for adr, v in zip(self._qpos_adr, q):
                    self.data.qpos[adr] = float(v)
                mujoco.mj_forward(self.model, self.data)
            time.sleep(self.step_dt)

    # ---- helpers ---------------------------------------------------------

    def _jid(self, name: str) -> int:
        jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise RuntimeError(f"joint_not_found:{name}")
        return jid

    def _resolve_tcp_body(self) -> int:
        for name in _TCP_CANDIDATES:
            bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            if bid >= 0:
                return int(bid)
        return 0

    def _tcp_pose(self) -> List[float]:
        pos = self.data.xpos[self._tcp_body].tolist()
        w, x, y, z = self.data.xquat[self._tcp_body].tolist()
        return [pos[0], pos[1], pos[2], x, y, z, w]

    @staticmethod
    def _trajectory_waypoints(traj) -> List[List[float]]:
        n_dof = traj.n_dof or len(_JOINTS)
        flat = list(traj.q_wp)
        return [flat[i : i + n_dof] for i in range(0, len(flat), n_dof)] if n_dof else []
