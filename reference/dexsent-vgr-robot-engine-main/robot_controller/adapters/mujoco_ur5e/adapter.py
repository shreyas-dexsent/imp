"""Implementation for `robot_controller.adapters.mujoco_ur5e.adapter`."""

import math
import threading
import time
from pathlib import Path
from typing import Any, Dict, List

import mujoco
import numpy as np
from robot_controller.core.models import CartesianVelocity, PoseSE3, TcpIkRequest
from robot_controller.logging import get_logger

log = get_logger("adapter.mujoco_ur5e")


class Adapter:
    def __init__(
        self,
        model_path: str,
        home_q=None,
        motion_cfg: Dict[str, Any] = None,
        base_frame_yaw_deg: float = 0.0,
    ):
        self.model_path = Path(model_path).resolve()
        if not self.model_path.exists():
            raise FileNotFoundError(self.model_path)
        self.model = mujoco.MjModel.from_xml_path(str(self.model_path))
        self.data = mujoco.MjData(self.model)
        self.mj_lock = threading.RLock()
        self.joint_names = [
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint",
        ]
        self._joint_ids = self._resolve_joint_ids()
        self._joint_qpos_adr = self._resolve_joint_qpos_adr()
        self._joint_dof_adr = self._resolve_joint_dof_adr()
        self._tcp_body_id = self._resolve_tcp_body_id()
        self.home_q = list(home_q) if home_q else None
        self.connected = False
        self.active_motion_id = None
        self._stop_requested = False
        self._mode = "DISCONNECTED"
        self._gripper_open = True
        self._freedrive_enabled = False
        motion_cfg = motion_cfg or {}
        self.joint_speed_rad_s = float(motion_cfg.get("joint_speed_rad_s", 0.7))
        self.linear_speed_m_s = float(motion_cfg.get("linear_speed_m_s", 0.15))
        self.step_dt = float(motion_cfg.get("step_dt", 0.02))
        self.ik_steps = int(motion_cfg.get("ik_steps", 200))
        self.ik_tol_pos = float(motion_cfg.get("ik_tol_pos", 1e-4))
        self.ik_tol_rot = float(motion_cfg.get("ik_tol_rot", 1e-3))
        self.ik_damping = float(motion_cfg.get("ik_damping", 1e-4))
        self.ik_step_size = float(motion_cfg.get("ik_step_size", 0.5))
        self.ik_orient_weight = float(motion_cfg.get("orientation_weight", 0.4))
        self.base_frame_yaw_deg = float(base_frame_yaw_deg or 0.0)
        self._base_rot_xyzw = self._yaw_quat_xyzw(math.radians(self.base_frame_yaw_deg))
        self._base_rot_inv_xyzw = self._quat_conj_xyzw(self._base_rot_xyzw)
        self._apply_home()

    def _yaw_quat_xyzw(self, yaw_rad: float) -> np.ndarray:
        half = yaw_rad / 2.0
        return np.array([0.0, 0.0, math.sin(half), math.cos(half)], dtype=float)

    def _resolve_joint_ids(self) -> List[int]:
        ids: List[int] = []
        for name in self.joint_names:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid < 0:
                raise RuntimeError(f"joint_not_found:{name}")
            ids.append(int(jid))
        return ids

    def _resolve_joint_qpos_adr(self) -> List[int]:
        addrs: List[int] = []
        for jid in self._joint_ids:
            addrs.append(int(self.model.jnt_qposadr[jid]))
        return addrs

    def _resolve_joint_dof_adr(self) -> List[int]:
        addrs: List[int] = []
        for jid in self._joint_ids:
            addrs.append(int(self.model.jnt_dofadr[jid]))
        return addrs

    def _set_joint_qpos(self, q: List[float]) -> None:
        if len(q) < len(self._joint_qpos_adr):
            raise ValueError("invalid_joint_count")
        with self.mj_lock:
            for idx, adr in enumerate(self._joint_qpos_adr):
                self.data.qpos[adr] = float(q[idx])

    def _get_joint_qpos(self) -> List[float]:
        with self.mj_lock:
            return [float(self.data.qpos[adr]) for adr in self._joint_qpos_adr]

    def _get_joint_qvel(self) -> List[float]:
        with self.mj_lock:
            return [float(self.data.qvel[adr]) for adr in self._joint_dof_adr]

    def _resolve_tcp_body_id(self) -> int:
        candidates = [
            "tool0",
            "tcp",
            "ee_link",
            "wrist_3_link",
            "wrist_3",
            "flange",
        ]
        for name in candidates:
            bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            if bid >= 0:
                log.info(f"TCP body resolved to '{name}'")
                return int(bid)
        log.warning("TCP body not found, using body 0")
        return 0

    def _tcp_pose(self) -> PoseSE3:
        if self._tcp_body_id < 0:
            return PoseSE3(position_m=[0.0, 0.0, 0.0], quat_xyzw=[0.0, 0.0, 0.0, 1.0])
        with self.mj_lock:
            pos = self.data.xpos[self._tcp_body_id].tolist()
            quat_wxyz = self.data.xquat[self._tcp_body_id].tolist()
        quat_xyzw = [quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]]
        return PoseSE3(position_m=pos, quat_xyzw=quat_xyzw, frame="base")

    def _apply_home(self) -> None:
        if not self.home_q:
            return
        with self.mj_lock:
            self._set_joint_qpos(self.home_q)
            mujoco.mj_forward(self.model, self.data)

    def connect(self) -> None:
        self.connected = True
        self._mode = "IDLE"
        log.info("MuJoCo UR5e adapter connected")

    def disconnect(self) -> None:
        self.connected = False
        self._mode = "DISCONNECTED"
        log.info("MuJoCo UR5e adapter disconnected")

    def get_state(self) -> Dict[str, Any]:
        q = self._get_joint_qpos()
        dq = self._get_joint_qvel()
        tcp_pose = self._to_external_pose(self._tcp_pose())
        return {
            "timestamp_ns": time.time_ns(),
            "mode": self._mode,
            "tcp_pose": tcp_pose.__dict__,
            "q": q,
            "dq": dq,
            "active_motion_id": self.active_motion_id,
            "gripper_open": self._gripper_open,
            "freedrive": self._freedrive_enabled,
        }

    def move_joints(self, q: List[float], motion_id: str, profile: str = "normal") -> None:
        if not self.connected:
            raise RuntimeError("robot_not_connected")
        self.active_motion_id = motion_id
        self._stop_requested = False
        self._mode = "MOVING"
        try:
            self._move_joints_smooth(q)
        finally:
            self.active_motion_id = None
            self._mode = "IDLE" if self.connected else "DISCONNECTED"

    def move_joint_path(self, q_waypoints: List[List[float]], motion_id: str, profile: str = "normal") -> None:
        if not self.connected:
            raise RuntimeError("robot_not_connected")
        path = [list(map(float, q)) for q in q_waypoints if isinstance(q, list)]
        if not path:
            return
        self.active_motion_id = motion_id
        self._stop_requested = False
        self._mode = "MOVING"
        try:
            for q in path:
                if self._stop_requested:
                    raise RuntimeError("motion_stopped")
                self._move_joints_smooth(q)
        finally:
            self.active_motion_id = None
            self._mode = "IDLE" if self.connected else "DISCONNECTED"

    def move_tcp(self, target: PoseSE3, motion_id: str, profile: str = "normal") -> None:
        if not self.connected:
            raise RuntimeError("robot_not_connected")
        target = self._to_internal_pose(target)
        target_pos = np.array(target.position_m, dtype=float)
        if target_pos.shape[0] != 3:
            raise RuntimeError("invalid_target_position")
        target_quat = np.array(target.quat_xyzw, dtype=float)
        if target_quat.shape[0] != 4:
            target_quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
        self.active_motion_id = motion_id
        self._stop_requested = False
        self._mode = "MOVING"
        start_q = self._get_joint_qpos()
        solved, target_q = self._solve_ik(target_pos, target_quat)
        if not solved or target_q is None:
            self.active_motion_id = None
            self._mode = "IDLE" if self.connected else "DISCONNECTED"
            raise RuntimeError("ik_failed")
        with self.mj_lock:
            self._set_joint_qpos(start_q)
            mujoco.mj_forward(self.model, self.data)
        self._move_joints_smooth(target_q)
        self.active_motion_id = None
        self._mode = "IDLE" if self.connected else "DISCONNECTED"

    def move_tcp_ik(
        self, request: TcpIkRequest, motion_id: str, profile: str = "normal"
    ) -> Dict[str, Any]:
        if request.preferred_joints is None:
            raise RuntimeError("ik_not_supported")
        self.move_tcp(request.target, motion_id, profile)
        return {
            "status": "simulated",
            "motion": "joint",
            "preferred_joints": list(request.preferred_joints),
        }

    def open_gripper(
        self,
        width_m: float | None = None,
        force_n: float | None = None,
    ) -> None:
        self._gripper_open = True

    def close_gripper(
        self,
        width_m: float | None = None,
        force_n: float | None = None,
    ) -> None:
        self._gripper_open = False

    def freedrive(self, enable: bool) -> None:
        self._freedrive_enabled = bool(enable)

    def set_mode(self, mode: int) -> None:
        # MuJoCo adapter does not model controller modes; accept for compatibility.
        return None

    def set_state(self, state: int) -> None:
        # MuJoCo adapter does not model controller states; accept for compatibility.
        return None

    def servo_tcp(self, target: PoseSE3, motion_id: str) -> None:
        # Best-effort servo for simulation: reuse IK move with small steps.
        self.move_tcp(target, motion_id)

    def servo_tcp_velocity(self, velocity: CartesianVelocity, motion_id: str) -> None:
        raise RuntimeError("cartesian_velocity_not_supported")

    def _quat_xyzw_to_wxyz(self, q: np.ndarray) -> np.ndarray:
        return np.array([q[3], q[0], q[1], q[2]], dtype=float)

    def _quat_wxyz_to_xyzw(self, q: np.ndarray) -> np.ndarray:
        return np.array([q[1], q[2], q[3], q[0]], dtype=float)

    def _quat_conj_xyzw(self, q: np.ndarray) -> np.ndarray:
        return np.array([-q[0], -q[1], -q[2], q[3]], dtype=float)

    def _quat_mul_xyzw(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        aw, ax, ay, az = a[3], a[0], a[1], a[2]
        bw, bx, by, bz = b[3], b[0], b[1], b[2]
        return np.array(
            [
                aw * bx + ax * bw + ay * bz - az * by,
                aw * by - ax * bz + ay * bw + az * bx,
                aw * bz + ax * by - ay * bx + az * bw,
                aw * bw - ax * bx - ay * by - az * bz,
            ],
            dtype=float,
        )

    def _quat_rotate_xyzw(self, q: np.ndarray, v: np.ndarray) -> np.ndarray:
        vq = np.array([v[0], v[1], v[2], 0.0], dtype=float)
        rq = self._quat_mul_xyzw(self._quat_mul_xyzw(q, vq), self._quat_conj_xyzw(q))
        return rq[:3]

    def _to_internal_pose(self, pose: PoseSE3) -> PoseSE3:
        if self.base_frame_yaw_deg == 0.0:
            return pose
        pos = np.array(pose.position_m, dtype=float)
        quat = np.array(pose.quat_xyzw, dtype=float)
        pos_i = self._quat_rotate_xyzw(self._base_rot_xyzw, pos)
        quat_i = self._quat_mul_xyzw(self._base_rot_xyzw, quat)
        return PoseSE3(
            position_m=pos_i.tolist(), quat_xyzw=quat_i.tolist(), frame=pose.frame
        )

    def _to_external_pose(self, pose: PoseSE3) -> PoseSE3:
        if self.base_frame_yaw_deg == 0.0:
            return pose
        pos = np.array(pose.position_m, dtype=float)
        quat = np.array(pose.quat_xyzw, dtype=float)
        pos_e = self._quat_rotate_xyzw(self._base_rot_inv_xyzw, pos)
        quat_e = self._quat_mul_xyzw(self._base_rot_inv_xyzw, quat)
        return PoseSE3(
            position_m=pos_e.tolist(), quat_xyzw=quat_e.tolist(), frame=pose.frame
        )

    def _quat_error(
        self, target_xyzw: np.ndarray, current_xyzw: np.ndarray
    ) -> np.ndarray:
        target_xyzw = target_xyzw / (np.linalg.norm(target_xyzw) + 1e-9)
        current_xyzw = current_xyzw / (np.linalg.norm(current_xyzw) + 1e-9)
        q_err = self._quat_mul_xyzw(target_xyzw, self._quat_conj_xyzw(current_xyzw))
        w = float(np.clip(q_err[3], -1.0, 1.0))
        angle = 2.0 * np.arccos(w)
        if angle > np.pi:
            angle -= 2.0 * np.pi
        s = np.sqrt(max(1e-9, 1.0 - w * w))
        axis = q_err[:3] / s
        return axis * angle

    def _solve_ik(
        self, target_pos: np.ndarray, target_quat_xyzw: np.ndarray
    ) -> (bool, List[float]):
        use_orient = self.ik_orient_weight > 0.0
        for _ in range(self.ik_steps):
            if self._stop_requested:
                return False, None
            with self.mj_lock:
                mujoco.mj_forward(self.model, self.data)
                cur_pos = self.data.xpos[self._tcp_body_id].copy()
            pos_err = target_pos - cur_pos

            with self.mj_lock:
                cur_quat_wxyz = self.data.xquat[self._tcp_body_id].copy()
            cur_quat_xyzw = self._quat_wxyz_to_xyzw(
                np.array(cur_quat_wxyz, dtype=float)
            )

            rot_err = np.zeros(3, dtype=float)
            if use_orient:
                rot_err = self._quat_error(target_quat_xyzw, cur_quat_xyzw)

            if np.linalg.norm(pos_err) < self.ik_tol_pos and (
                not use_orient or np.linalg.norm(rot_err) < self.ik_tol_rot
            ):
                return True, self._get_joint_qpos()

            jacp = np.zeros((3, self.model.nv))
            jacr = np.zeros((3, self.model.nv))
            with self.mj_lock:
                mujoco.mj_jacBody(self.model, self.data, jacp, jacr, self._tcp_body_id)
            j_pos = jacp[:, self._joint_dof_adr]
            if use_orient:
                j_rot = jacr[:, self._joint_dof_adr]
                err = np.concatenate([pos_err, rot_err * self.ik_orient_weight])
                j = np.vstack([j_pos, j_rot * self.ik_orient_weight])
            else:
                err = pos_err
                j = j_pos

            jj = j @ j.T + (self.ik_damping**2) * np.eye(j.shape[0])
            dq = j.T @ np.linalg.solve(jj, err)

            with self.mj_lock:
                for idx, qpos_adr in enumerate(self._joint_qpos_adr):
                    jid = self._joint_ids[idx]
                    lower = self.model.jnt_range[jid][0]
                    upper = self.model.jnt_range[jid][1]
                    new_q = self.data.qpos[qpos_adr] + self.ik_step_size * dq[idx]
                    self.data.qpos[qpos_adr] = float(np.clip(new_q, lower, upper))

        return False, None

    def _move_joints_smooth(self, target_q: List[float]) -> None:
        start = np.array(self._get_joint_qpos(), dtype=float)
        target = np.array(target_q, dtype=float)
        delta = target - start
        max_delta = float(np.max(np.abs(delta)))
        speed = max(1e-3, self.joint_speed_rad_s)
        duration = max_delta / speed if max_delta > 0 else 0.0
        dt = max(0.005, self.step_dt)
        steps = max(1, int(duration / dt)) if duration > 0 else 1
        for i in range(1, steps + 1):
            if self._stop_requested:
                break
            alpha = i / steps
            q = start + delta * alpha
            with self.mj_lock:
                self._set_joint_qpos(q.tolist())
                mujoco.mj_forward(self.model, self.data)
            time.sleep(dt)

    def stop(self) -> None:
        self._stop_requested = True
        self.active_motion_id = None
        self._mode = "IDLE" if self.connected else "DISCONNECTED"
