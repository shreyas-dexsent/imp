import math
import threading
import time
from typing import Any, Dict, List, Optional

from robot_controller.core.models import PoseSE3, TcpIkRequest
from robot_controller.logging import get_logger

log = get_logger("adapter.xarm")


def _quat_to_rpy_deg(quat_xyzw: List[float]) -> List[float]:
    x, y, z, w = [float(v) for v in quat_xyzw]
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1.0:
        pitch = math.copysign(math.pi / 2.0, sinp)
    else:
        pitch = math.asin(sinp)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return [math.degrees(roll), math.degrees(pitch), math.degrees(yaw)]


def _rpy_deg_to_quat(roll_deg: float, pitch_deg: float, yaw_deg: float) -> List[float]:
    roll = math.radians(roll_deg)
    pitch = math.radians(pitch_deg)
    yaw = math.radians(yaw_deg)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return [x, y, z, w]


class Adapter:
    """xArm hardware adapter using the official xarm-python-sdk."""

    def __init__(
        self,
        host: str,
        port: int = 18333,
        home_q: Optional[List[float]] = None,
        motion_cfg: Optional[Dict[str, Any]] = None,
        safety_cfg: Optional[Dict[str, Any]] = None,
        tool_frame: str = "xarm_tcp",
        gripper: str = "vacuum",
        joints_in_degrees: bool = False,
    ):
        self.host = str(host or "").strip()
        if not self.host:
            raise ValueError("missing_robot_ip")
        self.port = int(port)
        self.home_q = list(home_q or [])
        self.motion_cfg = dict(motion_cfg or {})
        self.safety_cfg = dict(safety_cfg or {})
        self.tool_frame = str(tool_frame or "xarm_tcp")
        self.gripper = str(gripper or "vacuum").strip().lower()
        self.joints_in_degrees = bool(joints_in_degrees)

        self.motion_timeout_s = self._read_scalar(self.safety_cfg.get("motion_timeout_s", 10.0), 10.0)
        self.linear_speed_mm_s = self._read_linear_speed_mm_s()
        self.linear_acceleration_mm_s2 = self._read_linear_acceleration_mm_s2()
        self.joint_speed_deg_s = self._read_joint_speed_deg_s()
        self.joint_acceleration_deg_s2 = self._read_joint_acceleration_deg_s2()
        self.gripper_open_position = int(self.motion_cfg.get("gripper_open_position", 850))
        self.gripper_close_position = int(self.motion_cfg.get("gripper_close_position", 0))
        self.gripper_speed = int(self.motion_cfg.get("gripper_speed", 3000))
        self.gripper_wait = bool(self.motion_cfg.get("gripper_wait", True))
        self.home_gripper_on_connect = bool(self.motion_cfg.get("gripper_home_on_connect", False))

        self.connected = False
        self.active_motion_id: Optional[str] = None
        self._mode = "DISCONNECTED"
        self._freedrive_enabled = False
        self._gripper_open = True
        self._last_error: Optional[str] = None
        self._q = list(self.home_q)
        self._dq = [0.0] * len(self.home_q)
        self._tcp_pose = PoseSE3(
            position_m=[0.0, 0.0, 0.0],
            quat_xyzw=[0.0, 0.0, 0.0, 1.0],
            frame="base",
        )

        self._lock = threading.Lock()
        self._backend = None
        self._arm = None

    def _load_backend(self):
        if self._backend is not None:
            return self._backend
        try:
            from xarm.wrapper import XArmAPI  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "xarm_backend_missing: install xarm-python-sdk before using robot.type=xarm_lite6"
            ) from exc
        self._backend = XArmAPI
        return XArmAPI

    def connect(self) -> None:
        XArmAPI = self._load_backend()
        host_arg = self.host if self.port == 18333 else f"{self.host}:{self.port}"
        try:
            try:
                self._arm = XArmAPI(host_arg, is_radian=False, do_not_open=False)
            except TypeError:
                self._arm = XArmAPI(self.host, port=self.port, is_radian=False, do_not_open=False)
            time.sleep(0.2)
            self._reset_motion_mode()
            self._setup_gripper()
            self.connected = True
            self._mode = "IDLE"
            self._last_error = None
            self._refresh_state()
        except Exception as exc:
            self.connected = False
            self._mode = "DISCONNECTED"
            self._last_error = f"xarm_connect_failed:{exc}"
            raise RuntimeError(self._last_error) from exc

        log.info(
            "xArm adapter connected to %s:%s (joint_speed_deg_s=%.3f, linear_speed_mm_s=%.3f, gripper=%s)",
            self.host,
            self.port,
            self.joint_speed_deg_s,
            self.linear_speed_mm_s,
            self.gripper,
        )

    def disconnect(self) -> None:
        with self._lock:
            if self._arm is not None:
                try:
                    self._arm.disconnect()
                except Exception:
                    pass
            self._arm = None
        self.connected = False
        self.active_motion_id = None
        self._mode = "DISCONNECTED"
        self._freedrive_enabled = False
        log.info("xArm adapter disconnected")

    def get_state(self) -> Dict[str, Any]:
        if self.connected:
            try:
                self._refresh_state()
            except Exception as exc:
                self._mode = "ERROR"
                self._last_error = f"xarm_state_refresh_failed:{exc}"
                log.warning(self._last_error)
        return {
            "timestamp_ns": time.time_ns(),
            "mode": self._mode,
            "tcp_pose": self._tcp_pose.__dict__,
            "q": list(self._q),
            "dq": list(self._dq),
            "active_motion_id": self.active_motion_id,
            "robot_ip": self.host,
            "tool_frame": self.tool_frame,
            "gripper_open": self._gripper_open,
            "freedrive": self._freedrive_enabled,
            "last_error": self._last_error,
        }

    def move_joints(self, q: List[float], motion_id: str, profile: str = "normal") -> None:
        if not self.connected or self._arm is None:
            raise RuntimeError("robot_not_connected")
        if self.home_q and len(q) != len(self.home_q):
            raise RuntimeError("invalid_joint_count")

        angles = [float(v) for v in q]
        if not self.joints_in_degrees:
            angles = [math.degrees(v) for v in angles]

        self.active_motion_id = motion_id
        self._mode = "MOVING"
        self._last_error = None
        try:
            self._reset_motion_mode()
            with self._lock:
                code = self._arm.set_servo_angle(
                    angle=angles,
                    speed=self.joint_speed_deg_s,
                    mvacc=self.joint_acceleration_deg_s2,
                    wait=True,
                )
            self._check(code, "move_joints")
            self._refresh_state()
        finally:
            self.active_motion_id = None
            self._mode = "IDLE" if self.connected else "DISCONNECTED"

    def move_joint_path(self, q_waypoints: List[List[float]], motion_id: str, profile: str = "normal") -> None:
        if not self.connected or self._arm is None:
            raise RuntimeError("robot_not_connected")
        path = [list(map(float, q)) for q in q_waypoints if isinstance(q, list)]
        if not path:
            return
        if self.home_q and any(len(q) != len(self.home_q) for q in path):
            raise RuntimeError("invalid_joint_count")

        self.active_motion_id = motion_id
        self._mode = "MOVING"
        self._last_error = None
        try:
            self._reset_motion_mode()
            for q in path:
                angles = [float(v) for v in q]
                if not self.joints_in_degrees:
                    angles = [math.degrees(v) for v in angles]
                with self._lock:
                    code = self._arm.set_servo_angle(
                        angle=angles,
                        speed=self.joint_speed_deg_s,
                        mvacc=self.joint_acceleration_deg_s2,
                        wait=True,
                    )
                self._check(code, "move_joint_path")
            self._refresh_state()
        finally:
            self.active_motion_id = None
            self._mode = "IDLE" if self.connected else "DISCONNECTED"

    def move_tcp(self, target: PoseSE3, motion_id: str, profile: str = "normal") -> None:
        if not self.connected or self._arm is None:
            raise RuntimeError("robot_not_connected")
        if len(target.position_m) != 3:
            raise RuntimeError("invalid_target_position")
        if len(target.quat_xyzw) != 4:
            raise RuntimeError("invalid_target_quaternion")

        rpy_deg = _quat_to_rpy_deg(target.quat_xyzw)
        x_mm = float(target.position_m[0]) * 1000.0
        y_mm = float(target.position_m[1]) * 1000.0
        z_mm = float(target.position_m[2]) * 1000.0

        self.active_motion_id = motion_id
        self._mode = "MOVING"
        self._last_error = None
        try:
            self._reset_motion_mode()
            with self._lock:
                try:
                    code = self._arm.set_position(
                        x_mm,
                        y_mm,
                        z_mm,
                        rpy_deg[0],
                        rpy_deg[1],
                        rpy_deg[2],
                        speed=self.linear_speed_mm_s,
                        mvacc=self.linear_acceleration_mm_s2,
                        radius=0.0,
                        wait=True,
                    )
                except TypeError:
                    code = self._arm.set_position(
                        x_mm,
                        y_mm,
                        z_mm,
                        rpy_deg[0],
                        rpy_deg[1],
                        rpy_deg[2],
                        speed=self.linear_speed_mm_s,
                        mvacc=self.linear_acceleration_mm_s2,
                        wait=True,
                    )
            self._check(code, "move_tcp")
            self._refresh_state()
        finally:
            self.active_motion_id = None
            self._mode = "IDLE" if self.connected else "DISCONNECTED"

    def move_tcp_ik(
        self, request: TcpIkRequest, motion_id: str, profile: str = "normal"
    ) -> Dict[str, Any]:
        raise RuntimeError("ik_not_supported")

    def open_gripper(
        self,
        width_m: float | None = None,
        force_n: float | None = None,
    ) -> None:
        if not self.connected or self._arm is None:
            raise RuntimeError("robot_not_connected")
        with self._lock:
            code = self._command_open_gripper()
        self._check(code, "open_gripper")
        self._gripper_open = True

    def close_gripper(
        self,
        width_m: float | None = None,
        force_n: float | None = None,
    ) -> None:
        if not self.connected or self._arm is None:
            raise RuntimeError("robot_not_connected")
        with self._lock:
            code = self._command_close_gripper()
        self._check(code, "close_gripper")
        self._gripper_open = False

    def freedrive(self, enable: bool) -> None:
        if not self.connected or self._arm is None:
            raise RuntimeError("robot_not_connected")
        mode = 2 if enable else 0
        with self._lock:
            code = self._arm.set_mode(mode)
            self._check(code, "set_mode")
            code = self._arm.set_state(0)
            self._check(code, "set_state")
        self._freedrive_enabled = bool(enable)
        self._mode = "GUIDING" if enable else "IDLE"

    def stop(self) -> None:
        self.active_motion_id = None
        self._mode = "IDLE" if self.connected else "DISCONNECTED"
        if self._arm is None:
            return
        with self._lock:
            try:
                if hasattr(self._arm, "emergency_stop"):
                    self._arm.emergency_stop()
                elif hasattr(self._arm, "stop_motion"):
                    self._arm.stop_motion()
                else:
                    self._arm.set_state(4)
            except Exception as exc:
                self._last_error = f"xarm_stop_failed:{exc}"
                log.warning(self._last_error)

    def _setup_gripper(self) -> None:
        if self._arm is None:
            return
        if self.gripper in ("vacuum", "suction"):
            return
        try:
            if hasattr(self._arm, "set_gripper_enable"):
                self._check(self._arm.set_gripper_enable(True), "set_gripper_enable")
            if hasattr(self._arm, "set_gripper_mode"):
                self._check(self._arm.set_gripper_mode(0), "set_gripper_mode")
            if hasattr(self._arm, "set_gripper_speed"):
                self._check(self._arm.set_gripper_speed(self.gripper_speed), "set_gripper_speed")
            if self.home_gripper_on_connect and hasattr(self._arm, "set_gripper_position"):
                self._check(
                    self._arm.set_gripper_position(
                        self.gripper_open_position,
                        wait=self.gripper_wait,
                        speed=self.gripper_speed,
                    ),
                    "gripper_home_on_connect",
                )
        except Exception as exc:
            log.warning("xArm gripper setup skipped: %s", exc)

    def _reset_motion_mode(self) -> None:
        if self._arm is None:
            raise RuntimeError("robot_not_connected")
        with self._lock:
            self._check(self._arm.clean_warn(), "clean_warn")
            self._check(self._arm.clean_error(), "clean_error")
            self._check(self._arm.motion_enable(True), "motion_enable")
            self._check(self._arm.set_mode(0), "set_mode")
            self._check(self._arm.set_state(0), "set_state")
        self._freedrive_enabled = False

    def _command_open_gripper(self) -> int:
        if self._arm is None:
            return 0
        if self.gripper in ("vacuum", "suction"):
            if hasattr(self._arm, "set_vacuum_gripper"):
                return self._normalize_code(self._arm.set_vacuum_gripper(False))
            if hasattr(self._arm, "set_suction_cup"):
                return self._normalize_code(self._arm.set_suction_cup(False))
            return 0
        if hasattr(self._arm, "set_gripper_position"):
            return self._normalize_code(
                self._arm.set_gripper_position(
                    self.gripper_open_position,
                    wait=self.gripper_wait,
                    speed=self.gripper_speed,
                )
            )
        if self.gripper == "lite6" and hasattr(self._arm, "open_lite6_gripper"):
            return self._normalize_code(self._arm.open_lite6_gripper())
        if hasattr(self._arm, "open_gripper"):
            return self._normalize_code(self._arm.open_gripper())
        return 0

    def _command_close_gripper(self) -> int:
        if self._arm is None:
            return 0
        if self.gripper in ("vacuum", "suction"):
            if hasattr(self._arm, "set_vacuum_gripper"):
                return self._normalize_code(self._arm.set_vacuum_gripper(True))
            if hasattr(self._arm, "set_suction_cup"):
                return self._normalize_code(self._arm.set_suction_cup(True))
            return 0
        if hasattr(self._arm, "set_gripper_position"):
            return self._normalize_code(
                self._arm.set_gripper_position(
                    self.gripper_close_position,
                    wait=self.gripper_wait,
                    speed=self.gripper_speed,
                )
            )
        if self.gripper == "lite6" and hasattr(self._arm, "close_lite6_gripper"):
            return self._normalize_code(self._arm.close_lite6_gripper())
        if hasattr(self._arm, "close_gripper"):
            return self._normalize_code(self._arm.close_gripper())
        return 0

    def _refresh_state(self) -> None:
        if self._arm is None:
            return
        with self._lock:
            state_ret = self._arm.get_state()
            pose_ret = self._arm.get_position(is_radian=False)
            joints_ret = self._arm.get_servo_angle(is_radian=False)
            moving_ret = self._arm.get_is_moving() if hasattr(self._arm, "get_is_moving") else 0
            err_ret = self._arm.get_err_warn_code() if hasattr(self._arm, "get_err_warn_code") else 0
            gripper_pos_ret = (
                self._arm.get_gripper_position() if hasattr(self._arm, "get_gripper_position") else None
            )

        state_code, state_val = self._unwrap_pair(state_ret)
        pose_code, pose_val = self._unwrap_pair(pose_ret)
        joints_code, joints_val = self._unwrap_pair(joints_ret)
        moving_code, moving_val = self._unwrap_pair(moving_ret)
        err_code, err_val = self._unwrap_pair(err_ret)
        gripper_code, gripper_val = self._unwrap_pair(gripper_pos_ret)

        if state_code != 0:
            raise RuntimeError(f"xarm_state_read_failed:{state_code}")
        if pose_code == 0 and isinstance(pose_val, (list, tuple)) and len(pose_val) >= 6:
            self._tcp_pose = PoseSE3(
                position_m=[
                    float(pose_val[0]) / 1000.0,
                    float(pose_val[1]) / 1000.0,
                    float(pose_val[2]) / 1000.0,
                ],
                quat_xyzw=_rpy_deg_to_quat(float(pose_val[3]), float(pose_val[4]), float(pose_val[5])),
                frame="base",
            )
        if joints_code == 0 and isinstance(joints_val, (list, tuple)):
            self._q = [math.radians(float(v)) for v in joints_val]
            self._dq = [0.0] * len(self._q)

        sdk_error = 0
        if err_code == 0 and isinstance(err_val, (list, tuple)) and err_val:
            try:
                sdk_error = int(err_val[0])
            except Exception:
                sdk_error = 0
        elif err_code == 0 and err_val is not None:
            try:
                sdk_error = int(err_val)
            except Exception:
                sdk_error = 0

        is_moving = False
        if moving_code == 0:
            if isinstance(moving_val, (list, tuple)) and moving_val:
                is_moving = bool(moving_val[0])
            else:
                is_moving = bool(moving_val)

        if gripper_code == 0 and gripper_val is not None and self.gripper not in ("vacuum", "suction"):
            try:
                position = float(gripper_val[0] if isinstance(gripper_val, (list, tuple)) else gripper_val)
                midpoint = (self.gripper_open_position + self.gripper_close_position) / 2.0
                self._gripper_open = position >= midpoint
            except Exception:
                pass

        if sdk_error != 0:
            self._mode = "ERROR"
            self._last_error = f"xarm_error:{sdk_error}"
        elif self._freedrive_enabled:
            self._mode = "GUIDING"
            self._last_error = None
        elif is_moving or self.active_motion_id:
            self._mode = "MOVING"
            self._last_error = None
        else:
            self._mode = "IDLE"
            self._last_error = None

        if state_val not in (None, 0, 1) and self._mode == "IDLE":
            self._mode = f"STATE_{state_val}"

    def _unwrap_pair(self, value: Any) -> Any:
        if isinstance(value, (tuple, list)):
            if not value:
                return 0, None
            code = self._normalize_code(value[0])
            payload = value[1] if len(value) > 1 else None
            return code, payload
        return 0, value

    def _check(self, code: Any, label: str) -> None:
        code_int = self._normalize_code(code)
        if code_int != 0:
            raise RuntimeError(f"{label}_failed:{code_int}")

    def _normalize_code(self, code: Any) -> int:
        if code is None:
            return 0
        if isinstance(code, bool):
            return 0 if code else -1
        if isinstance(code, (tuple, list)) and code:
            return self._normalize_code(code[0])
        try:
            return int(code)
        except Exception:
            return 0

    def _read_scalar(self, value: Any, default: float) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return float(default)

    def _read_joint_speed_deg_s(self) -> float:
        if "joint_speed_deg_s" in self.motion_cfg:
            return self._read_scalar(self.motion_cfg.get("joint_speed_deg_s"), 20.0)
        if "joint_speed_rad_s" in self.motion_cfg:
            return math.degrees(self._read_scalar(self.motion_cfg.get("joint_speed_rad_s"), 0.35))
        return self._read_scalar(self.motion_cfg.get("joint_speed", 20.0), 20.0)

    def _read_joint_acceleration_deg_s2(self) -> float:
        if "joint_acceleration_deg_s2" in self.motion_cfg:
            return self._read_scalar(self.motion_cfg.get("joint_acceleration_deg_s2"), 500.0)
        if "joint_acceleration_rad_s2" in self.motion_cfg:
            return math.degrees(self._read_scalar(self.motion_cfg.get("joint_acceleration_rad_s2"), 1.0))
        return self._read_scalar(self.motion_cfg.get("joint_acc", 500.0), 500.0)

    def _read_linear_speed_mm_s(self) -> float:
        if "linear_speed_mm_s" in self.motion_cfg:
            return self._read_scalar(self.motion_cfg.get("linear_speed_mm_s"), 87.0)
        if "linear_speed_m_s" in self.motion_cfg:
            return self._read_scalar(self.motion_cfg.get("linear_speed_m_s"), 0.087) * 1000.0
        return self._read_scalar(self.motion_cfg.get("tcp_speed", 87.0), 87.0)

    def _read_linear_acceleration_mm_s2(self) -> float:
        if "linear_acceleration_mm_s2" in self.motion_cfg:
            return self._read_scalar(self.motion_cfg.get("linear_acceleration_mm_s2"), 3000.0)
        if "linear_acceleration_m_s2" in self.motion_cfg:
            return self._read_scalar(self.motion_cfg.get("linear_acceleration_m_s2"), 3.0) * 1000.0
        return self._read_scalar(self.motion_cfg.get("tcp_acc", 3000.0), 3000.0)
