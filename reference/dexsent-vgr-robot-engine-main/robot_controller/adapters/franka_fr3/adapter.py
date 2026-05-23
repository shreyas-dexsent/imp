import math
import time
from typing import Any, Dict, List, Optional

import numpy as np

from robot_controller.core.models import PoseSE3, TcpIkRequest
from robot_controller.logging import get_logger

log = get_logger("adapter.franka_fr3")


FR3_JOINT_LIMIT_LOWER = np.array(
    [-2.7437, -1.7837, -2.9007, -3.0421, -2.8065, 0.5445, -3.0159],
    dtype=float,
)
FR3_JOINT_LIMIT_UPPER = np.array(
    [2.7437, 1.7837, 2.9007, -0.1518, 2.8065, 4.5169, 3.0159],
    dtype=float,
)


class Adapter:
    """Franka FR3 adapter backed by franky/libfranka 0.15.x."""

    def __init__(
        self,
        robot_ip: str,
        home_q: Optional[List[float]] = None,
        motion_cfg: Optional[Dict[str, Any]] = None,
        safety_cfg: Optional[Dict[str, Any]] = None,
        tool_frame: str = "fr3_tcp",
    ):
        self.robot_ip = str(robot_ip or "").strip()
        if not self.robot_ip:
            raise ValueError("missing_robot_ip")

        self.home_q = list(home_q) if home_q else [
            0.0,
            -0.785398,
            0.0,
            -2.356194,
            0.0,
            1.570796,
            0.785398,
        ]
        self.motion_cfg = dict(motion_cfg or {})
        self.safety_cfg = dict(safety_cfg or {})
        self.tool_frame = tool_frame
        self.motion_timeout_s = float(self.safety_cfg.get("motion_timeout_s", 10.0))
        self.joint_velocity_limit_rad_s = self._read_limit_vec(
            self.motion_cfg.get("joint_speed_rad_s", 0.5),
            len(self.home_q),
        )
        self.joint_acceleration_limit_rad_s2 = self._read_limit_vec(
            self.motion_cfg.get("joint_acceleration_rad_s2", 1.0),
            len(self.home_q),
        )
        self.joint_jerk_limit_rad_s3 = self._read_limit_vec(
            self.motion_cfg.get("joint_jerk_rad_s3", 4.0),
            len(self.home_q),
        )
        self.joint_lower_limit_rad = self._read_limit_vec(
            self.motion_cfg.get("joint_lower_limit_rad", FR3_JOINT_LIMIT_LOWER.tolist()),
            len(self.home_q),
        )
        self.joint_upper_limit_rad = self._read_limit_vec(
            self.motion_cfg.get("joint_upper_limit_rad", FR3_JOINT_LIMIT_UPPER.tolist()),
            len(self.home_q),
        )
        self.translation_velocity_limit_m_s = self._read_scalar(
            self.motion_cfg.get("linear_speed_m_s", self.safety_cfg.get("max_lin_vel_mps", 0.1)),
            0.1,
        )
        self.translation_acceleration_limit_m_s2 = self._read_scalar(
            self.motion_cfg.get("linear_acceleration_m_s2", 0.3),
            0.3,
        )
        self.translation_jerk_limit_m_s3 = self._read_scalar(
            self.motion_cfg.get("linear_jerk_m_s3", 1.0),
            1.0,
        )
        self.rotation_velocity_limit_rad_s = self._read_scalar(
            self.motion_cfg.get("angular_speed_rad_s", self.safety_cfg.get("max_ang_vel_rps", 0.8)),
            0.8,
        )
        self.rotation_acceleration_limit_rad_s2 = self._read_scalar(
            self.motion_cfg.get("angular_acceleration_rad_s2", 1.6),
            1.6,
        )
        self.rotation_jerk_limit_rad_s3 = self._read_scalar(
            self.motion_cfg.get("angular_jerk_rad_s3", 6.0),
            6.0,
        )
        self.elbow_velocity_limit_rad_s = self._read_scalar(
            self.motion_cfg.get("elbow_speed_rad_s", self.rotation_velocity_limit_rad_s),
            self.rotation_velocity_limit_rad_s,
        )
        self.elbow_acceleration_limit_rad_s2 = self._read_scalar(
            self.motion_cfg.get("elbow_acceleration_rad_s2", self.rotation_acceleration_limit_rad_s2),
            self.rotation_acceleration_limit_rad_s2,
        )
        self.elbow_jerk_limit_rad_s3 = self._read_scalar(
            self.motion_cfg.get("elbow_jerk_rad_s3", self.rotation_jerk_limit_rad_s3),
            self.rotation_jerk_limit_rad_s3,
        )
        self.gripper_open_width_m = float(self.motion_cfg.get("gripper_open_width_m", 0.08))
        self.gripper_close_width_m = float(self.motion_cfg.get("gripper_close_width_m", 0.0))
        self.gripper_open_threshold_m = float(self.motion_cfg.get("gripper_open_threshold_m", 0.02))
        self.gripper_speed_m_s = float(self.motion_cfg.get("gripper_speed_m_s", 0.05))
        self.gripper_force_n = float(self.motion_cfg.get("gripper_force_n", 40.0))
        self.gripper_epsilon_inner_m = float(self.motion_cfg.get("gripper_epsilon_inner_m", 0.005))
        self.gripper_epsilon_outer_m = float(self.motion_cfg.get("gripper_epsilon_outer_m", 0.005))
        self.home_gripper_on_connect = bool(self.motion_cfg.get("gripper_home_on_connect", False))

        self.connected = False
        self.active_motion_id: Optional[str] = None
        self._stop_requested = False
        self._mode = "DISCONNECTED"
        self._gripper_open = True
        self._gripper_width_m = self.gripper_open_width_m
        self._freedrive_enabled = False

        self._q = list(self.home_q)
        self._dq = [0.0] * len(self.home_q)
        self._tcp_pose = PoseSE3(
            position_m=[0.0, 0.0, 0.0],
            quat_xyzw=[0.0, 0.0, 0.0, 1.0],
            frame="base",
        )
        # Flange pose O_T_F = O_T_EE @ inv(F_T_EE). Always in base frame.
        self._flange_pose = PoseSE3(
            position_m=[0.0, 0.0, 0.0],
            quat_xyzw=[0.0, 0.0, 0.0, 1.0],
            frame="base",
        )
        # Cached F_T_EE from robot state (flange→EE, includes FR3's 45° yaw via F_T_NE).
        # Stored as franky.Affine so it can be passed directly to model.pose().
        self._f_t_ee_matrix: Optional[Any] = None

        self._backend = None
        self._robot = None
        self._gripper = None

    def _profile_scale(self, profile: Optional[str]) -> float:
        key = str(profile or "normal").strip().lower()
        default_mapping = {
            "verify": 0.18,
            "calibration": 0.18,
            "slow": 0.30,
            "normal": 0.50,
            "medium": 0.60,
            "fast": 0.75,
        }
        configured = self.motion_cfg.get("profile_scales")
        mapping = dict(default_mapping)
        if isinstance(configured, dict):
            for name, value in configured.items():
                try:
                    scale = float(value)
                except (TypeError, ValueError):
                    continue
                mapping[str(name).strip().lower()] = max(0.05, min(scale, 1.0))
        return float(mapping.get(key, mapping["normal"]))

    def _load_backend(self):
        if self._backend is not None:
            return self._backend
        try:
            import franky  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "franka_backend_missing: install franky-control built against libfranka 0.15.3 "
                "(expected module: franky) before using robot.type=franka_fr3"
            ) from exc
        self._backend = franky
        return franky

    def connect(self) -> None:
        backend = self._load_backend()
        try:
            self._robot = backend.Robot(
                self.robot_ip,
                relative_dynamics_factor=1.0,
                controller_mode=backend.ControllerMode.JointImpedance,
                realtime_config=backend.RealtimeConfig.Enforce,
            )
            if getattr(self._robot, "has_errors", False):
                recovered = bool(self._robot.recover_from_errors())
                if not recovered:
                    log.warning("Franka reported errors during connect and automatic recovery did not clear them")
            self._configure_robot_limits()
        except Exception as exc:
            raise RuntimeError(f"franka_connect_failed:{exc}") from exc

        try:
            self._gripper = backend.Gripper(self.robot_ip)
            if self.home_gripper_on_connect:
                self._gripper.homing()
        except Exception as exc:
            self._gripper = None
            log.warning(f"Franka gripper unavailable: {exc}")

        self.connected = True
        self._stop_requested = False
        self._refresh_state_from_robot()
        if self._mode == "DISCONNECTED":
            self._mode = "IDLE"
        log.info(
            f"Franka FR3 adapter connected to {self.robot_ip} "
            f"(joint_speed_rad_s={self.joint_velocity_limit_rad_s[0]:.3f}, "
            f"linear_speed_m_s={self.translation_velocity_limit_m_s:.3f}, "
            f"angular_speed_rad_s={self.rotation_velocity_limit_rad_s:.3f})"
        )

    def disconnect(self) -> None:
        self._robot = None
        self._gripper = None
        self.connected = False
        self.active_motion_id = None
        self._mode = "DISCONNECTED"
        log.info("Franka FR3 adapter disconnected")

    def get_state(self) -> Dict[str, Any]:
        if self.connected:
            try:
                self._refresh_state_from_robot()
            except Exception as exc:
                self._mode = "ERROR"
                log.warning(f"Franka state refresh failed: {exc}")
        return {
            "timestamp_ns": time.time_ns(),
            "mode": self._mode,
            "tcp_pose": self._tcp_pose.__dict__,
            "flange_pose": self._flange_pose.__dict__,
            "q": list(self._q),
            "dq": list(self._dq),
            "active_motion_id": self.active_motion_id,
            "robot_ip": self.robot_ip,
            "tool_frame": self.tool_frame,
            "gripper_open": self._gripper_open,
            "gripper_width_m": self._gripper_width_m,
            "freedrive": self._freedrive_enabled,
        }

    def move_joints(self, q: List[float], motion_id: str, profile: str = "normal") -> None:
        if not self.connected:
            raise RuntimeError("robot_not_connected")
        if len(q) != len(self.home_q):
            raise RuntimeError("invalid_joint_count")

        self.active_motion_id = motion_id
        self._stop_requested = False
        self._mode = "MOVING"
        try:
            self._execute_joint_move(q, profile)
            self._refresh_state_from_robot()
        finally:
            self.active_motion_id = None
            self._mode = "IDLE" if self.connected else "DISCONNECTED"

    def move_joint_path(self, q_waypoints: List[List[float]], motion_id: str, profile: str = "normal") -> None:
        if not self.connected:
            raise RuntimeError("robot_not_connected")
        path = [list(map(float, q)) for q in q_waypoints if isinstance(q, list)]
        if not path:
            return
        if any(len(q) != len(self.home_q) for q in path):
            raise RuntimeError("invalid_joint_count")

        self.active_motion_id = motion_id
        self._stop_requested = False
        self._mode = "MOVING"
        try:
            for q in path:
                if self._stop_requested:
                    raise RuntimeError("motion_stopped")
                self._execute_joint_move(q, profile)
            self._refresh_state_from_robot()
        finally:
            self.active_motion_id = None
            self._mode = "IDLE" if self.connected else "DISCONNECTED"

    def move_joint_trajectory(
        self,
        positions: List[List[float]],
        velocities: List[List[float]],
        motion_id: str,
        profile: str = "normal",
    ) -> None:
        """Execute a Ruckig-generated timed trajectory as one continuous move.

        Each (position, velocity) pair is a JointState waypoint — franky follows
        them as a feed-forward trajectory with no deceleration at intermediates.
        """
        if not self.connected:
            raise RuntimeError("robot_not_connected")
        if not positions:
            return
        if any(len(q) != len(self.home_q) for q in positions):
            raise RuntimeError("invalid_joint_count")

        backend = self._load_backend()
        dynamics = self._profile_scale(profile)

        self.active_motion_id = motion_id
        self._stop_requested = False
        self._mode = "MOVING"
        try:
            waypoints = [
                backend.JointWaypoint(
                    backend.JointState(
                        np.asarray(positions[i], dtype=float),
                        np.asarray(velocities[i], dtype=float) if i < len(velocities) else np.zeros(len(positions[i])),
                    ),
                    backend.ReferenceType.Absolute,
                    1.0,
                )
                for i in range(len(positions))
            ]
            motion = backend.JointWaypointMotion(waypoints, relative_dynamics_factor=dynamics)
            self._robot.move(motion)
            self._refresh_state_from_robot()
        finally:
            self.active_motion_id = None
            self._mode = "IDLE" if self.connected else "DISCONNECTED"

    def move_joint_waypoints(self, q_waypoints: List[List[float]], motion_id: str, profile: str = "normal") -> None:
        """Execute a smooth multi-waypoint joint motion in one continuous move.

        Uses franky JointWaypointMotion so the robot blends through all
        waypoints without stopping, producing a single smooth trajectory.
        """
        if not self.connected:
            raise RuntimeError("robot_not_connected")
        path = [list(map(float, q)) for q in q_waypoints if isinstance(q, list)]
        if not path:
            return
        if any(len(q) != len(self.home_q) for q in path):
            raise RuntimeError("invalid_joint_count")

        backend = self._load_backend()
        dynamics = self._profile_scale(profile)

        self.active_motion_id = motion_id
        self._stop_requested = False
        self._mode = "MOVING"
        try:
            # Use dynamics=1.0 per waypoint so franky doesn't decelerate at
            # intermediates — the motion-level factor controls overall speed.
            waypoints = [
                backend.JointWaypoint(
                    q,
                    backend.ReferenceType.Absolute,
                    1.0,
                )
                for q in path
            ]
            motion = backend.JointWaypointMotion(waypoints, relative_dynamics_factor=dynamics)
            self._robot.move(motion)
            self._refresh_state_from_robot()
        finally:
            self.active_motion_id = None
            self._mode = "IDLE" if self.connected else "DISCONNECTED"

    def move_tcp(self, target: PoseSE3, motion_id: str, profile: str = "normal") -> None:
        if not self.connected:
            raise RuntimeError("robot_not_connected")
        if len(target.position_m) != 3:
            raise RuntimeError("invalid_target_position")
        if len(target.quat_xyzw) != 4:
            raise RuntimeError("invalid_target_quaternion")

        self.active_motion_id = motion_id
        self._stop_requested = False
        self._mode = "MOVING"
        try:
            self._execute_cartesian_move(target, profile)
            self._refresh_state_from_robot()
        finally:
            self.active_motion_id = None
            self._mode = "IDLE" if self.connected else "DISCONNECTED"

    def move_tcp_ik(
        self, request: TcpIkRequest, motion_id: str, profile: str = "normal"
    ) -> Dict[str, Any]:
        if not self.connected:
            raise RuntimeError("robot_not_connected")
        if len(request.target.position_m) != 3:
            raise RuntimeError("invalid_target_position")
        if len(request.target.quat_xyzw) != 4:
            raise RuntimeError("invalid_target_quaternion")

        self._refresh_state_from_robot()
        seed = request.seed_joints if request.seed_joints is not None else self._q
        preferred = (
            request.preferred_joints
            if request.preferred_joints is not None
            else seed
        )
        if len(seed) != len(self.home_q):
            raise RuntimeError("invalid_seed_joint_count")
        if len(preferred) != len(self.home_q):
            raise RuntimeError("invalid_preferred_joint_count")

        solve = self._solve_tcp_ik(request, seed, preferred)
        joints = solve["joints"]
        self.active_motion_id = motion_id
        self._stop_requested = False
        self._mode = "MOVING"
        try:
            self._execute_joint_move(joints, profile)
            self._refresh_state_from_robot()
        finally:
            self.active_motion_id = None
            self._mode = "IDLE" if self.connected else "DISCONNECTED"
        result = dict(solve)
        result["profile"] = profile
        result["motion"] = "joint"
        return result

    def _clamp_gripper_width(self, width_m: float) -> float:
        lo = min(self.gripper_close_width_m, self.gripper_open_width_m)
        hi = max(self.gripper_close_width_m, self.gripper_open_width_m)
        return max(lo, min(hi, float(width_m)))

    def open_gripper(
        self,
        width_m: float | None = None,
        force_n: float | None = None,
    ) -> None:
        if not self.connected:
            raise RuntimeError("robot_not_connected")
        if self._gripper is None:
            raise RuntimeError("gripper_not_available")
        requested_force_n = (
            None if force_n is None else max(0.0, float(force_n))
        )
        if requested_force_n is not None and requested_force_n > 0.0:
            raise RuntimeError("franka_open_force_not_supported")
        target_width_m = self._clamp_gripper_width(
            self.gripper_open_width_m if width_m is None else width_m
        )
        try:
            if width_m is None:
                self._gripper.open(self.gripper_speed_m_s)
            else:
                self._gripper.move(target_width_m, self.gripper_speed_m_s)
        except Exception:
            self._gripper.move(target_width_m, self.gripper_speed_m_s)
        self._gripper_open = True

    def close_gripper(
        self,
        width_m: float | None = None,
        force_n: float | None = None,
    ) -> None:
        if not self.connected:
            raise RuntimeError("robot_not_connected")
        if self._gripper is None:
            raise RuntimeError("gripper_not_available")
        target_width_m = self._clamp_gripper_width(
            self.gripper_close_width_m if width_m is None else width_m
        )
        requested_force_n = (
            self.gripper_force_n if force_n is None else max(0.0, float(force_n))
        )
        if requested_force_n > 0.0:
            self._gripper.grasp(
                target_width_m,
                self.gripper_speed_m_s,
                requested_force_n,
                self.gripper_epsilon_inner_m,
                self.gripper_epsilon_outer_m,
            )
        else:
            self._gripper.move(target_width_m, self.gripper_speed_m_s)
        self._gripper_open = False

    def freedrive(self, enable: bool) -> None:
        self._freedrive_enabled = bool(enable)
        if not self.connected:
            raise RuntimeError("robot_not_connected")
        raise RuntimeError("freedrive_not_implemented")

    def stop(self) -> None:
        self._stop_requested = True
        self.active_motion_id = None
        self._mode = "IDLE" if self.connected else "DISCONNECTED"
        if self._robot is None:
            return
        try:
            self._robot.stop()
        except Exception as exc:
            log.warning(f"Franka stop failed: {exc}")

    def _execute_joint_move(self, target_q: List[float], profile: str = "normal") -> None:
        if self._robot is None:
            raise RuntimeError("robot_not_connected")
        if self._stop_requested:
            raise RuntimeError("motion_stopped")
        backend = self._load_backend()
        motion = backend.JointMotion(
            np.asarray(target_q, dtype=float),
            backend.ReferenceType.Absolute,
            self._profile_scale(profile),
            True,
        )
        self._robot.move(motion)

    def _execute_cartesian_move(self, target: PoseSE3, profile: str = "normal") -> None:
        if self._robot is None:
            raise RuntimeError("robot_not_connected")
        if self._stop_requested:
            raise RuntimeError("motion_stopped")
        backend = self._load_backend()
        target_pos = np.asarray(target.position_m, dtype=float)
        target_quat = self._normalize_quat(np.asarray(target.quat_xyzw, dtype=float))
        try:
            self._refresh_state_from_robot()
            current_quat = self._normalize_quat(np.asarray(self._tcp_pose.quat_xyzw, dtype=float))
            if float(np.dot(current_quat, target_quat)) < 0.0:
                target_quat = -target_quat
        except Exception as exc:
            log.warning(f"Franka state refresh before Cartesian move failed: {exc}")
        pose = backend.RobotPose(
            backend.Affine(
                target_pos,
                target_quat,
            )
        )
        motion = backend.CartesianMotion(
            backend.CartesianState(pose),
            backend.ReferenceType.Absolute,
            self._profile_scale(profile),
            True,
        )
        try:
            self._robot.move(motion)
        except Exception as exc:
            raise RuntimeError(
                "franka_cartesian_move_failed:"
                f"{exc}; target_pos_m={[round(float(v), 6) for v in target_pos.tolist()]}; "
                f"target_quat_xyzw={[round(float(v), 6) for v in target_quat.tolist()]}; "
                f"profile={profile}; scale={self._profile_scale(profile):.3f}"
            ) from exc

    def _solve_tcp_ik(
        self,
        request: TcpIkRequest,
        seed_joints: List[float],
        preferred_joints: List[float],
    ) -> Dict[str, Any]:
        target_pos = np.asarray(request.target.position_m, dtype=float)
        target_quat = self._normalize_quat(np.asarray(request.target.quat_xyzw, dtype=float))
        q = np.asarray(seed_joints, dtype=float)
        q_pref = np.asarray(preferred_joints, dtype=float)
        lo = np.asarray(self.joint_lower_limit_rad, dtype=float) + 1e-4
        hi = np.asarray(self.joint_upper_limit_rad, dtype=float) - 1e-4
        if not np.all(np.isfinite(q)) or not np.all(np.isfinite(q_pref)):
            raise RuntimeError("ik_invalid_nonfinite_seed")
        q = np.clip(q, lo, hi)
        q_pref = np.clip(q_pref, lo, hi)

        max_iter = max(1, min(int(request.max_iterations or 120), 400))
        pos_tol = max(1e-5, float(request.position_tolerance_m or 0.002))
        ori_tol_rad = math.radians(max(0.05, float(request.orientation_tolerance_deg or 2.0)))
        approx_pos_tol = max(pos_tol, float(request.approximate_position_tolerance_m or 0.015))
        approx_ori_tol_rad = max(
            ori_tol_rad,
            math.radians(float(request.approximate_orientation_tolerance_deg or 3.0)),
        )
        damping = max(1e-5, float(self.motion_cfg.get("ik_damping", 0.04)))
        null_gain = max(0.0, float(self.motion_cfg.get("ik_nullspace_gain", 0.08)))
        max_step = max(0.01, float(self.motion_cfg.get("ik_max_step_rad", 0.12)))

        last_pos_err = float("inf")
        last_ori_err = float("inf")
        best_q = q.copy()
        best_pos_err = float("inf")
        best_ori_err = float("inf")
        best_score = float("inf")
        for iteration in range(max_iter + 1):
            cur_pos, cur_quat = self._fk_pose(q)
            err_pos = target_pos - cur_pos
            err_rot = self._quat_error_rotvec(target_quat, cur_quat)
            last_pos_err = float(np.linalg.norm(err_pos))
            last_ori_err = float(np.linalg.norm(err_rot))
            score = last_pos_err / max(pos_tol, 1e-9) + last_ori_err / max(ori_tol_rad, 1e-9)
            if score < best_score:
                best_score = score
                best_q = q.copy()
                best_pos_err = last_pos_err
                best_ori_err = last_ori_err
            if last_pos_err <= pos_tol and last_ori_err <= ori_tol_rad:
                return {
                    "status": "ok",
                    "joints": [float(v) for v in q.tolist()],
                    "iterations": iteration,
                    "position_error_m": last_pos_err,
                    "orientation_error_deg": math.degrees(last_ori_err),
                    "seed_joints": [float(v) for v in np.asarray(seed_joints, dtype=float).tolist()],
                    "preferred_joints": [float(v) for v in q_pref.tolist()],
                }

            err = np.concatenate([err_pos, err_rot])
            jac = self._numeric_pose_jacobian(q, cur_pos, cur_quat)
            jj_t = jac @ jac.T
            jac_pinv = jac.T @ np.linalg.inv(jj_t + (damping * damping) * np.eye(6))
            dq_task = jac_pinv @ err
            null_projector = np.eye(len(q)) - jac_pinv @ jac
            dq_null = null_projector @ (null_gain * (q_pref - q))
            dq = dq_task + dq_null
            dq_norm = float(np.linalg.norm(dq))
            if not np.isfinite(dq_norm):
                break
            if dq_norm > max_step:
                dq *= max_step / dq_norm
            q = np.clip(q + dq, lo, hi)

        if best_pos_err <= approx_pos_tol and best_ori_err <= approx_ori_tol_rad:
            return {
                "status": "approximate",
                "joints": [float(v) for v in best_q.tolist()],
                "iterations": max_iter,
                "position_error_m": best_pos_err,
                "orientation_error_deg": math.degrees(best_ori_err),
                "strict_position_tolerance_m": pos_tol,
                "strict_orientation_tolerance_deg": math.degrees(ori_tol_rad),
                "approximate_position_tolerance_m": approx_pos_tol,
                "approximate_orientation_tolerance_deg": math.degrees(approx_ori_tol_rad),
                "seed_joints": [float(v) for v in np.asarray(seed_joints, dtype=float).tolist()],
                "preferred_joints": [float(v) for v in q_pref.tolist()],
            }

        raise RuntimeError(
            "ik_failed:"
            f"position_error_m={best_pos_err:.6f}; "
            f"orientation_error_deg={math.degrees(best_ori_err):.3f}; "
            f"iterations={max_iter}; "
            f"target_pos_m={[round(float(v), 6) for v in target_pos.tolist()]}; "
            f"target_quat_xyzw={[round(float(v), 6) for v in target_quat.tolist()]}"
        )

    def _numeric_pose_jacobian(
        self, q: np.ndarray, base_pos: np.ndarray, base_quat: np.ndarray
    ) -> np.ndarray:
        eps = max(1e-5, float(self.motion_cfg.get("ik_fd_eps_rad", 1e-4)))
        jac = np.zeros((6, len(q)), dtype=float)
        for idx in range(len(q)):
            q2 = q.copy()
            q2[idx] += eps
            pos2, quat2 = self._fk_pose(q2)
            jac[:3, idx] = (pos2 - base_pos) / eps
            jac[3:, idx] = self._quat_error_rotvec(quat2, base_quat) / eps
        return jac

    def _fk_pose(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self._robot is None:
            raise RuntimeError("robot_not_connected")
        backend = self._load_backend()
        model = getattr(self._robot, "model", None)
        if callable(model):
            model = model()
        if model is None:
            raise RuntimeError("franka_model_unavailable")
        frame = getattr(backend, "Frame").EndEffector
        # Use the robot's actual F_T_EE (not identity) so FK matches O_T_EE from state.
        # FR3 has a 45° yaw in F_T_NE that would otherwise cause an IK frame mismatch.
        # franky.RobotState.F_T_EE is already a franky.Affine — pass it through directly.
        f_t_ee = self._f_t_ee_matrix if self._f_t_ee_matrix is not None else self._identity_affine(backend)
        ee_t_k = self._identity_affine(backend)
        pose = model.pose(frame, np.asarray(q, dtype=float), f_t_ee, ee_t_k)
        ee_pose = self._maybe_call(getattr(pose, "end_effector_pose", pose))
        translation = self._maybe_call(getattr(ee_pose, "translation", None))
        quaternion = self._maybe_call(getattr(ee_pose, "quaternion", None))
        if translation is None or quaternion is None:
            raise RuntimeError("franka_fk_pose_unreadable")
        pos = np.asarray(translation, dtype=float)
        quat = self._normalize_quat(np.asarray(quaternion, dtype=float))
        return pos, quat

    def _quat_error_rotvec(self, target_quat: np.ndarray, current_quat: np.ndarray) -> np.ndarray:
        target = self._normalize_quat(target_quat)
        current = self._normalize_quat(current_quat)
        if float(np.dot(target, current)) < 0.0:
            target = -target
        q_err = self._quat_multiply(target, self._quat_conjugate(current))
        q_err = self._normalize_quat(q_err)
        if q_err[3] < 0.0:
            q_err = -q_err
        vec = q_err[:3]
        vec_norm = float(np.linalg.norm(vec))
        if vec_norm < 1e-12:
            return np.zeros(3, dtype=float)
        angle = 2.0 * math.atan2(vec_norm, float(q_err[3]))
        return vec / vec_norm * angle

    def _quat_multiply(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        ax, ay, az, aw = [float(v) for v in a]
        bx, by, bz, bw = [float(v) for v in b]
        return np.array(
            [
                aw * bx + ax * bw + ay * bz - az * by,
                aw * by - ax * bz + ay * bw + az * bx,
                aw * bz + ax * by - ay * bx + az * bw,
                aw * bw - ax * bx - ay * by - az * bz,
            ],
            dtype=float,
        )

    def _quat_conjugate(self, quat: np.ndarray) -> np.ndarray:
        return np.array([-quat[0], -quat[1], -quat[2], quat[3]], dtype=float)

    def _maybe_call(self, value: Any) -> Any:
        return value() if callable(value) else value

    def _identity_affine(self, backend: Any) -> Any:
        affine_identity = getattr(backend.Affine, "Identity", None)
        if callable(affine_identity):
            return affine_identity()
        try:
            return backend.Affine(
                np.zeros(3, dtype=float),
                np.array([0.0, 0.0, 0.0, 1.0], dtype=float),
            )
        except Exception:
            return backend.Affine()

    def _configure_robot_limits(self) -> None:
        if self._robot is None:
            return
        self._robot.joint_velocity_limit.set(self.joint_velocity_limit_rad_s)
        self._robot.joint_acceleration_limit.set(self.joint_acceleration_limit_rad_s2)
        self._robot.joint_jerk_limit.set(self.joint_jerk_limit_rad_s3)
        self._robot.translation_velocity_limit.set(self.translation_velocity_limit_m_s)
        self._robot.translation_acceleration_limit.set(self.translation_acceleration_limit_m_s2)
        self._robot.translation_jerk_limit.set(self.translation_jerk_limit_m_s3)
        self._robot.rotation_velocity_limit.set(self.rotation_velocity_limit_rad_s)
        self._robot.rotation_acceleration_limit.set(self.rotation_acceleration_limit_rad_s2)
        self._robot.rotation_jerk_limit.set(self.rotation_jerk_limit_rad_s3)
        self._robot.elbow_velocity_limit.set(self.elbow_velocity_limit_rad_s)
        self._robot.elbow_acceleration_limit.set(self.elbow_acceleration_limit_rad_s2)
        self._robot.elbow_jerk_limit.set(self.elbow_jerk_limit_rad_s3)

    def _refresh_state_from_robot(self) -> None:
        if self._robot is None:
            return

        # Single robot.state call — RobotState carries q, dq, O_T_EE, F_T_EE,
        # and robot_mode so we avoid three separate round-trips to libfranka.
        robot_state = self._robot.state

        self._q = [float(v) for v in np.asarray(robot_state.q, dtype=float).tolist()]
        self._dq = [float(v) for v in np.asarray(robot_state.dq, dtype=float).tolist()]

        ee_affine = robot_state.O_T_EE
        ee_pos = np.asarray(ee_affine.translation, dtype=float)
        quat = self._normalize_quat(np.asarray(ee_affine.quaternion, dtype=float))
        self._tcp_pose = PoseSE3(
            position_m=[float(v) for v in ee_pos.tolist()],
            quat_xyzw=[float(v) for v in quat.tolist()],
            frame="base",
        )

        self._mode = "MOVING" if self.active_motion_id else self._mode_from_backend(robot_state)

        # Cache F_T_EE so _fk_pose matches O_T_EE (includes FR3's F_T_NE 45° yaw).
        try:
            self._f_t_ee_matrix = robot_state.F_T_EE
        except Exception as exc:
            log.warning(f"Could not read F_T_EE from robot state: {exc}")
            self._f_t_ee_matrix = None

        # Compute flange pose: O_T_F = O_T_EE @ inv(F_T_EE).
        self._flange_pose = self._compute_flange_pose(ee_pos, quat)

        if self._gripper is not None:
            try:
                gripper_state = self._gripper.state
                self._gripper_width_m = float(gripper_state.width)
                self._gripper_open = bool(
                    not gripper_state.is_grasped and self._gripper_width_m >= self.gripper_open_threshold_m
                )
            except Exception as exc:
                log.warning(f"Franka gripper state refresh failed: {exc}")

    def _mode_from_backend(self, state: Any) -> str:
        mode = getattr(state, "robot_mode", None)
        if mode is None:
            return "IDLE" if self.connected else "DISCONNECTED"

        name = getattr(mode, "name", None) or str(mode)
        name = str(name).split(".")[-1]
        mapping = {
            "Idle": "IDLE",
            "Move": "MOVING",
            "Guiding": "GUIDING",
            "Reflex": "PROTECTIVE_STOP",
            "UserStopped": "ESTOP",
            "AutomaticErrorRecovery": "RECOVERING",
            "Other": "IDLE",
        }
        return mapping.get(name, "IDLE" if self.connected else "DISCONNECTED")

    def _normalize_quat(self, quat: np.ndarray) -> np.ndarray:
        norm = float(np.linalg.norm(quat))
        if norm < 1e-12:
            return np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
        return quat / norm

    def _read_scalar(self, value: Any, default: float) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return float(default)

    def _quat_xyzw_to_rotmat(self, q: np.ndarray) -> np.ndarray:
        x, y, z, w = q
        return np.array([
            [1 - 2*(y*y + z*z),   2*(x*y - z*w),     2*(x*z + y*w)],
            [2*(x*y + z*w),       1 - 2*(x*x + z*z), 2*(y*z - x*w)],
            [2*(x*z - y*w),       2*(y*z + x*w),     1 - 2*(x*x + y*y)],
        ], dtype=float)

    def _rotmat_to_quat_xyzw(self, R: np.ndarray) -> List[float]:
        trace = R[0, 0] + R[1, 1] + R[2, 2]
        if trace > 0:
            s = 0.5 / np.sqrt(trace + 1.0)
            w = 0.25 / s
            x = (R[2, 1] - R[1, 2]) * s
            y = (R[0, 2] - R[2, 0]) * s
            z = (R[1, 0] - R[0, 1]) * s
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s
        return [float(x), float(y), float(z), float(w)]

    def _compute_flange_pose(self, ee_pos: np.ndarray, ee_quat: np.ndarray) -> PoseSE3:
        """Compute O_T_F (flange pose in base frame) from O_T_EE by removing F_T_EE."""
        if self._f_t_ee_matrix is None:
            return PoseSE3(
                position_m=[float(v) for v in ee_pos.tolist()],
                quat_xyzw=[float(v) for v in ee_quat.tolist()],
                frame="base",
            )
        try:
            f_t_ee_t = np.asarray(self._f_t_ee_matrix.translation, dtype=float)
            f_t_ee_q = self._normalize_quat(np.asarray(self._f_t_ee_matrix.quaternion, dtype=float))
            R_f_ee = self._quat_xyzw_to_rotmat(f_t_ee_q)
            R_o_ee = self._quat_xyzw_to_rotmat(ee_quat)
            # inv(F_T_EE): R^T, -R^T @ t
            R_ee_f = R_f_ee.T
            t_ee_f = -R_ee_f @ f_t_ee_t
            R_o_f = R_o_ee @ R_ee_f
            t_o_f = R_o_ee @ t_ee_f + ee_pos
            return PoseSE3(
                position_m=[float(v) for v in t_o_f.tolist()],
                quat_xyzw=self._rotmat_to_quat_xyzw(R_o_f),
                frame="base",
            )
        except Exception as exc:
            log.warning(f"flange pose computation failed: {exc}")
            return PoseSE3(
                position_m=[float(v) for v in ee_pos.tolist()],
                quat_xyzw=[float(v) for v in ee_quat.tolist()],
                frame="base",
            )

    def _read_limit_vec(self, value: Any, size: int) -> List[float]:
        if isinstance(value, (list, tuple)):
            out = [float(v) for v in value]
            if len(out) != size:
                raise ValueError(f"invalid_limit_length_expected_{size}")
            return out
        scalar = self._read_scalar(value, 0.0)
        return [scalar] * size
