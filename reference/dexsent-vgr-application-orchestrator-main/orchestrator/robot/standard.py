"""Implementation for `orchestrator.robot.standard`."""

import time
from typing import Any, Dict, Tuple


class StandardRobotAdapter:
    def __init__(self):
        self._state = {
            "model": None,
            "mode": "idle",
            "tcp_pose": {
                "position_m": [0.0, 0.0, 0.0],
                "quat_xyzw": [0.0, 0.0, 0.0, 1.0],
                "frame": "base",
            },
            "joints": (),
            "gripper_open": True,
            "freedrive": False,
            "last_command": None,
        }

    def set_robot_model(self, model: str) -> None:
        self._state["model"] = model
        self._state["last_command"] = {"type": "set_robot_model", "model": model}

    def get_state(self) -> Dict[str, Any]:
        return dict(self._state)

    def movej(self, joints: Tuple[float, ...], profile: str) -> None:
        self._state["mode"] = "moving_joints"
        self._state["last_command"] = {
            "type": "movej",
            "joints": joints,
            "profile": profile,
        }
        time.sleep(0.05)
        self._state["joints"] = joints
        self._state["mode"] = "idle"

    def movej_path(self, waypoints: Tuple[Tuple[float, ...], ...], profile: str) -> None:
        self._state["mode"] = "moving_joint_path"
        self._state["last_command"] = {
            "type": "movej_path",
            "waypoint_count": len(waypoints),
            "profile": profile,
        }
        time.sleep(0.05)
        if waypoints:
            self._state["joints"] = tuple(waypoints[-1])
        self._state["mode"] = "idle"

    def move_joint_waypoints(self, waypoints: Tuple[Tuple[float, ...], ...], profile: str) -> None:
        self._state["mode"] = "moving_joint_waypoints"
        self._state["last_command"] = {
            "type": "move_joint_waypoints",
            "waypoint_count": len(waypoints),
            "profile": profile,
        }
        time.sleep(0.05)
        if waypoints:
            self._state["joints"] = tuple(waypoints[-1])
        self._state["mode"] = "idle"

    def move_joint_trajectory(self, positions: Tuple[Tuple[float, ...], ...], velocities: Tuple[Tuple[float, ...], ...], profile: str) -> None:
        self._state["mode"] = "moving_joint_trajectory"
        self._state["last_command"] = {
            "type": "move_joint_trajectory",
            "point_count": len(positions),
            "profile": profile,
        }
        time.sleep(0.05)
        if positions:
            self._state["joints"] = tuple(positions[-1])
        self._state["mode"] = "idle"

    def movel(self, target: Dict[str, Any], profile: str) -> None:
        self._state["mode"] = "moving_cartesian"
        self._state["last_command"] = {
            "type": "movel",
            "target": target,
            "profile": profile,
        }
        time.sleep(0.05)
        self._state["tcp_pose"] = dict(target)
        self._state["mode"] = "idle"

    def move_tcp_ik(
        self,
        target: Dict[str, Any],
        profile: str,
        seed_joints: Tuple[float, ...] | None = None,
        preferred_joints: Tuple[float, ...] | None = None,
        position_tolerance_m: float = 0.002,
        orientation_tolerance_deg: float = 2.0,
        approximate_position_tolerance_m: float = 0.015,
        approximate_orientation_tolerance_deg: float = 3.0,
        max_iterations: int = 120,
    ) -> Dict[str, Any]:
        self._state["mode"] = "moving_ik"
        self._state["last_command"] = {
            "type": "move_tcp_ik",
            "target": target,
            "profile": profile,
            "seed_joints": seed_joints,
            "preferred_joints": preferred_joints,
            "position_tolerance_m": position_tolerance_m,
            "orientation_tolerance_deg": orientation_tolerance_deg,
            "approximate_position_tolerance_m": approximate_position_tolerance_m,
            "approximate_orientation_tolerance_deg": approximate_orientation_tolerance_deg,
            "max_iterations": max_iterations,
        }
        time.sleep(0.05)
        self._state["tcp_pose"] = dict(target)
        self._state["mode"] = "idle"
        return {
            "status": "simulated",
            "motion": "joint",
            "joints": list(preferred_joints or seed_joints or ()),
            "position_error_m": 0.0,
            "orientation_error_deg": 0.0,
        }

    def servo_tcp(self, target: Dict[str, Any], profile: str = "normal") -> None:
        # Best-effort servo stub: apply target directly.
        self._state["mode"] = "servo"
        self._state["last_command"] = {
            "type": "servo_tcp",
            "target": target,
            "profile": profile,
        }
        self._state["tcp_pose"] = dict(target)

    def servo_tcp_velocity(
        self, velocity: Dict[str, Any], profile: str = "normal"
    ) -> None:
        self._state["mode"] = "servo_velocity"
        self._state["last_command"] = {
            "type": "servo_tcp_velocity",
            "velocity": velocity,
            "profile": profile,
        }

    def freedrive(self, enable: bool) -> None:
        self._state["freedrive"] = bool(enable)
        self._state["last_command"] = {"type": "freedrive", "enable": bool(enable)}

    def open_gripper(
        self,
        width_m: float | None = None,
        force_n: float | None = None,
    ) -> None:
        self._state["gripper_open"] = True
        self._state["last_command"] = {
            "type": "open_gripper",
            "width_m": width_m,
            "force_n": force_n,
        }

    def close_gripper(
        self,
        width_m: float | None = None,
        force_n: float | None = None,
    ) -> None:
        self._state["gripper_open"] = False
        self._state["last_command"] = {
            "type": "close_gripper",
            "width_m": width_m,
            "force_n": force_n,
        }

    def set_mode(self, mode: int) -> None:
        self._state["last_command"] = {"type": "set_mode", "mode": int(mode)}

    def set_state(self, state: int) -> None:
        self._state["last_command"] = {"type": "set_state", "state": int(state)}

    def stop(self) -> None:
        self._state["mode"] = "stopped"
        self._state["last_command"] = {"type": "stop"}
