"""Null robot adapter used for orchestrator vision-only mode."""

from typing import Any, Dict, Tuple


class NullRobotAdapter:
    def __init__(self):
        self._state = {
            "connected": False,
            "mode": "disabled",
            "tcp_pose": {
                "position_m": [0.0, 0.0, 0.0],
                "quat_xyzw": [0.0, 0.0, 0.0, 1.0],
                "frame": "base",
            },
            "q": [],
            "dq": [],
            "joints": [],
            "gripper_open": True,
            "freedrive": False,
            "robot_disabled": True,
            "last_command": None,
            "last_error": "robot_disabled",
        }

    def _record(self, command_type: str, **fields: Any) -> None:
        payload = {"type": command_type}
        payload.update(fields)
        self._state["last_command"] = payload

    def set_robot_model(self, model: str) -> None:
        self._record("set_robot_model", model=model)

    def get_state(self) -> Dict[str, Any]:
        return dict(self._state)

    def movej(self, joints: Tuple[float, ...], profile: str) -> None:
        self._record("movej_skipped", joints=list(joints), profile=profile)

    def movej_path(self, waypoints: Tuple[Tuple[float, ...], ...], profile: str) -> None:
        self._record("movej_path_skipped", waypoints=[list(q) for q in waypoints], profile=profile)

    def move_joint_waypoints(self, waypoints: Tuple[Tuple[float, ...], ...], profile: str) -> None:
        self._record("move_joint_waypoints_skipped", waypoints=[list(q) for q in waypoints], profile=profile)

    def move_joint_trajectory(self, positions: Tuple[Tuple[float, ...], ...], velocities: Tuple[Tuple[float, ...], ...], profile: str) -> None:
        self._record("move_joint_trajectory_skipped", point_count=len(positions), profile=profile)

    def movel(self, target: Dict[str, Any], profile: str) -> None:
        self._record("movel_skipped", target=dict(target), profile=profile)

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
        payload = {
            "target": dict(target),
            "profile": profile,
            "seed_joints": list(seed_joints) if seed_joints is not None else None,
            "preferred_joints": list(preferred_joints) if preferred_joints is not None else None,
            "position_tolerance_m": position_tolerance_m,
            "orientation_tolerance_deg": orientation_tolerance_deg,
            "approximate_position_tolerance_m": approximate_position_tolerance_m,
            "approximate_orientation_tolerance_deg": approximate_orientation_tolerance_deg,
            "max_iterations": max_iterations,
        }
        self._record("move_tcp_ik_skipped", **payload)
        return {"status": "skipped", "reason": "robot_disabled", **payload}

    def servo_tcp(self, target: Dict[str, Any], profile: str = "normal") -> None:
        self._record("servo_tcp_skipped", target=dict(target), profile=profile)

    def servo_tcp_velocity(
        self, velocity: Dict[str, Any], profile: str = "normal"
    ) -> None:
        self._record(
            "servo_tcp_velocity_skipped",
            velocity=dict(velocity),
            profile=profile,
        )

    def set_mode(self, mode: int) -> None:
        self._record("set_mode_skipped", mode=int(mode))

    def set_state(self, state: int) -> None:
        self._record("set_state_skipped", state=int(state))

    def freedrive(self, enable: bool) -> None:
        self._record("freedrive_skipped", enable=bool(enable))

    def open_gripper(
        self,
        width_m: float | None = None,
        force_n: float | None = None,
    ) -> None:
        self._record("open_gripper_skipped", width_m=width_m, force_n=force_n)

    def close_gripper(
        self,
        width_m: float | None = None,
        force_n: float | None = None,
    ) -> None:
        self._record("close_gripper_skipped", width_m=width_m, force_n=force_n)

    def stop(self) -> None:
        self._record("stop_skipped")

    def close(self) -> None:
        return None
