"""Implementation for `orchestrator.robot.base`."""

from typing import Any, Dict, Protocol, Tuple


class RobotAdapter(Protocol):
    def get_state(self) -> Dict[str, Any]: ...

    def set_robot_model(self, model: str) -> None: ...

    def movej(self, joints: Tuple[float, ...], profile: str) -> None: ...

    def movej_path(
        self,
        waypoints: Tuple[Tuple[float, ...], ...],
        profile: str,
    ) -> None: ...

    def move_joint_waypoints(
        self,
        waypoints: Tuple[Tuple[float, ...], ...],
        profile: str,
    ) -> None: ...

    def move_joint_trajectory(
        self,
        positions: Tuple[Tuple[float, ...], ...],
        velocities: Tuple[Tuple[float, ...], ...],
        profile: str,
    ) -> None: ...

    def movel(self, target: Dict[str, Any], profile: str) -> None: ...

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
    ) -> Dict[str, Any]: ...

    def servo_tcp(self, target: Dict[str, Any], profile: str = "normal") -> None: ...

    def servo_tcp_velocity(
        self, velocity: Dict[str, Any], profile: str = "normal"
    ) -> None: ...

    def set_mode(self, mode: int) -> None: ...

    def set_state(self, state: int) -> None: ...

    def freedrive(self, enable: bool) -> None: ...

    def open_gripper(
        self,
        width_m: float | None = None,
        force_n: float | None = None,
    ) -> None: ...

    def close_gripper(
        self,
        width_m: float | None = None,
        force_n: float | None = None,
    ) -> None: ...

    def stop(self) -> None: ...
