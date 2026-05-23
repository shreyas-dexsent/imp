"""Implementation for `robot_controller.core.adapter_base`."""

from typing import Any, Dict, List, Protocol

from robot_controller.core.models import CartesianVelocity, PoseSE3, TcpIkRequest


class RobotAdapter(Protocol):
    def connect(self) -> None: ...

    def disconnect(self) -> None: ...

    def get_state(self) -> Dict[str, Any]: ...

    def move_joints(self, q: List[float], motion_id: str, profile: str = "normal") -> None: ...

    def move_joint_path(
        self,
        q_waypoints: List[List[float]],
        motion_id: str,
        profile: str = "normal",
    ) -> None: ...

    def move_tcp(self, target: PoseSE3, motion_id: str, profile: str = "normal") -> None: ...

    def move_tcp_ik(
        self, request: TcpIkRequest, motion_id: str, profile: str = "normal"
    ) -> Dict[str, Any]: ...

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

    def freedrive(self, enable: bool) -> None: ...

    def set_mode(self, mode: int) -> None: ...

    def set_state(self, state: int) -> None: ...

    def servo_tcp(self, target: PoseSE3, motion_id: str) -> None: ...

    def servo_tcp_velocity(
        self, velocity: CartesianVelocity, motion_id: str
    ) -> None: ...

    def stop(self) -> None: ...
