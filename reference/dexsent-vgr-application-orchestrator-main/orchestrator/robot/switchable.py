"""Robot adapter wrapper that can switch to a null robot at runtime."""

from typing import Any, Dict, Tuple


class SwitchableRobotAdapter:
    def __init__(
        self,
        primary: Any,
        fallback: Any,
        runtime_state: Dict[str, Any],
    ):
        self._primary = primary
        self._fallback = fallback
        self._runtime_state = runtime_state

    def _enabled(self) -> bool:
        return bool(self._runtime_state.get("robot_enabled", True))

    def _active(self) -> Any:
        return self._primary if self._enabled() else self._fallback

    def set_robot_model(self, model: str) -> None:
        self._active().set_robot_model(model)

    def get_state(self) -> Dict[str, Any]:
        state = self._active().get_state() or {}
        merged = dict(state)
        merged["robot_disabled"] = not self._enabled()
        merged["runtime_robot_enabled"] = self._enabled()
        merged["runtime_robot_mode"] = "enabled" if self._enabled() else "disabled"
        if self._enabled() and "connected" not in merged:
            merged["connected"] = True
        if not self._enabled():
            merged["connected"] = False
            merged["mode"] = "disabled"
        return merged

    def movej(self, joints: Tuple[float, ...], profile: str) -> None:
        self._active().movej(joints, profile)

    def movej_path(self, waypoints: Tuple[Tuple[float, ...], ...], profile: str) -> None:
        active = self._active()
        if hasattr(active, "movej_path"):
            active.movej_path(waypoints, profile)
            return
        for joints in waypoints:
            active.movej(joints, profile)

    def move_joint_waypoints(self, waypoints: Tuple[Tuple[float, ...], ...], profile: str) -> None:
        active = self._active()
        if hasattr(active, "move_joint_waypoints"):
            active.move_joint_waypoints(waypoints, profile)
            return
        for joints in waypoints:
            active.movej(joints, profile)

    def move_joint_trajectory(self, positions: Tuple[Tuple[float, ...], ...], velocities: Tuple[Tuple[float, ...], ...], profile: str) -> None:
        active = self._active()
        if hasattr(active, "move_joint_trajectory"):
            active.move_joint_trajectory(positions, velocities, profile)
            return
        if hasattr(active, "move_joint_waypoints"):
            stride = max(1, len(positions) // 8)
            sparse = positions[::stride]
            if positions[-1] != sparse[-1]:
                sparse = sparse + (positions[-1],)
            active.move_joint_waypoints(sparse, profile)
            return
        if positions:
            active.movej(positions[-1], profile)

    def movel(self, target: Dict[str, Any], profile: str) -> None:
        self._active().movel(target, profile)

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
        return self._active().move_tcp_ik(
            target,
            profile,
            seed_joints=seed_joints,
            preferred_joints=preferred_joints,
            position_tolerance_m=position_tolerance_m,
            orientation_tolerance_deg=orientation_tolerance_deg,
            approximate_position_tolerance_m=approximate_position_tolerance_m,
            approximate_orientation_tolerance_deg=approximate_orientation_tolerance_deg,
            max_iterations=max_iterations,
        )

    def servo_tcp(self, target: Dict[str, Any], profile: str = "normal") -> None:
        self._active().servo_tcp(target, profile)

    def servo_tcp_velocity(
        self, velocity: Dict[str, Any], profile: str = "normal"
    ) -> None:
        self._active().servo_tcp_velocity(velocity, profile)

    def set_mode(self, mode: int) -> None:
        self._active().set_mode(mode)

    def set_state(self, state: int) -> None:
        self._active().set_state(state)

    def freedrive(self, enable: bool) -> None:
        self._active().freedrive(enable)

    def open_gripper(
        self,
        width_m: float | None = None,
        force_n: float | None = None,
    ) -> None:
        self._active().open_gripper(width_m, force_n)

    def close_gripper(
        self,
        width_m: float | None = None,
        force_n: float | None = None,
    ) -> None:
        self._active().close_gripper(width_m, force_n)

    def stop(self) -> None:
        self._active().stop()

    def close(self) -> None:
        close_fn = getattr(self._primary, "close", None)
        if callable(close_fn):
            close_fn()
