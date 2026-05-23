from typing import Any, Dict, Optional

from robot_controller.core.adapter_base import RobotAdapter
from robot_controller.core.models import PoseSE3, TcpIkRequest


class RobotController:
    def __init__(self, adapter: RobotAdapter):
        self.adapter = adapter
        self.connected = False

    def connect(self) -> None:
        if self.connected:
            return
        self.adapter.connect()
        self.connected = True

    def disconnect(self) -> None:
        if not self.connected:
            return
        self.adapter.disconnect()
        self.connected = False

    def get_state(self) -> Dict[str, Any]:
        return self.adapter.get_state()

    def handle_command(self, cmd: Dict[str, Any]) -> Dict[str, Any]:
        name = cmd.get("cmd")
        motion_id = cmd.get("motion_id", "")
        try:
            if name == "CONNECT":
                self.connect()
                return {"ok": True, "message": "connected"}

            if name == "GET_STATE":
                state = self.get_state()
                return {"ok": True, "message": "state", "state": state}

            if name == "DISCONNECT":
                self.disconnect()
                return {"ok": True, "message": "disconnected"}

            if name == "MOVE_JOINTS":
                q = cmd.get("q")
                if not isinstance(q, list):
                    return {"ok": False, "message": "reject", "reason": "missing_q"}
                self.adapter.move_joints(q, motion_id, str(cmd.get("profile", "normal")))
                return {"ok": True, "message": "accepted", "motion_id": motion_id}

            if name == "MOVE_JOINT_PATH":
                q_waypoints = cmd.get("q_waypoints")
                if not isinstance(q_waypoints, list) or not q_waypoints:
                    return {"ok": False, "message": "reject", "reason": "missing_q_waypoints"}
                if hasattr(self.adapter, "move_joint_path"):
                    self.adapter.move_joint_path(
                        q_waypoints,
                        motion_id,
                        str(cmd.get("profile", "normal")),
                    )
                else:
                    for q in q_waypoints:
                        self.adapter.move_joints(q, motion_id, str(cmd.get("profile", "normal")))
                return {
                    "ok": True,
                    "message": "accepted",
                    "motion_id": motion_id,
                    "waypoint_count": len(q_waypoints),
                }

            if name == "MOVE_JOINT_TRAJECTORY":
                positions = cmd.get("positions")
                velocities = cmd.get("velocities") or []
                if not isinstance(positions, list) or not positions:
                    return {"ok": False, "message": "reject", "reason": "missing_positions"}
                if hasattr(self.adapter, "move_joint_trajectory"):
                    self.adapter.move_joint_trajectory(
                        positions,
                        velocities,
                        motion_id,
                        str(cmd.get("profile", "normal")),
                    )
                else:
                    # fallback: sparse waypoint motion
                    stride = max(1, len(positions) // 8)
                    sparse = positions[::stride]
                    if positions[-1] != sparse[-1]:
                        sparse.append(positions[-1])
                    for q in sparse:
                        self.adapter.move_joints(q, motion_id, str(cmd.get("profile", "normal")))
                return {
                    "ok": True,
                    "message": "accepted",
                    "motion_id": motion_id,
                    "point_count": len(positions),
                }

            if name == "MOVE_JOINT_WAYPOINTS":
                q_waypoints = cmd.get("q_waypoints")
                if not isinstance(q_waypoints, list) or not q_waypoints:
                    return {"ok": False, "message": "reject", "reason": "missing_q_waypoints"}
                if hasattr(self.adapter, "move_joint_waypoints"):
                    self.adapter.move_joint_waypoints(
                        q_waypoints,
                        motion_id,
                        str(cmd.get("profile", "normal")),
                    )
                else:
                    # fallback: sequential moves
                    for q in q_waypoints:
                        self.adapter.move_joints(q, motion_id, str(cmd.get("profile", "normal")))
                return {
                    "ok": True,
                    "message": "accepted",
                    "motion_id": motion_id,
                    "waypoint_count": len(q_waypoints),
                }

            if name == "MOVE_TCP":
                target = cmd.get("target") or {}
                pose = PoseSE3(
                    position_m=target.get("position_m", [0.0, 0.0, 0.0]),
                    quat_xyzw=target.get("quat_xyzw", [0.0, 0.0, 0.0, 1.0]),
                    frame=target.get("frame", "base"),
                )
                self.adapter.move_tcp(pose, motion_id, str(cmd.get("profile", "normal")))
                return {"ok": True, "message": "accepted", "motion_id": motion_id}

            if name == "MOVE_TCP_IK":
                target = cmd.get("target") or {}
                pose = PoseSE3(
                    position_m=target.get("position_m", [0.0, 0.0, 0.0]),
                    quat_xyzw=target.get("quat_xyzw", [0.0, 0.0, 0.0, 1.0]),
                    frame=target.get("frame", "base"),
                )
                request = TcpIkRequest(
                    target=pose,
                    seed_joints=cmd.get("seed_joints"),
                    preferred_joints=cmd.get("preferred_joints"),
                    position_tolerance_m=float(cmd.get("position_tolerance_m", 0.002)),
                    orientation_tolerance_deg=float(cmd.get("orientation_tolerance_deg", 2.0)),
                    approximate_position_tolerance_m=float(
                        cmd.get("approximate_position_tolerance_m", 0.015)
                    ),
                    approximate_orientation_tolerance_deg=float(
                        cmd.get("approximate_orientation_tolerance_deg", 3.0)
                    ),
                    max_iterations=int(cmd.get("max_iterations", 120)),
                )
                result = self.adapter.move_tcp_ik(
                    request, motion_id, str(cmd.get("profile", "normal"))
                )
                return {
                    "ok": True,
                    "message": "accepted",
                    "motion_id": motion_id,
                    "result": result,
                }

            if name == "GRIPPER_OPEN":
                width_m = cmd.get("width_m")
                force_n = cmd.get("force_n")
                self.adapter.open_gripper(
                    None if width_m is None else float(width_m),
                    None if force_n is None else float(force_n),
                )
                return {"ok": True, "message": "accepted"}

            if name == "GRIPPER_CLOSE":
                width_m = cmd.get("width_m")
                force_n = cmd.get("force_n")
                self.adapter.close_gripper(
                    None if width_m is None else float(width_m),
                    None if force_n is None else float(force_n),
                )
                return {"ok": True, "message": "accepted"}

            if name == "FREEDRIVE":
                enabled = bool(cmd.get("enable", False))
                self.adapter.freedrive(enabled)
                return {"ok": True, "message": "accepted", "enabled": enabled}

            if name == "STOP":
                self.adapter.stop()
                return {"ok": True, "message": "stopped"}

            return {"ok": False, "message": "reject", "reason": "unknown_cmd"}
        except Exception as exc:
            if name == "CONNECT":
                self.connected = False
            return {"ok": False, "message": "reject", "reason": str(exc)}
