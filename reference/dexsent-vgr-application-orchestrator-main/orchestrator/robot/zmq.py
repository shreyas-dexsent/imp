"""Implementation for `orchestrator.robot.zmq`."""

import json
import threading
import time
from typing import Any, Dict, Optional, Tuple

import zmq


class ZmqRobotAdapter:
    """Thin transport adapter between orchestrator and robot-controller.

    Design intent:
    - Keep orchestrator free of vendor SDK details.
    - Send command RPCs over ZMQ REQ/REP.
    - Optionally subscribe to async robot state telemetry over ZMQ PUB/SUB.
    """

    def __init__(
        self,
        command_endpoint: str,
        state_endpoint: Optional[str] = None,
        timeout_ms: int = 2000,
        connect_on_init: bool = True,
    ):
        self._command_endpoint = command_endpoint
        self._timeout_ms = timeout_ms
        self._probe_timeout_ms = min(250, max(50, int(timeout_ms)))
        self._probe_interval_s = 1.0
        self._last_probe_monotonic = 0.0
        self.ctx = zmq.Context.instance()
        # REQ socket is used for all command/response transactions.
        self.req = self._build_req_socket()
        self._req_lock = threading.Lock()

        self._state = {
            "connected": False,
            "mode": "unknown",
            "tcp_pose": None,
            "q": [],
            "dq": [],
            "active_motion_id": None,
            "last_command": None,
            "last_error": None,
        }

        self._state_sock = None
        self._state_thread = None
        self._state_running = False

        if state_endpoint:
            # Optional state stream used by UI/logic that polls get_state().
            self._state_sock = self.ctx.socket(zmq.SUB)
            self._state_sock.connect(state_endpoint)
            self._state_sock.setsockopt_string(zmq.SUBSCRIBE, "robot")
            self._state_running = True
            self._state_thread = threading.Thread(
                target=self._state_loop, daemon=True
            )
            self._state_thread.start()

        # Initial best-effort connect handshake with robot-controller.
        if connect_on_init:
            resp = self._send({"cmd": "CONNECT"})
            if resp.get("ok", False):
                self._state["connected"] = True
            else:
                self._state["last_error"] = resp.get("reason") or resp.get("message")

    def _build_req_socket(self) -> zmq.Socket:
        sock = self.ctx.socket(zmq.REQ)
        sock.connect(self._command_endpoint)
        sock.setsockopt(zmq.RCVTIMEO, self._timeout_ms)
        sock.setsockopt(zmq.SNDTIMEO, self._timeout_ms)
        sock.setsockopt(zmq.LINGER, 0)
        return sock

    def _fast_probe_connect(self) -> bool:
        now = time.monotonic()
        if (now - self._last_probe_monotonic) < self._probe_interval_s:
            return bool(self._state.get("connected"))
        self._last_probe_monotonic = now
        sock = self.ctx.socket(zmq.REQ)
        try:
            sock.connect(self._command_endpoint)
            sock.setsockopt(zmq.RCVTIMEO, self._probe_timeout_ms)
            sock.setsockopt(zmq.SNDTIMEO, self._probe_timeout_ms)
            sock.setsockopt(zmq.LINGER, 0)
            sock.send_string(json.dumps({"cmd": "CONNECT"}, separators=(",", ":")))
            raw = sock.recv_string()
            resp = json.loads(raw)
        except Exception:
            self._state["connected"] = False
            self._state["last_error"] = "robot_not_connected"
            return False
        finally:
            try:
                sock.close(0)
            except Exception:
                pass
        if resp.get("ok", False):
            self._state["connected"] = True
            self._state["last_error"] = None
            return True
        self._state["connected"] = False
        self._state["last_error"] = resp.get("reason") or resp.get("message") or "robot_not_connected"
        return False

    def _reset_req_socket(self) -> None:
        try:
            self.req.close()
        except zmq.ZMQError:
            pass
        self.req = self._build_req_socket()

    def _state_loop(self) -> None:
        """Continuously merge published robot state into local cache.

        This loop is intentionally permissive:
        - malformed frames are skipped,
        - unknown events are ignored,
        - latest valid state always wins.
        """
        poller = zmq.Poller()
        poller.register(self._state_sock, zmq.POLLIN)
        while self._state_running and self._state_sock is not None:
            try:
                events = dict(poller.poll(200))
            except zmq.ZMQError:
                if not self._state_running:
                    break
                continue
            if self._state_sock not in events:
                continue
            try:
                raw = self._state_sock.recv_string()
            except zmq.ZMQError:
                if not self._state_running:
                    break
                continue
            if " " not in raw:
                continue
            _, payload = raw.split(" ", 1)
            try:
                data = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if data.get("event") != "ROBOT_STATE":
                continue
            merged = dict(self._state)
            merged.update(data)
            mode = str(merged.get("mode", "")).upper()
            if mode:
                merged["connected"] = mode != "DISCONNECTED"
            self._state = merged

    def _send(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Send one command over REQ/REP with recovery on socket-level failures."""
        with self._req_lock:
            try:
                self.req.send_string(
                    json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
                )
            except zmq.ZMQError as exc:
                self._reset_req_socket()
                self._state["connected"] = False
                self._state["last_error"] = str(exc)
                return {"ok": False, "message": "send_failed", "reason": str(exc)}
            try:
                resp_raw = self.req.recv_string()
            except zmq.Again:
                self._reset_req_socket()
                self._state["connected"] = False
                self._state["last_error"] = "no_reply"
                return {"ok": False, "message": "timeout", "reason": "no_reply"}
            except zmq.ZMQError as exc:
                self._reset_req_socket()
                self._state["connected"] = False
                self._state["last_error"] = str(exc)
                return {"ok": False, "message": "recv_failed", "reason": str(exc)}
        try:
            return json.loads(resp_raw)
        except json.JSONDecodeError:
            self._state["last_error"] = "bad_response"
            return {"ok": False, "message": "reject", "reason": "bad_response"}

    def _ensure_connected(self) -> None:
        """Reconnect lazily before motion/gripper commands."""
        if self._state.get("connected"):
            return
        resp = self._send({"cmd": "CONNECT"})
        if resp.get("ok", False):
            self._state["connected"] = True
            self._state["last_error"] = None
            return
        self._state["connected"] = False
        self._state["last_error"] = resp.get("reason") or resp.get("message")
        raise RuntimeError(self._state["last_error"] or "robot_not_connected")

    def set_robot_model(self, model: str) -> None:
        self._state["last_command"] = {"type": "set_robot_model", "model": model}

    def get_state(self) -> Dict[str, Any]:
        # Fast path: if a PUB/SUB state stream is active the _state_loop keeps
        # self._state fresh at the publisher's rate (typically 20–50 Hz).
        # Avoid a REQ/REP round-trip — it crosses to the robot controller over
        # TCP and on a real FR3 each trip calls robot.current_joint_state which
        # itself hits libfranka over Ethernet, adding 10–50 ms of latency.
        if self._state_sock is not None:
            if not self._state.get("connected"):
                self._fast_probe_connect()
            return dict(self._state)
        # No state subscription: fall back to REQ/REP.
        if not self._state.get("connected") and not self._fast_probe_connect():
            return dict(self._state)
        resp = self._send({"cmd": "GET_STATE"})
        if resp.get("ok", False) and isinstance(resp.get("state"), dict):
            merged = dict(self._state)
            merged.update(resp.get("state") or {})
            self._state = merged
        return dict(self._state)

    # Hard ceiling for blocking motion waits. A real motion that has not
    # cleared after 30 s is a stuck robot, not a long traverse.
    _MOTION_WAIT_TIMEOUT_S = 30.0
    _MOTION_POLL_INTERVAL_S = 0.05
    _MOTION_IDLE_MODES = frozenset({"idle", "stopped", "disconnected"})

    def _wait_for_motion(self, motion_id: str, fail_reason: str) -> None:
        """Block until the dispatched motion_id finishes on the robot server.

        movel/movej return the moment the server acks the command, while the
        robot is still traversing. Downstream code (vision capture, grasp
        planning) assumes the robot has arrived. Poll active_motion_id /
        mode until the server reports the motion is no longer running.
        """
        deadline = time.monotonic() + self._MOTION_WAIT_TIMEOUT_S
        cleared_streak = 0
        while True:
            resp = self._send({"cmd": "GET_STATE"})
            if resp.get("ok", False) and isinstance(resp.get("state"), dict):
                merged = dict(self._state)
                merged.update(resp.get("state") or {})
                self._state = merged
            active = self._state.get("active_motion_id")
            mode = str(self._state.get("mode") or "").strip().lower()
            connected = bool(self._state.get("connected", True))
            if not connected:
                raise RuntimeError("robot_not_connected")
            if active and str(active) != str(motion_id):
                # Server moved on to a different motion already; ours is done.
                return
            if not active or mode in self._MOTION_IDLE_MODES:
                cleared_streak += 1
                # Two consecutive clear polls debounce against a momentary
                # gap between command ack and active_motion_id being set.
                if cleared_streak >= 2:
                    return
            else:
                cleared_streak = 0
            if time.monotonic() >= deadline:
                raise RuntimeError(fail_reason)
            time.sleep(self._MOTION_POLL_INTERVAL_S)

    def movej(self, joints: Tuple[float, ...], profile: str) -> None:
        self._ensure_connected()
        motion_id = f"mj-{time.time_ns()}"
        resp = self._send(
            {
                "cmd": "MOVE_JOINTS",
                "q": list(joints),
                "profile": profile,
                "motion_id": motion_id,
            }
        )
        if not resp.get("ok", False):
            self._state["connected"] = (
                False
                if resp.get("reason") == "robot_not_connected"
                else self._state.get("connected")
            )
            raise RuntimeError(resp.get("reason", "movej_failed"))
        self._state["last_command"] = {"type": "movej", "profile": profile}
        self._wait_for_motion(motion_id, "movej_timeout")

    def movej_path(self, waypoints: Tuple[Tuple[float, ...], ...], profile: str) -> None:
        self._ensure_connected()
        path = [list(q) for q in waypoints if q]
        if not path:
            return
        motion_id = f"mjp-{time.time_ns()}"
        resp = self._send(
            {
                "cmd": "MOVE_JOINT_PATH",
                "q_waypoints": path,
                "profile": profile,
                "motion_id": motion_id,
            }
        )
        if not resp.get("ok", False):
            reason = str(resp.get("reason") or "")
            if reason == "unknown_cmd":
                for q in path:
                    self.movej(tuple(q), profile)
                return
            self._state["connected"] = (
                False
                if resp.get("reason") == "robot_not_connected"
                else self._state.get("connected")
            )
            raise RuntimeError(resp.get("reason", "movej_path_failed"))
        self._state["last_command"] = {
            "type": "movej_path",
            "profile": profile,
            "waypoint_count": len(path),
        }
        self._wait_for_motion(motion_id, "movej_path_timeout")

    def move_joint_waypoints(self, waypoints: Tuple[Tuple[float, ...], ...], profile: str) -> None:
        self._ensure_connected()
        path = [list(q) for q in waypoints if q]
        if not path:
            return
        motion_id = f"mjw-{time.time_ns()}"
        resp = self._send(
            {
                "cmd": "MOVE_JOINT_WAYPOINTS",
                "q_waypoints": path,
                "profile": profile,
                "motion_id": motion_id,
            }
        )
        if not resp.get("ok", False):
            reason = str(resp.get("reason") or "")
            if reason == "unknown_cmd":
                # server doesn't support it — fall back to sequential movej
                for q in path:
                    self.movej(tuple(q), profile)
                return
            self._state["connected"] = (
                False
                if resp.get("reason") == "robot_not_connected"
                else self._state.get("connected")
            )
            raise RuntimeError(resp.get("reason", "move_joint_waypoints_failed"))
        self._state["last_command"] = {
            "type": "move_joint_waypoints",
            "profile": profile,
            "waypoint_count": len(path),
        }
        self._wait_for_motion(motion_id, "move_joint_waypoints_timeout")

    def move_joint_trajectory(self, positions: Tuple[Tuple[float, ...], ...], velocities: Tuple[Tuple[float, ...], ...], profile: str) -> None:
        self._ensure_connected()
        pos_list = [list(q) for q in positions if q]
        vel_list = [list(v) for v in velocities if v]
        if not pos_list:
            return
        motion_id = f"mjt-{time.time_ns()}"
        resp = self._send(
            {
                "cmd": "MOVE_JOINT_TRAJECTORY",
                "positions": pos_list,
                "velocities": vel_list,
                "profile": profile,
                "motion_id": motion_id,
            }
        )
        if not resp.get("ok", False):
            reason = str(resp.get("reason") or "")
            if reason == "unknown_cmd":
                # fallback: sparse waypoint motion
                stride = max(1, len(pos_list) // 8)
                sparse = pos_list[::stride]
                if pos_list[-1] != sparse[-1]:
                    sparse.append(pos_list[-1])
                self.move_joint_waypoints(tuple(tuple(q) for q in sparse), profile)
                return
            raise RuntimeError(resp.get("reason", "move_joint_trajectory_failed"))
        self._state["last_command"] = {
            "type": "move_joint_trajectory",
            "profile": profile,
            "point_count": len(pos_list),
        }
        self._wait_for_motion(motion_id, "move_joint_trajectory_timeout")

    def movel(self, target: Dict[str, Any], profile: str) -> None:
        self._ensure_connected()
        motion_id = f"ml-{time.time_ns()}"
        resp = self._send(
            {
                "cmd": "MOVE_TCP",
                "target": target,
                "profile": profile,
                "motion_id": motion_id,
            }
        )
        if not resp.get("ok", False):
            self._state["connected"] = (
                False
                if resp.get("reason") == "robot_not_connected"
                else self._state.get("connected")
            )
            raise RuntimeError(resp.get("reason", "movel_failed"))
        self._state["last_command"] = {"type": "movel", "profile": profile}
        self._wait_for_motion(motion_id, "movel_timeout")

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
        self._ensure_connected()
        motion_id = f"ik-{time.time_ns()}"
        resp = self._send(
            {
                "cmd": "MOVE_TCP_IK",
                "target": target,
                "profile": profile,
                "seed_joints": list(seed_joints) if seed_joints is not None else None,
                "preferred_joints": list(preferred_joints) if preferred_joints is not None else None,
                "position_tolerance_m": position_tolerance_m,
                "orientation_tolerance_deg": orientation_tolerance_deg,
                "approximate_position_tolerance_m": approximate_position_tolerance_m,
                "approximate_orientation_tolerance_deg": approximate_orientation_tolerance_deg,
                "max_iterations": max_iterations,
                "motion_id": motion_id,
            }
        )
        if not resp.get("ok", False):
            self._state["connected"] = (
                False
                if resp.get("reason") == "robot_not_connected"
                else self._state.get("connected")
            )
            raise RuntimeError(resp.get("reason", "move_tcp_ik_failed"))
        self._state["last_command"] = {"type": "move_tcp_ik", "profile": profile}
        self._wait_for_motion(motion_id, "move_tcp_ik_timeout")
        return dict(resp.get("result") or {})

    def servo_tcp(self, target: Dict[str, Any], profile: str = "normal") -> None:
        self._ensure_connected()
        resp = self._send(
            {"cmd": "SERVO_TCP", "target": target, "motion_id": f"sv-{time.time_ns()}"}
        )
        if not resp.get("ok", False):
            self._state["connected"] = (
                False
                if resp.get("reason") == "robot_not_connected"
                else self._state.get("connected")
            )
            raise RuntimeError(resp.get("reason", "servo_tcp_failed"))
        self._state["last_command"] = {"type": "servo_tcp", "profile": profile}

    def servo_tcp_velocity(
        self, velocity: Dict[str, Any], profile: str = "normal"
    ) -> None:
        self._ensure_connected()
        resp = self._send(
            {
                "cmd": "SERVO_TCP_VEL",
                "velocity": velocity,
                "motion_id": f"svv-{time.time_ns()}",
            }
        )
        if not resp.get("ok", False):
            self._state["connected"] = (
                False
                if resp.get("reason") == "robot_not_connected"
                else self._state.get("connected")
            )
            raise RuntimeError(resp.get("reason", "servo_tcp_vel_failed"))
        self._state["last_command"] = {"type": "servo_tcp_velocity", "profile": profile}

    def set_mode(self, mode: int) -> None:
        self._ensure_connected()
        resp = self._send({"cmd": "SET_MODE", "mode": int(mode)})
        if not resp.get("ok", False):
            raise RuntimeError(resp.get("reason", "set_mode_failed"))
        self._state["last_command"] = {"type": "set_mode", "mode": int(mode)}

    def set_state(self, state: int) -> None:
        self._ensure_connected()
        resp = self._send({"cmd": "SET_STATE", "state": int(state)})
        if not resp.get("ok", False):
            raise RuntimeError(resp.get("reason", "set_state_failed"))
        self._state["last_command"] = {"type": "set_state", "state": int(state)}

    def freedrive(self, enable: bool) -> None:
        self._state["last_command"] = {"type": "freedrive", "enable": bool(enable)}

    def open_gripper(
        self,
        width_m: float | None = None,
        force_n: float | None = None,
    ) -> None:
        payload = {"cmd": "GRIPPER_OPEN"}
        if width_m is not None:
            payload["width_m"] = float(width_m)
        if force_n is not None:
            payload["force_n"] = float(force_n)
        resp = self._send(payload)
        if not resp.get("ok", False):
            raise RuntimeError(resp.get("reason", "gripper_open_failed"))
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
        payload = {"cmd": "GRIPPER_CLOSE"}
        if width_m is not None:
            payload["width_m"] = float(width_m)
        if force_n is not None:
            payload["force_n"] = float(force_n)
        resp = self._send(payload)
        if not resp.get("ok", False):
            raise RuntimeError(resp.get("reason", "gripper_close_failed"))
        self._state["last_command"] = {
            "type": "close_gripper",
            "width_m": width_m,
            "force_n": force_n,
        }

    def stop(self) -> None:
        self._send({"cmd": "STOP"})
        self._state["last_command"] = {"type": "stop"}

    def close(self) -> None:
        self._state_running = False
        if self._state_sock is not None:
            try:
                self._state_sock.close(0)
            except Exception:
                pass
        if self._state_thread is not None and self._state_thread.is_alive():
            self._state_thread.join(timeout=1.0)
        try:
            self.req.close(0)
        except Exception:
            pass
