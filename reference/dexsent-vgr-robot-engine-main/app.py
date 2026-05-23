import argparse
import threading
import time
from contextlib import nullcontext
from typing import Any, Dict

from robot_controller.config import load_config
from robot_controller.core.controller import RobotController
from robot_controller.io.command_plane.server import CommandServer
from robot_controller.io.state_plane.publisher import StatePublisher
from robot_controller.logging import setup_logging
from robot_controller.logging import get_logger

log = get_logger("app")


def build_adapter(cfg: Dict[str, Any]):
    robot_cfg = cfg.get("robot", {})
    motion_cfg = cfg.get("motion", {})
    robot_type = str(robot_cfg.get("type", "mujoco_ur5e")).strip().lower()

    if robot_type == "mujoco_ur5e":
        from robot_controller.adapters.mujoco_ur5e.adapter import Adapter as MujocoUr5eAdapter

        model_path = robot_cfg.get("model_path")
        home_q = robot_cfg.get("home_q")
        base_frame_yaw_deg = robot_cfg.get("base_frame_yaw_deg", 0.0)
        adapter = MujocoUr5eAdapter(
            model_path,
            home_q=home_q,
            motion_cfg=motion_cfg,
            base_frame_yaw_deg=base_frame_yaw_deg,
        )
        return adapter, f"MuJoCo model: {model_path}"

    if robot_type == "franka_fr3":
        from robot_controller.adapters.franka_fr3.adapter import Adapter as FrankaFr3Adapter

        robot_ip = robot_cfg.get("robot_ip")
        adapter = FrankaFr3Adapter(
            robot_ip=robot_ip,
            home_q=robot_cfg.get("home_q"),
            motion_cfg=motion_cfg,
            safety_cfg=cfg.get("safety", {}),
            tool_frame=robot_cfg.get("tool_frame", "fr3_tcp"),
        )
        return adapter, f"Franka FR3 target: {robot_ip}"

    if robot_type in ("xarm", "xarm_lite6", "lite6"):
        from robot_controller.adapters.xarm.adapter import Adapter as XArmAdapter

        host = str(robot_cfg.get("host") or robot_cfg.get("robot_ip") or "").strip()
        port = int(robot_cfg.get("port", 18333))
        adapter = XArmAdapter(
            host=host,
            port=port,
            home_q=robot_cfg.get("home_q"),
            motion_cfg=motion_cfg,
            safety_cfg=cfg.get("safety", {}),
            tool_frame=robot_cfg.get("tool_frame", "xarm_tcp"),
            gripper=str(robot_cfg.get("gripper", "vacuum")),
            joints_in_degrees=bool(robot_cfg.get("joints_in_degrees", False)),
        )
        return adapter, f"xArm target: {host}:{port}"

    raise ValueError(f"unsupported_robot_type:{robot_type}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/robot.local.json")
    args = ap.parse_args()

    cfg = load_config(args.config)
    setup_logging(cfg.get("logging", {}).get("level", "INFO"))
    adapter, startup_label = build_adapter(cfg)
    print(f"[robot-controller] Loaded adapter: {startup_label}")

    controller = RobotController(adapter)

    io_cfg = cfg.get("io", {})
    cmd_cfg = io_cfg.get("command_plane", {})
    state_cfg = io_cfg.get("state_plane", {})

    try:
        cmd_server = CommandServer(cmd_cfg.get("endpoint", "tcp://127.0.0.1:5571"), controller)
    except RuntimeError as e:
        if "command_plane_addr_in_use" in str(e):
            print(str(e))
            return
        raise
    state_pub = StatePublisher(
        state_cfg.get("endpoint", "tcp://127.0.0.1:5572"),
        controller,
        rate_hz=state_cfg.get("rate_hz", 20),
    )

    t_state = threading.Thread(target=state_pub.loop, daemon=True)
    t_state.start()

    viewer_cfg = cfg.get("viewer", {})
    if viewer_cfg.get("enabled", False):
        if not hasattr(adapter, "model") or not hasattr(adapter, "data"):
            log.warning("Viewer requested but adapter does not expose a MuJoCo model; skipping viewer startup")
        else:
            def _viewer_loop():
                import mujoco.viewer

                rate_hz = max(1.0, float(viewer_cfg.get("rate_hz", 30)))
                interval = 1.0 / rate_hz
                step_physics = bool(viewer_cfg.get("step_physics", False))
                lock = getattr(adapter, "mj_lock", None)
                guard = lock if lock is not None else nullcontext()
                with mujoco.viewer.launch_passive(adapter.model, adapter.data) as viewer:
                    while viewer.is_running():
                        with guard:
                            if step_physics:
                                mujoco.mj_step(adapter.model, adapter.data)
                            viewer.sync()
                        time.sleep(interval)

            t_viewer = threading.Thread(target=_viewer_loop, daemon=True)
            t_viewer.start()

    try:
        cmd_server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        cmd_server.stop()
        cmd_server.close()


if __name__ == "__main__":
    main()
