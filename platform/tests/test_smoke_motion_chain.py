"""Smoke gate: a synthetic perception pose is lifted, solved, and planned
end-to-end on the live bus.

This is the P4 required check (PLAN.md §4): one test, multiple modules
chained, prove the platform is more than a collection of independently
verifiable pieces. Stripped to the bare minimum to keep CI fast:

    synthetic Pose6D[camera]               (test publishes)
        --> spatial-transform              (Pose6D[cam] + tf -> PoseTarget[base])
            --> motion-pinocchio IK        (PoseTarget + RobotState -> JointSolution)
                --> motion-ompl plan       (RobotState + JointSolution -> Path)

The chain is run in a single process: each module gets its own
``ModuleNode`` thread, the test pumps inputs and asserts the final
``Path`` waypoints connect the start configuration to the IK solution.

Gated on ``pin + coal + ompl + zenoh`` (the bus stack + motion-core
native libs); skips cleanly on hosts without them.
"""

from __future__ import annotations

import contextlib
import threading
import time
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("zenoh")
pytest.importorskip("pinocchio")
pytest.importorskip("ompl")

from algorithms.descriptions import RobotSystemDescription, WorldDescription  # noqa: E402
from algorithms.kinematics import fk_local  # noqa: E402
from algorithms.resolved import KinematicModel  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402

from imp_sdk import Bus, QosClass, keyexpr  # noqa: E402
from imp_sdk.module import ModuleNode  # noqa: E402
from imp_sdk.schemas import imp_pb2  # noqa: E402


PLATFORM = Path(__file__).resolve().parents[1]
WORLDS = PLATFORM / "modules" / "motion-core" / "algorithms" / "configs" / "worlds"
ROBOTS = PLATFORM / "modules" / "motion-core" / "algorithms" / "configs" / "robots"

STATION, ROBOT = "smoke", "fr3"


@contextlib.contextmanager
def _running(*modules):
    """Spin every module in a daemon thread for the test body; clean up after."""
    nodes = [ModuleNode(m) for m in modules]
    threads = [threading.Thread(target=n.run, daemon=True) for n in nodes]
    for t in threads:
        t.start()
    try:
        time.sleep(0.6)  # let Zenoh discovery settle for every subscription
        yield
    finally:
        for n in nodes:
            n.stop()
        for t in threads:
            t.join(timeout=2.0)


def test_synthetic_chain_perception_to_planned_path():
    from imp_module_motion_ompl.plan import PlanModule
    from imp_module_motion_pinocchio.ik import IkModule
    from imp_module_spatial_transform import TransformModule

    # ------------------------------------------------------------------
    # Setup: pick a reachable target by FK-ing a perturbed home pose, then
    # encode it as a Pose6D in the camera frame plus a static base<-camera
    # hand-eye edge so spatial-transform can lift it.
    # ------------------------------------------------------------------
    rs_path = ROBOTS / "franka_fr3_with_franka_hand.yaml"
    world_path = WORLDS / "franka_robot_only_world.yaml"

    system = RobotSystemDescription.from_yaml(rs_path)
    model = KinematicModel.from_robot_system(system)
    spec = model.chain("arm")
    tcp = spec.tcp_frame or spec.tip_frame

    home = system.named_joint_state("home")
    q_home = np.array([home[n] for n in model.active_joint_names], dtype=float)

    q_target = q_home.copy()
    q_target[0] += 0.10
    T_base_tcp_target = fk_local(model, q_target, tcp)
    target_pos_base = T_base_tcp_target[:3, 3]

    # Hand-eye: camera offset from the robot base. The Pose6D the
    # perception side publishes is expressed in the camera frame; subtract
    # the offset so spatial-transform lifts it back to ``target_pos_base``.
    T_base_camera = np.eye(4)
    T_base_camera[:3, 3] = (0.5, 0.0, 0.5)
    T_camera_base = np.eye(4)
    T_camera_base[:3, :3] = T_base_camera[:3, :3].T
    T_camera_base[:3, 3] = -T_base_camera[:3, :3].T @ T_base_camera[:3, 3]

    pos_in_camera = (T_camera_base @ np.append(target_pos_base, 1.0))[:3]
    quat_target = Rotation.from_matrix(T_base_tcp_target[:3, :3]).as_quat()

    # ------------------------------------------------------------------
    # Wire the three modules.
    # ------------------------------------------------------------------
    pose_key = f"imp/{STATION}/perc/s1/pose"
    transform = TransformModule(
        station=STATION,
        pose_key=pose_key,
        out_plan="ik",
        base_frame=system.robot.base_frame,
    )
    ik = IkModule(
        station=STATION,
        robot=ROBOT,
        robot_system_path=str(rs_path),
    )
    plan = PlanModule(
        station=STATION,
        robot=ROBOT,
        world_path=str(world_path),
        random_seed=0,
    )

    with _running(transform, ik, plan):
        bus = Bus.open()
        try:
            path_sub = bus.subscribe(keyexpr.motion(STATION, "plan", "path"), imp_pb2.Path)

            # Robot state -- needed by IK as seed and by the planner as start.
            state = imp_pb2.RobotState(
                header=imp_pb2.Header(schema="imp.RobotState/1"),
                q=q_home.tolist(),
                mode="idle",
            )
            bus.put(keyexpr.hal(STATION, ROBOT, "state"), state, QosClass.STATE)
            time.sleep(0.05)

            # Hand-eye edge for spatial-transform.
            bus.put(
                keyexpr.tf(STATION),
                imp_pb2.TfEdge(
                    header=imp_pb2.Header(schema="imp.TfEdge/1"),
                    parent_frame=system.robot.base_frame,
                    child_frame="camera",
                    matrix=T_base_camera.flatten().tolist(),
                ),
                QosClass.STATE,
            )
            time.sleep(0.05)

            # The "perception" pose: the target in the camera frame.
            bus.put(
                pose_key,
                imp_pb2.Pose6D(
                    header=imp_pb2.Header(schema="imp.Pose6D/1", frame_id="camera"),
                    object_id="target",
                    position_m=pos_in_camera.tolist(),
                    quat_xyzw=quat_target.tolist(),
                    confidence=1.0,
                    valid=True,
                ),
                QosClass.STATE,
            )

            # The planner sometimes needs a second state nudge after the
            # goal lands to retrigger its compute().
            for _ in range(5):
                bus.put(keyexpr.hal(STATION, ROBOT, "state"), state, QosClass.STATE)
                time.sleep(0.2)

            path = path_sub.recv()
        finally:
            bus.close()

    # ------------------------------------------------------------------
    # Assertions: the final Path must connect start to a configuration
    # whose FK reaches the same TCP target.
    # ------------------------------------------------------------------
    assert path.n_dof == len(model.active_joint_names), \
        "planner produced a path with mismatched DOF"
    wp = np.array(path.q_wp).reshape(-1, path.n_dof)
    assert wp.shape[0] >= 2, "planner produced fewer than 2 waypoints"
    assert np.allclose(wp[0], q_home, atol=5e-3), "path doesn't start at home"

    # End-effector at the final waypoint should land near the target.
    T_final = fk_local(model, wp[-1], tcp)
    pos_err = float(np.linalg.norm(T_final[:3, 3] - target_pos_base))
    assert pos_err < 5e-3, f"final TCP {pos_err:.4f} m from target -- chain broke"
