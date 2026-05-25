"""P4 promotion of the per-module ``verify_*.py`` shell scripts into pytest.

One bus round-trip per module: start the module in a daemon thread, feed
its inputs over Zenoh, assert what it publishes matches motion-core's
direct call. Replaces manual ``python examples/verify_*.py`` runs as the
regression gate (debt **D5** in PLAN.md).

Gated with ``pytest.importorskip`` on the heavy native deps (``pinocchio``,
``coal``, ``ompl``, ``ruckig``, ``zenoh``); skips cleanly on hosts that
don't have the dev-container env.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("zenoh")
pytest.importorskip("pinocchio")

from scipy.spatial.transform import Rotation  # noqa: E402

from algorithms.collision import is_in_collision  # noqa: E402
from algorithms.descriptions import RobotSystemDescription, WorldDescription  # noqa: E402
from algorithms.kinematics import fk, fk_local  # noqa: E402
from algorithms.kinematics.ik import IKStatus, ik_local  # noqa: E402
from algorithms.planning import PlanOptions, plan_joint  # noqa: E402
from algorithms.resolved import CollisionModel, KinematicModel, Scene  # noqa: E402

from imp_sdk import QosClass, keyexpr  # noqa: E402
from imp_sdk.schemas import imp_pb2  # noqa: E402
from imp_sdk.testing import module_under_test  # noqa: E402


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

PLATFORM = Path(__file__).resolve().parents[1]
WORLDS = PLATFORM / "modules" / "motion-core" / "algorithms" / "configs" / "worlds"
ROBOTS = PLATFORM / "modules" / "motion-core" / "algorithms" / "configs" / "robots"

STATION, ROBOT = "devstation", "fr3"


def _home_q(robot_system: RobotSystemDescription) -> tuple[KinematicModel, np.ndarray]:
    model = KinematicModel.from_robot_system(robot_system)
    home = robot_system.named_joint_state("home")
    q = np.array([home[n] for n in model.active_joint_names], dtype=float)
    return model, q


def _state_msg(q: np.ndarray) -> imp_pb2.RobotState:
    return imp_pb2.RobotState(
        header=imp_pb2.Header(schema="imp.RobotState/1"),
        q=q.tolist(),
        mode="idle",
    )


# ---------------------------------------------------------------------------
# motion-pinocchio: FK (world-frame) + IK (round-trip)
# ---------------------------------------------------------------------------


def test_fk_module_round_trip_against_direct_fk():
    from imp_module_motion_pinocchio.fk import FkModule

    world_path = WORLDS / "franka_table_world.yaml"
    world = WorldDescription.from_yaml(world_path)
    model, q = _home_q(world.robot("arm").robot_system)
    scene = Scene.from_world(world)
    spec = model.chain("arm")
    tcp = spec.tcp_frame or spec.tip_frame
    want = fk(scene, "arm", q, tcp)

    module = FkModule(
        station=STATION,
        robot=ROBOT,
        world_path=str(world_path),
        world_robot="arm",
    )
    with module_under_test(module) as h:
        h.publish(keyexpr.hal(STATION, ROBOT, "state"), _state_msg(q))
        got = h.recv(keyexpr.motion(STATION, "fk", "tcp"), imp_pb2.Pose6D)

    got_pos = np.array(got.position_m)
    got_quat = np.array(got.quat_xyzw)
    want_quat = Rotation.from_matrix(want[:3, :3]).as_quat()

    assert got.header.frame_id == world.world_frame
    assert np.linalg.norm(got_pos - want[:3, 3]) < 1e-6
    assert min(np.linalg.norm(got_quat - want_quat),
               np.linalg.norm(got_quat + want_quat)) < 1e-6


def test_ik_module_round_trip_matches_direct_ik():
    from imp_module_motion_pinocchio.ik import IkModule

    rs_path = ROBOTS / "franka_fr3_with_franka_hand.yaml"
    system = RobotSystemDescription.from_yaml(rs_path)
    model, q_seed = _home_q(system)
    spec = model.chain("arm")
    tcp = spec.tcp_frame or spec.tip_frame

    # Build a reachable target: FK at a perturbed q.
    q_target = q_seed.copy()
    q_target[0] += 0.10
    T_target = fk_local(model, q_target, tcp)
    pos = T_target[:3, 3]
    quat = Rotation.from_matrix(T_target[:3, :3]).as_quat()

    direct = ik_local(model, T_target, q_seed, frame_id=tcp)
    assert direct.status == IKStatus.OK

    target_msg = imp_pb2.PoseTarget(
        header=imp_pb2.Header(schema="imp.PoseTarget/1"),
        target_frame=system.robot.base_frame,
        position_m=pos.tolist(),
        quat_xyzw=quat.tolist(),
    )

    module = IkModule(
        station=STATION,
        robot=ROBOT,
        robot_system_path=str(rs_path),
    )
    with module_under_test(module) as h:
        h.publish(keyexpr.hal(STATION, ROBOT, "state"), _state_msg(q_seed))
        h.publish(keyexpr.motion(STATION, "ik", "target"), target_msg)
        got = h.recv(keyexpr.motion(STATION, "ik", "solution"), imp_pb2.JointSolution)

    assert got.valid
    # FK on the returned q must hit the target.
    q_got = np.asarray(got.q, dtype=float)
    T_got = fk_local(model, q_got, tcp)
    assert np.linalg.norm(T_got[:3, 3] - pos) < 1e-3


# ---------------------------------------------------------------------------
# motion-coal: collision query matches direct
# ---------------------------------------------------------------------------


def test_collision_module_matches_direct_query():
    pytest.importorskip("coal")
    from imp_module_motion_coal.collision import CollisionModule

    world_path = WORLDS / "franka_table_world.yaml"
    world = WorldDescription.from_yaml(world_path)
    cm = CollisionModel.from_world(world)
    scene = Scene.from_world(world, cm)
    model, q = _home_q(world.robot("arm").robot_system)
    direct = is_in_collision(model, scene, q)
    want = float(len(direct.contacts)) if direct.in_collision else 0.0

    module = CollisionModule(
        station=STATION,
        robot=ROBOT,
        world_path=str(world_path),
        world_robot="arm",
    )
    with module_under_test(module) as h:
        h.publish(keyexpr.hal(STATION, ROBOT, "state"), _state_msg(q))
        got = h.recv(keyexpr.motion(STATION, "collision", "collision"), imp_pb2.Scalar)

    assert abs(got.value - want) < 1e-9


# ---------------------------------------------------------------------------
# motion-ompl: planning produces a valid path
# ---------------------------------------------------------------------------


def test_planning_module_produces_connected_path():
    pytest.importorskip("ompl")
    from imp_module_motion_ompl.plan import PlanModule

    world_path = WORLDS / "franka_robot_only_world.yaml"
    world = WorldDescription.from_yaml(world_path)
    model, q_home = _home_q(world.robots[0].robot_system)
    q_goal = q_home.copy()
    q_goal[0] += 0.8
    q_goal[2] -= 0.3

    direct = plan_joint(
        model,
        Scene.from_world(world, CollisionModel.from_world(world)),
        q_home,
        q_goal,
        options=PlanOptions(random_seed=0),
    )
    assert direct.success, "direct plan_joint failed; can't compare bus path"

    module = PlanModule(
        station=STATION,
        robot=ROBOT,
        world_path=str(world_path),
        random_seed=0,
    )
    with module_under_test(module) as h:
        goal_msg = imp_pb2.JointSolution(
            header=imp_pb2.Header(schema="imp.JointSolution/1"),
            q=q_goal.tolist(),
            valid=True,
        )
        h.publish(keyexpr.hal(STATION, ROBOT, "state"), _state_msg(q_home))
        h.publish(keyexpr.motion(STATION, "plan", "goal"), goal_msg)
        got = h.recv(keyexpr.motion(STATION, "plan", "path"), imp_pb2.Path, timeout_s=30.0)

    assert got.n_dof == len(model.active_joint_names)
    wp = np.array(got.q_wp).reshape(-1, got.n_dof)
    assert wp.shape[0] >= 2
    assert np.allclose(wp[0], q_home, atol=1e-3)
    assert np.allclose(wp[-1], q_goal, atol=1e-3)


# ---------------------------------------------------------------------------
# spatial-tf: graph composes edges arriving on imp/<st>/tf
# ---------------------------------------------------------------------------


def test_tf_module_composes_chain_from_published_edges():
    from imp_module_spatial_tf import TfModule

    T_wb = np.eye(4)
    T_wb[:3, 3] = (0.5, 0.0, 1.0)
    T_bt = np.eye(4)
    T_bt[:3, 3] = (0.1, 0.0, 0.3)

    def _edge(parent, child, T):
        return imp_pb2.TfEdge(
            header=imp_pb2.Header(schema="imp.TfEdge/1"),
            parent_frame=parent, child_frame=child,
            matrix=T.flatten().tolist(),
        )

    module = TfModule(station=STATION)
    with module_under_test(module) as h:
        tf_key = keyexpr.tf(STATION)
        h.publish(tf_key, _edge("world", "base", T_wb))
        h.publish(tf_key, _edge("base", "tcp", T_bt))
        # Wait for the frames-count heartbeat so we know both edges landed.
        _ = h.recv(keyexpr.motion(STATION, "tf", "frames"), imp_pb2.Scalar)
        _ = h.recv(keyexpr.motion(STATION, "tf", "frames"), imp_pb2.Scalar)

    composed = module.graph.lookup("world", "tcp")
    assert np.allclose(composed, T_wb @ T_bt)


# ---------------------------------------------------------------------------
# spatial-transform: Pose6D[cam] -> PoseTarget[base] via published tf
# ---------------------------------------------------------------------------


def test_transform_module_lifts_pose_through_tf():
    from imp_module_spatial_transform import TransformModule

    T_base_cam = np.eye(4)
    T_base_cam[:3, :3] = Rotation.from_euler("z", np.pi / 2).as_matrix()
    T_base_cam[:3, 3] = (0.2, 0.0, 0.5)

    pose_key = f"imp/{STATION}/perc/s1/pose"
    module = TransformModule(
        station=STATION,
        pose_key=pose_key,
        base_frame="base",
    )
    with module_under_test(module) as h:
        tf_msg = imp_pb2.TfEdge(
            header=imp_pb2.Header(schema="imp.TfEdge/1"),
            parent_frame="base", child_frame="camera",
            matrix=T_base_cam.flatten().tolist(),
        )
        pose_msg = imp_pb2.Pose6D(
            header=imp_pb2.Header(schema="imp.Pose6D/1", frame_id="camera"),
            position_m=[0.4, 0.0, 0.0],
            quat_xyzw=[0, 0, 0, 1],
            confidence=1.0,
            valid=True,
        )
        h.publish(keyexpr.tf(STATION), tf_msg)
        h.publish(pose_key, pose_msg)
        got = h.recv(keyexpr.motion(STATION, "transform", "target"), imp_pb2.PoseTarget)

    assert got.target_frame == "base"
    # Camera-frame (0.4, 0, 0) after the 90deg z-rotation + translation:
    assert np.allclose(np.array(got.position_m), (0.2, 0.4, 0.5))
