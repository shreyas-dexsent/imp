"""Cartesian path planning tests (Phase 6a)."""
from __future__ import annotations

from pathlib import Path as FsPath

import numpy as np
import pytest

pytest.importorskip("coal")
pytest.importorskip("pinocchio")

from algorithms.descriptions import WorldDescription
from algorithms.kinematics import fk_local
from algorithms.planning import PathStatus, PlanOptions, plan_cartesian
from algorithms.resolved import CollisionModel, KinematicModel, Scene
from algorithms.resolved.kinematic_model import _clear_cache

REPO_ROOT = FsPath(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _isolate_cache():
    _clear_cache()
    yield
    _clear_cache()


def _setup():
    world = WorldDescription.from_yaml(
        REPO_ROOT / "configs" / "worlds" / "franka_robot_only_world.yaml"
    )
    cm = CollisionModel.from_world(world)
    scene = Scene.from_world(world, cm)
    system = world.robots[0].robot_system
    model = KinematicModel.from_robot_system(system)
    home = system.named_joint_state("home")
    q_home = np.array([home[name] for name in model.active_joint_names], dtype=float)
    return scene, model, q_home


def test_plan_cartesian_straight_line_in_x_succeeds():
    scene, model, q_home = _setup()
    T_start = fk_local(model, q_home, "robot_tcp")
    T_goal = T_start.copy()
    T_goal[0, 3] += 0.10
    result = plan_cartesian(scene, "arm", "robot_tcp", T_start, T_goal, q_home)
    assert result.status is PathStatus.SUCCESS
    assert result.path is not None
    assert result.path.cartesian_waypoints is not None
    np.testing.assert_allclose(
        result.path.cartesian_waypoints[0][:3, 3], T_start[:3, 3], atol=1e-12,
    )
    np.testing.assert_allclose(
        result.path.cartesian_waypoints[-1][:3, 3], T_goal[:3, 3], atol=1e-12,
    )


def test_plan_cartesian_defaults_T_start_to_fk_of_seed():
    scene, model, q_home = _setup()
    T_start_via_fk = fk_local(model, q_home, "robot_tcp")
    T_goal = T_start_via_fk.copy()
    T_goal[1, 3] += 0.05
    # T_start = None → planner derives via FK on q_seed
    result = plan_cartesian(scene, "arm", "robot_tcp", None, T_goal, q_home)
    assert result.status is PathStatus.SUCCESS
    np.testing.assert_allclose(
        result.path.cartesian_waypoints[0][:3, 3], T_start_via_fk[:3, 3], atol=1e-12,
    )


def test_plan_cartesian_succeeds_with_short_translation():
    scene, model, q_home = _setup()
    T_start = fk_local(model, q_home, "robot_tcp")
    T_goal = T_start.copy()
    T_goal[2, 3] += 0.05
    result = plan_cartesian(scene, "arm", "robot_tcp", T_start, T_goal, q_home)
    assert result.status is PathStatus.SUCCESS
    # cartesian_waypoints is populated for cartesian paths
    assert result.path.cartesian_waypoints is not None
    assert result.path.cartesian_waypoints.shape[0] == result.path.num_waypoints


def test_plan_cartesian_metadata_carries_frame_id_and_robot_id():
    scene, model, q_home = _setup()
    T_start = fk_local(model, q_home, "robot_tcp")
    T_goal = T_start.copy()
    T_goal[0, 3] += 0.05
    result = plan_cartesian(scene, "arm", "robot_tcp", T_start, T_goal, q_home)
    assert result.path.metadata["frame_id"] == "robot_tcp"
    assert result.path.metadata["robot_id"] == "arm"
