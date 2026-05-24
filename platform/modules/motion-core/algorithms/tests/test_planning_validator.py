"""Path validator tests (Phase 6b).

Exercises every check the validator runs by deliberately constructing
paths that violate each one.
"""
from __future__ import annotations

from pathlib import Path as FsPath

import numpy as np
import pytest

pytest.importorskip("coal")
pytest.importorskip("pinocchio")

from algorithms.descriptions import WorldDescription
from algorithms.kinematics import fk_local
from algorithms.planning import (
    Path,
    PathValidationOptions,
    plan_cartesian,
    plan_joint,
    validate_path,
)
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


def test_validator_passes_well_formed_path():
    scene, model, q_home = _setup()
    q_goal = q_home.copy()
    q_goal[0] += 0.5
    result = plan_joint(model, scene, q_home, q_goal, backend="direct")
    assert result.path is not None
    report = validate_path(model, scene, result.path)
    assert report.passed
    assert report.first_failure is None


def test_validator_rejects_joint_limit_violation():
    scene, model, q_home = _setup()
    lower, upper = model.active_position_limits()
    # Construct a path whose middle waypoint is at upper + 1.0
    bad = upper + 1.0
    wps = np.stack([q_home, bad, q_home])
    path = Path(waypoints=wps, joint_names=tuple(model.active_joint_names))
    report = validate_path(model, scene, path)
    assert not report.passed
    assert report.first_failure is not None
    assert report.first_failure[0] == 1


def test_validator_rejects_branch_jump_on_cartesian_path():
    """Manufacture a Cartesian path with a large dq but tiny TCP delta."""
    scene, model, q_home = _setup()
    T_start = fk_local(model, q_home, "robot_tcp")
    T_goal = T_start.copy()
    T_goal[0, 3] += 0.02
    result = plan_cartesian(scene, "arm", "robot_tcp", T_start, T_goal, q_home)
    assert result.path is not None

    # Insert a deliberately large joint jump at waypoint 1.
    wps = result.path.waypoints.copy()
    cart = result.path.cartesian_waypoints.copy()
    wps[1, 0] = wps[0, 0] + 1.0  # 1 rad jump in joint 0
    # But TCP at waypoint 1 stays close to waypoint 0
    cart[1] = cart[0].copy()
    cart[1][0, 3] += 1e-4

    spoiled = Path(
        waypoints=wps,
        joint_names=result.path.joint_names,
        cartesian_waypoints=cart,
        metadata=dict(result.path.metadata),
    )
    # Need to override joint-limit check first by widening margin, then
    # the singularity check is skipped automatically if frame_id absent;
    # we keep frame_id so singularity runs, but the joint jump happens
    # before singularity in the check order — actually it happens after.
    # The validator checks joint_limits first (passes), then collision
    # (might fail since the joint jump configuration is novel), then
    # singularity, then branch_jump. To isolate branch_jump, disable
    # everything before it.
    opts = PathValidationOptions(reject_singular=False)
    # Disable collision by handing a scene with no collision_model
    scene_no_collision = Scene.from_world(scene.world, None)
    report = validate_path(model, scene_no_collision, spoiled, options=opts)
    assert not report.passed
    assert report.first_failure is not None
    assert "branch" in report.first_failure[1].lower()


def test_validator_velocity_envelope_flags_too_fast_segment():
    scene, model, q_home = _setup()
    q_goal = q_home.copy()
    q_goal[0] += 0.5
    wps = np.stack([q_home, q_goal])
    path = Path(waypoints=wps, joint_names=tuple(model.active_joint_names))
    # nominal time 0.001 s for 0.5 rad joint motion -> 500 rad/s, well above limit
    opts = PathValidationOptions(nominal_segment_time=0.001)
    report = validate_path(model, scene, path, options=opts)
    assert not report.passed
    assert "velocity" in report.first_failure[1].lower()
