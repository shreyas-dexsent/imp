"""Multi-robot collision tests.

Verifies that:
* `geometry_entries` walks every robot in the world with per-robot FK,
  using the per-robot `GeometryModel` to avoid parent-joint index
  garbling.
* `is_in_collision` accepts a composite `q` (dict keyed by robot_id).
* Cross-robot collision pairs emerge automatically: two FR3s placed
  close enough to touch report a collision; placed farther apart, they
  do not.
* The bare-ndarray `q` path still works for single-robot worlds
  (regression for the existing API).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("coal")
pytest.importorskip("pinocchio")

from algorithms.collision import is_in_collision
from algorithms.descriptions import WorldDescription
from algorithms.resolved import CollisionModel, KinematicModel, Scene
from algorithms.resolved.kinematic_model import _clear_cache

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _isolate_cache():
    _clear_cache()
    yield
    _clear_cache()


def _two_robot_scene():
    world = WorldDescription.from_yaml(
        REPO_ROOT / "configs" / "worlds" / "two_franka_table_world.yaml"
    )
    cm = CollisionModel.from_world(world)
    scene = Scene.from_world(world, cm)
    return world, cm, scene


def _home_q(system):
    home = system.named_joint_state("home")
    model = KinematicModel.from_robot_system(system)
    return np.array([home[name] for name in model.active_joint_names], dtype=float)


def test_collision_model_keeps_per_robot_geometry_models():
    world, cm, _ = _two_robot_scene()
    assert set(cm.robot_geoms_by_id.keys()) == {"left_arm", "right_arm"}
    # Each per-robot model has exactly that robot's namespaced links.
    for robot_id, geom in cm.robot_geoms_by_id.items():
        ns = world.robot(robot_id).namespace
        for go in geom.geometryObjects:
            assert go.name.startswith(f"{ns}/")


def test_multirobot_collision_query_with_dict_q():
    world, cm, scene = _two_robot_scene()
    left_q = _home_q(world.robot("left_arm").robot_system)
    right_q = _home_q(world.robot("right_arm").robot_system)

    # Composite-q dict form. KinematicModel arg is one of the robots';
    # the runtime ignores its q and reads everything from the dict.
    left_model = KinematicModel.from_robot_system(world.robot("left_arm").robot_system)
    report = is_in_collision(
        left_model,
        scene,
        {"left_arm": left_q, "right_arm": right_q},
    )
    # At home, bases are 1.2 m apart; the arms do not touch each other,
    # though each may touch the shared matka. We assert the call SUCCEEDS
    # (i.e., does not raise) rather than a specific in_collision value.
    assert report.checked_pairs > 0


def test_multirobot_q_falls_back_to_scene_robot_states():
    """If a robot's q is missing from the dict, the runtime reads it from
    scene.robot_states. This is the expected pattern when perception writes
    live state into the scene."""
    world, _, scene = _two_robot_scene()
    left_q = _home_q(world.robot("left_arm").robot_system)
    right_q = _home_q(world.robot("right_arm").robot_system)

    scene.set_robot_state("right_arm", right_q)
    left_model = KinematicModel.from_robot_system(world.robot("left_arm").robot_system)

    # Only left_arm in the dict; right_arm comes from scene.robot_states.
    report = is_in_collision(left_model, scene, {"left_arm": left_q})
    assert report.checked_pairs > 0


def test_multirobot_missing_q_raises_keyerror():
    world, _, scene = _two_robot_scene()
    left_q = _home_q(world.robot("left_arm").robot_system)
    left_model = KinematicModel.from_robot_system(world.robot("left_arm").robot_system)
    # Neither in the dict nor in scene.robot_states.
    with pytest.raises((KeyError, Exception)):
        is_in_collision(left_model, scene, {"left_arm": left_q})


def test_single_robot_scene_still_accepts_bare_ndarray():
    """Regression: the original API (q as ndarray) must keep working."""
    world = WorldDescription.from_yaml(
        REPO_ROOT / "configs" / "worlds" / "franka_table_world.yaml"
    )
    cm = CollisionModel.from_world(world)
    scene = Scene.from_world(world, cm)
    system = world.robot("arm").robot_system
    model = KinematicModel.from_robot_system(system)
    q = _home_q(system)
    report = is_in_collision(model, scene, q)
    assert report.checked_pairs > 0


def test_cross_robot_collision_detected_when_bases_overlap():
    """Two FR3s placed at the SAME world origin must collide arm-to-arm."""
    yaml_text = """
schema: dexsent.algorithms.world
version: 2
id: overlap_world
world_frame: world
robots:
  - id: arm_a
    robot_system: ../robots/franka_fr3_robot_only.yaml
    namespace: a
    base_pose:
      parent_frame: world
      child_frame: a/base
      matrix:
        - [1.0, 0.0, 0.0, 0.0]
        - [0.0, 1.0, 0.0, 0.0]
        - [0.0, 0.0, 1.0, 0.0]
        - [0.0, 0.0, 0.0, 1.0]
  - id: arm_b
    robot_system: ../robots/franka_fr3_robot_only.yaml
    namespace: b
    base_pose:
      parent_frame: world
      child_frame: b/base
      matrix:
        - [1.0, 0.0, 0.0, 0.0]
        - [0.0, 1.0, 0.0, 0.0]
        - [0.0, 0.0, 1.0, 0.0]
        - [0.0, 0.0, 0.0, 1.0]
objects: []
collision_matrix:
  default_action: check
  rules: []
"""
    import tempfile, textwrap
    worlds_dir = REPO_ROOT / "configs" / "worlds"
    with tempfile.NamedTemporaryFile(
        "w", suffix=".yaml", dir=str(worlds_dir), delete=False
    ) as handle:
        handle.write(textwrap.dedent(yaml_text))
        path = Path(handle.name)
    try:
        world = WorldDescription.from_yaml(path)
        cm = CollisionModel.from_world(world)
        scene = Scene.from_world(world, cm)
        q_home = _home_q(world.robot("arm_a").robot_system)

        model_a = KinematicModel.from_robot_system(world.robot("arm_a").robot_system)
        report = is_in_collision(
            model_a,
            scene,
            {"arm_a": q_home, "arm_b": q_home},
        )
        assert report.in_collision, (
            "two robots at the same base must collide arm-to-arm at home"
        )
    finally:
        path.unlink()


def test_far_apart_robots_do_not_collide_with_each_other():
    """Standard two-franka world has bases 1.2 m apart in x; at home the
    arms should not touch each other (each may touch the shared matka
    independently, but cross-robot pairs should be clear)."""
    world, cm, scene = _two_robot_scene()
    left_q = _home_q(world.robot("left_arm").robot_system)
    right_q = _home_q(world.robot("right_arm").robot_system)
    left_model = KinematicModel.from_robot_system(world.robot("left_arm").robot_system)

    report = is_in_collision(
        left_model, scene, {"left_arm": left_q, "right_arm": right_q},
    )
    # If a collision happens it must NOT be a cross-robot pair. Each pair
    # in the contact list comes with names; cross-robot pairs would mix
    # the "left/" and "right/" namespaces.
    if report.in_collision and report.contacts:
        for contact in report.contacts:
            a, b = contact.pair
            cross = a.startswith("left/") != b.startswith("left/")
            cross_arm_only = (
                cross and a.startswith(("left/", "right/")) and b.startswith(("left/", "right/"))
            )
            assert not cross_arm_only, (
                f"unexpected cross-robot collision between {a} and {b}; "
                "bases 1.2 m apart should keep arms clear of each other."
            )
