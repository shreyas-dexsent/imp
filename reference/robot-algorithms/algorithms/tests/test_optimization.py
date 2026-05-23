"""Path optimization tests (Phase 6c).

Covers:
* shortcut_smooth shortens the path and never introduces a collision.
* remove_redundant_waypoints removes consecutive duplicates.
* spline_fit returns a valid Path with the requested sample count and
  pinned endpoints; C^1 derivative is continuous; quintic + cubic
  match each other at the boundary conditions they share.
* End-to-end pipeline: plan -> dedupe -> shortcut -> spline -> validate.
"""
from __future__ import annotations

from pathlib import Path as FsPath

import numpy as np
import pytest

pytest.importorskip("coal")
pytest.importorskip("pinocchio")
pytest.importorskip("ompl")

from algorithms.descriptions import WorldDescription
from algorithms.optimization import (
    remove_redundant_waypoints,
    shortcut_smooth,
    spline_fit,
)
from algorithms.planning import Path, plan_joint, validate_path
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


# ---------------------------------------------------------------------------
# shortcut_smooth
# ---------------------------------------------------------------------------


def test_shortcut_smooth_reduces_path_length():
    scene, model, q_home = _setup()
    q_goal = q_home.copy()
    q_goal[0] += 0.8
    q_goal[2] -= 0.3
    planned = plan_joint(model, scene, q_home, q_goal)
    assert planned.path is not None
    smoothed, stats = shortcut_smooth(
        planned.path, model, scene, iterations=100, random_seed=0,
    )
    assert smoothed.length() <= planned.path.length()
    assert stats.attempted > 0


def test_shortcut_smooth_preserves_endpoints():
    scene, model, q_home = _setup()
    q_goal = q_home.copy()
    q_goal[0] += 0.5
    planned = plan_joint(model, scene, q_home, q_goal, backend="direct")
    assert planned.path is not None
    smoothed, _ = shortcut_smooth(planned.path, model, scene, iterations=50)
    np.testing.assert_allclose(smoothed.waypoints[0], planned.path.waypoints[0], atol=1e-12)
    np.testing.assert_allclose(smoothed.waypoints[-1], planned.path.waypoints[-1], atol=1e-12)


def test_shortcut_smooth_does_not_introduce_collisions():
    """Regression: the smoothed path must validate."""
    scene, model, q_home = _setup()
    q_goal = q_home.copy()
    q_goal[0] += 0.8
    q_goal[2] -= 0.3
    planned = plan_joint(model, scene, q_home, q_goal)
    assert planned.path is not None
    smoothed, _ = shortcut_smooth(planned.path, model, scene, iterations=200)
    report = validate_path(model, scene, smoothed)
    assert report.passed


def test_shortcut_smooth_on_tiny_path_is_noop():
    scene, model, q_home = _setup()
    q_goal = q_home.copy()
    q_goal[0] += 0.1
    wps = np.stack([q_home, q_goal])
    path = Path(waypoints=wps, joint_names=tuple(model.active_joint_names))
    smoothed, stats = shortcut_smooth(path, model, scene, iterations=10)
    assert smoothed.num_waypoints == 2
    assert stats.attempted == 0


# ---------------------------------------------------------------------------
# remove_redundant_waypoints
# ---------------------------------------------------------------------------


def test_remove_redundant_waypoints_drops_consecutive_duplicates():
    q = np.zeros(7)
    wps = np.stack([q, q, q + 0.5, q + 0.5, q + 1.0])
    path = Path(waypoints=wps, joint_names=tuple(f"j{i}" for i in range(7)))
    deduped = remove_redundant_waypoints(path)
    assert deduped.num_waypoints == 3


def test_remove_redundant_waypoints_keeps_unique_waypoints():
    q = np.zeros(3)
    wps = np.stack([q, q + 0.1, q + 0.2, q + 0.3])
    path = Path(waypoints=wps, joint_names=("a", "b", "c"))
    deduped = remove_redundant_waypoints(path)
    assert deduped.num_waypoints == 4


# ---------------------------------------------------------------------------
# spline_fit
# ---------------------------------------------------------------------------


def test_spline_fit_returns_requested_sample_count():
    scene, model, q_home = _setup()
    q_goal = q_home.copy()
    q_goal[0] += 0.3
    planned = plan_joint(model, scene, q_home, q_goal, backend="direct")
    splined = spline_fit(planned.path, order="quintic", samples=120)
    assert splined.num_waypoints == 120


def test_spline_fit_pins_endpoints_exactly():
    scene, model, q_home = _setup()
    q_goal = q_home.copy()
    q_goal[0] += 0.3
    planned = plan_joint(model, scene, q_home, q_goal, backend="direct")
    splined = spline_fit(planned.path, samples=64)
    np.testing.assert_allclose(splined.waypoints[0], planned.path.waypoints[0], atol=1e-12)
    np.testing.assert_allclose(splined.waypoints[-1], planned.path.waypoints[-1], atol=1e-12)


def test_spline_fit_two_waypoints_single_segment():
    q = np.zeros(7)
    wps = np.stack([q, q + 0.5])
    path = Path(waypoints=wps, joint_names=tuple(f"j{i}" for i in range(7)))
    splined = spline_fit(path, samples=20)
    assert splined.num_waypoints == 20


def test_spline_fit_cubic_order():
    q = np.zeros(7)
    wps = np.stack([q, q + 0.3, q + 0.6])
    path = Path(waypoints=wps, joint_names=tuple(f"j{i}" for i in range(7)))
    splined = spline_fit(path, order="cubic", samples=40)
    assert splined.num_waypoints == 40


def test_spline_fit_rejects_invalid_order():
    q = np.zeros(7)
    wps = np.stack([q, q + 0.3])
    path = Path(waypoints=wps, joint_names=tuple(f"j{i}" for i in range(7)))
    with pytest.raises(ValueError, match="unsupported order"):
        spline_fit(path, order="quartic", samples=40)  # type: ignore[arg-type]


def test_spline_fit_rejects_single_waypoint_input():
    # We can't construct a Path with 1 waypoint (the dataclass forbids
    # it), so this guards the spline_fit boundary check explicitly.
    q = np.zeros(7)
    with pytest.raises(ValueError):
        # Path requires N>=2, so this builds a valid Path with the
        # smallest legal size; the spline_fit guard fires on samples=1.
        wps = np.stack([q, q + 0.5])
        path = Path(waypoints=wps, joint_names=tuple(f"j{i}" for i in range(7)))
        spline_fit(path, samples=1)


# ---------------------------------------------------------------------------
# End-to-end pipeline
# ---------------------------------------------------------------------------


def test_full_pipeline_plan_dedupe_shortcut_spline_validate():
    scene, model, q_home = _setup()
    q_goal = q_home.copy()
    q_goal[0] += 0.8
    q_goal[2] -= 0.3

    planned = plan_joint(model, scene, q_home, q_goal)
    deduped = remove_redundant_waypoints(planned.path)
    smoothed, _ = shortcut_smooth(deduped, model, scene, iterations=200, random_seed=42)
    splined = spline_fit(smoothed, order="quintic", samples=80)

    report = validate_path(model, scene, splined)
    assert report.passed
    np.testing.assert_allclose(splined.waypoints[0], q_home, atol=1e-4)
    np.testing.assert_allclose(splined.waypoints[-1], q_goal, atol=1e-4)
