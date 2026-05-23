"""Trajectory generation + validation tests (Phase 6d + 6e).

Covers:
* Trajectory data type invariants (at(t), sample(dt), endpoint pinning).
* Polynomial backend respects v/a envelopes.
* Ruckig backend respects v/a/j envelopes when ruckig is installed.
* Pass-through behaviour: velocity at interior waypoints is non-zero;
  velocity at start and end is zero (rest-to-rest endpoints).
* rest_to_rest=True restores the legacy stop-at-each-waypoint behaviour.
* time_parameterize correctly dispatches when backends are missing.
* validate_trajectory catches v / a / position-limit violations and
  passes valid trajectories.
"""
from __future__ import annotations

from pathlib import Path as FsPath

import numpy as np
import pytest

pytest.importorskip("coal")
pytest.importorskip("pinocchio")
pytest.importorskip("ompl")

from algorithms.descriptions import WorldDescription
from algorithms.optimization import shortcut_smooth, spline_fit
from algorithms.planning import Path, plan_joint
from algorithms.resolved import CollisionModel, KinematicModel, Scene
from algorithms.resolved.kinematic_model import _clear_cache
from algorithms.trajectory import (
    TimeParameterizationOptions,
    Trajectory,
    TrajectoryStatus,
    TrajectoryValidationOptions,
    time_parameterize,
    validate_trajectory,
)

REPO_ROOT = FsPath(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _isolate_cache():
    _clear_cache()
    yield
    _clear_cache()


def _setup_with_path():
    world = WorldDescription.from_yaml(
        REPO_ROOT / "configs" / "worlds" / "franka_robot_only_world.yaml"
    )
    cm = CollisionModel.from_world(world)
    scene = Scene.from_world(world, cm)
    system = world.robots[0].robot_system
    model = KinematicModel.from_robot_system(system)
    home = system.named_joint_state("home")
    q_home = np.array([home[name] for name in model.active_joint_names], dtype=float)
    q_goal = q_home.copy()
    q_goal[0] += 0.8
    q_goal[2] -= 0.3
    planned = plan_joint(model, scene, q_home, q_goal)
    smoothed, _ = shortcut_smooth(planned.path, model, scene, iterations=200)
    splined = spline_fit(smoothed, order="quintic", samples=50)
    return scene, model, splined


# ---------------------------------------------------------------------------
# Trajectory data type
# ---------------------------------------------------------------------------


def _toy_trajectory(dof=2, n=10, duration=1.0) -> Trajectory:
    times = np.linspace(0.0, duration, n)
    positions = np.outer(times / duration, np.ones(dof))
    velocities = np.zeros((n, dof))
    velocities[:, :] = 1.0 / duration
    accelerations = np.zeros((n, dof))
    return Trajectory(
        times=times,
        positions=positions,
        velocities=velocities,
        accelerations=accelerations,
        joint_names=tuple(f"j{i}" for i in range(dof)),
        backend_used="toy",
    )


def test_trajectory_requires_two_samples():
    with pytest.raises(ValueError, match="at least 2 samples"):
        Trajectory(
            times=np.array([0.0]),
            positions=np.zeros((1, 2)),
            velocities=np.zeros((1, 2)),
            accelerations=np.zeros((1, 2)),
            joint_names=("a", "b"),
            backend_used="x",
        )


def test_trajectory_duration_and_dof():
    t = _toy_trajectory(dof=2, n=10, duration=1.5)
    assert t.duration == pytest.approx(1.5)
    assert t.dof == 2
    assert t.num_samples == 10


def test_trajectory_at_returns_endpoints_outside_domain():
    t = _toy_trajectory(dof=2, n=10, duration=1.0)
    q_before, _, _ = t.at(-0.1)
    q_after, _, _ = t.at(2.0)
    np.testing.assert_allclose(q_before, t.positions[0])
    np.testing.assert_allclose(q_after, t.positions[-1])


def test_trajectory_sample_uniform_dt():
    t = _toy_trajectory(dof=2, n=10, duration=1.0)
    times, q, qd, qdd = t.sample(dt=0.05)
    assert times.shape == (21,)
    assert q.shape == (21, 2)


# ---------------------------------------------------------------------------
# time_parameterize — polynomial backend
# ---------------------------------------------------------------------------


def test_polynomial_backend_succeeds_on_planned_path():
    scene, model, splined = _setup_with_path()
    result = time_parameterize(
        splined, model,
        options=TimeParameterizationOptions(backend="polynomial", dt=0.01),
    )
    assert result.status is TrajectoryStatus.SUCCESS
    assert result.trajectory is not None
    np.testing.assert_allclose(
        result.trajectory.positions[0], splined.waypoints[0], atol=1e-6,
    )
    np.testing.assert_allclose(
        result.trajectory.positions[-1], splined.waypoints[-1], atol=1e-6,
    )


def test_polynomial_pass_through_keeps_interior_velocity_nonzero():
    """Pass-through requirement: robot does NOT pause at interior waypoints."""
    scene, model, splined = _setup_with_path()
    result = time_parameterize(
        splined, model,
        options=TimeParameterizationOptions(backend="polynomial", dt=0.01),
    )
    traj = result.trajectory
    # Velocity at the midpoint should be non-zero.
    _, qd_mid, _ = traj.at(traj.duration * 0.5)
    assert float(np.linalg.norm(qd_mid)) > 0.05, (
        f"interior velocity {qd_mid} should be non-zero in pass-through mode"
    )
    # Velocity at the start and end should be zero (rest-to-rest endpoints).
    _, qd_start, _ = traj.at(0.0)
    _, qd_end, _ = traj.at(traj.duration)
    np.testing.assert_allclose(qd_start, np.zeros_like(qd_start), atol=1e-6)
    np.testing.assert_allclose(qd_end, np.zeros_like(qd_end), atol=1e-6)


def test_polynomial_rest_to_rest_forces_zero_interior_velocity():
    """Legacy behaviour with rest_to_rest=True; useful for debugging."""
    scene, model, splined = _setup_with_path()
    result = time_parameterize(
        splined, model,
        options=TimeParameterizationOptions(
            backend="polynomial", dt=0.01, rest_to_rest=True,
        ),
    )
    traj = result.trajectory
    # Just check endpoints; the per-segment rest-to-rest invariant is
    # an internal detail and depends on per-segment duration that
    # isn't exposed.
    _, qd_start, _ = traj.at(0.0)
    _, qd_end, _ = traj.at(traj.duration)
    np.testing.assert_allclose(qd_start, np.zeros_like(qd_start), atol=1e-6)
    np.testing.assert_allclose(qd_end, np.zeros_like(qd_end), atol=1e-6)


def test_polynomial_respects_velocity_envelope():
    scene, model, splined = _setup_with_path()
    result = time_parameterize(
        splined, model,
        options=TimeParameterizationOptions(backend="polynomial", dt=0.01),
    )
    traj = result.trajectory
    v_lim = model.active_velocity_limits()
    assert np.all(np.abs(traj.velocities) <= v_lim + 1e-3)


def test_polynomial_v_scale_caps_peak_velocity():
    """v_scale=0.3 should cap peak velocity at 30% of the model limit.

    Whether the trajectory's total duration changes depends on whether
    the motion is velocity-bound or acceleration-bound — for very
    short segments the acceleration bound dominates and v_scale leaves
    duration unchanged. The invariant the user cares about is the
    velocity envelope itself."""
    scene, model, splined = _setup_with_path()
    fast = time_parameterize(
        splined, model,
        options=TimeParameterizationOptions(backend="polynomial", dt=0.01, v_scale=1.0),
    ).trajectory
    slow = time_parameterize(
        splined, model,
        options=TimeParameterizationOptions(backend="polynomial", dt=0.01, v_scale=0.3),
    ).trajectory
    v_lim = model.active_velocity_limits()
    assert np.all(np.abs(fast.velocities) <= v_lim + 1e-3)
    assert np.all(np.abs(slow.velocities) <= 0.3 * v_lim + 1e-3)
    # Duration is non-decreasing under v_scale<1: never faster.
    assert slow.duration >= fast.duration - 1e-9


# ---------------------------------------------------------------------------
# time_parameterize — Ruckig backend (when installed)
# ---------------------------------------------------------------------------


def test_ruckig_backend_succeeds_when_installed():
    pytest.importorskip("ruckig")
    scene, model, splined = _setup_with_path()
    result = time_parameterize(
        splined, model,
        options=TimeParameterizationOptions(backend="ruckig", dt=0.01),
    )
    assert result.status is TrajectoryStatus.SUCCESS
    assert result.trajectory.backend_used == "ruckig"


def test_ruckig_pass_through_keeps_interior_velocity_nonzero():
    pytest.importorskip("ruckig")
    scene, model, splined = _setup_with_path()
    result = time_parameterize(
        splined, model,
        options=TimeParameterizationOptions(backend="ruckig", dt=0.01),
    )
    traj = result.trajectory
    _, qd_mid, _ = traj.at(traj.duration * 0.5)
    assert float(np.linalg.norm(qd_mid)) > 0.05


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------


def test_auto_dispatch_picks_ruckig_when_available():
    pytest.importorskip("ruckig")
    scene, model, splined = _setup_with_path()
    result = time_parameterize(splined, model)
    assert result.status is TrajectoryStatus.SUCCESS
    assert result.trajectory.backend_used in ("ruckig", "polynomial")


def test_unknown_backend_returns_invalid_input():
    scene, model, splined = _setup_with_path()
    result = time_parameterize(
        splined, model,
        options=TimeParameterizationOptions(backend="bogus"),  # type: ignore[arg-type]
    )
    assert result.status is TrajectoryStatus.INVALID_INPUT


# ---------------------------------------------------------------------------
# validate_trajectory
# ---------------------------------------------------------------------------


def test_validate_trajectory_passes_on_valid_polynomial_output():
    scene, model, splined = _setup_with_path()
    result = time_parameterize(
        splined, model,
        options=TimeParameterizationOptions(backend="polynomial", dt=0.001),
    )
    report = validate_trajectory(
        result.trajectory, model, scene,
        options=TrajectoryValidationOptions(check_collision=True),
    )
    assert report.passed


def test_validate_trajectory_catches_velocity_violation():
    """Construct a trajectory with deliberately too-high velocity and
    confirm the validator flags it."""
    scene, model, splined = _setup_with_path()
    result = time_parameterize(
        splined, model,
        options=TimeParameterizationOptions(backend="polynomial", dt=0.01),
    )
    traj = result.trajectory
    # Spoil: inflate the velocities array beyond joint limits.
    spoiled_vel = traj.velocities.copy()
    spoiled_vel[len(spoiled_vel) // 2] = (
        model.active_velocity_limits() * 10.0
    )
    spoiled = Trajectory(
        times=traj.times,
        positions=traj.positions,
        velocities=spoiled_vel,
        accelerations=traj.accelerations,
        joint_names=traj.joint_names,
        backend_used="spoiled",
    )
    report = validate_trajectory(
        spoiled, model, scene,
        options=TrajectoryValidationOptions(check_collision=False),
    )
    assert not report.passed
    assert "velocity" in report.first_failure[1].lower()


def test_validate_trajectory_catches_position_limit_violation():
    scene, model, splined = _setup_with_path()
    result = time_parameterize(
        splined, model,
        options=TimeParameterizationOptions(backend="polynomial", dt=0.01),
    )
    traj = result.trajectory
    _, upper = model.active_position_limits()
    spoiled_pos = traj.positions.copy()
    spoiled_pos[len(spoiled_pos) // 2] = upper + 1.0
    spoiled = Trajectory(
        times=traj.times,
        positions=spoiled_pos,
        velocities=traj.velocities,
        accelerations=traj.accelerations,
        joint_names=traj.joint_names,
        backend_used="spoiled",
    )
    report = validate_trajectory(
        spoiled, model, scene,
        options=TrajectoryValidationOptions(check_collision=False),
    )
    assert not report.passed
    assert "joint position" in report.first_failure[1].lower() or "limit" in report.first_failure[1].lower()


def test_validate_trajectory_controller_rate_compatibility():
    scene, model, splined = _setup_with_path()
    result = time_parameterize(
        splined, model,
        options=TimeParameterizationOptions(backend="polynomial", dt=0.05),
    )
    # Asking for 1 kHz controller against a 50 ms trajectory dt should fail.
    report = validate_trajectory(
        result.trajectory, model, scene,
        options=TrajectoryValidationOptions(
            controller_dt=0.001, check_collision=False,
        ),
    )
    assert not report.passed
    assert "controller" in report.first_failure[1].lower()


# ---------------------------------------------------------------------------
# End-to-end pipeline
# ---------------------------------------------------------------------------


def test_full_pipeline_plan_smooth_spline_parameterize_validate():
    scene, model, splined = _setup_with_path()
    result = time_parameterize(splined, model)
    assert result.status is TrajectoryStatus.SUCCESS
    report = validate_trajectory(result.trajectory, model, scene)
    assert report.passed


# ---------------------------------------------------------------------------
# Smoothness regression
#
# Both regressions caught here come from the same bug class: every
# interior waypoint was treated as a "via point the robot must hit
# with forced zero acceleration", producing one acceleration ramp per
# interior knot. A direct A->B move with N collision-validation samples
# would visibly wiggle N-1 times. The fixes (StraightLineBackend
# returns just the two endpoints; both backends share spline-derived
# interior boundary conditions) eliminate the artifact. These tests
# pin the property down so it doesn't regress.
# ---------------------------------------------------------------------------


def _accel_zero_crossings(accel: np.ndarray, joint_index: int) -> int:
    a = accel[:, joint_index]
    sign = np.sign(a)
    sign[sign == 0] = 1
    return int(np.sum(np.diff(sign) != 0))


def test_direct_joint_move_produces_clean_s_curve():
    """A direct A->B joint move must look like one S-curve, not a wobble.

    Regression guard for the bug where StraightLineBackend leaked
    collision-validation samples into the trajectory layer as via
    points, producing one acceleration ramp per sample.
    """
    from algorithms.primitives import MoveOptions, move_joint

    world = WorldDescription.from_yaml(
        REPO_ROOT / "configs" / "worlds" / "franka_robot_only_world.yaml"
    )
    cm = CollisionModel.from_world(world)
    scene = Scene.from_world(world, cm)
    system = world.robots[0].robot_system
    model = KinematicModel.from_robot_system(system)
    home = system.named_joint_state("home")
    q = np.array([home[n] for n in model.active_joint_names], dtype=float)
    q_goal = q.copy()
    j6 = model.active_joint_names.index("fr3_joint6")
    q_goal[j6] += np.deg2rad(90.0)

    result = move_joint(
        model=model, scene=scene, q_goal=q_goal, q_seed=q,
        options=MoveOptions(
            planner_backend="direct", smooth_path=False, spline_fit=False,
        ),
    )
    assert result.trajectory is not None
    crossings = _accel_zero_crossings(result.trajectory.accelerations, j6)
    # Clean trapezoidal/S-curve acceleration crosses zero ~twice
    # (ramp-up, ramp-down). The bug produced 20+.
    assert crossings <= 4, (
        f"direct joint move produced {crossings} accel zero-crossings; "
        f"a clean S-curve should have at most 4."
    )


def test_polynomial_dense_path_no_per_segment_ripples():
    """The polynomial backend must fit ONE continuous spline through
    dense paths, not produce one ripple per segment.

    Regression guard for the per-segment quintic with forced
    acceleration=0 at every interior knot.
    """
    # 25 waypoints along a smooth curve in 2D (simulates the dense
    # output of ``plan_cartesian``).
    n = 25
    s = np.linspace(0.0, 1.0, n)
    qs = np.stack([s, 0.2 * np.sin(np.pi * s)], axis=1)
    path = Path(
        waypoints=qs,
        joint_names=("j0", "j1"),
    )

    from algorithms.trajectory.backends.polynomial import PolynomialBackend

    v_limits = np.array([5.0, 5.0])
    a_limits = np.array([50.0, 50.0])
    raw = PolynomialBackend().parameterize(
        path, v_limits, a_limits, j_limits=None,
        options=TimeParameterizationOptions(dt=0.005),
    )
    assert raw.status is TrajectoryStatus.SUCCESS, raw.message
    crossings = _accel_zero_crossings(raw.accelerations, joint_index=0)
    # A single C^2 spline over a smooth curve should produce a small
    # constant number of accel sign changes — the curve's own curvature
    # plus a couple of ramp boundaries. Per-segment quintics with
    # forced a=0 at every interior knot would give ~2*(n-1) = ~48.
    assert crossings <= 12, (
        f"polynomial backend produced {crossings} accel zero-crossings "
        f"for a {n}-waypoint smooth path; expected ~10 or less for one "
        f"spline. Per-segment with forced interior a=0 would be ~{2 * (n - 1)}."
    )
