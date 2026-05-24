# Copyright (c) 2026 DexSent Robotics Pvt Ltd.
# All Rights Reserved.
#
# SPDX-License-Identifier: LicenseRef-Proprietary
# See LICENSE file for details.
#
# Author: Shreyas <shreyas.kumar@dexsentrobotics.com>
# Proprietary and confidential.

"""Inverse kinematics entrypoints.

Three public functions cover every IK use case:

* :func:`ik_local` — pose IK in the robot's base frame. Parallels
  :func:`algorithms.kinematics.fk_local`. The default entrypoint for
  application code.
* :func:`ik` — pose IK in the world frame. Parallels
  :func:`algorithms.kinematics.fk`. Composes the robot's world
  base pose into the target before solving.
* :func:`ik_velocity` — Cartesian velocity IK for servo loops. Returns
  ``qdot``, not :class:`IKResult`; the servo loop owns its live safety
  envelope.

Advanced callers building Drake-style modular problems with custom
constraints or costs call :func:`solve_problem` directly with a
:class:`IKProblem`.
"""
from __future__ import annotations

import time

import numpy as np

from algorithms.kinematics.ik.backends.analytical.base import AnalyticalIK
from algorithms.kinematics.ik.backends.base import BackendCandidate, BackendResult
from algorithms.kinematics.ik.constraints import JointPositionBounds, PoseTarget
from algorithms.kinematics.ik.costs import JointCenteringCost, SeedRegularization
from algorithms.kinematics.ik.dispatch import choose_backend, velocity_backend
from algorithms.kinematics.ik.options import IKOptions
from algorithms.kinematics.ik.problem import IKProblem, IKProblemSpec
from algorithms.kinematics.ik.result import (
    IKCandidate,
    IKDiagnostics,
    IKResult,
    IKStatus,
)
from algorithms.kinematics.ik.validator import validate
from algorithms.resolved.kinematic_model import KinematicModel
from algorithms.resolved.scene import Scene


# ---------------------------------------------------------------------------
# Tier 1: pose IK in the robot base frame
# ---------------------------------------------------------------------------


def ik_local(
    model: KinematicModel,
    frame_id: str,
    T_target: np.ndarray,
    q_seed: np.ndarray,
    *,
    pos_tol: float | None = None,
    rot_tol: float | None = None,
    backend: str | None = None,
    options: IKOptions | None = None,
    scene: Scene | None = None,
) -> IKResult:
    """Pose IK in the robot's base frame.

    On :class:`IKStatus.SUCCESS`, ``result.q`` is a joint configuration
    in ``model.active_joint_names`` order such that
    ``fk_local(model, result.q, frame_id) == T_target`` within the
    requested pose tolerance, with joint limits and the singularity
    threshold respected. Self-collision and environment-collision are
    also checked when ``scene`` is supplied with a built
    :class:`CollisionModel`.

    The function does not promise path collision-freeness, velocity /
    acceleration / jerk / torque feasibility, or human-safe execution.
    Those checks live downstream in planning, trajectory generation,
    and the runtime monitor.

    Parameters
    ----------
    model
        Resolved kinematic model for the robot system.
    frame_id
        Frame whose pose should match ``T_target``. Any resolved-model
        frame works (URDF frame or YAML-injected TCP).
    T_target
        Desired 4x4 homogeneous transform in the robot's base frame.
    q_seed
        Starting configuration. Required. Typical defaults are the
        current robot state or the named ``home`` pose.
    pos_tol, rot_tol
        Per-call override of the pose tolerance. When ``None``, the
        values from ``options`` apply.
    backend
        Force a specific backend (``"opw"``, ``"spherical_wrist_6r"``,
        ``"dls"``). When ``None``, dispatch picks: explicit hint →
        registered analytical for this robot id → generic constrained.
    options
        :class:`IKOptions` knob bag for sharing one config across many
        calls. When ``None``, defaults apply.
    scene
        Pass when collision validation against the world is required.
        When ``None``, validation runs every other check but skips
        collision.
    """
    opts = options or IKOptions()
    problem = _default_pose_problem(model, frame_id, T_target, q_seed, opts, pos_tol, rot_tol)
    return solve_problem(model, problem, q_seed, backend=backend, options=opts, scene=scene)


# ---------------------------------------------------------------------------
# Tier 1: pose IK in the world frame
# ---------------------------------------------------------------------------


def ik(
    scene: Scene,
    robot_id: str,
    frame_id: str,
    T_target: np.ndarray,
    q_seed: np.ndarray,
    *,
    pos_tol: float | None = None,
    rot_tol: float | None = None,
    backend: str | None = None,
    options: IKOptions | None = None,
    validate_collision: bool = True,
) -> IKResult:
    """Pose IK in the world frame.

    Composes the robot's world base pose into the target and delegates
    to :func:`ik_local`. Use this whenever the target comes from
    perception or a planner that works in world coordinates.

    Collision validation is enabled by default because a ``scene`` is
    always available at this entrypoint. Set ``validate_collision=False``
    to skip it (for example, when the collision model has not been
    built yet).
    """
    world_robot = scene.world.robot(robot_id)
    model = KinematicModel.from_robot_system(world_robot.robot_system)

    if world_robot.base_pose is None:
        T_world_base = np.eye(4)
    else:
        T_world_base = world_robot.base_pose.as_matrix()
    T_base_target = np.linalg.inv(T_world_base) @ T_target

    return ik_local(
        model,
        frame_id,
        T_base_target,
        q_seed,
        pos_tol=pos_tol,
        rot_tol=rot_tol,
        backend=backend,
        options=options,
        scene=scene if validate_collision else None,
    )


# ---------------------------------------------------------------------------
# Tier 1: Cartesian velocity IK for servo loops
# ---------------------------------------------------------------------------


def ik_velocity(
    model: KinematicModel,
    frame_id: str,
    target_twist: np.ndarray,
    q_current: np.ndarray,
    *,
    dt: float = 0.01,
) -> np.ndarray:
    """Cartesian velocity IK for servo loops.

    Returns a bounded ``qdot`` that best matches the desired 6D twist
    (linear three + angular three) at ``frame_id``. This is **not**
    pose IK: there is no :class:`IKResult`, no validator, and no
    failure status. The servo loop owns live safety. Typical cost on
    FR3 is around 100 microseconds per call; see the IK examples
    README for measured numbers.
    """
    return velocity_backend().solve_velocity(model, frame_id, target_twist, q_current, dt=dt)


# ---------------------------------------------------------------------------
# Tier 3: Drake-style modular problem
# ---------------------------------------------------------------------------


def solve_problem(
    model: KinematicModel,
    problem: IKProblem,
    q_seed: np.ndarray,
    *,
    backend: str | None = None,
    options: IKOptions | None = None,
    scene: Scene | None = None,
) -> IKResult:
    """Solve a user-built :class:`IKProblem` and validate every candidate.

    Use this when you need to add nonlinear constraints (tool axis,
    RCM, minimum distance, posture preference, ...) beyond what the
    default :func:`ik_local` problem ships. Build the problem with
    :meth:`IKProblem.add_task`, :meth:`add_constraint`, and
    :meth:`add_cost`, then pass it here.
    """
    opts = options or IKOptions()
    started = time.perf_counter()

    try:
        spec = problem.freeze()
        target = spec.pose_target
        q_seed = np.asarray(q_seed, dtype=float)
        if q_seed.shape != (len(model.active_joint_names),):
            return _result(
                IKStatus.INVALID_INPUT,
                None,
                backend or "none",
                (float("inf"), float("inf")),
                0,
                started,
                "q_seed has wrong shape",
            )
        if not model.pin_model.existFrame(target.frame_id):
            return _result(
                IKStatus.INVALID_INPUT,
                None,
                backend or "none",
                (float("inf"), float("inf")),
                0,
                started,
                "target frame not found",
            )
        backend_impl = choose_backend(model, backend)
    except Exception as exc:
        return _result(
            IKStatus.INVALID_INPUT,
            None,
            backend or "dispatch",
            (float("inf"), float("inf")),
            0,
            started,
            str(exc),
        )

    try:
        if hasattr(backend_impl, "solve_branches"):
            raw = _run_analytical_backend(backend_impl, model, spec, q_seed)
        else:
            raw = backend_impl.solve(model, spec, q_seed, opts)
    except ValueError as exc:
        return _result(
            IKStatus.INVALID_INPUT,
            None,
            getattr(backend_impl, "name", "unknown"),
            (float("inf"), float("inf")),
            0,
            started,
            str(exc),
        )
    except Exception as exc:
        return _result(
            IKStatus.NUMERICAL_FAILURE,
            None,
            getattr(backend_impl, "name", "unknown"),
            (float("inf"), float("inf")),
            0,
            started,
            str(exc),
        )

    valid: list[IKCandidate] = []
    validation_reports = []
    failure_statuses: list[IKStatus] = []

    for candidate in raw.candidates:
        report = validate(model, spec, candidate.q, opts, scene=scene)
        validation_reports.append(report)
        if report.success:
            valid.append(
                IKCandidate(
                    q=candidate.q,
                    pose_error=report.pose_error,
                    score=_candidate_score(candidate.q, report.pose_error, model),
                    backend=getattr(backend_impl, "name", "unknown"),
                    seed_index=candidate.seed_index,
                )
            )
        else:
            failure_statuses.append(report.status)

    valid = _dedupe_and_sort(valid)
    if valid:
        selected = valid[0]
        return IKResult(
            status=IKStatus.SUCCESS,
            q=selected.q,
            pose_error=selected.pose_error,
            iterations=raw.iterations,
            elapsed_ms=raw.elapsed_ms,
            backend_used=getattr(backend_impl, "name", "unknown"),
            candidates=tuple(valid if opts.return_all_candidates else [selected]),
            diagnostics=IKDiagnostics(
                message=raw.message,
                seed_reports=raw.seed_reports,
                validation_reports=tuple(validation_reports),
                backend_statuses=(raw.status,),
            ),
        )

    status = _failure_status(raw.status, failure_statuses)
    return IKResult(
        status=status,
        q=None,
        pose_error=(float("inf"), float("inf")),
        iterations=raw.iterations,
        elapsed_ms=raw.elapsed_ms,
        backend_used=getattr(backend_impl, "name", "unknown"),
        candidates=(),
        diagnostics=IKDiagnostics(
            message=raw.message or status.value,
            seed_reports=raw.seed_reports,
            validation_reports=tuple(validation_reports),
            backend_statuses=(raw.status, *failure_statuses),
        ),
    )


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _default_pose_problem(
    model: KinematicModel,
    frame_id: str,
    T_target: np.ndarray,
    q_seed: np.ndarray,
    options: IKOptions,
    pos_tol: float | None,
    rot_tol: float | None,
) -> IKProblem:
    """Build the default IKProblem used by ik_local/ik."""
    problem = IKProblem()
    problem.add_task(
        PoseTarget(
            frame_id=frame_id,
            T_target=np.asarray(T_target, dtype=float),
            pos_tol=pos_tol if pos_tol is not None else options.pos_tol,
            rot_tol=rot_tol if rot_tol is not None else options.rot_tol,
        )
    )
    q_min, q_max = model.active_position_limits()
    problem.add_constraint(JointPositionBounds(q_min, q_max, margin=options.joint_margin))
    problem.add_cost(
        SeedRegularization(np.asarray(q_seed, dtype=float), weight=options.seed_regularization_weight)
    )
    problem.add_cost(JointCenteringCost(weight=options.joint_centering_weight))
    return problem


def _run_analytical_backend(
    backend: AnalyticalIK,
    model: KinematicModel,
    spec: IKProblemSpec,
    q_seed: np.ndarray,
) -> BackendResult:
    branches = backend.solve_branches(model, spec, q_seed)
    candidates = tuple(
        BackendCandidate(
            q=np.asarray(q, dtype=float),
            pose_error=(float("inf"), float("inf")),
            iterations=0,
            seed_index=i,
        )
        for i, q in enumerate(branches)
    )
    return BackendResult(
        status=IKStatus.SUCCESS if candidates else IKStatus.NO_VALID_CANDIDATE,
        candidates=candidates,
        iterations=0,
        elapsed_ms=0.0,
        message="analytical branches produced",
    )


def _failure_status(raw_status: IKStatus, failure_statuses: list[IKStatus]) -> IKStatus:
    if raw_status in {
        IKStatus.TIMEOUT,
        IKStatus.MAX_ITERATIONS,
        IKStatus.NUMERICAL_FAILURE,
        IKStatus.UNREACHABLE,
    }:
        return raw_status
    priority = [
        IKStatus.INVALID_INPUT,
        IKStatus.JOINT_LIMIT_VIOLATION,
        IKStatus.POSE_ERROR_TOO_HIGH,
        IKStatus.SINGULARITY_RISK,
        IKStatus.FINAL_COLLISION,
        IKStatus.CONSTRAINT_VIOLATION,
    ]
    for status in priority:
        if status in failure_statuses:
            return status
    return IKStatus.NO_VALID_CANDIDATE


def _candidate_score(
    q: np.ndarray,
    pose_error: tuple[float, float],
    model: KinematicModel,
) -> float:
    q_min, q_max = model.active_position_limits()
    center = 0.5 * (q_min + q_max)
    span = np.maximum(q_max - q_min, 1e-9)
    center_cost = float(np.linalg.norm((q - center) / span))
    return pose_error[0] + pose_error[1] + 1e-4 * center_cost


def _dedupe_and_sort(candidates: list[IKCandidate]) -> list[IKCandidate]:
    ordered = sorted(candidates, key=lambda c: c.score)
    deduped: list[IKCandidate] = []
    for candidate in ordered:
        if not any(np.allclose(candidate.q, other.q, atol=1e-6) for other in deduped):
            deduped.append(candidate)
    return deduped


def _result(
    status: IKStatus,
    q: np.ndarray | None,
    backend: str,
    pose_error: tuple[float, float],
    iterations: int,
    started: float,
    message: str,
) -> IKResult:
    return IKResult(
        status=status,
        q=q,
        pose_error=pose_error,
        iterations=iterations,
        elapsed_ms=(time.perf_counter() - started) * 1000.0,
        backend_used=backend,
        candidates=(),
        diagnostics=IKDiagnostics(message=message),
    )
