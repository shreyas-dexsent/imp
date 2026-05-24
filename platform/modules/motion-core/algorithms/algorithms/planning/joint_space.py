# Copyright (c) 2026 DexSent Robotics Pvt Ltd.
# All Rights Reserved.
#
# SPDX-License-Identifier: LicenseRef-Proprietary
# See LICENSE file for details.
#
# Author: Shreyas <shreyas.kumar@dexsentrobotics.com>
# Proprietary and confidential.

"""Joint-space path planning entrypoint."""
from __future__ import annotations

import time
from typing import Mapping, Union

import numpy as np

from algorithms.planning.backends.base import PathPlannerBackend
from algorithms.planning.backends.ompl import OMPLBackend
from algorithms.planning.backends.straight_line import StraightLineBackend
from algorithms.planning.options import PlanOptions
from algorithms.planning.path import Path, PathStatus
from algorithms.planning.result import PathDiagnostics, PathPlanResult
from algorithms.planning.state_validity import (
    make_composite_state_validity_fn,
    make_state_validity_fn,
)
from algorithms.resolved.kinematic_model import KinematicModel
from algorithms.resolved.scene import Scene


_BACKENDS: dict[str, type[PathPlannerBackend]] = {
    "ompl": OMPLBackend,
    "direct": StraightLineBackend,
}


# Composite-q type for the multi-robot path.
QInput = Union[np.ndarray, Mapping[str, np.ndarray]]


def plan_joint(
    model: KinematicModel,
    scene: Scene,
    q_start: QInput,
    q_goal: QInput,
    *,
    backend: str = "ompl",
    options: PlanOptions | None = None,
    chain_id: str | None = None,
) -> PathPlanResult:
    """Plan a collision-free joint-space path from `q_start` to `q_goal`.

    On `PathStatus.SUCCESS`, ``result.path.waypoints`` is a sequence of
    configurations from start to goal, every waypoint inside joint
    limits with the configured margin, every segment collision-free at
    `options.max_joint_step` resolution.

    Parameters
    ----------
    model
        Resolved kinematic model of the robot being planned for. In
        multi-robot worlds this is the robot whose state varies; other
        robots remain fixed at `scene.robot_states[other_id]`.
    scene
        Scene the planner runs against. The collision pipeline uses the
        scene's `collision_model`; pose updates land in `object_poses`.
    q_start, q_goal
        Either bare ndarrays (single-robot path) or dicts keyed by
        `robot_id` (multi-robot composite-state path). When dicts are
        passed, the planner plans simultaneously over all robots in the
        dict.
    backend
        ``"ompl"`` (default; production sampling-based planner) or
        ``"direct"`` (straight-line interpolation + per-sample validity).
        Use ``"direct"`` when the straight line is likely free (fast
        first attempt).
    options
        :class:`PlanOptions` knob bag. Defaults from `PlanOptions()`.
    chain_id
        Restrict the collision check to a specific kinematic chain.
        Only meaningful for single-robot calls.
    """
    started = time.perf_counter()
    opts = options or PlanOptions()

    if backend not in _BACKENDS:
        return _result(
            PathStatus.INVALID_INPUT,
            None,
            backend or "none",
            0,
            started,
            f"unknown backend: {backend!r}. Choose 'ompl' or 'direct'.",
        )

    is_composite = isinstance(q_start, Mapping)
    if is_composite != isinstance(q_goal, Mapping):
        return _result(
            PathStatus.INVALID_INPUT,
            None,
            backend,
            0,
            started,
            "q_start and q_goal must both be ndarrays (single-robot) or both dicts (multi-robot)",
        )

    if is_composite:
        return _plan_composite(
            model, scene, q_start, q_goal, backend, opts, started
        )

    return _plan_single_robot(
        model, scene, q_start, q_goal, backend, opts, chain_id, started
    )


# ---------------------------------------------------------------------------
# Single-robot path
# ---------------------------------------------------------------------------


def _plan_single_robot(
    model: KinematicModel,
    scene: Scene,
    q_start: np.ndarray,
    q_goal: np.ndarray,
    backend: str,
    opts: PlanOptions,
    chain_id: str | None,
    started: float,
) -> PathPlanResult:
    q_start = np.asarray(q_start, dtype=float)
    q_goal = np.asarray(q_goal, dtype=float)

    n = len(model.active_joint_names)
    if q_start.shape != (n,) or q_goal.shape != (n,):
        return _result(
            PathStatus.INVALID_INPUT,
            None,
            backend,
            0,
            started,
            f"q_start/q_goal must have shape ({n},); got {q_start.shape} / {q_goal.shape}",
        )

    lower, upper = model.active_position_limits()
    margin = opts.joint_margin

    # Cheap up-front rejections so the planner doesn't waste time.
    pre = _precheck_endpoints(
        q_start, q_goal, lower, upper, margin, scene, model, chain_id
    )
    if pre is not None:
        return _result(pre, None, backend, 0, started, pre.value)

    validity_fn = make_state_validity_fn(
        model, scene, chain_id=chain_id, margin=margin
    )

    backend_impl: PathPlannerBackend = _BACKENDS[backend]()
    raw = backend_impl.plan(
        q_start, q_goal, lower, upper, validity_fn, opts,
    )

    if raw.status is not PathStatus.SUCCESS or raw.waypoints is None:
        return _result(
            raw.status,
            None,
            backend_impl.name,
            raw.iterations,
            started,
            raw.message,
            extra=raw.extra,
        )

    path = Path(
        waypoints=raw.waypoints,
        joint_names=tuple(model.active_joint_names),
        cartesian_waypoints=None,
        metadata={
            "planner_used": backend_impl.name,
            **raw.extra,
        },
    )

    return PathPlanResult(
        status=PathStatus.SUCCESS,
        path=path,
        planner_used=backend_impl.name,
        iterations=raw.iterations,
        elapsed_ms=(time.perf_counter() - started) * 1000.0,
        diagnostics=PathDiagnostics(message=raw.message, extra=raw.extra),
    )


def _precheck_endpoints(
    q_start: np.ndarray,
    q_goal: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    margin: float,
    scene: Scene,
    model: KinematicModel,
    chain_id: str | None,
) -> PathStatus | None:
    if np.any(q_start < lower + margin) or np.any(q_start > upper - margin):
        return PathStatus.START_OUT_OF_LIMITS
    if np.any(q_goal < lower + margin) or np.any(q_goal > upper - margin):
        return PathStatus.GOAL_OUT_OF_LIMITS

    if scene.collision_model is None:
        return None

    from algorithms.collision import is_in_collision

    if is_in_collision(model, scene, q_start, chain_id=chain_id).in_collision:
        return PathStatus.START_IN_COLLISION
    if is_in_collision(model, scene, q_goal, chain_id=chain_id).in_collision:
        return PathStatus.GOAL_IN_COLLISION
    return None


# ---------------------------------------------------------------------------
# Multi-robot composite-state path
# ---------------------------------------------------------------------------


def _plan_composite(
    model: KinematicModel,
    scene: Scene,
    q_start: Mapping[str, np.ndarray],
    q_goal: Mapping[str, np.ndarray],
    backend: str,
    opts: PlanOptions,
    started: float,
) -> PathPlanResult:
    """Composite-state planning over multiple robots.

    The robots that vary are the ones present in both `q_start` and
    `q_goal` dicts. Each robot contributes its dof to the composite
    vector, in the order it appears in `scene.world.robots`. Robots not
    in either dict are frozen at `scene.robot_states[robot_id]`.
    """
    moving_ids = [r.id for r in scene.world.robots if r.id in q_start and r.id in q_goal]
    if not moving_ids:
        return _result(
            PathStatus.INVALID_INPUT,
            None,
            backend,
            0,
            started,
            "composite plan: q_start and q_goal share no robot ids",
        )

    # Per-robot KinematicModels and limit slabs, in the moving_ids order.
    robot_models = {}
    lower_parts, upper_parts = [], []
    start_parts, goal_parts = [], []
    dofs = []
    for rid in moving_ids:
        world_robot = scene.world.robot(rid)
        rm = KinematicModel.from_robot_system(world_robot.robot_system)
        robot_models[rid] = rm
        rl, ru = rm.active_position_limits()
        lower_parts.append(rl)
        upper_parts.append(ru)
        start_parts.append(np.asarray(q_start[rid], dtype=float))
        goal_parts.append(np.asarray(q_goal[rid], dtype=float))
        dofs.append(len(rm.active_joint_names))

    composite_lower = np.concatenate(lower_parts)
    composite_upper = np.concatenate(upper_parts)
    composite_start = np.concatenate(start_parts)
    composite_goal = np.concatenate(goal_parts)

    # Composite validity adapter: flat ndarray -> per-robot dict -> bool.
    composite_validity = make_composite_state_validity_fn(scene, margin=opts.joint_margin)
    frozen = {
        rid: scene.robot_states[rid]
        for rid in (r.id for r in scene.world.robots)
        if rid not in moving_ids
    }

    def adapt(q_flat: np.ndarray) -> bool:
        idx = 0
        q_dict = dict(frozen)
        for rid, d in zip(moving_ids, dofs):
            q_dict[rid] = q_flat[idx : idx + d]
            idx += d
        return composite_validity(q_dict)

    # Endpoint precheck.
    if not adapt(composite_start):
        return _result(
            PathStatus.START_IN_COLLISION,
            None,
            backend,
            0,
            started,
            "composite plan: start state invalid",
        )
    if not adapt(composite_goal):
        return _result(
            PathStatus.GOAL_IN_COLLISION,
            None,
            backend,
            0,
            started,
            "composite plan: goal state invalid",
        )

    backend_impl = _BACKENDS[backend]()
    raw = backend_impl.plan(
        composite_start, composite_goal,
        composite_lower, composite_upper,
        adapt, opts,
    )

    if raw.status is not PathStatus.SUCCESS or raw.waypoints is None:
        return _result(
            raw.status,
            None,
            backend_impl.name,
            raw.iterations,
            started,
            raw.message,
            extra=raw.extra,
        )

    # Composite joint_names: prepend robot_id to each name so the order
    # is unambiguous to downstream consumers.
    composite_joint_names: list[str] = []
    for rid, rm in robot_models.items():
        composite_joint_names.extend(f"{rid}:{name}" for name in rm.active_joint_names)

    path = Path(
        waypoints=raw.waypoints,
        joint_names=tuple(composite_joint_names),
        cartesian_waypoints=None,
        metadata={
            "planner_used": backend_impl.name,
            "composite": True,
            "moving_robots": tuple(moving_ids),
            "dofs_per_robot": tuple(dofs),
            **raw.extra,
        },
    )

    return PathPlanResult(
        status=PathStatus.SUCCESS,
        path=path,
        planner_used=backend_impl.name,
        iterations=raw.iterations,
        elapsed_ms=(time.perf_counter() - started) * 1000.0,
        diagnostics=PathDiagnostics(message=raw.message, extra=raw.extra),
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _result(
    status: PathStatus,
    path: Path | None,
    planner: str,
    iterations: int,
    started: float,
    message: str,
    *,
    extra: dict | None = None,
) -> PathPlanResult:
    return PathPlanResult(
        status=status,
        path=path,
        planner_used=planner,
        iterations=iterations,
        elapsed_ms=(time.perf_counter() - started) * 1000.0,
        diagnostics=PathDiagnostics(message=message, extra=dict(extra or {})),
    )
