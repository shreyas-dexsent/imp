"""Robot-engine smoke task for collision and motion-planning validation."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

from orchestrator.core.context import StationContext
from orchestrator.core.runs import RunState
from orchestrator.robot_engine_bridge import (
    build_scene_request,
    evaluate_scene,
    matrix_to_transform,
)
from orchestrator.tasks._pick_runtime import _move_to_pose

_log = logging.getLogger(__name__)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _timeline(ctx: StationContext, state: RunState, event: str, **fields: Any) -> None:
    try:
        payload = {"event": event, "run_id": state.run_id, "timestamp_ns": time.time_ns()}
        payload.update({key: _json_safe(value) for key, value in fields.items()})
        ctx.runs.append_event(state.run_id, payload)
    except Exception:
        _log.exception("dummy_testing: failed to append timeline event %s", event)


def run_dummy_testing(ctx: StationContext, state: RunState, handle: Any) -> None:
    """Run a pose program while checking robot-engine scene health.

    The task consumes the same operator_flow used by the task-flow UI. Move Pose,
    Intermediate Pose, Place, Delay, Repeat, While TRUE, and Gripper blocks are
    enough for validation loops without pulling in vision.
    """

    task = state.task if isinstance(state.task, dict) else {}
    dummy_cfg = task.get("dummy_testing") if isinstance(task.get("dummy_testing"), dict) else {}
    flow = task.get("operator_flow") if isinstance(task.get("operator_flow"), list) else []
    if not flow:
        flow = dummy_cfg.get("program") if isinstance(dummy_cfg.get("program"), list) else []
    if not flow:
        raise RuntimeError("dummy_testing_flow_empty")

    poses = {p.get("name"): p for p in ctx.poses.list(state.process_id) if isinstance(p, dict) and p.get("name")}
    default_profile = str(dummy_cfg.get("profile") or "normal")
    dry_run = bool(dummy_cfg.get("dry_run", False) or state.params.get("dry_run", False))
    strict_collision = bool(dummy_cfg.get("strict_collision", True))
    max_loop_iterations = int(dummy_cfg.get("max_loop_iterations") or state.params.get("max_loop_iterations") or 1)
    use_collision_aware_planning = dummy_cfg.get("use_collision_aware_planning", True) is not False
    fail_if_planning_unavailable = dummy_cfg.get("fail_if_planning_unavailable", use_collision_aware_planning) is not False

    scene_req, meta = build_scene_request(ctx, state.process_id, object_id="dummy_testing_none")
    scene_yaml = meta.get("dummy_testing", {}).get("scene_yaml_path") or ""
    _timeline(
        ctx,
        state,
        "DUMMY_TESTING_CONFIG",
        dry_run=dry_run,
        strict_collision=strict_collision,
        use_collision_aware_planning=use_collision_aware_planning,
        fail_if_planning_unavailable=fail_if_planning_unavailable,
        max_loop_iterations=max_loop_iterations,
        scene_yaml=scene_yaml,
        obstacle_count=len((meta.get("dummy_testing") or {}).get("obstacles") or []),
    )

    # Build MotionPlanningPipeline from scene.yaml if collision-aware planning is enabled.
    # The pipeline owns self-collision + world collision (obstacles from environment.obstacles)
    # and uses OMPL RRTConnect via plan_to_configuration().
    pipeline = None
    if use_collision_aware_planning:
        _timeline(ctx, state, "DUMMY_PLANNER_BUILD_REQUEST", planner="MotionPlanningPipeline", scene_yaml=scene_yaml)
        pipeline = _build_pipeline(scene_yaml)
        _timeline(
            ctx,
            state,
            "DUMMY_PLANNER_BUILD_RESPONSE",
            planner="MotionPlanningPipeline",
            ok=pipeline is not None,
            scene_yaml=scene_yaml,
            obstacles=pipeline.list_obstacles() if pipeline is not None and hasattr(pipeline, "list_obstacles") else [],
        )

    # Fall back to the lightweight CollisionWorld-based planner_ctx when the
    # full pipeline is unavailable (e.g. Pinocchio not installed), or as secondary fallback
    # when pipeline cannot resolve goal joints (missing joint values, no IK exposed).
    planner_ctx = None
    if use_collision_aware_planning:
        _timeline(ctx, state, "DUMMY_PLANNER_BUILD_REQUEST", planner="CollisionAwarePlanner")
        planner_ctx = _build_planner_ctx(scene_req)
        _timeline(
            ctx,
            state,
            "DUMMY_PLANNER_BUILD_RESPONSE",
            planner="CollisionAwarePlanner",
            ok=planner_ctx is not None,
            joint_names=(planner_ctx or {}).get("joint_names") or [],
            lower_limits=(planner_ctx or {}).get("lower_limits") or [],
            upper_limits=(planner_ctx or {}).get("upper_limits") or [],
        )

    _check_scene(ctx, state, strict_collision)
    _execute_steps(
        ctx,
        state,
        handle,
        flow,
        poses,
        default_profile,
        dry_run=dry_run,
        strict_collision=strict_collision,
        max_loop_iterations=max_loop_iterations,
        pipeline=pipeline,
        planner_ctx=planner_ctx,
        require_planned_motion=fail_if_planning_unavailable,
    )


# ---------------------------------------------------------------------------
# Pipeline construction (MotionPlanningPipeline from scene.yaml)
# ---------------------------------------------------------------------------

def _build_pipeline(scene_yaml: str) -> Optional[Any]:
    """Build a MotionPlanningPipeline from the scene.yaml file."""
    try:
        from robot_engine.planning_core import MotionPlanningPipeline

        yaml_path = Path(scene_yaml)
        if not yaml_path.exists():
            _log.warning(f"dummy_testing: scene.yaml not found at {yaml_path}")
            return None

        pipeline = MotionPlanningPipeline.from_config(yaml_path)
        _log.info(f"dummy_testing: pipeline loaded from {yaml_path}, obstacles={pipeline.list_obstacles()}")
        return pipeline
    except Exception as exc:
        _log.warning(f"dummy_testing: MotionPlanningPipeline build failed: {exc}")
        return None


def _plan_with_pipeline(
    pipeline: Any,
    start_q: List[float],
    goal_q: List[float],
) -> Tuple[Optional[List[List[float]]], Dict[str, Any], Any]:
    """Plan via MotionPlanningPipeline.plan_to_configuration().

    Returns (path_waypoints, info, timed_trajectory).  timed_trajectory carries
    Ruckig positions+velocities for smooth feed-forward execution; it may be None
    if the pipeline does not produce one.
    """
    try:
        traj = pipeline.plan_to_configuration(
            "arm",
            np.asarray(start_q, dtype=float),
            np.asarray(goal_q, dtype=float),
        )
        if traj is None:
            return None, {"reason": "planner_returned_none"}, None
        waypoints = [np.asarray(q, dtype=float).tolist() for q in traj.path_waypoints]
        if not waypoints:
            return None, {"reason": "planner_returned_empty_path"}, None
        return waypoints, {"reason": "OK"}, traj
    except Exception as exc:
        _log.warning(f"dummy_testing: pipeline planning failed: {exc}")
        return None, {"reason": "exception", "error": str(exc)}, None


def _plan_with_pipeline_to_pose(
    pipeline: Any,
    start_q: List[float],
    pose: Dict[str, Any],
) -> Tuple[Optional[List[List[float]]], Dict[str, Any], Any]:
    """Plan via MotionPlanningPipeline.plan_to_pose(), including Pinocchio IK."""
    target = _target_matrix_from_saved_pose(pose)
    if target is None:
        return None, {"reason": "target_pose_unavailable"}, None
    try:
        seeds = _pipeline_ik_seeds(pipeline, start_q)
        traj = pipeline.plan_to_pose(
            "arm",
            np.asarray(start_q, dtype=float),
            target,
            ik_seeds=seeds,
        )
        if traj is None:
            return None, {
                "reason": "planner_returned_none",
                "target_matrix": target.tolist(),
                "seed_count": len(seeds),
            }, None
        waypoints = [np.asarray(q, dtype=float).tolist() for q in getattr(traj, "path_waypoints", [])]
        if not waypoints and getattr(traj, "positions", None):
            waypoints = [np.asarray(q, dtype=float).tolist() for q in traj.positions]
        timed_traj = traj if waypoints else None
        return (waypoints if waypoints else None), {
            "reason": "OK" if waypoints else "planner_returned_empty_path",
            "target_matrix": target.tolist(),
            "seed_count": len(seeds),
            "planner_used": getattr(traj, "planner_used", None),
            "duration": getattr(traj, "duration", None),
        }, timed_traj
    except Exception as exc:
        _log.warning(f"dummy_testing: pipeline pose planning failed: {exc}")
        return None, {
            "reason": "exception",
            "error": str(exc),
            "target_matrix": target.tolist(),
        }, None


def _target_matrix_from_saved_pose(pose: Dict[str, Any]) -> Optional[np.ndarray]:
    tcp_pose = pose.get("tcp_pose") if isinstance(pose.get("tcp_pose"), dict) else {}
    pos = tcp_pose.get("position_m")
    quat = tcp_pose.get("quat_xyzw")
    if not (isinstance(pos, list) and len(pos) == 3 and isinstance(quat, list) and len(quat) == 4):
        return None
    x, y, z, w = float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])
    norm = max(float(np.linalg.norm([x, y, z, w])), 1e-12)
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    target = np.array([
        [1 - 2*(y*y + z*z),   2*(x*y - z*w),   2*(x*z + y*w),   float(pos[0])],
        [  2*(x*y + z*w), 1 - 2*(x*x + z*z),   2*(y*z - x*w),   float(pos[1])],
        [  2*(x*z - y*w),   2*(y*z + x*w), 1 - 2*(x*x + y*y),   float(pos[2])],
        [              0,               0,               0,                1.0],
    ], dtype=float)
    return target


def _pipeline_ik_seeds(pipeline: Any, start_q: List[float]) -> List[np.ndarray]:
    seeds: List[np.ndarray] = [np.asarray(start_q, dtype=float)]
    try:
        group = pipeline.semantics.group("arm")
        lower = np.asarray(pipeline.semantics.joint_lower("arm"), dtype=float)
        upper = np.asarray(pipeline.semantics.joint_upper("arm"), dtype=float)
        if lower.shape == upper.shape and lower.size == len(start_q):
            seeds.append((lower + upper) * 0.5)
        if len(group.joints) == 7:
            seeds.append(np.asarray([0.0, -0.7853981633974483, 0.0, -2.356194490192345, 0.0, 1.5707963267948966, 0.7853981633974483], dtype=float))
        try:
            seeds.append(np.asarray(group.seed_array(), dtype=float))
        except Exception:
            pass
    except Exception:
        pass
    unique: List[np.ndarray] = []
    for seed in seeds:
        if seed.size != len(start_q):
            continue
        if not any(np.allclose(seed, old, atol=1e-9) for old in unique):
            unique.append(seed)
    return unique


# ---------------------------------------------------------------------------
# Fallback: lightweight CollisionWorld planner_ctx (no Pinocchio needed)
# ---------------------------------------------------------------------------

def _build_planner_ctx(scene_req: Any) -> Optional[Dict[str, Any]]:
    """Build a lightweight collision-aware planning context from the scene request."""
    try:
        from robot_engine.kinematics.kinematic_chain import KinematicChain
        from robot_engine.collision.collision_world import CollisionWorld
        from robot_engine.interfaces.schemas import CollisionMatrix

        chain = scene_req.chain
        if chain is None:
            return None

        kinematic_chain = KinematicChain(chain)
        joint_names = kinematic_chain.joint_names
        lower_limits = [j.lower for j in chain.joints if j.joint_type != "fixed"]
        upper_limits = [j.upper for j in chain.joints if j.joint_type != "fixed"]

        matrix = scene_req.collision_matrix or CollisionMatrix(default_action="check")
        world = CollisionWorld.from_configs(scene_req.collision_objects or [], matrix)

        # Track robot-link object IDs for FK re-posing during planning.
        robot_link_ids: Dict[str, str] = {}  # object_id -> FK link_name
        gripper_mount_link = str(getattr(chain, "tip_frame", "") or "")
        gripper_local_poses: Dict[str, np.ndarray] = {}
        fk0 = kinematic_chain.forward_matrices(scene_req.joint_state or {}).transforms
        mount0 = fk0.get(gripper_mount_link)
        for obj in (scene_req.collision_objects or []):
            if getattr(obj, "group", None) == "robot":
                parts = obj.object_id.split(":", 2)
                link_name = parts[1] if len(parts) >= 2 else obj.object_id
                robot_link_ids[obj.object_id] = link_name
            elif getattr(obj, "group", None) == "gripper" and mount0 is not None:
                gripper_local_poses[obj.object_id] = np.linalg.inv(np.asarray(mount0, dtype=float)) @ np.asarray(obj.pose.matrix, dtype=float)

        return {
            "chain": chain,
            "kinematic_chain": kinematic_chain,
            "joint_names": joint_names,
            "lower_limits": lower_limits,
            "upper_limits": upper_limits,
            "world": world,
            "robot_link_ids": robot_link_ids,
            "gripper_mount_link": gripper_mount_link,
            "gripper_local_poses": gripper_local_poses,
        }
    except Exception as exc:
        _log.warning(f"dummy_testing: planner context build failed: {exc}")
        return None


def _make_state_validity_fn(planner_ctx: Dict[str, Any]):
    """Return q_array -> bool (True = collision-free within joint limits)."""
    from robot_engine.collision.collision_checker import check_active_pairs

    kinematic_chain = planner_ctx["kinematic_chain"]
    joint_names: List[str] = planner_ctx["joint_names"]
    lower: List[float] = planner_ctx["lower_limits"]
    upper: List[float] = planner_ctx["upper_limits"]
    world = planner_ctx["world"]
    robot_link_ids: Dict[str, str] = planner_ctx["robot_link_ids"]
    gripper_mount_link: str = planner_ctx.get("gripper_mount_link") or ""
    gripper_local_poses: Dict[str, np.ndarray] = planner_ctx.get("gripper_local_poses") or {}

    def is_valid(q_array: Any) -> bool:
        q_list = list(np.asarray(q_array, dtype=float).tolist())
        for i, v in enumerate(q_list):
            if i >= len(lower):
                break
            if v < lower[i] - 1e-6 or v > upper[i] + 1e-6:
                return False
        if not robot_link_ids:
            return True
        q_dict = {name: float(q_list[i]) for i, name in enumerate(joint_names) if i < len(q_list)}
        fk = kinematic_chain.forward_matrices(q_dict).transforms
        for object_id, link_name in robot_link_ids.items():
            if link_name not in fk:
                continue
            try:
                world.update_pose(object_id, matrix_to_transform("world", object_id, fk[link_name]))
            except Exception:
                pass
        mount_tf = fk.get(gripper_mount_link)
        if mount_tf is not None:
            for object_id, local_tf in gripper_local_poses.items():
                try:
                    world.update_pose(object_id, matrix_to_transform("world", object_id, np.asarray(mount_tf) @ np.asarray(local_tf)))
                except Exception:
                    pass
        try:
            return not check_active_pairs(world).collision
        except Exception:
            return True

    return is_valid


def _plan_collision_free(
    planner_ctx: Dict[str, Any],
    start_q: List[float],
    goal_q: List[float],
) -> Tuple[Optional[List[List[float]]], Dict[str, Any]]:
    """Plan via CollisionAwarePlanner + state_validity_fn."""
    from robot_engine.path_planning.collision_aware_planner import CollisionAwarePlanner
    from robot_engine.path_planning.planner_base import PathRequest

    request = PathRequest(
        start=np.asarray(start_q, dtype=float),
        goal=np.asarray(goal_q, dtype=float),
        joint_limits=(planner_ctx["lower_limits"], planner_ctx["upper_limits"]),
        state_validity_fn=_make_state_validity_fn(planner_ctx),
        max_joint_step=0.05,
        max_iterations=5000,
        timeout=30.0,
        goal_bias=0.1,
        require_collision_aware_planning=True,
    )
    result = CollisionAwarePlanner().plan(request)
    path = [np.asarray(q, dtype=float).tolist() for q in result.q_waypoints]
    if result.success:
        if result.planner_used == "JOINT_DIRECT":
            return None, {
                "reason": "DIRECT_PATH_REJECTED_FOR_DUMMY_TESTING",
                "planner_used": result.planner_used,
                "waypoint_count": len(path),
                "length": result.length,
                "note": "Dummy Testing requires an actual collision-avoidance planner path; straight joint interpolation is not accepted.",
                "q_waypoints": path,
            }
        return path, {
            "reason": "OK",
            "planner_used": result.planner_used,
            "planning_time": result.planning_time,
            "length": result.length,
            "minimum_clearance": result.minimum_clearance,
        }
    return None, {
        "reason": result.rejection_reason,
        "planner_used": result.planner_used,
        "failed_stage": result.failed_stage,
        "failed_waypoint_index": result.failed_waypoint_index,
        "failed_segment_index": result.failed_segment_index,
        "colliding_pair": result.colliding_pair,
        "debug_info": result.debug_info,
    }


def _near_goal(start_q: List[float], goal_q: List[float], tolerance: float = 0.002) -> bool:
    if len(start_q) != len(goal_q):
        return False
    delta = np.asarray(goal_q, dtype=float) - np.asarray(start_q, dtype=float)
    return float(np.linalg.norm(delta)) <= tolerance


# ---------------------------------------------------------------------------
# IK: resolve tcp_pose → joint positions for planning
# ---------------------------------------------------------------------------

def _resolve_goal_joints(
    chain_or_planner_ctx: Any,
    pose: Dict[str, Any],
    current_q: List[float],
) -> Optional[List[float]]:
    """Return goal joint positions from pose, solving IK when only tcp_pose is stored."""
    goal_joints = pose.get("joints")
    if isinstance(goal_joints, list) and len(goal_joints) > 0:
        return [float(v) for v in goal_joints]

    tcp_pose = pose.get("tcp_pose") if isinstance(pose.get("tcp_pose"), dict) else {}
    pos = tcp_pose.get("position_m")
    quat = tcp_pose.get("quat_xyzw")
    if not (isinstance(pos, list) and len(pos) == 3 and isinstance(quat, list) and len(quat) == 4):
        return None

    # Warn if pose was recorded before TCP-offset fix (frame="base" with no explicit tcp frame).
    # Old poses store the EE/flange position; new poses store the fingertip (tcp frame) position.
    stored_frame = str(tcp_pose.get("frame") or "")
    if stored_frame == "base":
        _log.warning(
            "dummy_testing: pose '%s' has frame='base' — likely recorded before TCP-offset fix. "
            "Re-record this pose to get correct fingertip positioning.",
            pose.get("name"),
        )

    try:
        from robot_engine.interfaces.ui_api import solve_ik
        from robot_engine.interfaces.schemas import IKRequest
        from robot_engine.kinematics.kinematic_chain import KinematicChain

        # chain is either planner_ctx["chain"] (KinematicChainConfig) or passed directly
        chain = chain_or_planner_ctx["chain"] if isinstance(chain_or_planner_ctx, dict) else chain_or_planner_ctx
        joint_names = chain_or_planner_ctx["joint_names"] if isinstance(chain_or_planner_ctx, dict) else None
        if joint_names is None:
            joint_names = KinematicChain(chain).joint_names

        seed = {name: float(current_q[i]) for i, name in enumerate(joint_names) if i < len(current_q)}
        seeds = [seed]
        if isinstance(chain_or_planner_ctx, dict):
            lower = chain_or_planner_ctx.get("lower_limits") or []
            upper = chain_or_planner_ctx.get("upper_limits") or []
            if len(lower) >= len(joint_names) and len(upper) >= len(joint_names):
                seeds.append({
                    name: float((float(lower[i]) + float(upper[i])) * 0.5)
                    for i, name in enumerate(joint_names)
                })
        if len(joint_names) == 7:
            franka_home = [0.0, -0.7853981633974483, 0.0, -2.356194490192345, 0.0, 1.5707963267948966, 0.7853981633974483]
            seeds.append({name: franka_home[i] for i, name in enumerate(joint_names)})

        # Build 4x4 target matrix from quat_xyzw (x, y, z, w)
        x, y, z, w = float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])
        target_matrix = np.array([
            [1 - 2*(y*y + z*z),   2*(x*y - z*w),   2*(x*z + y*w),   float(pos[0])],
            [  2*(x*y + z*w), 1 - 2*(x*x + z*z),   2*(y*z - x*w),   float(pos[1])],
            [  2*(x*z - y*w),   2*(y*z + x*w), 1 - 2*(x*x + y*y),   float(pos[2])],
            [              0,               0,               0,                1.0],
        ], dtype=float)
        kinematic_chain = KinematicChain(chain)
        target = matrix_to_transform(chain.base_frame, kinematic_chain.tcp_frame, target_matrix)

        result = None
        for candidate_seed in seeds:
            result = solve_ik(IKRequest(
                chain=chain,
                target=target,
                seed=candidate_seed,
                max_iterations=500,
                position_tolerance=0.001,
                orientation_tolerance=0.05,
                damping=0.01,
            ))
            if result.ok:
                break
            result = solve_ik(IKRequest(
                chain=chain,
                target=target,
                seed=candidate_seed,
                max_iterations=500,
                position_tolerance=0.01,
                orientation_tolerance=10.0,
                damping=0.05,
                mode="position",
                minimum_joint_motion_weight=0.01,
                preferred_posture=candidate_seed,
                preferred_posture_weight=0.01,
                joint_limit_avoidance_weight=0.01,
            ))
            if result.ok:
                break
        if result.ok:
            return [float(result.joint_positions.get(name, current_q[i])) for i, name in enumerate(joint_names)]
        _log.warning(
            "dummy_testing: IK not converged for pose '%s': reason=%s position_error=%s orientation_error=%s",
            pose.get("name"),
            getattr(result, "reason", None),
            getattr(result, "position_error", None),
            getattr(result, "orientation_error", None),
        )
    except Exception as exc:
        _log.warning(f"dummy_testing: IK failed for pose '{pose.get('name')}': {exc}")
    return None


# ---------------------------------------------------------------------------
# Step execution
# ---------------------------------------------------------------------------

def _execute_steps(
    ctx: StationContext,
    state: RunState,
    handle: Any,
    steps: Iterable[Dict[str, Any]],
    poses: Dict[str, Dict[str, Any]],
    default_profile: str,
    *,
    dry_run: bool,
    strict_collision: bool,
    max_loop_iterations: int,
    pipeline: Optional[Any],
    planner_ctx: Optional[Dict[str, Any]],
    require_planned_motion: bool,
) -> None:
    for raw in steps:
        if handle.stop_event.is_set():
            raise RuntimeError("run_stopped")
        if not isinstance(raw, dict):
            continue
        step_type = str(raw.get("type") or "").strip().lower()
        if step_type in {"set_task_type", "config_vision_core", "config_vision_quality", "config_pick_profile", "config_follow_profile", "config_pallatizing_profile"}:
            continue
        if step_type == "while_true":
            body = raw.get("body") if isinstance(raw.get("body"), list) else []
            for _ in range(max(1, max_loop_iterations)):
                _execute_steps(
                    ctx,
                    state,
                    handle,
                    body,
                    poses,
                    default_profile,
                    dry_run=dry_run,
                    strict_collision=strict_collision,
                    max_loop_iterations=max_loop_iterations,
                    pipeline=pipeline,
                    planner_ctx=planner_ctx,
                    require_planned_motion=require_planned_motion,
                )
            continue
        if step_type == "delay":
            time.sleep(max(0.0, float(raw.get("ms") or 0) / 1000.0))
            continue
        if step_type in {"move_pose", "intermediate_pose", "place"}:
            pose_name = str(raw.get("pose_name") or "").strip()
            pose = poses.get(pose_name)
            if not pose:
                raise RuntimeError(f"pose_not_found:{pose_name}")
            _check_scene(ctx, state, strict_collision)
            if not dry_run:
                move_pose = dict(pose)
                move_pose["profile"] = str(raw.get("profile") or pose.get("profile") or default_profile)
                _execute_move(ctx, state, move_pose, move_pose["profile"], pipeline, planner_ctx, poses, require_planned_motion)
            continue
        if step_type == "gripper_action" and not dry_run:
            action = str(raw.get("action") or "close").strip().lower()
            api = "robot.open_gripper" if action == "open" else "robot.close_gripper"
            _timeline(ctx, state, "DUMMY_ROBOT_API_REQUEST", api=api)
            try:
                if action == "open":
                    ctx.robot.open_gripper()
                else:
                    ctx.robot.close_gripper()
                _timeline(ctx, state, "DUMMY_ROBOT_API_RESPONSE", api=api, ok=True)
            except Exception as exc:
                _timeline(ctx, state, "DUMMY_ROBOT_API_RESPONSE", api=api, ok=False, error=str(exc))
                raise


def _execute_move(
    ctx: StationContext,
    state: RunState,
    pose: Dict[str, Any],
    profile: str,
    pipeline: Optional[Any],
    planner_ctx: Optional[Dict[str, Any]],
    pose_index: Dict[str, Dict[str, Any]],
    require_planned_motion: bool,
) -> None:
    """Execute a move to pose through a collision-aware planner when requested."""
    pose_name = str(pose.get("name") or pose.get("pose_name") or "")
    _timeline(ctx, state, "DUMMY_ROBOT_API_REQUEST", api="robot.get_state", pose_name=pose_name)
    try:
        robot_state = ctx.robot.get_state() or {}
        _timeline(
            ctx,
            state,
            "DUMMY_ROBOT_API_RESPONSE",
            api="robot.get_state",
            ok=True,
            q=robot_state.get("q") or [],
        )
    except Exception as exc:
        _timeline(ctx, state, "DUMMY_ROBOT_API_RESPONSE", api="robot.get_state", ok=False, error=str(exc))
        raise
    current_q_raw = robot_state.get("q") or []
    if not isinstance(current_q_raw, list) or len(current_q_raw) == 0:
        if require_planned_motion:
            raise RuntimeError(f"dummy_testing_robot_state_missing:{pose_name}")
        _timeline(ctx, state, "DUMMY_DIRECT_MOVE_REQUEST", pose_name=pose_name, profile=profile, reason="missing_robot_state_q")
        _move_to_pose(ctx, pose, profile, prefer_cartesian=False, pose_index=pose_index)
        _timeline(ctx, state, "DUMMY_DIRECT_MOVE_RESPONSE", pose_name=pose_name, ok=True)
        return

    start_q = [float(v) for v in current_q_raw]
    _timeline(ctx, state, "DUMMY_MOVE_TARGET", pose_name=pose_name, profile=profile, start_q=start_q)

    # --- Try MotionPlanningPipeline (scene.yaml path, full self+world collision) ---
    if pipeline is not None:
        # Get goal joints from the pipeline's semantics (uses its own IK/joint_names)
        goal_q = _resolve_goal_joints_for_pipeline(pipeline, pose, start_q)
        if goal_q is not None:
            if _near_goal(start_q, goal_q):
                _timeline(
                    ctx,
                    state,
                    "DUMMY_MOVE_SKIPPED",
                    pose_name=pose_name,
                    reason="already_at_goal",
                    start_q=start_q,
                    goal_q=goal_q,
                )
                return
            _timeline(
                ctx,
                state,
                "DUMMY_PLANNER_REQUEST",
                planner="MotionPlanningPipeline",
                pose_name=pose_name,
                start_q=start_q,
                goal_q=goal_q,
            )
            waypoints, plan_info, timed_traj = _plan_with_pipeline(pipeline, start_q, goal_q)
            _timeline(
                ctx,
                state,
                "DUMMY_PLANNER_RESPONSE",
                planner="MotionPlanningPipeline",
                pose_name=pose_name,
                ok=waypoints is not None,
                waypoint_count=len(waypoints or []),
                q_waypoints=waypoints or [],
                details=plan_info,
            )
            if waypoints is not None:
                _execute_timed_trajectory(ctx, state, pose_name, profile, "MotionPlanningPipeline", timed_traj, waypoints)
                return
            _log.warning(f"dummy_testing: pipeline planning failed for '{pose.get('name')}', trying fallback")
        else:
            target_matrix = _target_matrix_from_saved_pose(pose)
            _timeline(
                ctx,
                state,
                "DUMMY_PLANNER_REQUEST",
                planner="MotionPlanningPipeline",
                mode="tcp_pose",
                pose_name=pose_name,
                start_q=start_q,
                target_matrix=target_matrix.tolist() if target_matrix is not None else None,
            )
            waypoints, plan_info, timed_traj = _plan_with_pipeline_to_pose(pipeline, start_q, pose)
            _timeline(
                ctx,
                state,
                "DUMMY_PLANNER_RESPONSE",
                planner="MotionPlanningPipeline",
                mode="tcp_pose",
                pose_name=pose_name,
                ok=waypoints is not None,
                waypoint_count=len(waypoints or []),
                q_waypoints=waypoints or [],
                details=plan_info,
            )
            if waypoints is not None:
                _execute_timed_trajectory(ctx, state, pose_name, profile, "MotionPlanningPipeline", timed_traj, waypoints)
                return
            _log.warning(f"dummy_testing: pipeline pose planning failed for '{pose.get('name')}', trying fallback")

    # --- Try lightweight CollisionAwarePlanner (CollisionWorld fallback) ---
    if planner_ctx is not None:
        _log_ik_seed_config(ctx, state, pose, start_q, planner_ctx)
        goal_q = _resolve_goal_joints(planner_ctx, pose, start_q)
        if goal_q is not None and len(goal_q) == len(start_q):
            if _near_goal(start_q, goal_q):
                _timeline(
                    ctx,
                    state,
                    "DUMMY_MOVE_SKIPPED",
                    pose_name=pose_name,
                    reason="already_at_goal",
                    start_q=start_q,
                    goal_q=goal_q,
                )
                return
            _timeline(
                ctx,
                state,
                "DUMMY_PLANNER_REQUEST",
                planner="CollisionAwarePlanner",
                pose_name=pose_name,
                start_q=start_q,
                goal_q=goal_q,
            )
            waypoints, plan_info = _plan_collision_free(planner_ctx, start_q, goal_q)
            _timeline(
                ctx,
                state,
                "DUMMY_PLANNER_RESPONSE",
                planner="CollisionAwarePlanner",
                pose_name=pose_name,
                ok=waypoints is not None,
                waypoint_count=len(waypoints or []),
                q_waypoints=waypoints or [],
                details=plan_info,
            )
            if waypoints is not None:
                _execute_joint_waypoints(ctx, state, pose_name, profile, "CollisionAwarePlanner", waypoints[1:])
                return
            _log.warning(f"dummy_testing: collision-aware planning failed for '{pose.get('name')}', using direct move")
        else:
            _timeline(
                ctx,
                state,
                "DUMMY_PLANNER_RESPONSE",
                planner="CollisionAwarePlanner",
                pose_name=pose_name,
                ok=False,
                reason="goal_joints_unavailable",
                goal_q=goal_q,
                start_q_len=len(start_q),
            )

    # --- Direct move (no collision avoidance) ---
    if require_planned_motion:
        raise RuntimeError(f"dummy_testing_planning_failed:{pose_name}")
    _timeline(ctx, state, "DUMMY_DIRECT_MOVE_REQUEST", pose_name=pose_name, profile=profile, reason="planning_not_required")
    _move_to_pose(ctx, pose, profile, prefer_cartesian=False, pose_index=pose_index)
    _timeline(ctx, state, "DUMMY_DIRECT_MOVE_RESPONSE", pose_name=pose_name, ok=True)


def _log_ik_seed_config(
    ctx: StationContext,
    state: RunState,
    pose: Dict[str, Any],
    start_q: List[float],
    planner_ctx: Dict[str, Any],
) -> None:
    if isinstance(pose.get("joints"), list) and pose.get("joints"):
        return
    tcp_pose = pose.get("tcp_pose") if isinstance(pose.get("tcp_pose"), dict) else {}
    if not tcp_pose:
        return
    joint_names = planner_ctx.get("joint_names") or []
    lower = planner_ctx.get("lower_limits") or []
    upper = planner_ctx.get("upper_limits") or []
    seeds: List[Dict[str, Any]] = [{
        "source": "current_robot_state_q",
        "q": start_q,
        "note": "Primary IK seed comes from ctx.robot.get_state().q at run time.",
    }]
    if len(lower) >= len(joint_names) and len(upper) >= len(joint_names):
        seeds.append({
            "source": "joint_limit_midpoint",
            "q": [float((float(lower[i]) + float(upper[i])) * 0.5) for i in range(len(joint_names))],
            "note": "Fallback seed computed from URDF/planning joint limits.",
        })
    if len(joint_names) == 7:
        seeds.append({
            "source": "franka_reference_home",
            "q": [0.0, -0.7853981633974483, 0.0, -2.356194490192345, 0.0, 1.5707963267948966, 0.7853981633974483],
            "note": "Generic Franka elbow-up home posture fallback used only if the stored pose has tcp_pose but no joints.",
        })
    _timeline(
        ctx,
        state,
        "DUMMY_IK_SEED_CONFIG",
        pose_name=str(pose.get("name") or pose.get("pose_name") or ""),
        joint_names=joint_names,
        seeds=seeds,
    )


def _execute_timed_trajectory(
    ctx: StationContext,
    state: RunState,
    pose_name: str,
    profile: str,
    planner: str,
    timed_traj: Any,
    fallback_waypoints: List[List[float]],
) -> None:
    """Execute collision-free path as one continuous franky motion.

    Uses the sparse OMPL keyframes (path_waypoints) from the timed trajectory
    so franky's JointWaypointMotion can blend through them without stopping.
    Dense Ruckig points are NOT sent to franky — franky generates its own
    smooth trajectory between the sparse keyframes.
    """
    # Use sparse_waypoints (shortcutted OMPL keyframes, pre-interpolation) so franky
    # can blend through them as one smooth motion. path_waypoints has 80-100 dense
    # linearly-interpolated points which cause franky to stop at each one.
    sparse: List[List[float]] = []
    if timed_traj is not None:
        raw_wps = getattr(timed_traj, "sparse_waypoints", None) or getattr(timed_traj, "path_waypoints", None)
        if raw_wps:
            sparse = [
                (q.tolist() if hasattr(q, "tolist") else list(q))
                for q in raw_wps
            ]
    if not sparse:
        sparse = fallback_waypoints

    _execute_joint_waypoints(ctx, state, pose_name, profile, planner, sparse)


def _execute_joint_waypoints(
    ctx: StationContext,
    state: RunState,
    pose_name: str,
    profile: str,
    planner: str,
    waypoints: List[List[float]],
) -> None:
    smoothed = _smooth_execution_waypoints(waypoints)
    total = len(smoothed)
    if total <= 0:
        return
    packed = tuple(tuple(float(v) for v in q) for q in smoothed)

    # Prefer smooth single-motion waypoint execution over sequential movej_path
    if hasattr(ctx.robot, "move_joint_waypoints"):
        _timeline(
            ctx,
            state,
            "DUMMY_TRAJECTORY_EXECUTE",
            planner=planner,
            pose_name=pose_name,
            waypoint_count=total,
            raw_waypoint_count=len(waypoints),
            execution_mode="move_joint_waypoints",
        )
        _timeline(
            ctx,
            state,
            "DUMMY_ROBOT_API_REQUEST",
            api="robot.move_joint_waypoints",
            planner=planner,
            pose_name=pose_name,
            profile=profile,
            waypoint_count=total,
            q_waypoints=smoothed,
        )
        try:
            ctx.robot.move_joint_waypoints(packed, profile)
            _timeline(
                ctx,
                state,
                "DUMMY_ROBOT_API_RESPONSE",
                api="robot.move_joint_waypoints",
                planner=planner,
                pose_name=pose_name,
                ok=True,
                waypoint_count=total,
            )
        except Exception as exc:
            _timeline(
                ctx,
                state,
                "DUMMY_ROBOT_API_RESPONSE",
                api="robot.move_joint_waypoints",
                planner=planner,
                pose_name=pose_name,
                ok=False,
                error=str(exc),
                waypoint_count=total,
            )
            raise
        return

    _timeline(
        ctx,
        state,
        "DUMMY_TRAJECTORY_EXECUTE",
        planner=planner,
        pose_name=pose_name,
        waypoint_count=total,
        raw_waypoint_count=len(waypoints),
        execution_mode="movej_path",
    )
    if hasattr(ctx.robot, "movej_path"):
        _timeline(
            ctx,
            state,
            "DUMMY_ROBOT_API_REQUEST",
            api="robot.movej_path",
            planner=planner,
            pose_name=pose_name,
            profile=profile,
            waypoint_count=total,
            q_waypoints=smoothed,
        )
        try:
            ctx.robot.movej_path(packed, profile)
            _timeline(
                ctx,
                state,
                "DUMMY_ROBOT_API_RESPONSE",
                api="robot.movej_path",
                planner=planner,
                pose_name=pose_name,
                ok=True,
                waypoint_count=total,
            )
        except Exception as exc:
            _timeline(
                ctx,
                state,
                "DUMMY_ROBOT_API_RESPONSE",
                api="robot.movej_path",
                planner=planner,
                pose_name=pose_name,
                ok=False,
                error=str(exc),
                waypoint_count=total,
            )
            raise
        return
    for idx, wp in enumerate(smoothed):
        _timeline(
            ctx,
            state,
            "DUMMY_ROBOT_API_REQUEST",
            api="robot.movej",
            planner=planner,
            pose_name=pose_name,
            profile=profile,
            waypoint_index=idx,
            waypoint_count=total,
            q=wp,
        )
        try:
            ctx.robot.movej(tuple(wp), profile)
            _timeline(
                ctx,
                state,
                "DUMMY_ROBOT_API_RESPONSE",
                api="robot.movej",
                planner=planner,
                pose_name=pose_name,
                ok=True,
                waypoint_index=idx,
                waypoint_count=total,
            )
        except Exception as exc:
            _timeline(
                ctx,
                state,
                "DUMMY_ROBOT_API_RESPONSE",
                api="robot.movej",
                planner=planner,
                pose_name=pose_name,
                ok=False,
                error=str(exc),
                waypoint_index=idx,
                waypoint_count=total,
            )
            raise


def _smooth_execution_waypoints(
    waypoints: List[List[float]],
    *,
    max_points: int = 8,
    min_joint_delta: float = 0.05,
) -> List[List[float]]:
    """Downsample to sparse keyframes so franky blends through them as one
    continuous s-curve instead of stopping at every dense collision-check point."""
    rows = [np.asarray(q, dtype=float) for q in waypoints if isinstance(q, list) and q]
    if len(rows) <= 2:
        return [q.tolist() for q in rows]
    # Always keep start and end; keep interior points only when joint delta is large
    kept = [rows[0]]
    for q in rows[1:-1]:
        if float(np.linalg.norm(q - kept[-1], ord=np.inf)) >= min_joint_delta:
            kept.append(q)
    kept.append(rows[-1])
    if len(kept) <= max_points:
        return [q.tolist() for q in kept]
    idxs = np.linspace(0, len(kept) - 1, max_points, dtype=int)
    compact = [kept[int(i)] for i in idxs]
    compact[-1] = kept[-1]
    return [q.tolist() for q in compact]


def _resolve_goal_joints_for_pipeline(
    pipeline: Any,
    pose: Dict[str, Any],
    current_q: List[float],
) -> Optional[List[float]]:
    """Resolve goal joints using the pipeline's planning group joint names."""
    try:
        joint_names = pipeline.semantics.group("arm").joints
        ctx_dict = {
            "chain": pipeline.semantics,  # not a KinematicChainConfig — use joint_names directly
            "joint_names": joint_names,
        }
        # Use the stored joints if available; otherwise fall back to IK via planner_ctx path
        goal_joints = pose.get("joints")
        if isinstance(goal_joints, list) and len(goal_joints) > 0:
            return [float(v) for v in goal_joints]

        # IK via robot_engine solve_ik using scene_req chain stored in planner_ctx
        # The pipeline doesn't expose KinematicChainConfig directly, so we skip IK here
        # and let the caller fall through to the planner_ctx path.
        return None
    except Exception as exc:
        _log.warning(f"dummy_testing: pipeline goal joint resolution failed: {exc}")
        return None


# ---------------------------------------------------------------------------
# Scene health check
# ---------------------------------------------------------------------------

def _check_scene(ctx: StationContext, state: RunState, strict_collision: bool) -> None:
    # Dummy testing validates robot/fixture collision. Product objects from old
    # pick tasks are intentionally excluded so stale vision config cannot fail
    # the smoke test before motion starts.
    scene_req, meta = build_scene_request(ctx, state.process_id, object_id="dummy_testing_none")
    _timeline(
        ctx,
        state,
        "DUMMY_SCENE_BUILD",
        object_id="dummy_testing_none",
        collision_object_count=len(scene_req.collision_objects or []),
        dummy_testing=meta.get("dummy_testing") or {},
    )
    payload = {"distance": True, "collision": True}
    _timeline(ctx, state, "DUMMY_SCENE_EVALUATE_REQUEST", object_id="dummy_testing_none", payload=payload)
    try:
        result = evaluate_scene(
            ctx,
            state.process_id,
            object_id="dummy_testing_none",
            payload=payload,
        )
        _timeline(
            ctx,
            state,
            "DUMMY_SCENE_EVALUATE_RESPONSE",
            ok=True,
            collision=((result.get("evaluation") or {}).get("collision") or {}),
            distance=((result.get("evaluation") or {}).get("distance") or {}),
        )
    except Exception as exc:
        _timeline(ctx, state, "DUMMY_SCENE_EVALUATE_RESPONSE", ok=False, error=str(exc))
        raise
    collision = ((result.get("evaluation") or {}).get("collision") or {})
    if strict_collision and collision.get("collision"):
        pairs = collision.get("colliding_pairs") or []
        _timeline(
            ctx,
            state,
            "DUMMY_COLLISION_DEBUG",
            pairs=pairs,
            debug=_collision_debug(scene_req, pairs),
        )
        raise RuntimeError(f"dummy_testing_collision:{pairs}")


def _collision_debug(scene_req: Any, pairs: List[List[str]]) -> List[Dict[str, Any]]:
    """Explain reported collision pairs using the exact objects given to robot_engine."""
    debug_rows: List[Dict[str, Any]] = []
    try:
        from robot_engine.collision.collision_checker import _coal_check_pair
        from robot_engine.collision.collision_world import CollisionWorld
        from robot_engine.collision.distance_queries import minimum_distance_pair
        from robot_engine.interfaces.schemas import CollisionMatrix

        matrix = scene_req.collision_matrix or CollisionMatrix(default_action="check")
        world = CollisionWorld.from_configs(scene_req.collision_objects or [], matrix)

        def describe(object_id: str) -> Dict[str, Any]:
            obj = world.objects.get(object_id)
            if obj is None:
                return {"object_id": object_id, "found": False}
            geom = obj.geometry
            mesh = geom.mesh
            return {
                "object_id": object_id,
                "found": True,
                "group": obj.group,
                "frame_id": geom.frame_id,
                "has_mesh": mesh is not None,
                "mesh_vertices": int(len(mesh.vertices)) if mesh is not None else 0,
                "mesh_faces": int(len(mesh.faces)) if mesh is not None else 0,
                "coal_ready": geom.coal_geometry is not None,
                "pose_matrix": obj.matrix.tolist(),
            }

        for pair in pairs:
            if not isinstance(pair, list) or len(pair) != 2:
                continue
            a_id, b_id = str(pair[0]), str(pair[1])
            a = world.objects.get(a_id)
            b = world.objects.get(b_id)
            row: Dict[str, Any] = {
                "pair": [a_id, b_id],
                "object_a": describe(a_id),
                "object_b": describe(b_id),
            }
            if a is not None and b is not None:
                dist = minimum_distance_pair(a, b)
                coal_result = _coal_check_pair(a, b)
                row.update(
                    {
                        "distance_query": dist.model_dump() if hasattr(dist, "model_dump") else dist.dict(),
                        "coal": {
                            "used": coal_result is not None,
                            "collision": coal_result[0] if coal_result is not None else None,
                            "contacts": coal_result[1] if coal_result is not None else [],
                        },
                    }
                )
            debug_rows.append(row)
    except Exception as exc:
        debug_rows.append({"error": str(exc)})
    return debug_rows
