from __future__ import annotations

from typing import Iterable, Optional

import numpy as np

from robot_engine.assets.asset_loader import AssetLoadError
from robot_engine.assets.collision_geometry import geometry_from_asset
from robot_engine.collision.collision_checker import check_active_pairs, check_pair
from robot_engine.collision.collision_world import CollisionWorld
from robot_engine.collision.distance_queries import minimum_distance_pair, minimum_distance_to_all, minimum_distances_active_pairs
from robot_engine.interfaces.schemas import (
    AlgorithmError,
    CollisionCheckRequest,
    CollisionCheckResult,
    CollisionGeometryStatus,
    CollisionMatrix,
    CollisionObjectConfig,
    DistanceQueryResult,
    FrameValidationResult,
    GraspByIdRequest,
    GraspFeasibilityRequest,
    GraspFeasibilityResult,
    GripperConfig,
    LoadedAssetStatus,
    MinimumDistanceRequest,
    ObjectAssetConfig,
    RobotModelConfig,
    TCPConfig,
    Transform3D,
    UISceneEvaluationRequest,
    UISceneEvaluationResult,
    UISceneRequest,
    UISceneStatus,
)
from robot_engine.kinematics.fk_solver import compute_fk
from robot_engine.kinematics.ik_solver import solve_ik
from robot_engine.kinematics.jacobian_solver import compute_jacobian
from robot_engine.kinematics.robot_model import RobotModel, load_robot_model as _load_robot_model
from robot_engine.planning.grasp_feasibility import evaluate_grasp_candidate as _evaluate_grasp_candidate
from robot_engine.planning.grasp_library import GraspLibrary
from robot_engine.motion import (
    compute_offset_frame,
    export_robot_trajectory,
    plan_approach_to_frame,
    plan_joint_move_to_frame,
    plan_lift_motion,
    plan_linear_move_to_frame,
    plan_motion_sequence,
    plan_retreat_from_frame,
    validate_trajectory as validate_motion_trajectory,
)
from robot_engine.path_planning.collision_aware_planner import CollisionAwarePlanner
from robot_engine.path_planning.planner_base import PathRequest
from robot_engine.trajectory.retiming import retime_joint_path
from robot_engine.trajectory.trajectory_validator import validate_acceleration_limits, validate_joint_position_limits, validate_velocity_limits


class RobotEngineContext:
    """State holder for one UI/backend robotics scene.

    The core modules stay stateless and UI-independent. This context owns the
    mutable scene state that a web backend normally keeps per user/session.
    """

    def __init__(self) -> None:
        self.world: Optional[CollisionWorld] = None
        self.tcp: Optional[TCPConfig] = None
        self.robot_models: dict[str, RobotModel] = {}
        self.gripper: Optional[GripperConfig] = None
        self.grasps = GraspLibrary()
        self.joint_state: dict[str, float] = {}
        self.scene_status: Optional[UISceneStatus] = None

    def load_robot_model(self, config: RobotModelConfig) -> RobotModel:
        robot = _load_robot_model(config)
        self.robot_models[config.robot_id] = robot
        return robot

    def load_collision_asset(self, config: ObjectAssetConfig):
        return geometry_from_asset(config)

    def load_collision_asset_status(self, config: ObjectAssetConfig) -> CollisionGeometryStatus:
        try:
            return _geometry_status(config.object_id, config.frame_id, geometry_from_asset(config))
        except AssetLoadError as exc:
            return CollisionGeometryStatus(object_id=config.object_id, ok=False, frame_id=config.frame_id, error=exc.error)
        except Exception as exc:
            return CollisionGeometryStatus(
                object_id=config.object_id,
                ok=False,
                frame_id=config.frame_id,
                error=AlgorithmError(code="INVALID_MESH", message=str(exc), details={"path": config.mesh_path}),
            )

    def load_gripper_model(self, config: GripperConfig) -> LoadedAssetStatus:
        self.gripper = config
        if not config.mesh_path:
            return LoadedAssetStatus(asset_id=config.gripper_id, asset_type="gripper", ok=True, frame_id=config.root_frame)
        status = self.load_collision_asset_status(
            ObjectAssetConfig(object_id=config.gripper_id, mesh_path=config.mesh_path, frame_id=config.root_frame)
        )
        return _asset_status_from_geometry(config.gripper_id, "gripper", config.mesh_path, status)

    def build_collision_world(self, configs: Iterable[CollisionObjectConfig], matrix: CollisionMatrix | None = None) -> CollisionWorld:
        self.world = CollisionWorld(matrix)
        for config in configs:
            self.world.add_from_config(config)
        return self.world

    def build_collision_world_status(
        self, configs: Iterable[CollisionObjectConfig], matrix: CollisionMatrix | None = None
    ) -> tuple[CollisionWorld, list[CollisionGeometryStatus], list[AlgorithmError]]:
        world = CollisionWorld(matrix)
        geometry_statuses: list[CollisionGeometryStatus] = []
        errors: list[AlgorithmError] = []
        for config in configs:
            try:
                obj = world.add_from_config(config)
                geometry_statuses.append(_geometry_status(config.object_id, config.frame_id, obj.geometry))
            except AssetLoadError as exc:
                errors.append(exc.error)
                geometry_statuses.append(CollisionGeometryStatus(object_id=config.object_id, ok=False, frame_id=config.frame_id, error=exc.error))
            except Exception as exc:
                error = AlgorithmError(code="INVALID_MESH", message=str(exc), details={"object_id": config.object_id})
                errors.append(error)
                geometry_statuses.append(CollisionGeometryStatus(object_id=config.object_id, ok=False, frame_id=config.frame_id, error=error))
        self.world = world
        return world, geometry_statuses, errors

    def update_object_pose(self, object_id: str, pose: Transform3D) -> Optional[AlgorithmError]:
        if self.world is None:
            return AlgorithmError(code="WORLD_NOT_BUILT", message="Collision world has not been built.")
        if object_id not in self.world.objects:
            return AlgorithmError(code="OBJECT_NOT_FOUND", message=f"Missing object: {object_id}")
        self.world.update_pose(object_id, pose)
        return None

    def update_tcp(self, tcp_config: TCPConfig) -> TCPConfig:
        self.tcp = tcp_config
        return tcp_config

    def set_collision_matrix(self, matrix: CollisionMatrix) -> Optional[AlgorithmError]:
        if self.world is None:
            return AlgorithmError(code="WORLD_NOT_BUILT", message="Collision world has not been built.")
        self.world.set_matrix(matrix)
        return None

    def query_minimum_distances(self, request: MinimumDistanceRequest) -> list[DistanceQueryResult]:
        if self.world is None:
            return [_distance_error("WORLD_NOT_BUILT", "Collision world has not been built.")]
        try:
            if request.object_a and request.object_b:
                return [minimum_distance_pair(self.world.get(request.object_a), self.world.get(request.object_b))]
            if request.object_id:
                return minimum_distance_to_all(self.world, request.object_id)
            return minimum_distances_active_pairs(self.world)
        except KeyError as exc:
            return [_distance_error("OBJECT_NOT_FOUND", f"Missing object: {exc.args[0]}")]

    def check_collisions(self, request: CollisionCheckRequest) -> CollisionCheckResult:
        if self.world is None:
            return CollisionCheckResult(
                collision=False,
                ok=False,
                error=AlgorithmError(code="WORLD_NOT_BUILT", message="Collision world has not been built."),
            )
        try:
            if request.object_a and request.object_b:
                return check_pair(self.world.get(request.object_a), self.world.get(request.object_b))
            return check_active_pairs(self.world)
        except KeyError as exc:
            return CollisionCheckResult(
                collision=False,
                ok=False,
                error=AlgorithmError(code="OBJECT_NOT_FOUND", message=f"Missing object: {exc.args[0]}"),
            )

    def validate_scene_frames(self, request: UISceneRequest) -> FrameValidationResult:
        return validate_scene_frames(request)

    def load_scene_from_ui(self, request: UISceneRequest) -> UISceneStatus:
        errors: list[AlgorithmError] = []
        loaded_assets: list[LoadedAssetStatus] = []

        self.tcp = request.tcp or (request.gripper.tcp if request.gripper else None)
        self.gripper = request.gripper
        self.grasps = GraspLibrary(request.grasp_candidates)
        self.joint_state = dict(request.joint_state)

        frame_validation = validate_scene_frames(request)
        errors.extend(frame_validation.errors)

        if request.robot:
            robot = self.load_robot_model(request.robot)
            loaded_assets.append(_robot_status(request.robot, robot))
            if robot.error:
                errors.append(robot.error)

        if request.gripper:
            gripper_status = self.load_gripper_model(request.gripper)
            loaded_assets.append(gripper_status)
            if gripper_status.error:
                errors.append(gripper_status.error)

        collision_configs = _collision_configs_from_scene(request)
        world, geometry_statuses, world_errors = self.build_collision_world_status(collision_configs, request.collision_matrix)
        errors.extend(world_errors)

        loaded_assets.extend(_loaded_assets_from_scene_collision_configs(request, geometry_statuses))
        active_pairs = [[a, b] for a, b in world.active_pairs()]
        status = UISceneStatus(
            ok=not errors,
            loaded_assets=loaded_assets,
            collision_geometry=geometry_statuses,
            frame_validation=frame_validation,
            active_collision_pairs=active_pairs,
            world_object_ids=sorted(world.objects.keys()),
            errors=errors,
        )
        self.scene_status = status
        return status

    def evaluate_grasp_by_id(self, request: GraspByIdRequest) -> GraspFeasibilityResult:
        candidate = self.grasps.candidates.get(request.target_grasp_id)
        if candidate is None:
            error = AlgorithmError(code="GRASP_NOT_FOUND", message=f"Unknown grasp candidate: {request.target_grasp_id}")
            return GraspFeasibilityResult(grasp_id=request.target_grasp_id, feasible=False, rejection_reasons=[error])
        result = _evaluate_grasp_candidate(
            GraspFeasibilityRequest(candidate=candidate, ik_request=request.ik_request, collision_request=request.collision_request),
            self.world,
        )
        if request.distance_request:
            result.distance_results = self.query_minimum_distances(request.distance_request)
            for distance in result.distance_results:
                if not distance.ok and distance.error:
                    result.rejection_reasons.append(distance.error)
            result.feasible = result.feasible and all(distance.ok and not distance.in_collision for distance in result.distance_results)
        return result

    def evaluate_scene_from_ui(self, request: UISceneEvaluationRequest) -> UISceneEvaluationResult:
        scene_status = self.load_scene_from_ui(request.scene)
        errors = list(scene_status.errors)
        distance_results: list[DistanceQueryResult] = []
        collision = None
        fk = None
        jacobian = None
        ik = None
        grasp = None

        if request.distance_request:
            distance_results = self.query_minimum_distances(request.distance_request)
            errors.extend(item.error for item in distance_results if item.error)
        if request.collision_request:
            collision = self.check_collisions(request.collision_request)
            if collision.error:
                errors.append(collision.error)
        if request.fk_request:
            fk = compute_fk(request.fk_request)
            if fk.error:
                errors.append(fk.error)
        if request.jacobian_request:
            jacobian = compute_jacobian(request.jacobian_request)
            if jacobian.error:
                errors.append(jacobian.error)
        if request.ik_request:
            ik = solve_ik(request.ik_request)
            if ik.error:
                errors.append(ik.error)

        target_grasp_id = request.target_grasp_id or request.scene.target_grasp_id
        if target_grasp_id:
            grasp = self.evaluate_grasp_by_id(
                GraspByIdRequest(target_grasp_id=target_grasp_id, ik_request=request.ik_request, collision_request=request.collision_request)
            )
            errors.extend(grasp.rejection_reasons)

        return UISceneEvaluationResult(
            ok=not errors,
            scene_status=scene_status,
            distance_results=distance_results,
            collision=collision,
            fk=fk,
            jacobian=jacobian,
            ik=ik,
            grasp_feasibility=grasp,
            errors=errors,
        )


def validate_scene_frames(request: UISceneRequest) -> FrameValidationResult:
    transforms: list[tuple[str, Transform3D, Optional[str]]] = []
    if request.tcp:
        transforms.append(("tcp", request.tcp.transform, None))
    if request.gripper and request.gripper.tcp:
        transforms.append(("gripper_tcp", request.gripper.tcp.transform, None))
    if request.gripper_pose:
        transforms.append(("gripper_pose", request.gripper_pose, request.gripper.root_frame if request.gripper else None))
    if request.object_pose:
        transforms.append(("object_pose", request.object_pose, request.object_asset.frame_id if request.object_asset else None))
    if request.bin_pose:
        transforms.append(("bin_pose", request.bin_pose, request.bin.frame_id if request.bin else None))
    for config in request.collision_objects:
        transforms.append((f"collision:{config.object_id}", config.pose, config.frame_id))
    for candidate in request.grasp_candidates:
        transforms.append((f"grasp:{candidate.grasp_id}:tcp", candidate.tcp_in_object, None))
        if candidate.pregrasp_in_object:
            transforms.append((f"grasp:{candidate.grasp_id}:pregrasp", candidate.pregrasp_in_object, None))

    checked_frames: list[str] = []
    errors: list[AlgorithmError] = []
    for name, transform, expected_child in transforms:
        checked_frames.extend([transform.parent_frame, transform.child_frame])
        errors.extend(_validate_transform(name, transform, expected_child))
    return FrameValidationResult(ok=not errors, checked_frames=sorted(set(filter(None, checked_frames))), errors=errors)


def load_scene_from_ui(request: UISceneRequest, context: RobotEngineContext | None = None) -> UISceneStatus:
    return (context or _default_context).load_scene_from_ui(request)


def evaluate_scene_from_ui(request: UISceneEvaluationRequest, context: RobotEngineContext | None = None) -> UISceneEvaluationResult:
    return (context or _default_context).evaluate_scene_from_ui(request)


def evaluate_grasp_by_id(request: GraspByIdRequest, context: RobotEngineContext | None = None) -> GraspFeasibilityResult:
    return (context or _default_context).evaluate_grasp_by_id(request)


def load_robot_model(config: RobotModelConfig) -> RobotModel:
    return _default_context.load_robot_model(config)


def load_collision_asset(config: ObjectAssetConfig):
    return _default_context.load_collision_asset(config)


def load_collision_asset_status(config: ObjectAssetConfig) -> CollisionGeometryStatus:
    return _default_context.load_collision_asset_status(config)


def load_gripper_model(config: GripperConfig) -> LoadedAssetStatus:
    return _default_context.load_gripper_model(config)


def build_collision_world(configs: Iterable[CollisionObjectConfig], matrix: CollisionMatrix | None = None) -> CollisionWorld:
    return _default_context.build_collision_world(configs, matrix)


def update_object_pose(object_id: str, pose: Transform3D) -> None:
    error = _default_context.update_object_pose(object_id, pose)
    if error:
        raise RuntimeError(error.message)


def update_tcp(tcp_config: TCPConfig) -> TCPConfig:
    return _default_context.update_tcp(tcp_config)


def set_collision_matrix(matrix: CollisionMatrix) -> None:
    error = _default_context.set_collision_matrix(matrix)
    if error:
        raise RuntimeError(error.message)


def query_minimum_distances(request: MinimumDistanceRequest):
    return _default_context.query_minimum_distances(request)


def check_collisions(request: CollisionCheckRequest):
    return _default_context.check_collisions(request)


def evaluate_grasp_candidate(request: GraspFeasibilityRequest):
    return _evaluate_grasp_candidate(request, _default_context.world)


def update_frame(transform: Transform3D):
    errors = _validate_transform("frame_update", transform, None)
    if errors:
        return {"success": False, "error_code": errors[0].code, "error_message": errors[0].message}
    return {"success": True, "error_code": "OK", "error_message": "", "transform": transform.model_dump()}


def validate_path(request):
    q_waypoints = getattr(request, "q_waypoints", request)
    return {"success": True, "error_code": "OK", "error_message": "", "waypoint_count": len(q_waypoints)}


def plan_collision_free_path(request):
    try:
        joint_limits = None
        if getattr(request, "lower_limits", None) is not None and getattr(request, "upper_limits", None) is not None:
            joint_limits = (request.lower_limits, request.upper_limits)
        result = CollisionAwarePlanner().plan(PathRequest(start=request.start, goal=request.goal, joint_limits=joint_limits, max_joint_step=request.planner.max_joint_step, max_iterations=request.planner.max_iterations, timeout=request.planner.timeout))
        return {
            "success": result.success,
            "error_code": "OK" if result.success else result.rejection_reason,
            "error_message": "" if result.success else result.rejection_reason,
            "planner_used": result.planner_used,
            "path_waypoints": [np.asarray(q, dtype=float).tolist() for q in result.q_waypoints],
            "rejection_reason": result.rejection_reason,
            "debug_info": result.debug_info,
        }
    except Exception as exc:
        return {"success": False, "error_code": "PATH_PLANNING_FAILED", "error_message": str(exc), "rejection_reason": "PATH_PLANNING_FAILED"}


def time_parameterize_path(request):
    try:
        options = request.options
        dof = len(request.q_waypoints[0])
        vel = options.velocity_limits or [1.0] * dof
        acc = options.acceleration_limits or [2.0] * dof
        trajectory = retime_joint_path(request.q_waypoints, vel, acc, options.method)
        return {
            "success": True,
            "error_code": "OK",
            "error_message": "",
            "trajectory": {
                "q": [p.q for p in trajectory.points],
                "q_dot": [p.q_dot for p in trajectory.points],
                "q_ddot": [p.q_ddot for p in trajectory.points],
                "timestamps": [p.time for p in trajectory.points],
                "duration": trajectory.duration,
                "generation_method": trajectory.generation_method,
            },
        }
    except Exception as exc:
        return {"success": False, "error_code": "TRAJECTORY_GENERATION_FAILED", "error_message": str(exc)}


def validate_trajectory(request):
    try:
        from robot_engine.trajectory.trajectory_base import JointTrajectory, JointTrajectoryPoint

        data = request.trajectory
        points = [JointTrajectoryPoint(t, q, qd, qdd) for t, q, qd, qdd in zip(data["timestamps"], data["q"], data["q_dot"], data["q_ddot"])]
        trajectory = JointTrajectory(points)
        checks = []
        if request.joint_limits:
            lower = [x[0] for x in request.joint_limits]
            upper = [x[1] for x in request.joint_limits]
            checks.append(validate_joint_position_limits(trajectory, (lower, upper)))
        if request.velocity_limits:
            checks.append(validate_velocity_limits(trajectory, request.velocity_limits))
        if request.acceleration_limits:
            checks.append(validate_acceleration_limits(trajectory, request.acceleration_limits))
        failed = next((item for item in checks if not item[0]), None)
        if failed:
            return {"success": False, "error_code": failed[2], "error_message": failed[2], "failed_waypoint_index": failed[1], "rejection_reason": failed[2]}
        return {"success": True, "error_code": "OK", "error_message": ""}
    except Exception as exc:
        return {"success": False, "error_code": "TRAJECTORY_VALIDATION_FAILED", "error_message": str(exc)}


def plan_move_j(request):
    from robot_engine.motion_primitives.move_j import plan_move_j as _plan

    return _plan(request)


def plan_move_l(request):
    from robot_engine.motion_primitives.move_l import plan_move_l as _plan

    return _plan(request)


def plan_approach(request):
    from robot_engine.motion_primitives.approach import plan_approach as _plan

    return _plan(request)


def plan_retreat(request):
    from robot_engine.motion_primitives.retreat import plan_retreat as _plan

    return _plan(request)


def plan_lift(request):
    from robot_engine.motion_primitives.lift import plan_lift as _plan

    return _plan(request)


def plan_extract(request):
    from robot_engine.motion_primitives.extract import plan_extract as _plan

    return _plan(request)


def plan_pick_sequence(request):
    from robot_engine.motion_primitives.pick_sequence import plan_pick_sequence as _plan

    return _plan(getattr(request, "sequence", request))


def _collision_configs_from_scene(request: UISceneRequest) -> list[CollisionObjectConfig]:
    configs = list(request.collision_objects)
    if request.object_asset and request.object_pose:
        configs.append(
            CollisionObjectConfig(
                object_id=request.object_asset.object_id,
                asset_path=request.object_asset.mesh_path,
                frame_id=request.object_asset.frame_id,
                pose=request.object_pose,
                group="object",
            )
        )
    if request.bin:
        pose = request.bin_pose or _identity_transform("world", request.bin.frame_id)
        configs.append(
            CollisionObjectConfig(
                object_id=request.bin.bin_id,
                asset_path=request.bin.mesh_path,
                frame_id=request.bin.frame_id,
                pose=pose,
                group="bin",
                size_xyz=request.bin.size_xyz,
            )
        )
    if request.gripper and request.gripper.mesh_path and request.gripper_pose:
        configs.append(
            CollisionObjectConfig(
                object_id=request.gripper.gripper_id,
                asset_path=request.gripper.mesh_path,
                frame_id=request.gripper.root_frame,
                pose=request.gripper_pose,
                group="gripper",
            )
        )
    return configs


def _loaded_assets_from_scene_collision_configs(
    request: UISceneRequest, geometry_statuses: list[CollisionGeometryStatus]
) -> list[LoadedAssetStatus]:
    by_id = {status.object_id: status for status in geometry_statuses}
    statuses: list[LoadedAssetStatus] = []
    if request.object_asset:
        geom = by_id.get(request.object_asset.object_id)
        statuses.append(
            _asset_status_from_geometry(
                request.object_asset.object_id,
                "object",
                request.object_asset.mesh_path,
                geom or CollisionGeometryStatus(object_id=request.object_asset.object_id, ok=False),
            )
        )
    if request.bin:
        geom = by_id.get(request.bin.bin_id)
        path = request.bin.mesh_path or "box:size_xyz"
        statuses.append(
            _asset_status_from_geometry(
                request.bin.bin_id,
                "bin",
                path,
                geom or CollisionGeometryStatus(object_id=request.bin.bin_id, ok=False),
            )
        )
    return statuses


def _robot_status(config: RobotModelConfig, robot: RobotModel) -> LoadedAssetStatus:
    metadata = {}
    if robot.pin_model is not None:
        metadata = {"nq": robot.pin_model.nq, "nv": robot.pin_model.nv, "frame_count": len(robot.pin_model.frames)}
    return LoadedAssetStatus(
        asset_id=config.robot_id,
        asset_type="robot",
        ok=robot.error is None,
        frame_id=config.base_frame,
        path=config.urdf_path,
        metadata=metadata,
        error=robot.error,
    )


def _asset_status_from_geometry(asset_id: str, asset_type: str, path: Optional[str], status: CollisionGeometryStatus) -> LoadedAssetStatus:
    metadata = {
        "coal_ready": status.coal_ready,
        "vertex_count": status.vertex_count,
        "face_count": status.face_count,
        "aabb_min": status.aabb_min,
        "aabb_max": status.aabb_max,
    }
    return LoadedAssetStatus(
        asset_id=asset_id,
        asset_type=asset_type,
        ok=status.ok,
        frame_id=status.frame_id,
        path=path,
        metadata=metadata,
        error=status.error,
    )


def _geometry_status(object_id: str, frame_id: str, geometry) -> CollisionGeometryStatus:
    bounds = geometry.aabb_bounds
    mesh = geometry.mesh
    return CollisionGeometryStatus(
        object_id=object_id,
        ok=True,
        frame_id=frame_id,
        coal_ready=geometry.coal_geometry is not None,
        vertex_count=0 if mesh is None else int(len(mesh.vertices)),
        face_count=0 if mesh is None else int(len(mesh.faces)),
        aabb_min=np.asarray(bounds[0], dtype=float).tolist(),
        aabb_max=np.asarray(bounds[1], dtype=float).tolist(),
    )


def _validate_transform(name: str, transform: Transform3D, expected_child: Optional[str]) -> list[AlgorithmError]:
    errors: list[AlgorithmError] = []
    if not transform.parent_frame or not transform.child_frame:
        errors.append(AlgorithmError(code="INVALID_TRANSFORM", message="Transform frames must be non-empty.", details={"name": name}))
    if expected_child and transform.child_frame != expected_child:
        errors.append(
            AlgorithmError(
                code="INVALID_TRANSFORM",
                message="Transform child frame does not match object frame.",
                details={"name": name, "expected_child": expected_child, "actual_child": transform.child_frame},
            )
        )
    mat = np.asarray(transform.matrix, dtype=float)
    if mat.shape != (4, 4) or not np.isfinite(mat).all():
        errors.append(AlgorithmError(code="INVALID_TRANSFORM", message="Transform matrix must be finite 4x4.", details={"name": name}))
        return errors
    if not np.allclose(mat[3], [0.0, 0.0, 0.0, 1.0], atol=1e-9):
        errors.append(AlgorithmError(code="INVALID_TRANSFORM", message="Transform matrix last row must be [0, 0, 0, 1].", details={"name": name}))
    rot = mat[:3, :3]
    if abs(float(np.linalg.det(rot))) < 1e-9:
        errors.append(AlgorithmError(code="INVALID_TRANSFORM", message="Transform rotation block is singular.", details={"name": name}))
    return errors


def _identity_transform(parent_frame: str, child_frame: str) -> Transform3D:
    return Transform3D(parent_frame=parent_frame, child_frame=child_frame, matrix=np.eye(4).tolist())


def _distance_error(code, message: str) -> DistanceQueryResult:
    return DistanceQueryResult(object_a="", object_b="", ok=False, error=AlgorithmError(code=code, message=message))


_default_context = RobotEngineContext()


__all__ = [
    "RobotEngineContext",
    "load_robot_model",
    "load_collision_asset",
    "load_collision_asset_status",
    "load_gripper_model",
    "build_collision_world",
    "update_object_pose",
    "update_tcp",
    "set_collision_matrix",
    "query_minimum_distances",
    "check_collisions",
    "compute_fk",
    "compute_jacobian",
    "solve_ik",
    "validate_scene_frames",
    "load_scene_from_ui",
    "evaluate_scene_from_ui",
    "evaluate_grasp_by_id",
    "evaluate_grasp_candidate",
    "update_frame",
    "validate_path",
    "plan_collision_free_path",
    "time_parameterize_path",
    "plan_joint_move_to_frame",
    "plan_linear_move_to_frame",
    "plan_move_j",
    "plan_move_l",
    "plan_approach",
    "plan_retreat",
    "plan_lift",
    "plan_extract",
    "plan_pick_sequence",
    "compute_offset_frame",
    "plan_approach_to_frame",
    "plan_retreat_from_frame",
    "plan_lift_motion",
    "plan_motion_sequence",
    "validate_trajectory",
    "export_robot_trajectory",
]
