from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np

from robot_engine.environment.planning_collision_world import PlanningCollisionWorld
from robot_engine.environment.pointcloud_processing import PointCloudCollisionConfig
from robot_engine.environment.primitive_obstacle import PrimitiveShape
from robot_engine.planning_core.ompl_planner import OMPLPlanner, PlannerResult
from robot_engine.planning_core.path_processor import PathProcessor
from robot_engine.planning_core.pinocchio_robot import PinocchioRobot
from robot_engine.planning_core.ruckig_generator import RuckigTrajectoryGenerator, TimedTrajectory
from robot_engine.planning_core.semantic_config import SemanticConfig


class MotionPlanningPipeline:
    """
    End-to-end motion planning pipeline.

        URDF + YAML semantics
            ↓
        PinocchioRobot
            ↓
        PlanningCollisionWorld (mesh + point-cloud + primitive obstacles)
            ↓
        is_state_valid(q)  ← joint limits + self-collision + world collision
            ↓
        IK candidates
            ↓
        OMPL RRTConnect
            ↓
        PathProcessor (shortcut + interpolate)
            ↓
        RuckigTrajectoryGenerator
            ↓
        TimedTrajectory

    Usage
    -----
    ::

        pipeline = MotionPlanningPipeline.from_config("config/robot_semantics.yaml")

        pipeline.add_mesh_obstacle("table", "table.stl", T_world_table)
        pipeline.add_pointcloud_obstacle("scene", "scene.ply", np.eye(4))

        traj = pipeline.plan_to_pose(
            group_name="arm",
            q_start=current_q,
            T_goal=T_world_tool_goal,
        )
    """

    def __init__(
        self,
        robot: PinocchioRobot,
        semantics: SemanticConfig,
    ) -> None:
        self._robot = robot
        self._semantics = semantics
        self._world = PlanningCollisionWorld()

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, yaml_path: str | Path) -> "MotionPlanningPipeline":
        """Build the pipeline from a YAML semantics file."""
        sem = SemanticConfig.from_yaml(yaml_path)
        return cls._from_semantics(sem)

    @classmethod
    def from_semantics(cls, semantics: SemanticConfig) -> "MotionPlanningPipeline":
        return cls._from_semantics(semantics)

    @classmethod
    def _from_semantics(cls, sem: SemanticConfig) -> "MotionPlanningPipeline":
        robot = PinocchioRobot(
            urdf_path=sem.urdf_path,
            package_dirs=sem.package_dirs,
            allowed_collision_pairs=sem.allowed_collisions,
            urdf_xml=sem.urdf_xml,
        )
        pipeline = cls(robot, sem)
        pipeline._load_environment_obstacles()
        return pipeline

    # ------------------------------------------------------------------
    # Obstacle management
    # ------------------------------------------------------------------

    def add_mesh_obstacle(
        self,
        name: str,
        mesh_path: str,
        T_world_mesh: np.ndarray,
        scale: float = 1.0,
        simplify_faces: Optional[int] = None,
    ) -> None:
        self._world.add_mesh_obstacle(name, mesh_path, T_world_mesh, scale=scale, simplify_faces=simplify_faces)

    def add_primitive_obstacle(
        self,
        name: str,
        shape: PrimitiveShape | str,
        T_world_primitive: np.ndarray,
        size: dict,
    ) -> None:
        self._world.add_primitive_obstacle(name, shape, T_world_primitive, size)

    def add_pointcloud_obstacle(
        self,
        name: str,
        pointcloud_path: str,
        T_world_cloud: np.ndarray,
        config: Optional[PointCloudCollisionConfig] = None,
    ) -> None:
        cfg = config or self._semantics.collision.pointcloud
        self._world.add_pointcloud_obstacle_from_file(name, pointcloud_path, T_world_cloud, cfg)

    def add_pointcloud_obstacle_from_numpy(
        self,
        name: str,
        points_xyz: np.ndarray,
        T_world_cloud: np.ndarray,
        config: Optional[PointCloudCollisionConfig] = None,
    ) -> None:
        cfg = config or self._semantics.collision.pointcloud
        self._world.add_pointcloud_obstacle_from_numpy(name, points_xyz, T_world_cloud, cfg)

    def update_pointcloud_obstacle_from_numpy(
        self,
        name: str,
        points_xyz: np.ndarray,
        T_world_cloud: np.ndarray,
        config: Optional[PointCloudCollisionConfig] = None,
    ) -> None:
        """Replace a live point-cloud obstacle. Safe to call at sensor rate."""
        cfg = config or self._semantics.collision.pointcloud
        self._world.update_pointcloud_obstacle_from_numpy(name, points_xyz, T_world_cloud, cfg)

    def remove_obstacle(self, name: str) -> None:
        self._world.remove_obstacle(name)

    def clear_obstacles(self) -> None:
        self._world.clear_obstacles()

    def list_obstacles(self) -> List[str]:
        return self._world.list_obstacles()

    def _load_environment_obstacles(self) -> None:
        for obs in self._semantics.environment_obstacles:
            if obs.type == "mesh":
                self.add_mesh_obstacle(
                    name=obs.name,
                    mesh_path=obs.path,
                    T_world_mesh=obs.transform,
                    scale=obs.scale,
                    simplify_faces=obs.simplify_faces,
                )
            elif obs.type == "primitive":
                self.add_primitive_obstacle(
                    name=obs.name,
                    shape=obs.shape,
                    T_world_primitive=obs.transform,
                    size=obs.size,
                )
            elif obs.type == "pointcloud":
                if not self._semantics.collision.pointcloud_enabled:
                    continue
                self.add_pointcloud_obstacle(
                    name=obs.name,
                    pointcloud_path=obs.path,
                    T_world_cloud=obs.transform,
                    config=obs.pointcloud or self._semantics.collision.pointcloud,
                )

    # ------------------------------------------------------------------
    # State validity
    # ------------------------------------------------------------------

    def is_state_valid(self, group_name: str, q_group: np.ndarray) -> bool:
        """
        Full state validity check: joint limits + self-collision + world collision.

        Parameters
        ----------
        group_name : str
        q_group : (n_group,) array of joint angles for the planning group

        Returns
        -------
        bool
        """
        sem = self._semantics
        robot = self._robot

        # 1. Joint limits
        group = sem.group(group_name)
        q_group = np.asarray(q_group, dtype=float)
        lower = sem.joint_lower(group_name)
        upper = sem.joint_upper(group_name)
        if len(lower) == len(q_group):
            if np.any(q_group < lower - 1e-6) or np.any(q_group > upper + 1e-6):
                return False

        # 2. Map to full model q
        q_full = _group_q_to_full(robot, group, q_group)

        # 3. Self-collision
        if sem.collision.self_collision_enabled:
            if robot.in_self_collision(q_full):
                return False

        # 4. World collision
        if sem.collision.world_collision_enabled and len(self._world) > 0:
            robot.update_geometry(q_full)
            geoms = robot.link_coal_geometries()
            transforms = robot.link_coal_transforms()
            if self._world.robot_in_world_collision(geoms, transforms):
                return False

        return True

    # ------------------------------------------------------------------
    # IK
    # ------------------------------------------------------------------

    def solve_ik(
        self,
        group_name: str,
        T_goal,
        seeds: Optional[List[np.ndarray]] = None,
        max_iters: int = 200,
    ) -> List[np.ndarray]:
        """
        Return a list of IK solutions (as full-model q vectors) that pass state validity.

        Parameters
        ----------
        T_goal : pin.SE3 or (4, 4) array
        seeds : list of (n_group,) seed vectors, or None (uses default seed)
        """
        group = self._semantics.group(group_name)
        tip_link = group.tip_link
        robot = self._robot

        if seeds is None:
            default_seed = group.seed_array()
            seeds = [default_seed]
            # Add a neutral seed too
            seeds.append(np.zeros(len(group.joints)))

        # If the group has a TCP offset (tip_link → tcp), the stored pose is in tcp frame.
        # Pinocchio IK targets tip_link, so convert: T_tip = T_tcp @ inv(tip_T_tcp).
        T_tip_goal = T_goal
        tcp_offset = group.tcp_offset
        if tcp_offset is not None:
            T_goal_mat = np.asarray(T_goal.homogeneous if hasattr(T_goal, "homogeneous") else T_goal, dtype=float)
            # tip_T_tcp is the transform from tip_link to tcp in tip_link frame.
            # To place tcp at T_goal, tip must be at: T_goal @ inv(tip_T_tcp)
            T_tip_goal = T_goal_mat @ np.linalg.inv(tcp_offset)

        full_seeds = [_group_q_to_full(robot, group, s) for s in seeds]
        raw_solutions = robot.solve_ik_multi_seed(tip_link, T_tip_goal, full_seeds, max_iters=max_iters)

        # Filter by validity
        valid = []
        for q_full in raw_solutions:
            q_group = _full_q_to_group(robot, group, q_full)
            if self.is_state_valid(group_name, q_group):
                valid.append(q_full)
        return valid

    # ------------------------------------------------------------------
    # Planning
    # ------------------------------------------------------------------

    def plan_to_pose(
        self,
        group_name: str,
        q_start: np.ndarray,
        T_goal,
        ik_seeds: Optional[List[np.ndarray]] = None,
        timeout: Optional[float] = None,
    ) -> Optional[TimedTrajectory]:
        """
        Full planning pipeline: IK → RRTConnect → PathProcessor → Ruckig.

        Parameters
        ----------
        group_name : str
        q_start : (n_group,) array
        T_goal : pin.SE3 or (4, 4) array
        ik_seeds : optional list of (n_group,) IK seeds
        timeout : override planner timeout (seconds)

        Returns
        -------
        TimedTrajectory or None if planning failed
        """
        sem = self._semantics
        robot = self._robot
        group = sem.group(group_name)

        q_start = np.asarray(q_start, dtype=float)

        # Validate start
        if not self.is_state_valid(group_name, q_start):
            return None

        # IK
        ik_solutions = self.solve_ik(group_name, T_goal, seeds=ik_seeds)
        if not ik_solutions:
            return None

        # Build validity function (closure over current world)
        def valid_fn(q_group: np.ndarray) -> bool:
            return self.is_state_valid(group_name, q_group)

        lower = sem.joint_lower(group_name)
        upper = sem.joint_upper(group_name)

        planner_timeout = timeout or sem.planner.timeout
        planner = OMPLPlanner(
            lower_limits=lower,
            upper_limits=upper,
            is_state_valid=valid_fn,
            timeout=planner_timeout,
            interpolation_waypoints=sem.planner.interpolation_waypoints,
            max_joint_step=sem.planner.max_joint_step,
        )

        # Try each IK solution
        plan_result: Optional[PlannerResult] = None
        for q_goal_full in ik_solutions:
            q_goal_group = _full_q_to_group(robot, group, q_goal_full)
            result = planner.plan(q_start, q_goal_group)
            if result.success:
                plan_result = result
                break

        if plan_result is None or not plan_result.success:
            return None

        # Path processing: shortcut first, then interpolate for collision validation
        processor = PathProcessor(
            is_state_valid=valid_fn,
            interpolation_waypoints=sem.planner.interpolation_waypoints,
        )
        sparse = processor.shortcut(plan_result.q_waypoints)
        processed = processor.interpolate(sparse, sem.planner.interpolation_waypoints)

        if not processor.validate(processed):
            return None

        # Trajectory timing
        vel = sem.velocity_limits(group_name)
        acc = sem.acceleration_limits(group_name)
        jrk = sem.jerk_limits(group_name)

        generator = RuckigTrajectoryGenerator(
            velocity_limits=vel,
            acceleration_limits=acc,
            jerk_limits=jrk,
            dt=sem.trajectory.dt,
        )
        traj = generator.generate_smooth(processed)
        if traj is None:
            return None
        traj.path_waypoints = [np.asarray(q, dtype=float) for q in processed]
        traj.sparse_waypoints = [np.asarray(q, dtype=float) for q in sparse]
        traj.planner_used = plan_result.planner_used
        return traj

    def plan_to_configuration(
        self,
        group_name: str,
        q_start: np.ndarray,
        q_goal: np.ndarray,
        timeout: Optional[float] = None,
    ) -> Optional[TimedTrajectory]:
        """
        Plan directly to a target joint configuration (skips IK).
        """
        sem = self._semantics
        q_start = np.asarray(q_start, dtype=float)
        q_goal = np.asarray(q_goal, dtype=float)

        if not self.is_state_valid(group_name, q_start):
            return None
        if not self.is_state_valid(group_name, q_goal):
            return None

        def valid_fn(q: np.ndarray) -> bool:
            return self.is_state_valid(group_name, q)

        lower = sem.joint_lower(group_name)
        upper = sem.joint_upper(group_name)

        planner = OMPLPlanner(
            lower_limits=lower,
            upper_limits=upper,
            is_state_valid=valid_fn,
            timeout=timeout or sem.planner.timeout,
            interpolation_waypoints=sem.planner.interpolation_waypoints,
            max_joint_step=sem.planner.max_joint_step,
        )
        result = planner.plan(q_start, q_goal)
        if not result.success:
            return None

        processor = PathProcessor(
            is_state_valid=valid_fn,
            interpolation_waypoints=sem.planner.interpolation_waypoints,
        )
        sparse = processor.shortcut(result.q_waypoints)
        processed = processor.interpolate(sparse, sem.planner.interpolation_waypoints)

        generator = RuckigTrajectoryGenerator(
            velocity_limits=sem.velocity_limits(group_name),
            acceleration_limits=sem.acceleration_limits(group_name),
            jerk_limits=sem.jerk_limits(group_name),
            dt=sem.trajectory.dt,
        )
        traj = generator.generate_smooth(processed)
        if traj is None:
            return None
        traj.path_waypoints = [np.asarray(q, dtype=float) for q in processed]
        traj.sparse_waypoints = [np.asarray(q, dtype=float) for q in sparse]
        traj.planner_used = result.planner_used
        return traj

    def _validate_timed_trajectory(self, group_name: str, traj: TimedTrajectory) -> bool:
        """Validate the sampled timed trajectory after Ruckig retiming."""
        return all(self.is_state_valid(group_name, q) for q in traj.positions)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def robot(self) -> PinocchioRobot:
        return self._robot

    @property
    def semantics(self) -> SemanticConfig:
        return self._semantics

    @property
    def world(self) -> PlanningCollisionWorld:
        return self._world


# ------------------------------------------------------------------
# Private helpers
# ------------------------------------------------------------------

def _group_q_to_full(
    robot: PinocchioRobot,
    group,
    q_group: np.ndarray,
) -> np.ndarray:
    """Map a group-sized vector to full model q using joint name matching."""
    all_names = robot.joint_names
    q_full = robot.neutral_q()
    for idx, jname in enumerate(group.joints):
        if jname in all_names:
            fidx = all_names.index(jname)
            q_full[fidx] = q_group[idx]
    return q_full


def _full_q_to_group(
    robot: PinocchioRobot,
    group,
    q_full: np.ndarray,
) -> np.ndarray:
    """Extract group-joint values from a full model q vector."""
    all_names = robot.joint_names
    result = np.zeros(len(group.joints))
    for idx, jname in enumerate(group.joints):
        if jname in all_names:
            fidx = all_names.index(jname)
            result[idx] = q_full[fidx]
    return result
