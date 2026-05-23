from __future__ import annotations

import math
import shutil
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import numpy as np
from pydantic import BaseModel, Field

from robot_engine.collision.continuous_collision import conservative_continuous_path_collision
from robot_engine.collision.path_collision_checker import PathCollisionChecker
from robot_engine.interfaces.schemas import (
    CollisionMatrix,
    CollisionObjectConfig,
    GripperConfig,
    TCPConfig,
    Transform3D,
)
from robot_engine.interfaces.ui_api import RobotEngineContext
from robot_engine.api.sim_runtime import SimRuntime
from robot_engine.path_planning.collision_aware_planner import CollisionAwarePlanner
from robot_engine.path_planning.joint_direct_planner import JointDirectPlanner
from robot_engine.path_planning.planner_base import PathRequest
from robot_engine.trajectory.retiming import retime_joint_path


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def identity_transform(parent: str, child: str) -> Transform3D:
    return Transform3D(parent_frame=parent, child_frame=child, matrix=np.eye(4).tolist())


class RobotSelection(BaseModel):
    robot_id: str
    asset_path: Optional[str] = None
    urdf_path: Optional[str] = None
    base_frame: str = "robot_base"
    flange_frame: str = "robot_flange"
    joint_names: List[str] = Field(default_factory=list)
    lower_limits: List[float] = Field(default_factory=list)
    upper_limits: List[float] = Field(default_factory=list)


class GripperSelection(BaseModel):
    gripper_id: str
    asset_path: Optional[str] = None
    mesh_path: Optional[str] = None
    root_frame: str = "gripper_flange"
    flange_frame: Optional[Transform3D] = None
    tcp_frame: Optional[Transform3D] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class GeometryRecord(BaseModel):
    geometry_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    type: Literal["bin", "object", "fixture", "obstacle", "world"] = "obstacle"
    mesh_url: Optional[str] = None
    asset_path: Optional[str] = None
    frame_id: str = "world"
    pose: Transform3D = Field(default_factory=lambda: identity_transform("world", "geometry"))
    size_xyz: Optional[List[float]] = None
    collision_enabled: bool = True
    visual_enabled: bool = True


class PoseRecord(BaseModel):
    pose_id: str
    name: str
    tcp_pose: Optional[Dict[str, Any]] = None
    joint_state: List[float] = Field(default_factory=list)
    created_at: str = Field(default_factory=now_iso)


class TestCaseRecord(BaseModel):
    test_case_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    robot_id: Optional[str] = None
    gripper_id: Optional[str] = None
    object_id: Optional[str] = None
    object_pose: Optional[Dict[str, Any]] = None
    bin_id: Optional[str] = None
    environment_ids: List[str] = Field(default_factory=list)
    start_pose_id: Optional[str] = None
    target_pose_id: Optional[str] = None
    grasp_candidate_id: Optional[str] = None
    collision_matrix_id: str = "default"
    planner_options: Dict[str, Any] = Field(default_factory=dict)


class CellContext(BaseModel):
    cell_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str = "Untitled cell"
    cell_type: str = "custom"
    robot: Optional[RobotSelection] = None
    gripper: Optional[GripperSelection] = None
    tcp: Optional[TCPConfig] = None
    geometries: Dict[str, GeometryRecord] = Field(default_factory=dict)
    poses: Dict[str, PoseRecord] = Field(default_factory=dict)
    test_cases: Dict[str, TestCaseRecord] = Field(default_factory=dict)
    test_results: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    collision_matrix: CollisionMatrix = Field(default_factory=CollisionMatrix)
    latest_joint_state: List[float] = Field(default_factory=list)
    latest_tcp_pose: Dict[str, Any] = Field(default_factory=lambda: {
        "position_m": [0.0, 0.0, 0.0],
        "quat_xyzw": [0.0, 0.0, 0.0, 1.0],
        "frame": "base",
    })
    created_at: str = Field(default_factory=now_iso)
    updated_at: str = Field(default_factory=now_iso)

    def touch(self) -> None:
        self.updated_at = now_iso()


class CellRuntime:
    def __init__(self, cell: CellContext, tmp_dir: Path) -> None:
        self.cell = cell
        self.engine = RobotEngineContext()
        self.sim = SimRuntime()
        self.tmp_dir = tmp_dir

    def rebuild_collision_world(self):
        configs = []
        for geom in self.cell.geometries.values():
            if not geom.collision_enabled:
                continue
            configs.append(
                CollisionObjectConfig(
                    object_id=geom.geometry_id,
                    asset_path=geom.asset_path,
                    frame_id=geom.frame_id,
                    pose=geom.pose,
                    group="bin" if geom.type == "bin" else "object" if geom.type == "object" else "fixture",
                    size_xyz=geom.size_xyz,
                )
            )
        return self.engine.build_collision_world_status(configs, self.cell.collision_matrix)


class OrchestratorStore:
    def __init__(self) -> None:
        self._cells: Dict[str, CellRuntime] = {}
        self._lock = threading.Lock()

    def create_cell(self, name: str = "Untitled cell", cell_type: str = "custom") -> CellRuntime:
        cell = CellContext(name=name, cell_type=cell_type)
        tmp_dir = Path(tempfile.mkdtemp(prefix=f"imp_cell_{cell.cell_id[:8]}_"))
        runtime = CellRuntime(cell, tmp_dir)
        with self._lock:
            self._cells[cell.cell_id] = runtime
        return runtime

    def list_cells(self) -> List[CellContext]:
        with self._lock:
            return [runtime.cell for runtime in self._cells.values()]

    def get(self, cell_id: str) -> CellRuntime:
        with self._lock:
            runtime = self._cells.get(cell_id)
        if runtime is None:
            raise KeyError(cell_id)
        return runtime

    def update_cell(self, cell_id: str, patch: Dict[str, Any]) -> CellRuntime:
        runtime = self.get(cell_id)
        for key in ("name", "cell_type"):
            if key in patch and patch[key] is not None:
                setattr(runtime.cell, key, patch[key])
        runtime.cell.touch()
        return runtime

    def delete_cell(self, cell_id: str) -> bool:
        with self._lock:
            runtime = self._cells.pop(cell_id, None)
        if runtime is None:
            return False
        shutil.rmtree(runtime.tmp_dir, ignore_errors=True)
        return True


def pose_from_tcp_dict(parent: str, child: str, pose: Dict[str, Any]) -> Transform3D:
    matrix = np.eye(4)
    position = pose.get("position_m") or [0.0, 0.0, 0.0]
    matrix[:3, 3] = np.asarray(position[:3], dtype=float)
    return Transform3D(parent_frame=parent, child_frame=child, matrix=matrix.tolist())


def offset_tcp_pose(pose: Dict[str, Any], axis: str, direction: str, distance_m: float) -> Dict[str, Any]:
    if distance_m < 0 or not math.isfinite(distance_m):
        raise ValueError("distance_m must be non-negative and finite")
    idx = {"X": 0, "Y": 1, "Z": 2}.get(axis.upper())
    if idx is None:
        raise ValueError("axis must be X, Y, or Z")
    sign = 1.0 if direction.upper() == "POSITIVE" else -1.0 if direction.upper() == "NEGATIVE" else None
    if sign is None:
        raise ValueError("direction must be POSITIVE or NEGATIVE")
    next_pose = dict(pose)
    position = list(next_pose.get("position_m") or [0.0, 0.0, 0.0])
    position[idx] = float(position[idx]) + sign * float(distance_m)
    next_pose["position_m"] = position
    next_pose.setdefault("quat_xyzw", [0.0, 0.0, 0.0, 1.0])
    next_pose.setdefault("frame", "base")
    return next_pose


def state_validity_from_cspace_boxes(boxes: List[Dict[str, Any]]):
    def is_valid(q) -> bool:
        point = np.asarray(q, dtype=float)
        for box in boxes:
            lower = np.asarray(box.get("lower", []), dtype=float)
            upper = np.asarray(box.get("upper", []), dtype=float)
            if lower.shape == point.shape and upper.shape == point.shape and np.all(point >= lower) and np.all(point <= upper):
                return False
        return True
    return is_valid


def plan_joint_path(start, goal, planner_options: Dict[str, Any]) -> Dict[str, Any]:
    lower = planner_options.get("lower_limits")
    upper = planner_options.get("upper_limits")
    joint_limits = (lower, upper) if lower is not None and upper is not None else None
    state_validity_fn = None
    if planner_options.get("cspace_obstacles"):
        state_validity_fn = state_validity_from_cspace_boxes(planner_options["cspace_obstacles"])
    req = PathRequest(
        start=start,
        goal=goal,
        joint_limits=joint_limits,
        state_validity_fn=state_validity_fn,
        max_joint_step=float(planner_options.get("max_joint_step", 0.1)),
        max_iterations=int(planner_options.get("max_iterations", 1000)),
        timeout=float(planner_options.get("timeout", 5.0)),
        goal_bias=float(planner_options.get("goal_bias", 0.1)),
    )
    direct = JointDirectPlanner().plan(req)
    result = direct if direct.success or not planner_options.get("collision_aware", True) else CollisionAwarePlanner().plan(req)
    waypoints = [np.asarray(q, dtype=float).tolist() for q in result.q_waypoints]
    trajectory = None
    if result.success and len(waypoints) >= 2:
        dof = len(waypoints[0])
        vel = planner_options.get("velocity_limits") or [1.0] * dof
        acc = planner_options.get("acceleration_limits") or [2.0] * dof
        traj = retime_joint_path(waypoints, vel, acc, planner_options.get("trajectory_method", "trapezoidal"))
        trajectory = {
            "q": [p.q for p in traj.points],
            "q_dot": [p.q_dot for p in traj.points],
            "q_ddot": [p.q_ddot for p in traj.points],
            "timestamps": [p.time for p in traj.points],
            "duration": traj.duration,
            "generation_method": traj.generation_method,
        }
    return {
        "success": result.success,
        "collision_free": result.success,
        "planner_used": result.planner_used,
        "q_waypoints": waypoints,
        "trajectory": trajectory,
        "minimum_clearance": result.minimum_clearance,
        "colliding_pairs": [] if result.success else ([result.colliding_pair] if result.colliding_pair else []),
        "rejection_reason": None if result.success else result.rejection_reason,
        "debug_info": result.debug_info,
    }


def check_path_with_world(runtime: CellRuntime, q_waypoints: List[List[float]], resolution: float = 0.05) -> Dict[str, Any]:
    if runtime.engine.world is None:
        runtime.rebuild_collision_world()
    result = conservative_continuous_path_collision(q_waypoints, world=runtime.engine.world, resolution=resolution)
    return {
        "success": result.success,
        "collision_free": bool(result.success and not result.collision),
        "collision": result.collision,
        "minimum_clearance": result.minimum_clearance,
        "first_collision_waypoint": result.first_collision_waypoint,
        "first_collision_segment": result.first_collision_segment,
        "colliding_pairs": [result.colliding_pair] if getattr(result, "colliding_pair", None) else [],
        "rejection_reason": None if result.success else result.rejection_reason,
        "debug_info": result.debug_info,
    }


_store = OrchestratorStore()


def get_orchestrator_store() -> OrchestratorStore:
    return _store
