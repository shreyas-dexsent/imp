"""Orch-facing helpers for the task-neutral robot_engine package."""

from __future__ import annotations

import json
import math
import importlib.util
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

from orchestrator.core.context import StationContext
from robot_engine.interfaces.schemas import (
    BinAssetConfig,
    CollisionCheckRequest,
    CollisionMatrix,
    CollisionObjectConfig,
    CollisionPairRule,
    GraspCandidate,
    GripperConfig,
    KinematicChainConfig,
    KinematicJointConfig,
    MinimumDistanceRequest,
    ObjectAssetConfig,
    RobotModelConfig,
    TCPConfig,
    Transform3D,
    UISceneEvaluationRequest,
    UISceneRequest,
)


MESH_EXTS = ("stl", "obj", "step", "stp", "iges", "igs", "dae", "ply", "glb")


def read_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists() or not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def matrix_to_transform(parent: str, child: str, matrix: Any) -> Transform3D:
    return Transform3D(parent_frame=parent, child_frame=child, matrix=np.asarray(matrix, dtype=float).reshape(4, 4).tolist())


def identity_transform(parent: str, child: str) -> Transform3D:
    return matrix_to_transform(parent, child, np.eye(4))


def rpy_matrix(position: Iterable[float], rpy: Iterable[float], *, degrees: bool = False) -> np.ndarray:
    pos = list(position or [0.0, 0.0, 0.0])[:3]
    r = list(rpy or [0.0, 0.0, 0.0])[:3]
    if degrees:
        r = [math.radians(float(v)) for v in r]
    cr, sr = math.cos(float(r[0])), math.sin(float(r[0]))
    cp, sp = math.cos(float(r[1])), math.sin(float(r[1]))
    cy, sy = math.cos(float(r[2])), math.sin(float(r[2]))
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=float)
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=float)
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=float)
    out = np.eye(4)
    out[:3, :3] = rz @ ry @ rx
    out[:3, 3] = [float(pos[0] or 0.0), float(pos[1] or 0.0), float(pos[2] or 0.0)]
    return out


def quat_to_matrix(position: Iterable[float], quat_xyzw: Iterable[float]) -> np.ndarray:
    pos = list(position or [0.0, 0.0, 0.0])[:3]
    q = np.asarray(list(quat_xyzw or [0.0, 0.0, 0.0, 1.0])[:4], dtype=float)
    norm = float(np.linalg.norm(q))
    if norm < 1e-12:
      q = np.asarray([0.0, 0.0, 0.0, 1.0], dtype=float)
    else:
      q = q / norm
    x, y, z, w = q.tolist()
    out = np.eye(4)
    out[:3, :3] = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=float,
    )
    out[:3, 3] = [float(pos[0] or 0.0), float(pos[1] or 0.0), float(pos[2] or 0.0)]
    return out


def process_and_station(ctx: StationContext, asset_id: str) -> Tuple[str, Dict[str, Any]]:
    process = ctx.processes.get(asset_id)
    if not process:
        raise ValueError("process_not_found")
    station_id = str(process.get("station_id") or "")
    if not station_id:
        raise ValueError("station_not_set")
    return station_id, process


def asset_dir(ctx: StationContext, asset_id: str, kind: str) -> Path:
    station_id, _ = process_and_station(ctx, asset_id)
    if kind == "robot":
        return ctx.data_paths.process_robot_dir(station_id, asset_id)
    if kind == "gripper":
        return ctx.data_paths.process_gripper_dir(station_id, asset_id)
    raise ValueError("unsupported_asset_kind")


def manifest_summary(root: Path, public_base: str = "") -> Optional[Dict[str, Any]]:
    manifest = read_json(root / "manifest.json")
    if manifest is None:
        return None
    frames = read_json(root / "frames.json")
    urdf = str(manifest.get("urdf") or "").strip()
    item: Dict[str, Any] = {
        "id": str(manifest.get("name") or root.name),
        "name": str(manifest.get("display_name") or manifest.get("model") or manifest.get("name") or root.name),
        "path": root.name,
        "manifest": manifest,
        "has_frames": frames is not None,
        "frames": frames,
        "asset_root": str(root),
        "visual_count": len(list((root / "visual").glob("*"))) if (root / "visual").exists() else 0,
        "collision_count": len(list((root / "collision").glob("*"))) if (root / "collision").exists() else 0,
    }
    if urdf:
        item["urdf_path"] = str((root / urdf).resolve())
        item["urdf_url"] = f"{public_base}/{urdf}".replace("//", "/") if public_base else urdf
    return item


def find_object_mesh(ctx: StationContext, asset_id: str, object_id: str) -> Optional[Path]:
    station_id, _ = process_and_station(ctx, asset_id)
    obj_dir = ctx.data_paths.process_objects_dir(station_id, asset_id) / object_id
    if not obj_dir.exists():
        return None
    candidates: List[Path] = []
    for ext in MESH_EXTS:
        candidate = obj_dir / f"{object_id}.{ext}"
        if candidate.exists() and candidate.is_file():
            candidates.append(candidate)
    visual_dir = obj_dir / "visual"
    if visual_dir.exists():
        for ext in MESH_EXTS:
            candidates.extend(sorted(visual_dir.glob(f"model.{ext}")))
            candidates.extend(sorted(visual_dir.glob(f"*.{ext}")))
    if not candidates:
        for ext in MESH_EXTS:
            candidates.extend(sorted(obj_dir.glob(f"*.{ext}")))
    return candidates[0] if candidates else None


def load_object_grasps(ctx: StationContext, asset_id: str, object_id: str) -> List[Dict[str, Any]]:
    station_id, _ = process_and_station(ctx, asset_id)
    path = ctx.data_paths.process_objects_dir(station_id, asset_id) / object_id / "grasp_authoring.json"
    data = read_json(path) or {}
    grasps = data.get("grasps") if isinstance(data, dict) else []
    return grasps if isinstance(grasps, list) else []


def grasp_candidates_from_authoring(ctx: StationContext, asset_id: str, object_id: str) -> List[GraspCandidate]:
    candidates: List[GraspCandidate] = []
    for raw in load_object_grasps(ctx, asset_id, object_id):
        if not isinstance(raw, dict):
            continue
        grasp_id = str(raw.get("id") or raw.get("grasp_id") or f"grasp_{len(candidates) + 1}")
        if isinstance(raw.get("center_local_m"), list):
            tcp = _pick_contact_tcp_matrix(raw)
        else:
            pos_mm = raw.get("position") if isinstance(raw.get("position"), list) else [0.0, 0.0, 0.0]
            rpy_deg = raw.get("rotation") if isinstance(raw.get("rotation"), list) else [0.0, 0.0, 0.0]
            tcp = rpy_matrix([float(v or 0.0) / 1000.0 for v in pos_mm[:3]], rpy_deg[:3], degrees=True)
        pre = tcp @ rpy_matrix([0.0, 0.0, -0.10], [0.0, 0.0, 0.0])
        metadata = dict(raw)
        candidates.append(
            GraspCandidate(
                grasp_id=grasp_id,
                object_id=object_id,
                tcp_in_object=matrix_to_transform(object_id, f"{grasp_id}_tcp", tcp),
                pregrasp_in_object=matrix_to_transform(object_id, f"{grasp_id}_pregrasp", pre),
                score=float(raw.get("score") or 0.0),
                metadata=metadata,
            )
        )
        candidates.extend(_invariance_candidates(object_id, grasp_id, tcp, raw))
    return candidates


def _pick_contact_tcp_matrix(raw: Dict[str, Any]) -> np.ndarray:
    center = np.asarray(raw.get("center_local_m") or [0.0, 0.0, 0.0], dtype=float)
    jaw_y = _safe_unit(raw.get("jaw_axis_local") or [0.0, 1.0, 0.0], np.array([0.0, 1.0, 0.0]))
    approach_z = _safe_unit(raw.get("approach_axis_local") or [0.0, 0.0, -1.0], np.array([0.0, 0.0, -1.0]))
    x_axis = _safe_unit(np.cross(jaw_y, approach_z), np.array([1.0, 0.0, 0.0]))
    y_axis = _safe_unit(np.cross(approach_z, x_axis), jaw_y)
    out = np.eye(4)
    out[:3, 0] = x_axis
    out[:3, 1] = y_axis
    out[:3, 2] = approach_z
    out[:3, 3] = center
    return out


def _safe_unit(value: Any, fallback: np.ndarray) -> np.ndarray:
    vec = np.asarray(value, dtype=float).reshape(3)
    norm = float(np.linalg.norm(vec))
    if norm < 1e-12 or not np.isfinite(vec).all():
        return fallback.astype(float)
    return vec / norm


def _invariance_candidates(object_id: str, grasp_id: str, base_tcp: np.ndarray, raw: Dict[str, Any]) -> List[GraspCandidate]:
    inv = raw.get("invariance") if isinstance(raw.get("invariance"), dict) else {}
    if not inv or not bool(inv.get("enabled")):
        return []
    steps = max(1, min(36, int(float(inv.get("steps") or 8))))
    enabled = inv.get("enabledSteps")
    enabled_steps = set(int(v) for v in enabled) if isinstance(enabled, list) else set(range(steps + 1))
    lower = math.radians(float(inv.get("lowerLimit") or 0.0))
    upper = math.radians(float(inv.get("upperLimit") or 180.0))
    axis_name = str(inv.get("axis") or "z").lower()
    axis = {"x": np.array([1.0, 0.0, 0.0]), "y": np.array([0.0, 1.0, 0.0]), "z": np.array([0.0, 0.0, 1.0])}.get(axis_name, np.array([0.0, 0.0, 1.0]))
    axis_pos = np.asarray([(float(v or 0.0) / 1000.0) for v in (inv.get("axisPos") or [0.0, 0.0, 0.0])[:3]], dtype=float)
    out: List[GraspCandidate] = []
    for step in range(steps + 1):
        if step == 0 or step not in enabled_steps:
            continue
        angle = lower + (upper - lower) * (step / steps)
        tf = _axis_rotation(axis, angle, axis_pos) @ base_tcp
        gid = f"{grasp_id}_inv_{step}"
        out.append(
            GraspCandidate(
                grasp_id=gid,
                object_id=object_id,
                tcp_in_object=matrix_to_transform(object_id, f"{gid}_tcp", tf),
                score=float(raw.get("score") or 0.0),
                metadata={"source_grasp_id": grasp_id, "invariance_step": step, "invariance": inv},
            )
        )
    return out


def _axis_rotation(axis: np.ndarray, angle: float, center: np.ndarray) -> np.ndarray:
    axis = axis / max(float(np.linalg.norm(axis)), 1e-12)
    x, y, z = axis.tolist()
    c, s = math.cos(angle), math.sin(angle)
    r = np.array(
        [
            [c + x * x * (1 - c), x * y * (1 - c) - z * s, x * z * (1 - c) + y * s],
            [y * x * (1 - c) + z * s, c + y * y * (1 - c), y * z * (1 - c) - x * s],
            [z * x * (1 - c) - y * s, z * y * (1 - c) + x * s, c + z * z * (1 - c)],
        ],
        dtype=float,
    )
    out = np.eye(4)
    out[:3, :3] = r
    t1 = np.eye(4)
    t1[:3, 3] = center
    t2 = np.eye(4)
    t2[:3, 3] = -center
    return t1 @ out @ t2


def joint_state_from_robot_state(robot_state: Dict[str, Any], chain: Optional[KinematicChainConfig] = None) -> Dict[str, float]:
    raw = robot_state.get("joints") or robot_state.get("joint_positions") or robot_state.get("joint_state") or robot_state.get("q") or {}
    if isinstance(raw, dict):
        return {str(k): float(v or 0.0) for k, v in raw.items()}
    if isinstance(raw, (list, tuple)):
        names = [j.name for j in (chain.joints if chain else []) if j.joint_type != "fixed"]
        return {name: float(raw[idx] or 0.0) for idx, name in enumerate(names) if idx < len(raw)}
    return {}


def build_chain_from_urdf(urdf_path: Path, manifest: Dict[str, Any], gripper_frames: Optional[Dict[str, Any]] = None) -> Optional[KinematicChainConfig]:
    if not urdf_path.exists():
        return None
    root = ET.parse(urdf_path).getroot()
    by_child: Dict[str, ET.Element] = {}
    children: Dict[str, List[ET.Element]] = {}
    for joint in root.findall("joint"):
        parent = joint.find("parent")
        child = joint.find("child")
        if parent is None or child is None:
            continue
        parent_link = str(parent.attrib.get("link") or "")
        child_link = str(child.attrib.get("link") or "")
        if not parent_link or not child_link:
            continue
        by_child[child_link] = joint
        children.setdefault(parent_link, []).append(joint)
    planning = manifest.get("planning_groups") if isinstance(manifest.get("planning_groups"), dict) else {}
    arm = planning.get("arm") if isinstance(planning.get("arm"), dict) else {}
    base_frame = str(arm.get("base_link") or manifest.get("base_link") or "base")
    tip_frame = str(arm.get("tip_link") or manifest.get("tcp_link") or manifest.get("mount_link") or "")
    if not tip_frame:
        tip_frame = _leaf_from(base_frame, children) or base_frame
    joints: List[KinematicJointConfig] = []
    current = tip_frame
    reversed_joints: List[ET.Element] = []
    while current != base_frame and current in by_child:
        joint = by_child[current]
        reversed_joints.append(joint)
        parent = joint.find("parent")
        current = str(parent.attrib.get("link") if parent is not None else "")
    if current != base_frame:
        return None
    for joint in reversed(reversed_joints):
        joints.append(_joint_config(joint))
    tcp_cfg = None
    if isinstance(gripper_frames, dict):
        tcp_raw = gripper_frames.get("tcp") if isinstance(gripper_frames.get("tcp"), dict) else None
        if tcp_raw:
            tcp_matrix = rpy_matrix(
                [(float(v or 0.0) / 1000.0) for v in (tcp_raw.get("position") or [0, 0, 0])[:3]],
                (tcp_raw.get("rotation") or [0, 0, 0])[:3],
                degrees=True,
            )
            tcp_cfg = TCPConfig(tcp_id="tcp", transform=matrix_to_transform(tip_frame, "tcp", tcp_matrix))
    return KinematicChainConfig(chain_id=str(manifest.get("name") or "robot"), base_frame=base_frame, tip_frame=tip_frame, joints=joints, tcp=tcp_cfg)


def _leaf_from(base: str, children: Dict[str, List[ET.Element]]) -> Optional[str]:
    current = base
    seen = set()
    while current not in seen:
        seen.add(current)
        joints = children.get(current) or []
        if not joints:
            return current
        child = joints[-1].find("child")
        current = str(child.attrib.get("link") if child is not None else "")
    return None


def _joint_config(joint: ET.Element) -> KinematicJointConfig:
    parent = joint.find("parent")
    child = joint.find("child")
    origin = joint.find("origin")
    axis = joint.find("axis")
    limit = joint.find("limit")
    parent_link = str(parent.attrib.get("link") if parent is not None else "")
    child_link = str(child.attrib.get("link") if child is not None else "")
    xyz = _triple(origin.attrib.get("xyz") if origin is not None else None, [0.0, 0.0, 0.0])
    rpy = _triple(origin.attrib.get("rpy") if origin is not None else None, [0.0, 0.0, 0.0])
    joint_type = str(joint.attrib.get("type") or "fixed")
    if joint_type == "continuous":
        joint_type = "revolute"
    if joint_type not in {"revolute", "prismatic", "fixed"}:
        joint_type = "fixed"
    return KinematicJointConfig(
        name=str(joint.attrib.get("name") or f"{parent_link}_to_{child_link}"),
        parent_frame=parent_link,
        child_frame=child_link,
        joint_type=joint_type,
        axis=_triple(axis.attrib.get("xyz") if axis is not None else None, [1.0, 0.0, 0.0]),
        origin=matrix_to_transform(parent_link, child_link, rpy_matrix(xyz, rpy)),
        lower=float(limit.attrib.get("lower", -math.pi)) if limit is not None else (-math.pi if joint_type != "fixed" else 0.0),
        upper=float(limit.attrib.get("upper", math.pi)) if limit is not None else (math.pi if joint_type != "fixed" else 0.0),
    )


def _triple(raw: Optional[str], fallback: List[float]) -> List[float]:
    if not raw:
        return list(fallback)
    vals = [float(v) for v in str(raw).split()]
    return vals[:3] if len(vals) >= 3 else list(fallback)


def selected_object_id(ctx: StationContext, asset_id: str, requested: Optional[str] = None) -> Optional[str]:
    if requested is not None and not str(requested).strip():
        return None
    if str(requested or "").strip().lower() in {"none", "__none__", "dummy_testing_none"}:
        return None
    if requested:
        return requested
    station_id, _ = process_and_station(ctx, asset_id)
    objects_root = ctx.data_paths.process_objects_dir(station_id, asset_id)
    if not objects_root.exists():
        return None
    dirs = [p.name for p in sorted(objects_root.iterdir()) if p.is_dir()]
    return dirs[0] if dirs else None


def build_scene_request(ctx: StationContext, asset_id: str, object_id: Optional[str] = None, task_type: Optional[str] = None, task_id: Optional[str] = None) -> Tuple[UISceneRequest, Dict[str, Any]]:
    station_id, process = process_and_station(ctx, asset_id)
    task_type = str(task_type or process.get("task_type") or "").strip().lower()
    task_id = "".join(ch for ch in str(task_id or "").strip() if ch.isalnum() or ch in ("-", "_"))
    robot_root = ctx.data_paths.process_robot_dir(station_id, asset_id)
    gripper_root = ctx.data_paths.process_gripper_dir(station_id, asset_id)
    robot = manifest_summary(robot_root, f"/processes/{asset_id}/bin-picking/files/robot")
    gripper = manifest_summary(gripper_root, f"/processes/{asset_id}/bin-picking/files/gripper")
    robot_manifest = robot.get("manifest") if robot else {}
    gripper_frames = gripper.get("frames") if gripper else None
    chain = build_chain_from_urdf(Path(robot["urdf_path"]), robot_manifest, gripper_frames) if robot and robot.get("urdf_path") else None
    robot_state = ctx.robot.get_state() or {}
    joints = joint_state_from_robot_state(robot_state, chain)

    pin_available = importlib.util.find_spec("pinocchio") is not None
    robot_cfg = RobotModelConfig(
        robot_id=str(robot_manifest.get("name") or "robot"),
        urdf_path=str(robot.get("urdf_path")) if pin_available and robot and robot.get("urdf_path") else None,
        package_dirs=[str(robot_root)],
        base_frame=str((robot_manifest or {}).get("base_link") or "base"),
    ) if robot else None

    tcp_cfg = None
    gripper_cfg = None
    if gripper:
        frames = gripper.get("frames") if isinstance(gripper.get("frames"), dict) else {}
        tcp_raw = frames.get("tcp") if isinstance(frames.get("tcp"), dict) else None
        if tcp_raw:
            tcp_matrix = rpy_matrix(
                [(float(v or 0.0) / 1000.0) for v in (tcp_raw.get("position") or [0, 0, 0])[:3]],
                (tcp_raw.get("rotation") or [0, 0, 0])[:3],
                degrees=True,
            )
            root_frame = str((gripper.get("manifest") or {}).get("root_link") or "gripper")
            tcp_cfg = TCPConfig(tcp_id="tcp", transform=matrix_to_transform(root_frame, "tcp", tcp_matrix))
        collision_mesh = _first_mesh(gripper_root / "collision") or _first_mesh(gripper_root / "visual")
        gripper_cfg = GripperConfig(
            gripper_id=str((gripper.get("manifest") or {}).get("name") or "gripper"),
            mesh_path=str(collision_mesh) if collision_mesh else None,
            root_frame=str((gripper.get("manifest") or {}).get("root_link") or "gripper"),
            tcp=tcp_cfg,
        )

    chosen_object_id = selected_object_id(ctx, asset_id, object_id)
    object_asset = None
    object_pose = None
    grasps: List[GraspCandidate] = []
    if chosen_object_id:
        mesh = find_object_mesh(ctx, asset_id, chosen_object_id)
        if mesh:
            object_asset = ObjectAssetConfig(object_id=chosen_object_id, mesh_path=str(mesh), frame_id=chosen_object_id)
            object_pose = identity_transform("world", chosen_object_id)
        grasps = grasp_candidates_from_authoring(ctx, asset_id, chosen_object_id)

    bin_cfg = None
    bin_pose = None
    bin_path = ctx.data_paths.process_bin_path(station_id, asset_id)
    bin_data = read_json(bin_path) or {}
    size = _bin_size(bin_data)
    if size:
        bin_cfg = BinAssetConfig(bin_id="bin", frame_id="bin", size_xyz=size)
        bin_pose = identity_transform("world", "bin")

    gripper_manifest = gripper.get("manifest") if gripper else {}
    mount_parent_link = _robot_parent_link(robot_root, robot_manifest, str(robot_manifest.get("mount_link") or robot_manifest.get("tcp_link") or ""))
    collision_matrix = CollisionMatrix(default_action="check", rules=_collision_rules(
        robot_manifest,
        chosen_object_id,
        gripper_manifest,
        mount_parent_link=mount_parent_link,
    ))
    bin_picking_objects = []
    task_scene_yaml = (
        ctx.data_paths.process_task_scene_dir(station_id, asset_id, task_id) / "scene.yaml"
        if task_id
        else None
    )
    if task_scene_yaml is not None and task_scene_yaml.exists():
        dummy_objects, dummy_obstacle_meta = _task_collision_objects(task_scene_yaml.parent, task_scene_yaml, asset_id, task_id)
        dummy_meta = {"obstacles": dummy_obstacle_meta, "scene_path": str(task_scene_yaml)}
    elif task_type == "bin_picking" or task_type == "bin-picking":
        dummy_objects = []
        bin_picking_objects, bin_picking_obstacle_meta = _bin_picking_collision_objects(ctx, station_id, asset_id)
        dummy_meta = {"obstacles": bin_picking_obstacle_meta}
    else:
        dummy_objects, dummy_meta = _dummy_testing_collision_objects(ctx, station_id, asset_id)

    scene = UISceneRequest(
        robot=robot_cfg,
        gripper=gripper_cfg,
        tcp=tcp_cfg,
        object_asset=object_asset,
        object_pose=object_pose,
        bin=bin_cfg,
        bin_pose=bin_pose,
        collision_objects=(
            _robot_collision_objects(robot_root, chain, joints, robot_manifest) if chain else []
        ) + (
            _gripper_collision_objects(gripper_root, gripper, robot_manifest, chain, joints) if chain and gripper else []
        ) + dummy_objects + bin_picking_objects,
        collision_matrix=collision_matrix,
        grasp_candidates=grasps,
        target_grasp_id=grasps[0].grasp_id if grasps else None,
        joint_state=joints,
        chain=chain,
    )
    meta = {
        "station_id": station_id,
        "asset_id": asset_id,
        "object_id": chosen_object_id,
        "robot": robot,
        "gripper": gripper,
        "robot_state": robot_state,
        "bin": bin_data,
        "dummy_testing": dummy_meta,
    }
    return scene, meta


def collision_debug_scene(ctx: StationContext, asset_id: str, object_id: Optional[str] = None, task_type: Optional[str] = None, task_id: Optional[str] = None) -> Dict[str, Any]:
    """Return the exact collision objects robot_engine will evaluate."""
    from robot_engine.collision.collision_checker import check_active_pairs
    from robot_engine.collision.collision_world import CollisionWorld

    scene, meta = build_scene_request(ctx, asset_id, object_id, task_type=task_type, task_id=task_id)
    world = CollisionWorld.from_configs(scene.collision_objects or [], scene.collision_matrix)
    collision = check_active_pairs(world)
    objects: List[Dict[str, Any]] = []
    for cfg in scene.collision_objects or []:
        obj = world.objects.get(cfg.object_id)
        if obj is None:
            continue
        geom = obj.geometry
        mesh = geom.mesh
        objects.append({
            "object_id": obj.object_id,
            "group": obj.group,
            "frame_id": geom.frame_id,
            "asset_path": str(getattr(cfg, "asset_path", "") or ""),
            "scale": float(getattr(cfg, "scale", 1.0) or 1.0),
            "size_xyz": getattr(cfg, "size_xyz", None),
            "has_mesh": mesh is not None,
            "mesh_vertices": int(len(mesh.vertices)) if mesh is not None else 0,
            "mesh_faces": int(len(mesh.faces)) if mesh is not None else 0,
            "coal_ready": geom.coal_geometry is not None,
            "coal_error": str(getattr(geom, "coal_error", "") or ""),
            "backend": "coal" if geom.coal_geometry is not None else "exact_unavailable",
            "pose_matrix": obj.matrix.tolist(),
        })
    return {
        "status": "ok",
        "scene": scene.model_dump(),
        "meta": meta,
        "debug": {
            "objects": objects,
            "collision": collision.model_dump(),
            "notes": [
                "Robot link poses come from the selected robot URDF joints plus live robot state.",
                "Robot collision objects come from <collision> tags in the selected robot URDF.",
                "Dummy Testing obstacle poses come from dummy_testing/scene.yaml.",
                "Objects with coal_ready=false are not evaluated with AABB fallback; exact mesh collision is required.",
            ],
        },
    }


def _dummy_testing_collision_objects(
    ctx: StationContext,
    station_id: str,
    asset_id: str,
) -> Tuple[List[CollisionObjectConfig], Dict[str, Any]]:
    root = ctx.data_paths.process_dummy_testing_dir(station_id, asset_id)

    # Prefer scene.yaml (reference format); fall back to scene.json for compatibility.
    yaml_path = root / "scene.yaml"
    json_path = root / "scene.json"

    if yaml_path.exists():
        return _dummy_testing_collision_objects_from_yaml(root, yaml_path, asset_id)
    return _dummy_testing_collision_objects_from_json(root, json_path, asset_id)


def _dummy_testing_collision_objects_from_yaml(
    root: Any,
    yaml_path: Any,
    asset_id: str,
) -> Tuple[List[CollisionObjectConfig], Dict[str, Any]]:
    import yaml as _yaml

    with open(yaml_path) as f:
        data = _yaml.safe_load(f) or {}

    env = data.get("environment") or {}
    raw_obstacles = env.get("obstacles") if isinstance(env.get("obstacles"), list) else []
    out: List[CollisionObjectConfig] = []
    kept: List[Dict[str, Any]] = []

    for idx, raw in enumerate(raw_obstacles):
        if not isinstance(raw, dict):
            continue
        obstacle_id = str(raw.get("name") or f"obstacle_{idx + 1}")
        rel_path = str(raw.get("path") or "").strip()
        mesh_path = (root / rel_path).resolve() if rel_path else None
        if not mesh_path or not mesh_path.exists() or not mesh_path.is_file():
            continue

        # Parse transform: {translation: [x,y,z], rotation: [[r0],[r1],[r2]]} or 4x4 matrix
        tf = raw.get("transform") or {}
        if isinstance(tf, dict):
            translation = tf.get("translation") or [0.0, 0.0, 0.0]
            rotation = tf.get("rotation") or [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
            matrix = np.eye(4)
            matrix[:3, :3] = np.asarray(rotation, dtype=float)
            matrix[:3, 3] = np.asarray(translation, dtype=float)
        else:
            matrix = np.asarray(tf, dtype=float).reshape(4, 4) if tf else np.eye(4)

        frame_id = f"dummy:{obstacle_id}"
        out.append(
            CollisionObjectConfig(
                object_id=frame_id,
                asset_path=str(mesh_path),
                frame_id=frame_id,
                pose=matrix_to_transform("world", frame_id, matrix),
                group="fixture",
            )
        )
        kept.append({
            "id": obstacle_id,
            "name": obstacle_id,
            "mesh": rel_path,
            "url": f"/processes/{asset_id}/dummy-testing/files/{rel_path}",
            "pose": {"position_m": matrix[:3, 3].tolist(), "rotation_matrix": matrix[:3, :3].tolist()},
        })

    meta = {"scene_path": str(yaml_path), "scene_yaml_path": str(yaml_path), "obstacles": kept}
    return out, meta


def _dummy_testing_collision_objects_from_json(
    root: Any,
    json_path: Any,
    asset_id: str,
) -> Tuple[List[CollisionObjectConfig], Dict[str, Any]]:
    data = read_json(json_path) or {}
    raw_obstacles = data.get("obstacles") if isinstance(data.get("obstacles"), list) else []
    out: List[CollisionObjectConfig] = []
    kept: List[Dict[str, Any]] = []
    for idx, raw in enumerate(raw_obstacles):
        if not isinstance(raw, dict):
            continue
        obstacle_id = str(raw.get("id") or raw.get("name") or f"obstacle_{idx + 1}")
        rel_mesh = str(raw.get("mesh") or "").strip()
        mesh_path = (root / rel_mesh).resolve() if rel_mesh else None
        if not mesh_path or not mesh_path.exists() or not mesh_path.is_file():
            continue
        pose = raw.get("pose") if isinstance(raw.get("pose"), dict) else {}
        position_m = _pose_position_m(pose)
        rpy_deg = pose.get("rotation_rpy_deg") or pose.get("rotation_deg") or [0.0, 0.0, 0.0]
        matrix = rpy_matrix(position_m, rpy_deg, degrees=True)
        frame_id = f"dummy:{obstacle_id}"
        out.append(
            CollisionObjectConfig(
                object_id=frame_id,
                asset_path=str(mesh_path),
                frame_id=frame_id,
                pose=matrix_to_transform("world", frame_id, matrix),
                group="fixture",
            )
        )
        item = dict(raw)
        item["id"] = obstacle_id
        item["url"] = f"/processes/{asset_id}/dummy-testing/files/{rel_mesh}"
        kept.append(item)
    meta = {"scene_path": str(json_path), "obstacles": kept}
    return out, meta


def _bin_picking_collision_objects(
    ctx: StationContext,
    station_id: str,
    asset_id: str,
) -> Tuple[List[CollisionObjectConfig], List[Dict[str, Any]]]:
    import yaml as _yaml

    root = ctx.data_paths.process_dir(station_id, asset_id) / "bin_picking"
    yaml_path = root / "scene.yaml"
    if not yaml_path.exists():
        return [], []

    with open(yaml_path) as f:
        data = _yaml.safe_load(f) or {}

    env = data.get("environment") or {}
    raw_obstacles = env.get("obstacles") if isinstance(env.get("obstacles"), list) else []
    out: List[CollisionObjectConfig] = []
    kept: List[Dict[str, Any]] = []

    for idx, raw in enumerate(raw_obstacles):
        if not isinstance(raw, dict):
            continue
        obstacle_id = str(raw.get("name") or f"bin_obstacle_{idx + 1}")
        rel_path = str(raw.get("path") or "").strip()
        mesh_path = (root / rel_path).resolve() if rel_path else None
        if not mesh_path or not mesh_path.exists() or not mesh_path.is_file():
            continue

        tf = raw.get("transform") or {}
        if isinstance(tf, dict):
            translation = tf.get("translation") or [0.0, 0.0, 0.0]
            rotation = tf.get("rotation") or [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
            matrix = np.eye(4)
            matrix[:3, :3] = np.asarray(rotation, dtype=float)
            matrix[:3, 3] = np.asarray(translation, dtype=float)
        else:
            matrix = np.asarray(tf, dtype=float).reshape(4, 4) if tf else np.eye(4)

        frame_id = f"bin_picking:{obstacle_id}"
        out.append(
            CollisionObjectConfig(
                object_id=frame_id,
                asset_path=str(mesh_path),
                frame_id=frame_id,
                pose=matrix_to_transform("world", frame_id, matrix),
                group="fixture",
            )
        )
        kept.append({
            "id": obstacle_id,
            "name": obstacle_id,
            "mesh": rel_path,
            "url": f"/processes/{asset_id}/bin-picking/scene-files/{rel_path}",
            "pose": {"position_m": matrix[:3, 3].tolist(), "rotation_matrix": matrix[:3, :3].tolist()},
        })

    return out, kept


def _task_collision_objects(
    root: Any,
    yaml_path: Any,
    asset_id: str,
    task_id: str,
) -> Tuple[List[CollisionObjectConfig], List[Dict[str, Any]]]:
    import yaml as _yaml

    with open(yaml_path) as f:
        data = _yaml.safe_load(f) or {}

    env = data.get("environment") or {}
    raw_obstacles = env.get("obstacles") if isinstance(env.get("obstacles"), list) else []
    out: List[CollisionObjectConfig] = []
    kept: List[Dict[str, Any]] = []

    for idx, raw in enumerate(raw_obstacles):
        if not isinstance(raw, dict):
            continue
        obstacle_id = str(raw.get("name") or f"obstacle_{idx + 1}")
        rel_path = str(raw.get("path") or "").strip()
        mesh_path = (root / rel_path).resolve() if rel_path else None
        if not mesh_path or not mesh_path.exists() or not mesh_path.is_file():
            continue

        tf = raw.get("transform") or {}
        if isinstance(tf, dict):
            translation = tf.get("translation") or [0.0, 0.0, 0.0]
            rotation = tf.get("rotation") or [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
            matrix = np.eye(4)
            matrix[:3, :3] = np.asarray(rotation, dtype=float)
            matrix[:3, 3] = np.asarray(translation, dtype=float)
        else:
            matrix = np.asarray(tf, dtype=float).reshape(4, 4) if tf else np.eye(4)

        frame_id = f"task:{task_id}:{obstacle_id}"
        out.append(
            CollisionObjectConfig(
                object_id=frame_id,
                asset_path=str(mesh_path),
                frame_id=frame_id,
                pose=matrix_to_transform("world", frame_id, matrix),
                group="fixture",
            )
        )
        kept.append({
            "id": obstacle_id,
            "name": obstacle_id,
            "mesh": rel_path,
            "url": f"/processes/{asset_id}/tasks/{task_id}/scene-files/{rel_path}",
            "pose": {"position_m": matrix[:3, 3].tolist(), "rotation_matrix": matrix[:3, :3].tolist()},
        })

    return out, kept


def _pose_position_m(pose: Dict[str, Any]) -> List[float]:
    if isinstance(pose.get("position_m"), list):
        vals = pose.get("position_m")[:3]
        return [float(v or 0.0) for v in vals]
    if isinstance(pose.get("position_mm"), list):
        vals = pose.get("position_mm")[:3]
        return [float(v or 0.0) / 1000.0 for v in vals]
    return [0.0, 0.0, 0.0]


def _bin_size(data: Dict[str, Any]) -> Optional[List[float]]:
    for key in ("size_xyz", "dimensions_m", "size_m"):
        raw = data.get(key)
        if isinstance(raw, list) and len(raw) >= 3:
            return [float(raw[0]), float(raw[1]), float(raw[2])]
    dims = data.get("dimensions") if isinstance(data.get("dimensions"), dict) else {}
    vals = [dims.get("x") or dims.get("length"), dims.get("y") or dims.get("width"), dims.get("z") or dims.get("height")]
    if all(v is not None for v in vals):
        return [float(v) for v in vals]
    return None


def build_chain_for_process(ctx: StationContext, asset_id: str) -> Optional[KinematicChainConfig]:
    """Return the KinematicChainConfig for an asset, or None if robot/URDF unavailable."""
    try:
        station_id, _ = process_and_station(ctx, asset_id)
        robot_root = ctx.data_paths.process_robot_dir(station_id, asset_id)
        gripper_root = ctx.data_paths.process_gripper_dir(station_id, asset_id)
        robot = manifest_summary(robot_root)
        gripper = manifest_summary(gripper_root)
        if not robot or not robot.get("urdf_path"):
            return None
        gripper_frames = gripper.get("frames") if gripper else None
        return build_chain_from_urdf(Path(robot["urdf_path"]), robot.get("manifest") or {}, gripper_frames)
    except Exception:
        return None


def apply_tcp_offset_to_ee_pose(
    ee_position_m: List[float],
    ee_quat_xyzw: List[float],
    chain: KinematicChainConfig,
) -> Tuple[List[float], List[float], str]:
    """Transform an EE/flange pose into the TCP/fingertip frame.

    Returns (position_m, quat_xyzw, tcp_frame_name).
    If no TCP offset is configured on the chain, returns the inputs unchanged
    with tip_frame as the frame name.
    """
    if chain.tcp is None:
        return list(ee_position_m), list(ee_quat_xyzw), chain.tip_frame

    # Build O_T_EE from position+quaternion
    ee_mat = quat_to_matrix(ee_position_m, ee_quat_xyzw)

    # TCP offset: tip_frame -> tcp_frame (e.g. fr3_link8 -> tcp)
    tcp_offset = np.asarray(chain.tcp.transform.matrix, dtype=float)

    # O_T_tcp = O_T_EE @ tip_T_tcp
    tcp_mat = ee_mat @ tcp_offset

    pos = tcp_mat[:3, 3].tolist()

    # Extract quat from rotation matrix (xyzw)
    R = tcp_mat[:3, :3]
    trace = float(R[0, 0] + R[1, 1] + R[2, 2])
    if trace > 0:
        s = 0.5 / math.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s

    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm < 1e-12:
        quat = [0.0, 0.0, 0.0, 1.0]
    else:
        quat = [x / norm, y / norm, z / norm, w / norm]

    tcp_frame_name = chain.tcp.transform.child_frame
    return pos, quat, tcp_frame_name


def _first_mesh(root: Path) -> Optional[Path]:
    if not root.exists():
        return None
    for ext in MESH_EXTS:
        found = sorted(root.rglob(f"*.{ext}"))
        if found:
            return found[0]
    return None


def _collision_rules(
    robot_manifest: Dict[str, Any],
    object_id: Optional[str],
    gripper_manifest: Optional[Dict[str, Any]] = None,
    *,
    mount_parent_link: Optional[str] = None,
) -> List[CollisionPairRule]:
    rules: List[CollisionPairRule] = []
    for pair in robot_manifest.get("disabled_self_collisions") or []:
        if isinstance(pair, list) and len(pair) >= 2:
            rules.append(CollisionPairRule(object_a=f"robot:{pair[0]}", object_b=f"robot:{pair[1]}", action="ignore", reason="robot_disabled_self_collision"))
    for pair in (gripper_manifest or {}).get("disabled_self_collisions") or []:
        if isinstance(pair, list) and len(pair) >= 2:
            rules.append(CollisionPairRule(object_a=f"gripper:{pair[0]}", object_b=f"gripper:{pair[1]}", action="ignore", reason="gripper_disabled_self_collision"))
    gripper_root = str((gripper_manifest or {}).get("root_link") or "")
    robot_mount_links = {
        str(robot_manifest.get("mount_link") or ""),
        str(robot_manifest.get("tcp_link") or ""),
        str(mount_parent_link or ""),
    }
    if gripper_root:
        for link in sorted(v for v in robot_mount_links if v):
            rules.append(CollisionPairRule(
                object_a=f"gripper:{gripper_root}",
                object_b=f"robot:{link}",
                action="ignore",
                reason="mounted_gripper_base_contact",
            ))
    if object_id:
        rules.append(CollisionPairRule(object_a=str(object_id), object_b=str(object_id), action="ignore", reason="same_object"))
    return rules


def _robot_parent_link(robot_root: Path, manifest: Dict[str, Any], child_link: str) -> Optional[str]:
    if not child_link:
        return None
    urdf_path = robot_root / str(manifest.get("urdf") or "")
    if not urdf_path.exists():
        return None
    try:
        root = ET.parse(urdf_path).getroot()
    except Exception:
        return None
    for joint in root.findall("joint"):
        parent = joint.find("parent")
        child = joint.find("child")
        if child is None or parent is None:
            continue
        if str(child.attrib.get("link") or "") == child_link:
            return str(parent.attrib.get("link") or "") or None
    return None


def _robot_collision_objects(robot_root: Path, chain: KinematicChainConfig, joints: Dict[str, float], manifest: Dict[str, Any]) -> List[CollisionObjectConfig]:
    # Keep this v1 chain lightweight: one link-level collision object per mesh, posed from FK.
    from robot_engine.kinematics.kinematic_chain import KinematicChain

    fk = KinematicChain(chain).forward_matrices(joints).transforms
    urdf_path = robot_root / str(manifest.get("urdf") or "")
    if not urdf_path.exists():
        return []
    base_dir = urdf_path.parent
    root = ET.parse(urdf_path).getroot()
    out: List[CollisionObjectConfig] = []
    for link in root.findall("link"):
        link_name = str(link.attrib.get("name") or "")
        if link_name not in fk:
            continue
        for idx, coll in enumerate(link.findall("collision")):
            mesh = coll.find("geometry/mesh")
            if mesh is None:
                continue
            filename = str(mesh.attrib.get("filename") or "")
            if not filename:
                continue
            path = (base_dir / filename).resolve()
            scale = _mesh_uniform_scale(mesh)
            origin = coll.find("origin")
            rel = rpy_matrix(
                _triple(origin.attrib.get("xyz") if origin is not None else None, [0.0, 0.0, 0.0]),
                _triple(origin.attrib.get("rpy") if origin is not None else None, [0.0, 0.0, 0.0]),
            )
            object_id = f"robot:{link_name}" if idx == 0 else f"robot:{link_name}:{idx}"
            out.append(
                CollisionObjectConfig(
                    object_id=object_id,
                    asset_path=str(path),
                    frame_id=object_id,
                    pose=matrix_to_transform("world", object_id, fk[link_name] @ rel),
                    group="robot",
                    scale=scale,
                )
            )
    return out


def _gripper_collision_objects(
    gripper_root: Path,
    gripper_summary: Dict[str, Any],
    robot_manifest: Dict[str, Any],
    chain: KinematicChainConfig,
    robot_joints: Dict[str, float],
) -> List[CollisionObjectConfig]:
    manifest = gripper_summary.get("manifest") if isinstance(gripper_summary.get("manifest"), dict) else {}
    urdf_rel = str(manifest.get("urdf") or "")
    urdf_path = gripper_root / urdf_rel
    if not urdf_path.exists():
        return []
    mount_link = str(robot_manifest.get("mount_link") or robot_manifest.get("tcp_link") or chain.tip_frame)
    from robot_engine.kinematics.kinematic_chain import KinematicChain

    robot_fk = KinematicChain(chain).forward_matrices(robot_joints).transforms
    mount_world = robot_fk.get(mount_link)
    if mount_world is None:
        mount_world = robot_fk.get(chain.tip_frame)
    if mount_world is None:
        return []
    frames = gripper_summary.get("frames") if isinstance(gripper_summary.get("frames"), dict) else {}
    flange = frames.get("flange") if isinstance(frames.get("flange"), dict) else {}
    mount_to_gripper = rpy_matrix(
        [(float(v or 0.0) / 1000.0) for v in (flange.get("position") or [0, 0, 0])[:3]],
        (flange.get("rotation") or [0, 0, 0])[:3],
        degrees=True,
    )
    root_world = np.asarray(mount_world, dtype=float) @ mount_to_gripper
    gripper_joints = _gripper_joint_defaults(manifest)
    link_fk = _urdf_link_matrices(urdf_path, str(manifest.get("root_link") or ""), gripper_joints)
    if not link_fk:
        return []
    base_dir = urdf_path.parent
    root = ET.parse(urdf_path).getroot()
    out: List[CollisionObjectConfig] = []
    for link in root.findall("link"):
        link_name = str(link.attrib.get("name") or "")
        if link_name not in link_fk:
            continue
        for idx, coll in enumerate(link.findall("collision")):
            mesh = coll.find("geometry/mesh")
            if mesh is None:
                continue
            filename = str(mesh.attrib.get("filename") or "")
            if not filename:
                continue
            origin = coll.find("origin")
            rel = rpy_matrix(
                _triple(origin.attrib.get("xyz") if origin is not None else None, [0.0, 0.0, 0.0]),
                _triple(origin.attrib.get("rpy") if origin is not None else None, [0.0, 0.0, 0.0]),
            )
            object_id = f"gripper:{link_name}" if idx == 0 else f"gripper:{link_name}:{idx}"
            out.append(
                CollisionObjectConfig(
                    object_id=object_id,
                    asset_path=str((base_dir / filename).resolve()),
                    frame_id=object_id,
                    pose=matrix_to_transform("world", object_id, root_world @ link_fk[link_name] @ rel),
                    group="gripper",
                    scale=_mesh_uniform_scale(mesh),
                )
            )
    return out


def _urdf_link_matrices(urdf_path: Path, root_link: str, joints: Dict[str, float]) -> Dict[str, np.ndarray]:
    root = ET.parse(urdf_path).getroot()
    children: Dict[str, List[ET.Element]] = {}
    all_links = [str(link.attrib.get("name") or "") for link in root.findall("link")]
    child_links = set()
    for joint in root.findall("joint"):
        parent = joint.find("parent")
        child = joint.find("child")
        parent_link = str(parent.attrib.get("link") if parent is not None else "")
        child_link = str(child.attrib.get("link") if child is not None else "")
        if not parent_link or not child_link:
            continue
        children.setdefault(parent_link, []).append(joint)
        child_links.add(child_link)
    if not root_link:
        root_link = next((name for name in all_links if name not in child_links), all_links[0] if all_links else "")
    if not root_link:
        return {}
    out: Dict[str, np.ndarray] = {root_link: np.eye(4)}
    stack = [root_link]
    while stack:
        parent_link = stack.pop()
        parent_tf = out[parent_link]
        for joint in children.get(parent_link, []):
            child = joint.find("child")
            child_link = str(child.attrib.get("link") if child is not None else "")
            if not child_link:
                continue
            joint_tf = _joint_matrix_with_value(joint, joints)
            out[child_link] = parent_tf @ joint_tf
            stack.append(child_link)
    return out


def _joint_matrix_with_value(joint: ET.Element, joints: Dict[str, float]) -> np.ndarray:
    origin = joint.find("origin")
    mat = rpy_matrix(
        _triple(origin.attrib.get("xyz") if origin is not None else None, [0.0, 0.0, 0.0]),
        _triple(origin.attrib.get("rpy") if origin is not None else None, [0.0, 0.0, 0.0]),
    )
    joint_type = str(joint.attrib.get("type") or "fixed")
    if joint_type == "fixed":
        return mat
    name = str(joint.attrib.get("name") or "")
    axis_el = joint.find("axis")
    axis = np.asarray(_triple(axis_el.attrib.get("xyz") if axis_el is not None else None, [1.0, 0.0, 0.0]), dtype=float)
    norm = float(np.linalg.norm(axis))
    axis = axis / norm if norm > 1e-12 else np.asarray([1.0, 0.0, 0.0], dtype=float)
    value = float(joints.get(name, 0.0) or 0.0)
    motion = np.eye(4)
    if joint_type == "prismatic":
        motion[:3, 3] = axis * value
    elif joint_type in {"revolute", "continuous"}:
        motion[:3, :3] = _axis_angle_matrix(axis, value)
    return mat @ motion


def _axis_angle_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    x, y, z = axis.tolist()
    c, s = math.cos(angle), math.sin(angle)
    C = 1.0 - c
    return np.asarray([
        [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
        [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
        [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
    ], dtype=float)


def _gripper_joint_defaults(manifest: Dict[str, Any]) -> Dict[str, float]:
    named = manifest.get("named_states") if isinstance(manifest.get("named_states"), dict) else {}
    state = named.get("open") if isinstance(named.get("open"), dict) else {}
    out = {str(k): float(v or 0.0) for k, v in state.items()}
    mimic = manifest.get("mimic_joints") if isinstance(manifest.get("mimic_joints"), dict) else {}
    for name, cfg in mimic.items():
        if not isinstance(cfg, dict):
            continue
        src = str(cfg.get("joint") or "")
        out[str(name)] = float(out.get(src, 0.0)) * float(cfg.get("multiplier") or 1.0) + float(cfg.get("offset") or 0.0)
    return out


def _mesh_uniform_scale(mesh: Optional[ET.Element]) -> float:
    vals = _triple(mesh.attrib.get("scale") if mesh is not None else None, [1.0, 1.0, 1.0])
    if max(vals) - min(vals) > 1e-9:
        return 1.0
    return float(vals[0])


def evaluate_scene(ctx: StationContext, asset_id: str, object_id: Optional[str] = None, payload: Optional[Dict[str, Any]] = None, task_type: Optional[str] = None, task_id: Optional[str] = None) -> Dict[str, Any]:
    from robot_engine.interfaces.ui_api import RobotEngineContext
    from robot_engine.motion.motion_request import MotionRequest, MotionType
    from robot_engine.motion.path_planner import plan_motion

    scene, meta = build_scene_request(ctx, asset_id, object_id, task_type=task_type, task_id=task_id)
    request_payload = payload or {}
    context = RobotEngineContext()
    evaluation = context.evaluate_scene_from_ui(
        UISceneEvaluationRequest(
            scene=scene,
            distance_request=MinimumDistanceRequest() if request_payload.get("distance", True) else None,
            collision_request=CollisionCheckRequest() if request_payload.get("collision", True) else None,
            target_grasp_id=request_payload.get("target_grasp_id") or scene.target_grasp_id,
        )
    )
    motion = None
    if scene.chain and scene.target_grasp_id and scene.grasp_candidates:
        target = scene.grasp_candidates[0].tcp_in_object
        if scene.object_pose:
            target_matrix = np.asarray(scene.object_pose.matrix) @ np.asarray(target.matrix)
            target = matrix_to_transform(scene.chain.base_frame, "tcp", target_matrix)
        try:
            motion = plan_motion(
                MotionRequest(
                    motion_type=MotionType.JOINT,
                    chain=scene.chain,
                    current_joint_state=scene.joint_state,
                    target_frame=target,
                )
            )
        except Exception as exc:
            motion = {"success": False, "error": str(exc)}
    return {
        "status": "ok",
        "scene": scene.model_dump(),
        "meta": meta,
        "evaluation": evaluation.model_dump(),
        "motion": motion.model_dump() if hasattr(motion, "model_dump") else motion,
    }
