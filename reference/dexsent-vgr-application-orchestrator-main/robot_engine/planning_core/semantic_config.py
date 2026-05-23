from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import yaml

from robot_engine.environment.pointcloud_processing import PointCloudCollisionConfig


@dataclass
class JointLimit:
    name: str
    lower: float
    upper: float
    velocity: float = 1.5
    acceleration: float = 2.0
    jerk: float = 20.0


@dataclass
class PlanningGroup:
    name: str
    base_link: str
    tip_link: str
    joints: List[str]
    default_seed: List[float] = field(default_factory=list)
    # tip_T_tcp: 4x4 transform from tip_link frame to the TCP frame.
    # When set, IK targets the TCP frame; poses are expressed in this frame.
    # None means tip_link IS the control point (no offset).
    tcp_offset: Optional[np.ndarray] = field(default=None)

    def seed_array(self) -> np.ndarray:
        return np.asarray(self.default_seed, dtype=float)

    def tip_T_tcp(self) -> Optional[np.ndarray]:
        """4x4 tip_link→tcp transform, or None if not configured."""
        return self.tcp_offset

    def tcp_T_tip(self) -> Optional[np.ndarray]:
        """Inverse: 4x4 tcp→tip_link transform, or None if not configured."""
        if self.tcp_offset is None:
            return None
        return np.linalg.inv(self.tcp_offset)


@dataclass
class CollisionConfig:
    safety_margin: float = 0.02
    self_collision_enabled: bool = True
    world_collision_enabled: bool = True
    pointcloud: PointCloudCollisionConfig = field(default_factory=PointCloudCollisionConfig)
    pointcloud_enabled: bool = True


@dataclass
class PlannerConfig:
    type: str = "RRTConnect"
    timeout: float = 5.0
    interpolation_waypoints: int = 100
    simplify_path: bool = True
    max_joint_step: float = 0.2


@dataclass
class TrajectoryConfig:
    dt: float = 0.01


@dataclass
class EnvironmentObstacleConfig:
    name: str
    type: str
    path: str = ""
    transform: np.ndarray = field(default_factory=lambda: np.eye(4))
    scale: float = 1.0
    simplify_faces: Optional[int] = None
    shape: str = ""
    size: dict = field(default_factory=dict)
    pointcloud: Optional[PointCloudCollisionConfig] = None


@dataclass
class SemanticConfig:
    urdf_path: str
    package_dirs: List[str]
    groups: Dict[str, PlanningGroup]
    joint_limits: Dict[str, JointLimit]
    allowed_collisions: List[Tuple[str, str]]
    collision: CollisionConfig
    planner: PlannerConfig
    trajectory: TrajectoryConfig
    environment_obstacles: List[EnvironmentObstacleConfig] = field(default_factory=list)
    # When a gripper is attached at load time, the merged URDF XML is stored here.
    # PinocchioRobot uses this instead of urdf_path when present.
    urdf_xml: Optional[str] = field(default=None)

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    def group(self, name: str) -> PlanningGroup:
        if name not in self.groups:
            raise KeyError(f"Planning group '{name}' not found in semantics. Available: {list(self.groups)}")
        return self.groups[name]

    def velocity_limits(self, group_name: str) -> np.ndarray:
        g = self.group(group_name)
        return np.array([self.joint_limits[j].velocity for j in g.joints if j in self.joint_limits])

    def acceleration_limits(self, group_name: str) -> np.ndarray:
        g = self.group(group_name)
        return np.array([self.joint_limits[j].acceleration for j in g.joints if j in self.joint_limits])

    def jerk_limits(self, group_name: str) -> np.ndarray:
        g = self.group(group_name)
        return np.array([self.joint_limits[j].jerk for j in g.joints if j in self.joint_limits])

    def joint_lower(self, group_name: str) -> np.ndarray:
        g = self.group(group_name)
        return np.array([self.joint_limits[j].lower for j in g.joints if j in self.joint_limits])

    def joint_upper(self, group_name: str) -> np.ndarray:
        g = self.group(group_name)
        return np.array([self.joint_limits[j].upper for j in g.joints if j in self.joint_limits])

    # ------------------------------------------------------------------
    # Loader
    # ------------------------------------------------------------------

    @classmethod
    def from_yaml(cls, path: str | Path) -> "SemanticConfig":
        path = Path(path)
        with open(path) as f:
            d = yaml.safe_load(f)

        robot_d = d.get("robot", {})
        urdf_path = robot_d.get("urdf_path", "")
        package_dirs = robot_d.get("package_dirs", [])

        # Resolve urdf_path relative to the YAML file if not absolute
        if urdf_path and not Path(urdf_path).is_absolute():
            urdf_path = str(path.parent / urdf_path)

        package_dirs_abs = []
        for p in package_dirs:
            pp = Path(p)
            if not pp.is_absolute():
                pp = path.parent / pp
            package_dirs_abs.append(str(pp))

        # Optionally attach a gripper URDF at runtime (no combined file needed).
        # scene.yaml: gripper: { urdf_path: ../gripper/urdf/franka_hand.urdf,
        #                         attach_to_link: fr3_link8,
        #                         joint_name: fr3_hand_joint }
        urdf_xml: Optional[str] = None
        gripper_d = d.get("gripper")
        if gripper_d and urdf_path:
            gripper_urdf_rel = gripper_d.get("urdf_path", "")
            if gripper_urdf_rel:
                gripper_urdf_path = Path(gripper_urdf_rel)
                if not gripper_urdf_path.is_absolute():
                    gripper_urdf_path = path.parent / gripper_urdf_path
                attach_to = gripper_d.get("attach_to_link", "")
                joint_name = gripper_d.get("joint_name", "robot_gripper_joint")
                if gripper_urdf_path.exists() and attach_to:
                    urdf_xml = _merge_urdfs(
                        Path(urdf_path),
                        gripper_urdf_path,
                        attach_to_link=attach_to,
                        joint_name=joint_name,
                        package_dirs=package_dirs_abs,
                    )

        # Planning groups
        groups: Dict[str, PlanningGroup] = {}
        for gname, gd in d.get("planning_groups", {}).items():
            tcp_offset = _parse_tcp_offset(gd.get("tcp"))
            groups[gname] = PlanningGroup(
                name=gname,
                base_link=gd.get("base_link", ""),
                tip_link=gd.get("tip_link", ""),
                joints=gd.get("joints", []),
                default_seed=gd.get("default_seed", []),
                tcp_offset=tcp_offset,
            )

        # Joint limits
        joint_limits: Dict[str, JointLimit] = {}
        for jname, jd in d.get("joint_limits", {}).items():
            joint_limits[jname] = JointLimit(
                name=jname,
                lower=float(jd.get("lower", -np.pi)),
                upper=float(jd.get("upper", np.pi)),
                velocity=float(jd.get("velocity", 1.5)),
                acceleration=float(jd.get("acceleration", 2.0)),
                jerk=float(jd.get("jerk", 20.0)),
            )

        # Allowed collisions
        allowed: List[Tuple[str, str]] = []
        for pair in d.get("allowed_collisions", []):
            if len(pair) >= 2:
                allowed.append((str(pair[0]), str(pair[1])))

        # Collision config
        col_d = d.get("collision", {})
        pc_d = col_d.get("pointcloud", {})
        collision = CollisionConfig(
            safety_margin=float(col_d.get("safety_margin", 0.02)),
            self_collision_enabled=col_d.get("self_collision", {}).get("enabled", True),
            world_collision_enabled=col_d.get("world_collision", {}).get("enabled", True),
            pointcloud=PointCloudCollisionConfig.from_dict(pc_d),
            pointcloud_enabled=bool(pc_d.get("enabled", True)),
        )

        # Planner config
        plan_d = d.get("planner", {})
        planner = PlannerConfig(
            type=plan_d.get("type", "RRTConnect"),
            timeout=float(plan_d.get("timeout", 5.0)),
            interpolation_waypoints=int(plan_d.get("interpolation_waypoints", 100)),
            simplify_path=bool(plan_d.get("simplify_path", True)),
            max_joint_step=float(plan_d.get("max_joint_step", 0.2)),
        )

        # Trajectory config
        traj_d = d.get("trajectory", {})
        trajectory = TrajectoryConfig(dt=float(traj_d.get("dt", 0.01)))

        environment_obstacles = _load_environment_obstacles(d, path.parent, collision.pointcloud)

        return cls(
            urdf_path=urdf_path,
            package_dirs=package_dirs_abs,
            groups=groups,
            joint_limits=joint_limits,
            allowed_collisions=allowed,
            collision=collision,
            planner=planner,
            trajectory=trajectory,
            environment_obstacles=environment_obstacles,
            urdf_xml=urdf_xml,
        )

    @classmethod
    def minimal(
        cls,
        urdf_path: str,
        group_name: str,
        base_link: str,
        tip_link: str,
        joint_names: List[str],
        joint_lower: Optional[List[float]] = None,
        joint_upper: Optional[List[float]] = None,
        package_dirs: Optional[List[str]] = None,
    ) -> "SemanticConfig":
        """Build a minimal SemanticConfig without a YAML file (for testing / scripting)."""
        lower = joint_lower or [-np.pi] * len(joint_names)
        upper = joint_upper or [np.pi] * len(joint_names)
        jl = {
            name: JointLimit(name=name, lower=float(l), upper=float(u))
            for name, l, u in zip(joint_names, lower, upper)
        }
        g = PlanningGroup(
            name=group_name,
            base_link=base_link,
            tip_link=tip_link,
            joints=joint_names,
            default_seed=[0.0] * len(joint_names),
        )
        return cls(
            urdf_path=urdf_path,
            package_dirs=package_dirs or [],
            groups={group_name: g},
            joint_limits=jl,
            allowed_collisions=[],
            collision=CollisionConfig(),
            planner=PlannerConfig(),
            trajectory=TrajectoryConfig(),
        )


def _parse_tcp_offset(tcp_d) -> Optional[np.ndarray]:
    """Parse a tcp section dict into a 4x4 tip_T_tcp matrix, or None if absent."""
    if tcp_d is None:
        return None
    if isinstance(tcp_d, dict):
        pos = [float(v) for v in (tcp_d.get("position_m") or [0.0, 0.0, 0.0])[:3]]
        rpy_deg = [float(v) for v in (tcp_d.get("rotation_rpy_deg") or [0.0, 0.0, 0.0])[:3]]
        import math
        r, p, y = [math.radians(v) for v in rpy_deg]
        cr, sr = math.cos(r), math.sin(r)
        cp, sp = math.cos(p), math.sin(p)
        cy, sy = math.cos(y), math.sin(y)
        Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
        Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
        Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
        T = np.eye(4)
        T[:3, :3] = Rz @ Ry @ Rx
        T[:3, 3] = pos
        return T
    # Allow raw 4x4 list
    try:
        T = np.asarray(tcp_d, dtype=float)
        if T.shape == (4, 4):
            return T
    except Exception:
        pass
    return None


def _load_environment_obstacles(
    document: dict,
    base_dir: Path,
    default_pointcloud: PointCloudCollisionConfig,
) -> List[EnvironmentObstacleConfig]:
    env = document.get("environment", {})
    entries = env.get("obstacles", [])
    obstacles: List[EnvironmentObstacleConfig] = []

    for entry in entries:
        typ = str(entry.get("type", "")).lower()
        name = str(entry.get("name", ""))
        if not name:
            raise ValueError("Every environment obstacle requires a non-empty 'name'.")
        if typ not in {"mesh", "primitive", "pointcloud"}:
            raise ValueError(f"Unsupported environment obstacle type for '{name}': {typ!r}")

        path = str(entry.get("path", ""))
        if path:
            p = Path(path)
            if not p.is_absolute():
                p = base_dir / p
            path = str(p)

        transform = _parse_transform(entry.get("transform", np.eye(4)))
        pointcloud_cfg = None
        if typ == "pointcloud":
            override = dict(entry.get("pointcloud", {}))
            if override:
                merged = {
                    "voxel_size": default_pointcloud.voxel_size,
                    "inflation_radius": default_pointcloud.inflation_radius,
                    "remove_statistical_outliers": default_pointcloud.remove_statistical_outliers,
                    "statistical_nb_neighbors": default_pointcloud.statistical_nb_neighbors,
                    "statistical_std_ratio": default_pointcloud.statistical_std_ratio,
                    "crop": {
                        "enabled": default_pointcloud.crop_enabled,
                        "min": default_pointcloud.crop_min.tolist(),
                        "max": default_pointcloud.crop_max.tolist(),
                    },
                }
                merged.update(override)
                pointcloud_cfg = PointCloudCollisionConfig.from_dict(merged)

        obstacles.append(
            EnvironmentObstacleConfig(
                name=name,
                type=typ,
                path=path,
                transform=transform,
                scale=float(entry.get("scale", 1.0)),
                simplify_faces=entry.get("simplify_faces"),
                shape=str(entry.get("shape", "")),
                size=dict(entry.get("size", {})),
                pointcloud=pointcloud_cfg,
            )
        )

    return obstacles


def _merge_urdfs(
    robot_urdf: Path,
    gripper_urdf: Path,
    attach_to_link: str,
    joint_name: str,
    package_dirs: List[str],
) -> str:
    """Merge two URDF files in-memory and return the combined XML string.

    All mesh paths are made absolute so Pinocchio can resolve them from the
    XML string without a filesystem URDF file.
    """
    def _absolutize_mesh_paths(tree_root: ET.Element, base_dir: Path) -> None:
        for geom in tree_root.iter("geometry"):
            mesh = geom.find("mesh")
            if mesh is None:
                continue
            fname = mesh.get("filename", "")
            if not fname:
                continue
            # Handle package:// URIs
            if fname.startswith("package://"):
                rel = fname[len("package://"):]
                for pkg_dir in package_dirs:
                    candidate = Path(pkg_dir) / rel
                    if candidate.exists():
                        mesh.set("filename", str(candidate.resolve()))
                        break
                # If still package://, try stripping the first path component (package name)
                if mesh.get("filename", "").startswith("package://"):
                    parts = rel.split("/", 1)
                    if len(parts) == 2:
                        for pkg_dir in package_dirs:
                            candidate = Path(pkg_dir) / parts[1]
                            if candidate.exists():
                                mesh.set("filename", str(candidate.resolve()))
                                break
            elif not Path(fname).is_absolute():
                mesh.set("filename", str((base_dir / fname).resolve()))

    robot_tree = ET.parse(robot_urdf)
    robot_root = robot_tree.getroot()
    _absolutize_mesh_paths(robot_root, robot_urdf.parent)

    gripper_tree = ET.parse(gripper_urdf)
    gripper_root = gripper_tree.getroot()
    _absolutize_mesh_paths(gripper_root, gripper_urdf.parent)

    # Find the gripper's root link name
    gripper_child_links = {
        j.find("child").get("link")
        for j in gripper_root.findall("joint")
        if j.find("child") is not None
    }
    gripper_root_links = [
        l.get("name")
        for l in gripper_root.findall("link")
        if l.get("name") not in gripper_child_links
    ]
    gripper_root_link = gripper_root_links[0] if gripper_root_links else None

    # Copy all links and joints from gripper into robot tree
    for elem in list(gripper_root):
        if elem.tag in ("link", "joint"):
            robot_root.append(elem)

    # Add fixed attachment joint
    if gripper_root_link:
        attach_joint = ET.SubElement(robot_root, "joint", name=joint_name, type="fixed")
        ET.SubElement(attach_joint, "parent", link=attach_to_link)
        ET.SubElement(attach_joint, "child", link=gripper_root_link)
        ET.SubElement(attach_joint, "origin", xyz="0 0 0", rpy="0 0 0")

    ET.register_namespace("", "")
    return ET.tostring(robot_root, encoding="unicode")


def _parse_transform(value) -> np.ndarray:
    if isinstance(value, dict):
        translation = np.asarray(value.get("translation", [0.0, 0.0, 0.0]), dtype=float)
        rotation = np.asarray(value.get("rotation", np.eye(3)), dtype=float)
        if rotation.shape != (3, 3):
            raise ValueError("Environment obstacle rotation must be a 3x3 matrix.")
        T = np.eye(4)
        T[:3, :3] = rotation
        T[:3, 3] = translation
        return T

    T = np.asarray(value, dtype=float)
    if T.shape != (4, 4):
        raise ValueError("Environment obstacle transform must be a 4x4 matrix or {translation, rotation}.")
    return T
