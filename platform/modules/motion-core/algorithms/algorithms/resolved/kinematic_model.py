# Copyright (c) 2026 DexSent Robotics Pvt Ltd.
# All Rights Reserved.
#
# SPDX-License-Identifier: LicenseRef-Proprietary
# See LICENSE file for details.
#
# Author: Shreyas <shreyas.kumar@dexsentrobotics.com>
# Proprietary and confidential.

"""Resolved kinematic representation of a robot system.

One `KinematicModel` per robot system. Built once at startup (~100 ms -
URDF parsing, Pinocchio model construction, mimic-joint analysis, chain
slicing) and reused by every downstream operation (FK, Jacobian, IK,
collision, planning).

Design contracts
----------------
* **Composed Pinocchio model.** Robot URDF and (optional) gripper URDF
  are joined into a single `pin.Model` via `pin.appendModel`. One
  call evaluates kinematics for the whole chain.

* **Mimic-joint handling.** URDF `<mimic>` followers are kept in the
  underlying Pinocchio model but excluded from the user-facing active
  DOF list. The class exposes a linear expansion map
  `q_full = active_to_full @ q_active + active_offset` that operations
  apply internally; callers never see the mimic followers.

* **Limit resolution.** Position, velocity, and effort limits default
  from URDF; YAML overrides per joint. Acceleration and jerk are not in
  URDF - YAML supplies them, and missing values raise at build time for
  any joint participating in a kinematic chain.

* **No `pin.Data` ownership.** A KinematicModel may back several
  world-robot instances (cache hit on identical YAMLs). Sharing one
  `pin.Data` would corrupt FK across them. Operations allocate fresh
  `pin.Data` per call via `model.pin_model.createData()`.

See `docs/architecture.md` for the full layered design.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from algorithms.descriptions import (
    JointLimitsSpec,
    KinematicChainSpec,
    RobotSystemDescription,
)


# ---------------------------------------------------------------------------
# Mimic-joint parsing
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MimicRelation:
    """One URDF `<mimic>` relation: `follower = multiplier * driver + offset`."""

    follower: str
    driver: str
    multiplier: float = 1.0
    offset: float = 0.0


def parse_mimic_relations(urdf_path: Path) -> List[MimicRelation]:
    """Parse URDF `<mimic>` tags from one URDF file.

    Pinocchio's experimental mimic support is fragile; this module handles
    expansion explicitly via the `active_to_full` map instead.

    Parameters
    ----------
    urdf_path : Path
        Path to a URDF file.

    Returns
    -------
    list[MimicRelation]
        One entry per `<mimic>` tag found, with defaults
        `multiplier=1.0` and `offset=0.0` if not declared.
    """
    tree = ET.parse(str(urdf_path))
    relations: List[MimicRelation] = []
    for joint in tree.getroot().findall("joint"):
        mimic = joint.find("mimic")
        if mimic is None:
            continue
        follower = joint.get("name")
        driver = mimic.get("joint")
        if follower is None or driver is None:
            raise ValueError(f"malformed <mimic> tag in {urdf_path}")
        multiplier = float(mimic.get("multiplier", "1.0"))
        offset = float(mimic.get("offset", "0.0"))
        relations.append(
            MimicRelation(
                follower=follower, driver=driver, multiplier=multiplier, offset=offset
            )
        )
    return relations


# ---------------------------------------------------------------------------
# Per-chain joint limits view
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class JointLimits:
    """Resolved joint limits for one joint, as NumPy 0-d arrays.

    Returned in chain joint order from `KinematicModel.*_limits` methods.
    The 0-d array shape preserves NumPy semantics when stacking via
    `np.asarray(...)`.
    """

    lower: np.ndarray
    upper: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray
    jerk: np.ndarray
    effort: np.ndarray


# ---------------------------------------------------------------------------
# KinematicModel
# ---------------------------------------------------------------------------


class KinematicModel:
    """Heavy, cached resolved kinematic representation of a robot system.

    Build with :meth:`from_robot_system`. The result is reusable by every
    downstream operation; do not hold onto its `pin.Data` companion -
    operations allocate fresh buffers per call (see module docstring).

    Public attributes
    -----------------
    pin_model : pinocchio.Model
        The composed model. `pin_model.nq` is the full DOF count
        including mimic-follower joints.
    active_joint_names : list[str]
        User-facing joint names in Pinocchio q-index order, excluding
        mimic followers.
    full_joint_names : list[str]
        All Pinocchio joints in q-index order, including mimic followers.
    active_to_full : np.ndarray, shape (full_dof, active_dof)
        Linear map from active-DOF q to full-DOF q. Operations use this
        for FK input expansion and Jacobian column folding.
    active_offset : np.ndarray, shape (full_dof,)
        Affine offset for the expansion (non-zero only when a mimic
        relation declares a non-zero offset).
    """

    def __init__(
        self,
        *,
        system: RobotSystemDescription,
        pin_model,
        active_joint_names: List[str],
        full_joint_names: List[str],
        active_to_full: np.ndarray,
        active_offset: np.ndarray,
        per_joint_limits: Dict[str, JointLimits],
    ) -> None:
        self._system = system
        self.pin_model = pin_model
        self.active_joint_names = active_joint_names
        self.full_joint_names = full_joint_names
        self.active_to_full = active_to_full
        self.active_offset = active_offset
        self._per_joint_limits = per_joint_limits

        # Internal: active joint name -> index into active q vector.
        # Kept private; callers use the public chain/index helpers below.
        self._active_index: Dict[str, int] = {
            name: idx for idx, name in enumerate(active_joint_names)
        }

    # ------------------------------------------------------------------
    # Chain helpers
    # ------------------------------------------------------------------

    @property
    def system(self) -> RobotSystemDescription:
        """The originating description object (immutable)."""
        return self._system

    def chain(self, chain_id: str) -> KinematicChainSpec:
        """Return the chain specification for the given id."""
        return self._system.chain(chain_id)

    def chain_indices(self, chain_id: str) -> np.ndarray:
        """Indices into the active q vector corresponding to a chain's joints.

        Use to slice an active q vector down to one chain's DOF:
        `q_arm = q_active[model.chain_indices("arm")]`.
        """
        chain = self.chain(chain_id)
        missing = [j for j in chain.joints if j not in self._active_index]
        if missing:
            raise ValueError(
                f"chain {chain_id!r} references joints not present as active DOF "
                f"in the composed model: {missing}"
            )
        return np.asarray([self._active_index[j] for j in chain.joints], dtype=int)

    def chain_dof(self, chain_id: str) -> int:
        """Number of active degrees of freedom in a chain."""
        return len(self.chain(chain_id).joints)

    # ------------------------------------------------------------------
    # DOF expansion
    # ------------------------------------------------------------------

    def expand(self, q_active: np.ndarray) -> np.ndarray:
        """Expand an active-DOF q vector into the full Pinocchio q vector.

        Computes `q_full = active_to_full @ q_active + active_offset`.
        Mimic followers receive `multiplier * driver_value + offset`.

        Parameters
        ----------
        q_active : np.ndarray
            Shape `(len(active_joint_names),)`.

        Returns
        -------
        np.ndarray
            Shape `(pin_model.nq,)`.

        Raises
        ------
        ValueError
            If `q_active` does not have the expected shape.
        """
        q_active = np.asarray(q_active, dtype=float)
        if q_active.shape != (len(self.active_joint_names),):
            raise ValueError(
                f"q_active has shape {q_active.shape}; "
                f"expected ({len(self.active_joint_names)},)"
            )
        return self.active_to_full @ q_active + self.active_offset

    # ------------------------------------------------------------------
    # Limits accessors - chain-ordered NumPy vectors
    # ------------------------------------------------------------------

    def _stack(self, chain_id: str, field: str) -> np.ndarray:
        chain = self.chain(chain_id)
        return np.asarray(
            [getattr(self._per_joint_limits[j], field).item() for j in chain.joints],
            dtype=float,
        )

    def position_limits(self, chain_id: str) -> Tuple[np.ndarray, np.ndarray]:
        """Return `(lower, upper)` joint position limits for a chain."""
        chain = self.chain(chain_id)
        lower = np.asarray(
            [self._per_joint_limits[j].lower.item() for j in chain.joints], dtype=float
        )
        upper = np.asarray(
            [self._per_joint_limits[j].upper.item() for j in chain.joints], dtype=float
        )
        return lower, upper

    def active_position_limits(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return `(lower, upper)` limits in `active_joint_names` order."""
        lower = np.asarray(
            [self._per_joint_limits[j].lower.item() for j in self.active_joint_names],
            dtype=float,
        )
        upper = np.asarray(
            [self._per_joint_limits[j].upper.item() for j in self.active_joint_names],
            dtype=float,
        )
        return lower, upper

    def active_velocity_limits(self) -> np.ndarray:
        """Return velocity limits in `active_joint_names` order."""
        return np.asarray(
            [
                self._per_joint_limits[j].velocity.item()
                for j in self.active_joint_names
            ],
            dtype=float,
        )

    def active_acceleration_limits(self) -> np.ndarray:
        """Return acceleration limits in `active_joint_names` order.

        Raises if any active joint has no acceleration limit declared
        — the trajectory layer requires complete envelopes.
        """
        return np.asarray(
            [
                self._per_joint_limits[j].acceleration.item()
                for j in self.active_joint_names
            ],
            dtype=float,
        )

    def active_jerk_limits(self) -> np.ndarray:
        """Return jerk limits in `active_joint_names` order."""
        return np.asarray(
            [
                self._per_joint_limits[j].jerk.item()
                for j in self.active_joint_names
            ],
            dtype=float,
        )

    def active_effort_limits(self) -> np.ndarray:
        """Return effort / torque limits in `active_joint_names` order."""
        return np.asarray(
            [
                self._per_joint_limits[j].effort.item()
                for j in self.active_joint_names
            ],
            dtype=float,
        )

    def velocity_limits(self, chain_id: str) -> np.ndarray:
        """Return joint velocity limits for a chain."""
        return self._stack(chain_id, "velocity")

    def acceleration_limits(self, chain_id: str) -> np.ndarray:
        """Return joint acceleration limits for a chain."""
        return self._stack(chain_id, "acceleration")

    def jerk_limits(self, chain_id: str) -> np.ndarray:
        """Return joint jerk limits for a chain."""
        return self._stack(chain_id, "jerk")

    def effort_limits(self, chain_id: str) -> np.ndarray:
        """Return joint effort/torque limits for a chain."""
        return self._stack(chain_id, "effort")

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_robot_system(cls, system: RobotSystemDescription) -> "KinematicModel":
        """Build or fetch the cached kinematic model for a robot system.

        The cache key is `(yaml_path, urdf_mtimes)`; touching any URDF
        invalidates the cached entry automatically.
        """
        if system.source_path is None:
            return _build_kinematic_model(system)

        urdf_paths = [system.resolve_path(system.robot.urdf_path)]
        if system.gripper is not None:
            urdf_paths.append(system.resolve_path(system.gripper.urdf_path))

        key = (str(system.source_path), tuple(p.stat().st_mtime for p in urdf_paths))
        cached = _CACHE.get(key)
        if cached is not None:
            _CACHE.move_to_end(key)
            return cached

        built = _build_kinematic_model(system)
        _CACHE[key] = built
        _CACHE.move_to_end(key)
        while len(_CACHE) > _CACHE_MAXSIZE:
            _CACHE.popitem(last=False)
        return built


# ---------------------------------------------------------------------------
# Module-level cache
# ---------------------------------------------------------------------------


# Pydantic models are unhashable, so functools.lru_cache cannot key on the
# description directly. The cache is keyed on (yaml_path, urdf_mtimes), which
# is also what enables auto-invalidation on URDF edits.
_CACHE_MAXSIZE = 16
_CACHE: "OrderedDict[Tuple[str, Tuple[float, ...]], KinematicModel]" = OrderedDict()


def _clear_cache() -> None:
    """Empty the kinematic-model cache. Intended for tests."""
    _CACHE.clear()


# ---------------------------------------------------------------------------
# Build path
# ---------------------------------------------------------------------------


def _build_kinematic_model(system: RobotSystemDescription) -> KinematicModel:
    """Construct a fresh KinematicModel from a RobotSystemDescription."""
    import pinocchio as pin

    # ---- Load and compose Pinocchio models -----------------------------
    robot_urdf = _require_existing(system.resolve_path(system.robot.urdf_path))
    pin_model = pin.buildModelFromUrdf(str(robot_urdf))
    mimic_relations: List[MimicRelation] = list(parse_mimic_relations(robot_urdf))

    if system.gripper is not None:
        gripper_urdf = _require_existing(system.resolve_path(system.gripper.urdf_path))
        gripper_model = pin.buildModelFromUrdf(str(gripper_urdf))
        mimic_relations.extend(parse_mimic_relations(gripper_urdf))

        parent_frame_name = system.gripper.mount.parent_frame
        if not pin_model.existFrame(parent_frame_name):
            raise ValueError(
                f"gripper mount parent frame {parent_frame_name!r} not found in robot URDF"
            )
        # pin.appendModel's third argument is a FRAME id (not a joint id).
        # Passing a joint id silently attaches the gripper at the wrong frame.
        parent_frame_id = pin_model.getFrameId(parent_frame_name)
        mount_se3 = _se3_from_matrix(system.gripper.mount.as_matrix())
        pin_model = pin.appendModel(
            pin_model, gripper_model, parent_frame_id, mount_se3
        )

    _add_yaml_tcp_frames(pin_model, system)

    # ---- Determine active vs full DOF ordering ------------------------
    # Skip the "universe" root joint (index 0, nq=0). Limit to 1-DoF joints
    # because the rest of the limits API assumes one position per joint.
    full_joint_names: List[str] = [
        pin_model.names[i]
        for i in range(1, pin_model.njoints)
        if pin_model.joints[i].nq == 1
    ]
    followers = {rel.follower: rel for rel in mimic_relations}
    active_joint_names: List[str] = [j for j in full_joint_names if j not in followers]

    # ---- Build the linear expansion map -------------------------------
    # q_full = active_to_full @ q_active + active_offset
    full_dof = pin_model.nq
    active_dof = len(active_joint_names)
    active_to_full = np.zeros((full_dof, active_dof), dtype=float)
    active_offset = np.zeros(full_dof, dtype=float)

    active_index = {name: idx for idx, name in enumerate(active_joint_names)}
    for joint_name in full_joint_names:
        joint_id = pin_model.getJointId(joint_name)
        idx_q = pin_model.joints[joint_id].idx_q
        if joint_name in followers:
            rel = followers[joint_name]
            if rel.driver not in active_index:
                raise ValueError(
                    f"mimic joint {joint_name!r} drives off {rel.driver!r}, "
                    "which is not an active DOF"
                )
            active_to_full[idx_q, active_index[rel.driver]] = rel.multiplier
            active_offset[idx_q] = rel.offset
        else:
            active_to_full[idx_q, active_index[joint_name]] = 1.0

    # ---- Resolve per-joint limits -------------------------------------
    per_joint_limits = _build_joint_limits(
        system=system,
        pin_model=pin_model,
        active_joint_names=active_joint_names,
    )

    return KinematicModel(
        system=system,
        pin_model=pin_model,
        active_joint_names=active_joint_names,
        full_joint_names=full_joint_names,
        active_to_full=active_to_full,
        active_offset=active_offset,
        per_joint_limits=per_joint_limits,
    )


# ---------------------------------------------------------------------------
# Limit resolution: URDF defaults overlaid with YAML overrides
# ---------------------------------------------------------------------------


def _build_joint_limits(
    *,
    system: RobotSystemDescription,
    pin_model,
    active_joint_names: List[str],
) -> Dict[str, JointLimits]:
    """Resolve per-active-joint limits.

    For each active joint, position / velocity / effort default to the
    URDF value with optional YAML override. Acceleration and jerk are
    not carried by URDF; they must be supplied in YAML for any joint
    that participates in a kinematic chain. The resolution raises if
    that contract is violated.
    """
    # Combine YAML overrides from both robot and gripper into one dict.
    yaml_overrides: Dict[str, JointLimitsSpec] = {}
    yaml_overrides.update(system.robot.joint_limits)
    if system.gripper is not None:
        yaml_overrides.update(system.gripper.joint_limits)

    # Joints that appear in any chain need accel and jerk (trajectory generation).
    chained_joints: set[str] = set()
    for chain in system.kinematic_chains:
        chained_joints.update(chain.joints)

    out: Dict[str, JointLimits] = {}
    missing_traj: List[str] = []

    for joint_name in active_joint_names:
        joint_id = pin_model.getJointId(joint_name)
        idx_q = pin_model.joints[joint_id].idx_q
        idx_v = pin_model.joints[joint_id].idx_v

        urdf_lower = float(pin_model.lowerPositionLimit[idx_q])
        urdf_upper = float(pin_model.upperPositionLimit[idx_q])
        urdf_velocity = float(pin_model.velocityLimit[idx_v])
        urdf_effort = float(pin_model.effortLimit[idx_v])

        override = yaml_overrides.get(joint_name)
        lower, upper = urdf_lower, urdf_upper
        velocity, effort = urdf_velocity, urdf_effort
        acceleration: Optional[float] = None
        jerk: Optional[float] = None

        if override is not None:
            if override.position is not None:
                lower, upper = float(override.position[0]), float(override.position[1])
            if override.velocity is not None:
                velocity = float(override.velocity)
            if override.effort is not None:
                effort = float(override.effort)
            if override.acceleration is not None:
                acceleration = float(override.acceleration)
            if override.jerk is not None:
                jerk = float(override.jerk)

        if joint_name in chained_joints:
            if acceleration is None or jerk is None:
                missing_traj.append(joint_name)
                continue

        out[joint_name] = JointLimits(
            lower=np.array(lower),
            upper=np.array(upper),
            velocity=np.array(velocity),
            acceleration=np.array(acceleration if acceleration is not None else np.nan),
            jerk=np.array(jerk if jerk is not None else np.nan),
            effort=np.array(effort),
        )

    if missing_traj:
        raise ValueError(
            "acceleration and jerk are required in YAML joint_limits for joints "
            "that appear in any kinematic_chain (URDF does not carry them). "
            f"Missing: {sorted(missing_traj)}"
        )

    return out


# ---------------------------------------------------------------------------
# YAML frame injection
# ---------------------------------------------------------------------------


def _add_yaml_tcp_frames(pin_model, system: RobotSystemDescription) -> None:
    """Add TCP frames declared in YAML to the composed Pinocchio model.

    Manufacturer URDFs sometimes already contain useful tool frames. When the
    YAML TCP child frame already exists in the composed model, the URDF frame
    remains authoritative and no duplicate frame is added. Missing TCP child
    frames are added as fixed operational frames under their declared parent.
    """
    import pinocchio as pin

    for tcp in system.tcps:
        transform = tcp.transform
        parent_name = transform.parent_frame
        child_name = transform.child_frame

        if pin_model.existFrame(child_name):
            continue

        if not pin_model.existFrame(parent_name):
            raise ValueError(
                f"TCP parent frame {parent_name!r} not found in composed model"
            )

        parent_frame_id = pin_model.getFrameId(parent_name)
        parent_frame = pin_model.frames[parent_frame_id]
        parent_joint_id = parent_frame.parentJoint
        placement = parent_frame.placement * _se3_from_matrix(transform.as_matrix())

        pin_model.addFrame(
            pin.Frame(
                child_name,
                parent_joint_id,
                parent_frame_id,
                placement,
                pin.FrameType.OP_FRAME,
            )
        )


# ---------------------------------------------------------------------------
# Pinocchio helpers
# ---------------------------------------------------------------------------


def _se3_from_matrix(matrix: np.ndarray):
    """Convert a 4x4 NumPy matrix to a `pin.SE3`."""
    import pinocchio as pin

    se3 = pin.SE3.Identity()
    se3.rotation = np.asarray(matrix[:3, :3], dtype=float)
    se3.translation = np.asarray(matrix[:3, 3], dtype=float)
    return se3


def _require_existing(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"URDF not found: {path}")
    return path
