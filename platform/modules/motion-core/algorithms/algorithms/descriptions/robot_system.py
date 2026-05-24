# Copyright (c) 2026 DexSent Robotics Pvt Ltd.
# All Rights Reserved.
#
# SPDX-License-Identifier: LicenseRef-Proprietary
# See LICENSE file for details.
#
# Author: Shreyas <shreyas.kumar@dexsentrobotics.com>
# Proprietary and confidential.

"""Pydantic schema for a robot system: one robot plus an optional gripper.

A `RobotSystemDescription` is the YAML-facing typed representation of a
manipulator system. It is Layer 1 of the architecture (see
`docs/architecture.md`) - parsing and validation only, no computation.

Contracts:

* URDF is the source of truth for joint position, velocity, and effort
  limits; YAML may override per joint.
* Acceleration and jerk are NOT carried by URDF. YAML must supply them
  for every joint used in a chain that will undergo trajectory generation.
  The check is enforced in the resolution layer at build time.
* Kinematic chains, named TCPs, and named joint states are system-scoped
  (top-level) because chains routinely span the robot and the gripper.
* URDF `<mimic>` joints are NOT declared in chain joint lists. The
  resolution layer expands them internally so user-facing q vectors
  remain in active-DOF order.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

import yaml
from pydantic import BaseModel, ConfigDict, Field

from algorithms.descriptions.transforms import TransformSpec


# ---------------------------------------------------------------------------
# Joint-level specifications
# ---------------------------------------------------------------------------


class JointLimitsSpec(BaseModel):
    """Per-joint kinematic limit overrides on top of URDF.

    All fields are optional. Position, velocity, and effort fall back to
    URDF defaults when omitted. Acceleration and jerk have no URDF source
    and must be supplied here for joints that participate in any chain
    used for trajectory generation; the resolution layer enforces this.
    """

    position: Optional[Tuple[float, float]] = None
    velocity: Optional[float] = None
    acceleration: Optional[float] = None
    jerk: Optional[float] = None
    effort: Optional[float] = None

    model_config = ConfigDict(extra="forbid")


class AllowedLinkPairSpec(BaseModel):
    """One within-system link pair allowed to be in contact by design.

    Use for static contacts that should not be flagged as collisions, for
    example a parallel-jaw fingertip resting against its palm at rest.

    Task-driven dynamic allowances (e.g. end-effector touching a workpiece
    during a grasp phase) belong to the runtime Scene, not here.
    """

    a: str
    b: str
    reason: Optional[str] = None

    model_config = ConfigDict(extra="forbid")


class CollisionSpec(BaseModel):
    """Static, within-system collision configuration for a robot or gripper.

    Parameters
    ----------
    enabled : bool
        Whether collision checking is active for this component.
    source : Literal["urdf"]
        Where collision geometry comes from. Only `"urdf"` is supported
        in v2; future sources (SDF, custom) can be added without schema
        breakage by widening this Literal.
    allowed_pairs : list[AllowedLinkPairSpec]
        Static link pairs to exclude from collision checks.
    disabled_links : list[str]
        Link names whose collision geometry should be omitted entirely.
    """

    enabled: bool = True
    source: Literal["urdf"] = "urdf"
    allowed_pairs: List[AllowedLinkPairSpec] = Field(default_factory=list)
    disabled_links: List[str] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid")


# ---------------------------------------------------------------------------
# System-level specifications (chains, TCPs, named states)
# ---------------------------------------------------------------------------


class TcpDescription(BaseModel):
    """A named Tool Center Point as a static transform off some parent frame.

    Multiple TCPs may exist per system (e.g. a fingertip TCP and a palm
    TCP). Algorithms accept TCPs by id, not by transform.
    """

    id: str
    transform: TransformSpec

    model_config = ConfigDict(extra="forbid")


class KinematicChainSpec(BaseModel):
    """An ordered group of degrees of freedom used by algorithms.

    A chain defines the joint ordering that operations like FK, IK, and
    planning see. `q` vectors for a chain are NumPy arrays indexed in
    the order given by `joints`.

    Notes
    -----
    * URDF `<mimic>` joints (e.g. `fr3_finger_joint2`) are NOT listed;
      the resolution layer expands them internally.
    * `id` has no default; every chain must be named explicitly.
    """

    id: str
    base_frame: str
    tip_frame: str
    tcp_frame: Optional[str] = None
    joints: List[str] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid")


class NamedJointStateSpec(BaseModel):
    """A named joint state stored as a `{joint_name: position}` dict.

    Replaces the legacy parallel-list (names + positions) form. The
    resolution layer converts a named state into a chain-ordered NumPy
    vector on demand.
    """

    joints: Dict[str, float]

    model_config = ConfigDict(extra="forbid")


# ---------------------------------------------------------------------------
# Robot and gripper
# ---------------------------------------------------------------------------


class RobotDescription(BaseModel):
    """Description of the manipulator portion of a robot system."""

    id: str
    urdf_path: str
    package_dirs: List[str] = Field(default_factory=list)
    base_frame: str = "base"
    joint_limits: Dict[str, JointLimitsSpec] = Field(default_factory=dict)
    collision: CollisionSpec = Field(default_factory=CollisionSpec)

    model_config = ConfigDict(extra="forbid")


class GripperDescription(BaseModel):
    """Description of an end-effector that mounts on the manipulator.

    The `mount` transform attaches the gripper's root frame to a frame
    on the robot (typically the flange).
    """

    id: str
    urdf_path: str
    package_dirs: List[str] = Field(default_factory=list)
    root_frame: str
    mount: TransformSpec
    joint_limits: Dict[str, JointLimitsSpec] = Field(default_factory=dict)
    collision: CollisionSpec = Field(default_factory=CollisionSpec)

    model_config = ConfigDict(extra="forbid")


# ---------------------------------------------------------------------------
# Top-level robot system
# ---------------------------------------------------------------------------


class RobotSystemDescription(BaseModel):
    """A robot plus an optional gripper, with system-level structural facts.

    The system-level fields (`tcps`, `kinematic_chains`,
    `named_joint_states`) deliberately sit above `robot` and
    `gripper` because they routinely span both: a chain may include
    arm joints and a gripper finger, and TCPs are typically defined on
    the gripper but used by the system as a whole.
    """

    schema_name: Literal["dexsent.algorithms.robot_system"] = Field(
        "dexsent.algorithms.robot_system", alias="schema"
    )
    version: Literal[2] = 2
    id: str
    name: str = "untitled robot system"

    robot: RobotDescription
    gripper: Optional[GripperDescription] = None

    tcps: List[TcpDescription] = Field(default_factory=list)
    kinematic_chains: List[KinematicChainSpec] = Field(default_factory=list)
    named_joint_states: Dict[str, NamedJointStateSpec] = Field(default_factory=dict)

    source_path: Optional[Path] = Field(default=None, exclude=True)

    model_config = ConfigDict(
        populate_by_name=True, extra="forbid", arbitrary_types_allowed=True
    )

    @classmethod
    def from_yaml(cls, path: str | Path) -> "RobotSystemDescription":
        """Parse and validate a robot-system YAML file.

        Parameters
        ----------
        path : str or Path
            Path to the YAML file.

        Returns
        -------
        RobotSystemDescription
            The parsed description with `source_path` populated for
            relative path resolution.
        """
        path = Path(path).resolve()
        with path.open("r", encoding="utf-8") as handle:
            description = cls.model_validate(yaml.safe_load(handle) or {})
        description.source_path = path
        return description

    def resolve_path(self, path: str) -> Path:
        """Resolve a relative path against the YAML's source directory.

        Absolute paths pass through unchanged. Relative paths are resolved
        against the directory containing the YAML file this description
        was loaded from.
        """
        raw = Path(path)
        if raw.is_absolute():
            return raw
        if self.source_path is None:
            return raw.resolve()
        return (self.source_path.parent / raw).resolve()

    def chain(self, chain_id: str) -> KinematicChainSpec:
        """Look up a kinematic chain by id.

        Raises
        ------
        KeyError
            If no chain with that id exists.
        """
        for chain in self.kinematic_chains:
            if chain.id == chain_id:
                return chain
        raise KeyError(f"kinematic chain not found: {chain_id}")

    def tcp(self, tcp_id: str) -> TcpDescription:
        """Look up a TCP by id.

        Raises
        ------
        KeyError
            If no TCP with that id exists.
        """
        for tcp in self.tcps:
            if tcp.id == tcp_id:
                return tcp
        raise KeyError(f"tcp not found: {tcp_id}")

    def named_joint_state(self, name: str = "home") -> Dict[str, float]:
        """Return a named joint state as a flat `{joint_name: value}` dict.

        Raises
        ------
        KeyError
            If no named state with that name exists.
        """
        if name not in self.named_joint_states:
            raise KeyError(f"named joint state not found: {name}")
        return dict(self.named_joint_states[name].joints)
