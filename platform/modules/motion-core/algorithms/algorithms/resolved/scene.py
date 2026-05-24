# Copyright (c) 2026 DexSent Robotics Pvt Ltd.
# All Rights Reserved.
#
# SPDX-License-Identifier: LicenseRef-Proprietary
# See LICENSE file for details.
#
# Author: Shreyas <shreyas.kumar@dexsentrobotics.com>
# Proprietary and confidential.

"""Mutable runtime state container.

A `Scene` is the runtime sibling of `WorldDescription`. The description
is immutable YAML-loaded data; the scene carries everything that changes
at runtime:

* live object poses (perception updates),
* live robot configurations (the current `q` per robot),
* attached objects (e.g. workpieces rigidly grasped by an end-effector),
* a dynamic Allowed-Collision-Matrix overlay (task-driven allowances).

Operations (FK, IK, collision queries, planning) accept a `Scene` and
never accept `WorldDescription` directly.

This class is a plain in-process state container. It does not subscribe
to ROS topics, time-stamp updates, lock across threads, or handle network
failures - those concerns belong to application middleware that writes
into the scene. Keeping the engine transport-agnostic is what lets the
same code run on ROS2, libfranka direct, or any other transport.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, FrozenSet, Optional, Set, Tuple

import numpy as np

from algorithms.descriptions import (
    CollisionGeometrySpec,
    GeometrySpec,
    VisualSpec,
    WorldDescription,
)
from algorithms.resolved.collision_model import CollisionModel, _canonical_pair


# ---------------------------------------------------------------------------
# Attached objects
# ---------------------------------------------------------------------------


@dataclass
class AttachedObject:
    """Runtime state of an object rigidly attached to a robot link.

    Attributes
    ----------
    object_id : str
        Geometry name matching the object's id in the world catalogue.
    parent_frame : str
        Frame the object is attached to (e.g. `"fr3_hand_tcp"`,
        namespaced if needed). Live FK reads this frame to compute the
        object's world pose at query time.
    T_parent_obj : np.ndarray, shape (4, 4)
        Fixed transform from `parent_frame` to the object's local frame.
    robot_id : str or None
        Which world robot owns the parent frame. Used by the multi-robot
        collision pipeline to pick the right `KinematicModel` for FK on
        `parent_frame`. `None` is permitted only for single-robot worlds
        and is resolved to the sole robot at query time.
    """

    object_id: str
    parent_frame: str
    T_parent_obj: np.ndarray
    robot_id: Optional[str] = None


# ---------------------------------------------------------------------------
# Dynamic ACM overlay
# ---------------------------------------------------------------------------


@dataclass
class CollisionOverlay:
    """Runtime additions and removals on top of the static ACM.

    Two sets:

    * `allowed` - pairs that should be skipped at runtime in addition
      to the static allow list (task-driven allowances; e.g. EE <-> workpiece
      during a grasp; populated automatically by :meth:`Scene.attach`).
    * `disallowed` - pairs that should be checked at runtime even if
      the static ACM said allow. Rare; used to force a normally-allowed
      pair back on during a specific phase.
    """

    allowed: Set[Tuple[str, str]] = field(default_factory=set)
    disallowed: Set[Tuple[str, str]] = field(default_factory=set)
    version: int = 0

    def bump(self) -> None:
        """Increment the overlay version after any effective mutation."""
        self.version += 1

    def allow(self, a: str, b: str) -> None:
        """Mark a pair as allowed-to-touch dynamically."""
        pair = _canonical_pair(a, b)
        before = (pair in self.allowed, pair in self.disallowed)
        self.allowed.add(pair)
        self.disallowed.discard(pair)
        after = (pair in self.allowed, pair in self.disallowed)
        if before != after:
            self.bump()

    def disallow(self, a: str, b: str) -> None:
        """Force a pair back to `check` even if statically allowed."""
        pair = _canonical_pair(a, b)
        before = (pair in self.allowed, pair in self.disallowed)
        self.disallowed.add(pair)
        self.allowed.discard(pair)
        after = (pair in self.allowed, pair in self.disallowed)
        if before != after:
            self.bump()

    def clear(self, a: str, b: str) -> None:
        """Drop both runtime allow and disallow entries for a pair."""
        pair = _canonical_pair(a, b)
        before = (pair in self.allowed, pair in self.disallowed)
        self.allowed.discard(pair)
        self.disallowed.discard(pair)
        after = (pair in self.allowed, pair in self.disallowed)
        if before != after:
            self.bump()


# ---------------------------------------------------------------------------
# Scene
# ---------------------------------------------------------------------------


@dataclass
class Scene:
    """Mutable runtime state container for the world's live state.

    Construct via :meth:`from_world`; never mutate `world` directly.
    """

    world: WorldDescription
    collision_model: Optional[CollisionModel] = None

    object_poses: Dict[str, np.ndarray] = field(default_factory=dict)
    """Live 4x4 world-to-object transform per free-standing world-object id."""

    robot_states: Dict[str, np.ndarray] = field(default_factory=dict)
    """Live active-q vector per world-robot id (chain order is the caller's choice)."""

    attached: Dict[str, AttachedObject] = field(default_factory=dict)
    """Objects rigidly attached to a robot frame at runtime."""

    collision_overlay: CollisionOverlay = field(default_factory=CollisionOverlay)

    _visual_overrides: Dict[str, "VisualSpec"] = field(default_factory=dict)
    """Visual geometry for runtime-added perception objects. UI reads this
    via :meth:`get_visual_spec`. YAML-declared objects' visuals are read
    from the WorldDescription directly."""

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_world(
        cls,
        world: WorldDescription,
        collision_model: Optional[CollisionModel] = None,
    ) -> "Scene":
        """Initialise a Scene from a WorldDescription's defaults.

        Each world object's description-time pose is copied into the
        scene's mutable `object_poses` dict. Robot states start empty;
        callers populate them via :meth:`set_robot_state` as state
        information becomes available.
        """
        object_poses: Dict[str, np.ndarray] = {}
        for obj in world.objects:
            object_poses[obj.id] = obj.pose.as_matrix()
        return cls(
            world=world,
            collision_model=collision_model,
            object_poses=object_poses,
        )

    # ------------------------------------------------------------------
    # Pose / state mutators
    # ------------------------------------------------------------------

    def set_object_pose(self, object_id: str, T_world_obj: np.ndarray) -> None:
        """Update the world pose of a free-standing object.

        Raises
        ------
        ValueError
            If the pose is not 4x4, or the object is currently attached
            (attached objects have their pose computed by FK on the
            parent frame; use :meth:`detach` first to free them).
        """
        T = np.asarray(T_world_obj, dtype=float)
        if T.shape != (4, 4):
            raise ValueError(f"pose must be (4,4); got {T.shape}")
        if object_id in self.attached:
            raise ValueError(
                f"object {object_id!r} is currently attached; "
                "set_object_pose is invalid until detach()"
            )
        self.object_poses[object_id] = T

    def get_object_pose(self, object_id: str) -> np.ndarray:
        """Return the live world pose of a free-standing object."""
        if object_id not in self.object_poses:
            raise KeyError(f"object {object_id!r} not in scene")
        return self.object_poses[object_id]

    def set_robot_state(self, robot_id: str, q: np.ndarray) -> None:
        """Update the live joint configuration of a robot."""
        q = np.asarray(q, dtype=float)
        if q.ndim != 1:
            raise ValueError(f"q must be 1-D; got shape {q.shape}")
        self.robot_states[robot_id] = q

    def get_robot_state(self, robot_id: str) -> np.ndarray:
        """Return the live joint configuration of a robot."""
        if robot_id not in self.robot_states:
            raise KeyError(f"robot {robot_id!r} state not set")
        return self.robot_states[robot_id]

    # ------------------------------------------------------------------
    # Attach / detach lifecycle
    # ------------------------------------------------------------------

    def attach(
        self,
        object_id: str,
        parent_frame: str,
        T_parent_obj: np.ndarray,
        *,
        allow_collision_with: Optional[list[str]] = None,
    ) -> None:
        """Rigidly attach a world object to a robot frame.

        While attached, the object moves with `parent_frame`: collision
        queries transform the object's geometry by FK on the parent frame
        rather than reading from `object_poses`. The free-standing pose
        is removed from `object_poses` for the duration of the attach.

        Parameters
        ----------
        object_id : str
            Id of the object to attach.
        parent_frame : str
            Frame the object becomes rigidly attached to.
        T_parent_obj : np.ndarray
            4x4 transform from `parent_frame` to the object's local frame.
        allow_collision_with : list[str], optional
            Geometry names that the attached object should be allowed to
            touch (typically the end-effector links). Each is added to
            the dynamic ACM overlay; entries are revoked on detach.
        """
        T = np.asarray(T_parent_obj, dtype=float)
        if T.shape != (4, 4):
            raise ValueError(f"T_parent_obj must be (4,4); got {T.shape}")
        if object_id in self.attached:
            raise ValueError(f"object {object_id!r} is already attached")

        self.attached[object_id] = AttachedObject(
            object_id=object_id,
            parent_frame=parent_frame,
            T_parent_obj=T,
        )
        # While attached, the static world pose is no longer authoritative.
        self.object_poses.pop(object_id, None)

        for partner in allow_collision_with or []:
            self.collision_overlay.allow(object_id, partner)
        self.collision_overlay.bump()

    def detach(self, object_id: str, T_world_obj: np.ndarray) -> None:
        """Detach an object and re-place it in the world at the given pose.

        The caller supplies the world pose at detach time (typically
        computed by FK on the parent frame at the instant of release).
        Any dynamic ACM allowances added by :meth:`attach` for this
        object are revoked.

        Raises
        ------
        KeyError
            If the object is not currently attached.
        ValueError
            If `T_world_obj` is not 4x4.
        """
        if object_id not in self.attached:
            raise KeyError(f"object {object_id!r} is not attached")
        self.attached.pop(object_id)

        # Revoke any dynamic allowances referencing this object.
        before_allowed = set(self.collision_overlay.allowed)
        self.collision_overlay.allowed = {
            pair for pair in self.collision_overlay.allowed if object_id not in pair
        }
        if self.collision_overlay.allowed != before_allowed:
            self.collision_overlay.bump()

        T = np.asarray(T_world_obj, dtype=float)
        if T.shape != (4, 4):
            raise ValueError(f"T_world_obj must be (4,4); got {T.shape}")
        self.object_poses[object_id] = T
        self.collision_overlay.bump()

    # ------------------------------------------------------------------
    # ACM convenience
    # ------------------------------------------------------------------

    def allow_collision(self, a: str, b: str, *, reason: Optional[str] = None) -> None:
        """Dynamically allow a pair of geometry names to touch."""
        del reason  # accepted for self-documenting call sites; not persisted
        self.collision_overlay.allow(a, b)

    def disallow_collision(self, a: str, b: str) -> None:
        """Force a pair back to `check` even if the static ACM said allow."""
        self.collision_overlay.disallow(a, b)

    def is_pair_allowed(self, a: str, b: str) -> bool:
        """Effective ACM result combining static rules and runtime overlay.

        Returns `True` if collision checking for `(a, b)` should be
        skipped. Precedence (highest first): dynamic disallow, dynamic
        allow, static allow.
        """
        pair = _canonical_pair(a, b)
        if pair in self.collision_overlay.disallowed:
            return False
        if pair in self.collision_overlay.allowed:
            return True
        if self.collision_model is not None:
            return self.collision_model.is_statically_allowed(a, b)
        return False

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def known_object_ids(self) -> FrozenSet[str]:
        """Object ids the Scene tracks (free-standing or attached)."""
        return frozenset(set(self.object_poses) | set(self.attached))

    # ------------------------------------------------------------------
    # Runtime perception integration (Pattern A: pose-only updates after add)
    # ------------------------------------------------------------------

    def add_object(
        self,
        object_id: str,
        *,
        collision: Optional[GeometrySpec] = None,
        visual: Optional[GeometrySpec] = None,
        pose: Optional[np.ndarray] = None,
    ) -> None:
        """Inject a perception-supplied object into the runtime scene.

        The library treats this object exactly like a YAML-declared world
        object: its pose lives in `object_poses`, the collision shape is
        registered in `collision_model.world_geom`, and the UI can read
        it back via `collision_model.shapes_for(object_id)`. Subsequent
        pose updates use the standard :meth:`set_object_pose`.

        Geometry is built once at add time (Pattern A). For objects
        whose shape changes during execution, call :meth:`remove_object`
        followed by :meth:`add_object` again with the new geometry.

        Parameters
        ----------
        object_id
            Unique identity used by all subsequent lookups. Must not
            collide with any existing object id (YAML or runtime).
        collision
            Collision geometry from any of the supported GeometrySpec
            variants (Box / Sphere / Cylinder / Capsule / ConvexHull /
            Octree / HeightField / MeshGeometrySpec by path /
            MeshDataGeometrySpec in memory). When ``None``, no
            collision shape is registered and the object is purely
            visual.
        visual
            Visual geometry. Retained on the Scene for the UI; not used
            by the planner. May be ``None``.
        pose
            Initial world pose. Defaults to identity. Updated later via
            :meth:`set_object_pose`.
        """
        if object_id in self.object_poses or object_id in self.attached:
            raise ValueError(f"object {object_id!r} already exists in scene")

        T = (
            np.asarray(pose, dtype=float)
            if pose is not None
            else np.eye(4, dtype=float)
        )
        if T.shape != (4, 4):
            raise ValueError(f"pose must be (4,4); got {T.shape}")

        if collision is not None:
            if self.collision_model is None:
                raise ValueError(
                    "Scene has no collision_model; cannot add a collision-bearing "
                    "object. Build the scene with `Scene.from_world(world, "
                    "collision_model)` first."
                )
            spec = CollisionGeometrySpec(geometry=collision, enabled=True)
            self.collision_model.add_world_object(object_id, spec)

        # Retain the visual spec for UI introspection. Stored alongside
        # object_poses; the UI reads this when rendering perception inputs.
        if visual is not None:
            self._visual_overrides[object_id] = VisualSpec(geometry=visual, enabled=True)

        self.object_poses[object_id] = T

    def remove_object(self, object_id: str) -> None:
        """Remove a runtime-added perception object.

        Removes the collision-model entry (if any), the visual overlay
        (if any), and the live pose. Raises if the object id is not in
        the scene or was declared in YAML (YAML objects are not
        runtime-removable; they live as long as the WorldDescription).
        """
        if object_id not in self.object_poses:
            raise KeyError(f"object {object_id!r} not in scene")

        # YAML-declared objects are not removable at runtime; the world
        # description treats them as static physical facts.
        if any(obj.id == object_id for obj in self.world.objects):
            raise ValueError(
                f"object {object_id!r} is YAML-declared and cannot be removed at runtime"
            )

        if self.collision_model is not None and self.collision_model.has_object(object_id):
            self.collision_model.remove_world_object(object_id)
        self._visual_overrides.pop(object_id, None)
        self.object_poses.pop(object_id)

    def get_visual_spec(self, object_id: str) -> Optional[VisualSpec]:
        """Return the visual spec for an object, if one is set.

        Looks first at runtime-added objects (perception overlays),
        then falls back to the YAML-declared world object's visual.
        Returns ``None`` when neither is present.
        """
        if object_id in self._visual_overrides:
            return self._visual_overrides[object_id]
        for obj in self.world.objects:
            if obj.id == object_id:
                return obj.visual
        return None
