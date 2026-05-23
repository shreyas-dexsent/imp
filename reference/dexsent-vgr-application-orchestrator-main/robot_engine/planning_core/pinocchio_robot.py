from __future__ import annotations

from typing import List, Optional, Set, Tuple

import numpy as np


class PinocchioRobot:
    """
    Robot model backed by Pinocchio + Coal.

    Responsibilities
    ----------------
    - Load URDF (kinematic + collision geometry model).
    - Forward kinematics and frame placement.
    - Joint limit validation.
    - Self-collision checking via Pinocchio + Coal collision pairs.
    - Expose per-link Coal geometries and world transforms so that
      PlanningCollisionWorld can do robot-vs-environment collision.

    Parameters
    ----------
    urdf_path : str
    package_dirs : list[str], optional
        Mesh search paths passed to Pinocchio.
    allowed_collision_pairs : list[(str, str)], optional
        Link name pairs that are always allowed to collide (adjacent links etc.).
    group_joint_names : list[str], optional
        If given, FK / IK / limits only consider these joints (group sub-set).
    """

    def __init__(
        self,
        urdf_path: str,
        package_dirs: Optional[List[str]] = None,
        allowed_collision_pairs: Optional[List[Tuple[str, str]]] = None,
        group_joint_names: Optional[List[str]] = None,
        urdf_xml: Optional[str] = None,
    ) -> None:
        import pinocchio as pin

        self._urdf_path = urdf_path
        self._package_dirs = package_dirs or []

        # Kinematic model — prefer in-memory XML (merged URDF) when available
        if urdf_xml:
            self.model: pin.Model = pin.buildModelFromXML(urdf_xml)
        else:
            self.model: pin.Model = pin.buildModelFromUrdf(urdf_path)
        self.data: pin.Data = self.model.createData()

        # Collision geometry model (Coal BVH per link)
        try:
            if urdf_xml:
                self.geom_model: pin.GeometryModel = pin.buildGeomFromUrdfString(
                    self.model,
                    urdf_xml,
                    pin.GeometryType.COLLISION,
                    package_dirs=self._package_dirs,
                )
            else:
                self.geom_model: pin.GeometryModel = pin.buildGeomFromUrdf(
                    self.model,
                    urdf_path,
                    pin.GeometryType.COLLISION,
                    package_dirs=self._package_dirs,
                )
        except Exception:
            self.geom_model = pin.GeometryModel()
        self.geom_data: pin.GeometryData = self.geom_model.createData()

        # Build the collision pair set (all pairs minus the allowed ones)
        self._allowed: Set[Tuple[str, str]] = set()
        for a, b in (allowed_collision_pairs or []):
            self._allowed.add((a, b))
            self._allowed.add((b, a))
        self._setup_collision_pairs()

        # Group joint indices (subset of model joints, if provided)
        self._group_joint_names: List[str] = group_joint_names or []
        self._group_joint_indices: List[int] = []
        if self._group_joint_names:
            all_names = list(self.model.names[1:])  # skip 'universe'
            for name in self._group_joint_names:
                if name in all_names:
                    self._group_joint_indices.append(all_names.index(name))

        # Full neutral configuration
        self._q_neutral = pin.neutral(self.model).copy()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _setup_collision_pairs(self) -> None:
        import pinocchio as pin

        n = len(self.geom_model.geometryObjects)
        for i in range(n):
            for j in range(i + 1, n):
                gi = self.geom_model.geometryObjects[i]
                gj = self.geom_model.geometryObjects[j]
                # Skip pairs between the same parent joint
                if gi.parentJoint == gj.parentJoint:
                    continue
                # Skip allowed pairs (by geometry name prefix matching link name)
                li = _link_name(gi.name)
                lj = _link_name(gj.name)
                if (li, lj) in self._allowed:
                    continue
                self.geom_model.addCollisionPair(pin.CollisionPair(i, j))
        self.geom_data = self.geom_model.createData()

    # ------------------------------------------------------------------
    # Joint interface
    # ------------------------------------------------------------------

    @property
    def nq(self) -> int:
        return self.model.nq

    @property
    def joint_names(self) -> List[str]:
        return list(self.model.names[1:])

    @property
    def lower_limits(self) -> np.ndarray:
        return np.asarray(self.model.lowerPositionLimit, dtype=float)

    @property
    def upper_limits(self) -> np.ndarray:
        return np.asarray(self.model.upperPositionLimit, dtype=float)

    def neutral_q(self) -> np.ndarray:
        return self._q_neutral.copy()

    def clip_q(self, q: np.ndarray) -> np.ndarray:
        return np.clip(np.asarray(q, dtype=float), self.lower_limits, self.upper_limits)

    def within_limits(self, q: np.ndarray, tol: float = 1e-6) -> bool:
        q = np.asarray(q, dtype=float)
        return bool(
            np.all(q >= self.lower_limits - tol)
            and np.all(q <= self.upper_limits + tol)
        )

    # ------------------------------------------------------------------
    # Group helpers (map group q vector ↔ full q vector)
    # ------------------------------------------------------------------

    def group_q_to_full(self, q_group: np.ndarray) -> np.ndarray:
        """Map a group-sized joint vector to the full model joint vector."""
        if not self._group_joint_indices:
            return np.asarray(q_group, dtype=float)
        q_full = self._q_neutral.copy()
        for group_idx, full_idx in enumerate(self._group_joint_indices):
            q_full[full_idx] = q_group[group_idx]
        return q_full

    def full_q_to_group(self, q_full: np.ndarray) -> np.ndarray:
        if not self._group_joint_indices:
            return np.asarray(q_full, dtype=float)
        return np.asarray(q_full, dtype=float)[self._group_joint_indices]

    def within_group_limits(self, q_group: np.ndarray, tol: float = 1e-6) -> bool:
        if not self._group_joint_indices:
            return self.within_limits(q_group, tol)
        ll = self.lower_limits[self._group_joint_indices]
        ul = self.upper_limits[self._group_joint_indices]
        q = np.asarray(q_group, dtype=float)
        return bool(np.all(q >= ll - tol) and np.all(q <= ul + tol))

    @property
    def group_lower_limits(self) -> np.ndarray:
        if not self._group_joint_indices:
            return self.lower_limits
        return self.lower_limits[self._group_joint_indices]

    @property
    def group_upper_limits(self) -> np.ndarray:
        if not self._group_joint_indices:
            return self.upper_limits
        return self.upper_limits[self._group_joint_indices]

    # ------------------------------------------------------------------
    # Kinematics
    # ------------------------------------------------------------------

    def update_kinematics(self, q: np.ndarray) -> None:
        """Run FK and update frame placements."""
        import pinocchio as pin

        q = np.asarray(q, dtype=float)
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)

    def update_geometry(self, q: np.ndarray) -> None:
        """Update collision geometry placements (calls forwardKinematics internally)."""
        import pinocchio as pin

        q = np.asarray(q, dtype=float)
        pin.updateGeometryPlacements(self.model, self.data, self.geom_model, self.geom_data, q)

    def get_frame_placement(self, frame_name: str):
        """Return pin.SE3 of a named frame after update_kinematics."""
        import pinocchio as pin

        fid = self.model.getFrameId(frame_name)
        if fid >= len(self.model.frames):
            raise ValueError(f"Frame '{frame_name}' not found in model.")
        return self.data.oMf[fid]

    def get_tcp_placement(self, tip_link: str):
        return self.get_frame_placement(tip_link)

    # ------------------------------------------------------------------
    # Collision geometry accessors
    # ------------------------------------------------------------------

    def link_coal_geometries(self) -> List:
        """Return list of Coal geometry objects for all collision links."""
        return [go.geometry for go in self.geom_model.geometryObjects]

    def link_coal_transforms(self) -> List:
        """
        Return current Coal.Transform3s for each collision geometry.
        Call update_geometry() first.
        """
        from robot_engine.assets.mesh_converter import se3_to_coal_transform

        return [se3_to_coal_transform(self.geom_data.oMg[i])
                for i in range(len(self.geom_model.geometryObjects))]

    # ------------------------------------------------------------------
    # Self-collision
    # ------------------------------------------------------------------

    def in_self_collision(self, q: np.ndarray) -> bool:
        """
        Return True if the configuration q causes any self-collision.

        Uses Pinocchio's built-in collision pair checking with Coal backend.
        """
        import pinocchio as pin

        q = np.asarray(q, dtype=float)
        if len(self.geom_model.collisionPairs) == 0:
            return False
        return bool(pin.computeCollisions(
            self.model,
            self.data,
            self.geom_model,
            self.geom_data,
            q,
            True,  # stop_at_first_collision
        ))

    # ------------------------------------------------------------------
    # IK (Pinocchio DLS, group joints only)
    # ------------------------------------------------------------------

    def solve_ik(
        self,
        tip_link: str,
        T_world_target,
        q_seed: Optional[np.ndarray] = None,
        max_iters: int = 200,
        pos_tol: float = 1e-4,
        rot_tol: float = 1e-4,
        damping: float = 1e-6,
    ) -> Optional[np.ndarray]:
        """
        Pinocchio DLS IK.  Returns full-model q or None on failure.

        Parameters
        ----------
        T_world_target : pin.SE3 or (4,4) array
        """
        import pinocchio as pin

        if q_seed is None:
            q = self._q_neutral.copy()
        else:
            q = np.asarray(q_seed, dtype=float).copy()
            if len(q) == len(self._group_joint_indices) and self._group_joint_indices:
                q = self.group_q_to_full(q)

        # Target as SE3
        if hasattr(T_world_target, "rotation"):
            target = T_world_target
        else:
            M = np.asarray(T_world_target, dtype=float)
            target = pin.SE3(M[:3, :3], M[:3, 3])

        frame_id = self.model.getFrameId(tip_link)
        if frame_id >= len(self.model.frames):
            return None

        data = self.model.createData()
        for _ in range(max_iters):
            pin.forwardKinematics(self.model, data, q)
            pin.updateFramePlacements(self.model, data)
            current = data.oMf[frame_id]
            err = pin.log(current.inverse() * target).vector
            if np.linalg.norm(err[:3]) < pos_tol and np.linalg.norm(err[3:]) < rot_tol:
                return q
            J = pin.computeFrameJacobian(self.model, data, q, frame_id, pin.ReferenceFrame.LOCAL)
            lhs = J @ J.T + damping * np.eye(6)
            dq = J.T @ np.linalg.solve(lhs, err)
            q = pin.integrate(self.model, q, dq)
            q = np.clip(q, self.lower_limits, self.upper_limits)

        return None

    def solve_ik_multi_seed(
        self,
        tip_link: str,
        T_world_target,
        seeds: List[np.ndarray],
        **kwargs,
    ) -> List[np.ndarray]:
        """Try multiple seeds and return all converged solutions."""
        solutions = []
        for seed in seeds:
            q = self.solve_ik(tip_link, T_world_target, q_seed=seed, **kwargs)
            if q is not None:
                solutions.append(q)
        return solutions


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _link_name(geometry_name: str) -> str:
    """Heuristically extract link name from geometry object name (strip _0, _collision etc.)."""
    # Pinocchio names geom objects like "shoulder_link_0", "forearm_link_collision"
    parts = geometry_name.rsplit("_", 1)
    if len(parts) == 2 and (parts[1].isdigit() or parts[1] in ("collision", "visual")):
        return parts[0]
    return geometry_name
