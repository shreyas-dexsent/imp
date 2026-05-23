"""
Cuboid 6D Pose Estimator
Implements depth-based 6D pose estimation for cuboid objects using plane fitting + PCA.
"""

from typing import Any, Dict, Optional, Tuple

import numpy as np
from scipy.spatial.transform import Rotation
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA


class CuboidPoseEstimator:
    """
    Estimates 6D pose of a cuboid object from depth data.

    Uses plane fitting + PCA to determine:
    - One principal axis from plane normal (height H)
    - Two in-plane axes from PCA (length L, breadth B)
    """

    def __init__(
        self,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        depth_min_m: float = 0.05,
        depth_max_m: float = 5.0,
        ransac_iterations: int = 100,
        ransac_thresh_m: float = 0.01,
        min_inliers: int = 50,
    ):
        """
        Args:
            fx, fy, cx, cy: Camera intrinsics
            depth_min_m: Minimum valid depth in meters
            depth_max_m: Maximum valid depth in meters
            ransac_iterations: Max RANSAC iterations for plane fitting
            ransac_thresh_m: RANSAC inlier distance threshold in meters
            min_inliers: Minimum inliers required for valid plane
        """
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.depth_min_m = depth_min_m
        self.depth_max_m = depth_max_m
        self.ransac_iterations = ransac_iterations
        self.ransac_thresh_m = ransac_thresh_m
        self.min_inliers = min_inliers

    def depth_to_points(
        self, depth_roi: np.ndarray, roi_bbox: Tuple[int, int, int, int]
    ) -> np.ndarray:
        """
        Convert depth ROI to 3D points in camera frame.

        Args:
            depth_roi: (H, W) depth image in meters
            roi_bbox: (x, y, w, h) bounding box in original image coordinates

        Returns:
            (N, 3) array of 3D points in camera frame [X, Y, Z]
        """
        h, w = depth_roi.shape
        x_roi, y_roi, roi_w, roi_h = roi_bbox

        # Create coordinate grids in ROI
        u, v = np.meshgrid(np.arange(w), np.arange(h))

        # Map back to original image coordinates
        u_img = u + x_roi
        v_img = v + y_roi

        # Backproject to 3D
        z = depth_roi
        x = (u_img - self.cx) * z / self.fx
        y = (v_img - self.cy) * z / self.fy

        # Stack into point cloud
        points = np.stack([x, y, z], axis=-1)  # (H, W, 3)
        points = points.reshape(-1, 3)  # (N, 3)

        # Filter invalid points
        valid = (z.flatten() >= self.depth_min_m) & (z.flatten() <= self.depth_max_m)
        valid &= ~np.isnan(depth_roi.flatten())
        valid &= ~np.isinf(depth_roi.flatten())

        return points[valid]

    def _fit_plane_ransac(
        self, points: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Fit a plane to point cloud using RANSAC.

        Returns:
            normal: (3,) plane normal (unit vector)
            centroid: (3,) point on plane
            inlier_ratio: fraction of inliers
        """
        if len(points) < 3:
            return np.array([0, 0, 1.0]), points.mean(axis=0), 0.0

        best_normal = None
        best_centroid = None
        best_inliers = 0

        for _ in range(self.ransac_iterations):
            # Sample 3 random points
            idx = np.random.choice(len(points), 3, replace=False)
            p1, p2, p3 = points[idx]

            # Compute normal
            v1 = p2 - p1
            v2 = p3 - p1
            normal = np.cross(v1, v2)
            norm = np.linalg.norm(normal)
            if norm < 1e-6:
                continue
            normal = normal / norm

            # Count inliers
            centroid = p1
            dist = np.abs(np.dot(points - centroid, normal))
            inliers = np.sum(dist < self.ransac_thresh_m)

            if inliers > best_inliers:
                best_inliers = inliers
                best_normal = normal
                best_centroid = centroid

        if best_normal is None:
            # Fallback: use PCA normal
            points_centered = points - points.mean(axis=0)
            pca = PCA(n_components=3)
            pca.fit(points_centered)
            best_normal = pca.components_[-1]  # smallest variance direction
            best_centroid = points.mean(axis=0)
            best_inliers = len(points)

        inlier_ratio = best_inliers / len(points)
        return best_normal, best_centroid, inlier_ratio

    def _get_inplane_axes(
        self, points: np.ndarray, plane_normal: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get two orthogonal axes on the plane using PCA.

        Returns:
            u, v: (3,) unit vectors on plane, sorted by variance
        """
        # Project points onto plane
        points_centered = points - points.mean(axis=0)

        # Get projection matrix for plane
        n_outer = np.outer(plane_normal, plane_normal)
        proj_mat = np.eye(3) - n_outer

        points_projected = points_centered @ proj_mat.T

        # PCA on projected points
        pca = PCA(n_components=2)
        pca.fit(points_projected)

        # Get two principal directions
        u = pca.components_[0]  # largest variance
        v = pca.components_[1]  # second largest

        # Ensure orthonormality
        u = u / np.linalg.norm(u)
        v = v / np.linalg.norm(v)

        return u, v

    def estimate_pose(
        self,
        depth_roi: np.ndarray,
        roi_bbox: Tuple[int, int, int, int],
        cuboid_dims: Dict[str, float],
        previous_pose: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Estimate 6D pose of cuboid from depth ROI.

        Args:
            depth_roi: (H, W) depth in meters
            roi_bbox: (x, y, w, h) in original image
            cuboid_dims: {"L": mm, "B": mm, "H": mm}
            previous_pose: Optional previous pose for axis disambiguation

        Returns:
            result dict with:
                - pose_cam: {"t_m": [x,y,z], "q_xyzw": [x,y,z,w]}
                - confidence: float 0-1
                - quality: dict with diagnostic info
                - debug: dict with intermediate data
        """
        # Convert depth to point cloud
        points = self.depth_to_points(depth_roi, roi_bbox)

        if len(points) < self.min_inliers:
            return {
                "pose_cam": None,
                "confidence": 0.0,
                "quality": {"error": "insufficient_points", "point_count": len(points)},
                "debug": {},
            }

        # Fit dominant plane
        plane_normal, plane_centroid, plane_inlier_ratio = self._fit_plane_ransac(
            points
        )

        if plane_inlier_ratio < 0.1:
            return {
                "pose_cam": None,
                "confidence": 0.0,
                "quality": {
                    "error": "poor_plane_fit",
                    "inlier_ratio": plane_inlier_ratio,
                },
                "debug": {},
            }

        # Get in-plane axes
        u, v = self._get_inplane_axes(points, plane_normal)

        # Disambiguate axes: which is L vs B?
        # Use previous pose hint if available, else use extent
        L_mm = cuboid_dims["L"]
        B_mm = cuboid_dims["B"]
        H_mm = cuboid_dims["H"]

        # Estimate extents
        extents = self._estimate_extents(points, u, v, plane_normal)
        extent_u, extent_v, extent_h = extents

        # Choose L vs B based on extents (L > B assumed, or use previous)
        if extent_u > extent_v:
            axis_l = u
            axis_b = v
        else:
            axis_l = v
            axis_b = u

        # Flip axes to face camera direction (sign convention)
        axis_h = plane_normal
        if np.dot(axis_h, np.array([0, 0, 1])) < 0:
            axis_h = -axis_h

        # Ensure right-handed frame
        axis_l_normalized = axis_l / np.linalg.norm(axis_l)
        axis_b_normalized = axis_b / np.linalg.norm(axis_b)
        axis_h_normalized = axis_h / np.linalg.norm(axis_h)

        # Recompute axis_b to ensure orthogonality
        axis_b_normalized = np.cross(axis_h_normalized, axis_l_normalized)
        axis_b_normalized = axis_b_normalized / np.linalg.norm(axis_b_normalized)

        # Estimate center: plane centroid ± H/2 * normal
        H_m = H_mm / 1000.0
        center = plane_centroid - 0.5 * H_m * axis_h_normalized

        # Refine fit (optional but recommended)
        refined_center, refined_rot_mat = self._refine_box_fit(
            points,
            center,
            axis_l_normalized,
            axis_b_normalized,
            axis_h_normalized,
            L_mm / 1000.0,
            B_mm / 1000.0,
            H_mm / 1000.0,
        )

        # Convert rotation matrix to quaternion
        rot = Rotation.from_matrix(refined_rot_mat)
        q_xyzw = rot.as_quat()  # xyzw order

        # Compute confidence and residuals
        residual = self._compute_fit_residual(
            points,
            refined_center,
            refined_rot_mat,
            L_mm / 1000.0,
            B_mm / 1000.0,
            H_mm / 1000.0,
        )

        # Inlier-based confidence
        inlier_count = np.sum(residual < 0.02)  # 20mm threshold
        inlier_ratio = inlier_count / len(points)
        confidence = min(1.0, inlier_ratio * plane_inlier_ratio)

        return {
            "pose_cam": {"t_m": refined_center.tolist(), "q_xyzw": q_xyzw.tolist()},
            "confidence": float(confidence),
            "quality": {
                "template_score": 0.0,  # To be filled by module
                "depth_inliers": int(inlier_count),
                "depth_residual_m": float(np.mean(residual)),
                "plane_inlier_ratio": float(plane_inlier_ratio),
                "axis_flip_flag": False,
            },
            "debug": {
                "plane_normal": plane_normal.tolist(),
                "point_count": len(points),
                "estimated_extents_m": extents,
            },
        }

    def _estimate_extents(
        self, points: np.ndarray, u: np.ndarray, v: np.ndarray, normal: np.ndarray
    ) -> Tuple[float, float, float]:
        """Estimate dimensions of point cloud along u, v, normal directions."""
        proj_u = np.abs(np.dot(points, u))
        proj_v = np.abs(np.dot(points, v))
        proj_n = np.abs(np.dot(points, normal))

        extent_u = np.percentile(proj_u, 95) - np.percentile(proj_u, 5)
        extent_v = np.percentile(proj_v, 95) - np.percentile(proj_v, 5)
        extent_n = np.percentile(proj_n, 95) - np.percentile(proj_n, 5)

        return extent_u, extent_v, extent_n

    def _refine_box_fit(
        self,
        points: np.ndarray,
        center: np.ndarray,
        axis_l: np.ndarray,
        axis_b: np.ndarray,
        axis_h: np.ndarray,
        L_m: float,
        B_m: float,
        H_m: float,
        max_iterations: int = 8,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Refine box position and orientation to minimize point-to-box distance.

        Returns:
            refined_center, refined_rot_mat
        """
        rot_mat = np.stack([axis_l, axis_b, axis_h], axis=0)  # 3x3 rotation

        best_center = center.copy()
        best_rot = rot_mat.copy()
        best_residual = np.inf

        for it in range(max_iterations):
            # Compute residual
            residual = self._compute_fit_residual(
                points, best_center, best_rot, L_m, B_m, H_m
            )
            mean_residual = np.mean(residual)

            if mean_residual >= best_residual:
                break  # No improvement

            best_residual = mean_residual

            # Refine translation by moving to inlier centroid
            inliers = residual < 0.03
            if np.sum(inliers) > 0:
                best_center = points[inliers].mean(axis=0)

        return best_center, best_rot

    def _compute_fit_residual(
        self,
        points: np.ndarray,
        center: np.ndarray,
        rot_mat: np.ndarray,
        L_m: float,
        B_m: float,
        H_m: float,
    ) -> np.ndarray:
        """
        Compute point-to-box distance for each point.

        Returns:
            (N,) array of distances in meters
        """
        # Transform points to box frame
        points_centered = points - center
        points_box = points_centered @ rot_mat.T

        # Box half-extents
        L_h, B_h, H_h = L_m / 2, B_m / 2, H_m / 2

        # Compute distance to box surface
        abs_x = np.abs(points_box[:, 0])
        abs_y = np.abs(points_box[:, 1])
        abs_z = np.abs(points_box[:, 2])

        # Clamp to box extents
        dx = np.maximum(abs_x - L_h, 0)
        dy = np.maximum(abs_y - B_h, 0)
        dz = np.maximum(abs_z - H_h, 0)

        residual = np.sqrt(dx**2 + dy**2 + dz**2)
        return residual
