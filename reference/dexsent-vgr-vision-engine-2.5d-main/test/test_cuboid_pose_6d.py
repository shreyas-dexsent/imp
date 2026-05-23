"""
Unit tests for cuboid 6D pose estimation.
"""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from vision_engine.modules.cuboid_pose_6d.pose_estimator import CuboidPoseEstimator
from vision_engine.modules.cuboid_pose_6d.temporal_filter import TemporalPoseFilter


class TestCuboidPoseEstimator:
    """Test pose estimation from synthetic cuboid point clouds."""

    @pytest.fixture
    def estimator(self):
        """Create estimator with standard intrinsics."""
        return CuboidPoseEstimator(
            fx=500.0,
            fy=500.0,
            cx=320.0,
            cy=240.0,
            depth_min_m=0.05,
            depth_max_m=5.0,
            ransac_iterations=100,
            ransac_thresh_m=0.01,
            min_inliers=50,
        )

    def _generate_cuboid_points(
        self, center, rotation_mat, L, B, H, noise_std=0.001, num_points=500
    ):
        """
        Generate synthetic point cloud for a cuboid.

        Args:
            center: (3,) center in camera frame
            rotation_mat: (3, 3) rotation matrix
            L, B, H: dimensions in meters
            noise_std: noise standard deviation
            num_points: number of points

        Returns:
            (N, 3) point cloud in camera frame
        """
        # Generate points on cuboid surfaces
        points_local = []

        # Top and bottom faces
        for z_sign in [-1, 1]:
            for _ in range(num_points // 4):
                x = np.random.uniform(-L / 2, L / 2)
                y = np.random.uniform(-B / 2, B / 2)
                z = z_sign * H / 2
                points_local.append([x, y, z])

        # Side faces
        for _ in range(num_points // 4):
            x = np.random.choice([-L / 2, L / 2])
            y = np.random.uniform(-B / 2, B / 2)
            z = np.random.uniform(-H / 2, H / 2)
            points_local.append([x, y, z])

        points_local = np.array(points_local)

        # Transform to camera frame
        points_camera = points_local @ rotation_mat.T + center

        # Add noise
        points_camera += np.random.normal(0, noise_std, points_camera.shape)

        return points_camera

    def test_plane_fitting(self, estimator):
        """Test plane fitting RANSAC."""
        # Create a simple plane
        points = np.random.randn(100, 3)
        points[:, 2] = 1.0  # z=1 plane

        normal, centroid, inlier_ratio = estimator._fit_plane_ransac(points)

        assert normal is not None
        assert np.allclose(np.abs(normal[2]), 1.0, atol=0.1)
        assert inlier_ratio > 0.8

    def test_depth_to_points(self, estimator):
        """Test depth to 3D point conversion."""
        depth_roi = np.ones((100, 100), dtype=np.float32) * 1.0  # 1 meter away
        roi_bbox = (100, 100, 100, 100)  # x, y, w, h

        points = estimator.depth_to_points(depth_roi, roi_bbox)

        assert points.shape[1] == 3
        assert len(points) == 100 * 100
        assert np.all(points[:, 2] >= 0.99)  # depth around 1.0m

    def test_pose_estimation_perfect_cuboid(self, estimator):
        """Test pose estimation on a perfect cuboid."""
        # Create a known cuboid
        center_true = np.array([0.0, 0.0, 2.0])
        rot_true = Rotation.from_euler("xyz", [10, 20, 30], degrees=True).as_matrix()

        L, B, H = 0.1, 0.08, 0.06  # 100x80x60 mm

        points = self._generate_cuboid_points(
            center_true, rot_true, L, B, H, noise_std=0.0005
        )

        # Project to image
        depth = np.zeros((480, 640), dtype=np.float32)
        roi = np.zeros((640, 480), dtype=np.uint16)

        for p in points:
            if p[2] > 0:
                u = int(estimator.fx * p[0] / p[2] + estimator.cx)
                v = int(estimator.fy * p[1] / p[2] + estimator.cy)
                if 0 <= u < 640 and 0 <= v < 480:
                    depth[v, u] = p[2]

        # Extract depth ROI
        x1, x2 = 150, 490
        y1, y2 = 100, 380
        depth_roi = depth[y1:y2, x1:x2]
        roi_bbox = (x1, y1, x2 - x1, y2 - y1)

        # Estimate pose
        result = estimator.estimate_pose(
            depth_roi, roi_bbox, {"L": L * 1000, "B": B * 1000, "H": H * 1000}
        )

        assert result["confidence"] > 0.3
        assert result["pose_cam"] is not None

        est_center = np.array(result["pose_cam"]["t_m"])
        est_error = np.linalg.norm(est_center - center_true)

        # Position should be close (within ~30mm)
        assert est_error < 0.03, f"Position error too large: {est_error}"


class TestTemporalFilter:
    """Test temporal filtering and gating."""

    @pytest.fixture
    def filter_obj(self):
        """Create temporal filter."""
        return TemporalPoseFilter(
            alpha_trans=0.3,
            alpha_rot=0.3,
            max_jump_m=0.05,
            max_rot_deg=15.0,
            lost_grace_s=0.8,
        )

    def test_initialization(self, filter_obj):
        """Test filter initialization."""
        pose = {"t_m": [0.0, 0.0, 1.0], "q_xyzw": [0.0, 0.0, 0.0, 1.0]}

        filtered, state, reason = filter_obj.update(pose, 0.8, 0.0)

        assert state == "TARGET"
        assert filtered is not None
        assert np.allclose(filtered["t_m"], [0.0, 0.0, 1.0])

    def test_smoothing(self, filter_obj):
        """Test pose smoothing."""
        pose1 = {"t_m": [0.0, 0.0, 1.0], "q_xyzw": [0.0, 0.0, 0.0, 1.0]}
        pose2 = {"t_m": [0.01, 0.0, 1.0], "q_xyzw": [0.0, 0.0, 0.0, 1.0]}

        filtered1, _, _ = filter_obj.update(pose1, 0.8, 0.0)
        filtered2, _, _ = filter_obj.update(pose2, 0.8, 0.033)  # 30ms later

        # Smoothed result should be between input 1 and 2
        x2 = filtered2["t_m"][0]
        assert 0.0 < x2 < 0.01

    def test_gating_high_jump(self, filter_obj):
        """Test gating of high jumps."""
        pose1 = {"t_m": [0.0, 0.0, 1.0], "q_xyzw": [0.0, 0.0, 0.0, 1.0]}
        pose_jump = {
            "t_m": [0.2, 0.0, 1.0],
            "q_xyzw": [0.0, 0.0, 0.0, 1.0],
        }  # 200mm jump

        filter_obj.update(pose1, 0.8, 0.0)
        filtered, state, reason = filter_obj.update(pose_jump, 0.8, 0.033)

        assert state == "LOST"
        assert "HIGH_JUMP" in reason or "HIGH_JUMP" in str(reason)

    def test_lost_grace_prediction(self, filter_obj):
        """Test prediction during grace period."""
        pose = {"t_m": [0.0, 0.0, 1.0], "q_xyzw": [0.0, 0.0, 0.0, 1.0]}

        # Initialize and move
        filter_obj.update(pose, 0.8, 0.0)

        # Move with velocity
        pose2 = {"t_m": [0.01, 0.0, 1.0], "q_xyzw": [0.0, 0.0, 0.0, 1.0]}
        filter_obj.update(pose2, 0.8, 0.033)

        # Lose target
        filtered, state, _ = filter_obj.update(None, 0.0, 0.066)

        assert state == "LOST"
        # Should still have a prediction
        if filtered is not None:
            assert "t_m" in filtered


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
