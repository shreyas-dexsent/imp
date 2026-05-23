"""
Cuboid 6D Pose Vision Module
Fuses template matching with depth-based pose estimation for cuboid objects.
"""

import base64
import json
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
from vision_engine.core.module_base import VisionModule
from vision_engine.io.data_plane.frame_bundle import FrameBundle
from vision_engine.modules.cuboid_pose_6d.pose_estimator import CuboidPoseEstimator
from vision_engine.modules.cuboid_pose_6d.temporal_filter import TemporalPoseFilter


class CuboidPose6DModule(VisionModule):
    """
    Estimates and tracks 6D pose of cuboid objects.

    Inputs:
    - RGB and depth frames from camera
    - Template matching result (ROI, score, optional face hint)
    - Cuboid geometry (L, B, H in mm)

    Outputs:
    - 6D pose in camera frame (translation + quaternion rotation)
    - Confidence score
    - Quality metrics (inliers, residual, etc.)
    """

    def __init__(self, name: str, params: Dict[str, Any]):
        super().__init__(name, params)

        # Intrinsics
        intrinsics = params.get("intrinsics", {})
        self.fx = float(intrinsics.get("fx", 500.0))
        self.fy = float(intrinsics.get("fy", 500.0))
        self.cx = float(intrinsics.get("cx", 320.0))
        self.cy = float(intrinsics.get("cy", 240.0))

        # Depth filtering
        self.depth_min_m = float(params.get("depth_min_m", 0.05))
        self.depth_max_m = float(params.get("depth_max_m", 5.0))
        self.depth_downsample_stride = int(params.get("depth_downsample_stride", 1))
        self.depth_roi_margin_px = int(params.get("depth_roi_margin_px", 20))

        # Pose estimation
        self.ransac_iterations = int(params.get("ransac_iterations", 100))
        self.ransac_thresh_m = float(params.get("ransac_thresh_m", 0.01))
        self.min_inliers = int(params.get("min_inliers", 50))

        # Temporal filtering
        self.alpha_trans = float(params.get("alpha_trans", 0.3))
        self.alpha_rot = float(params.get("alpha_rot", 0.3))
        self.max_jump_m = float(params.get("max_jump_m", 0.05))
        self.max_rot_deg = float(params.get("max_rot_deg", 15.0))
        self.lost_grace_s = float(params.get("lost_grace_s", 0.8))

        # Template integration
        self.template_score_threshold = float(
            params.get("template_score_threshold", 0.6)
        )
        self.require_template_match = bool(params.get("require_template_match", True))

        # Image output
        self.include_image = bool(params.get("include_image", False))
        self.image_format = str(params.get("image_format", "jpg")).lower().strip(".")
        if self.image_format not in ("jpg", "jpeg", "png"):
            self.image_format = "jpg"
        self.image_quality = int(params.get("image_quality", 80))
        self.image_fps_limit = float(params.get("image_fps_limit", 0.0))
        if self.image_fps_limit < 0:
            self.image_fps_limit = 0.0
        self._image_min_interval_s = (
            1.0 / self.image_fps_limit if self.image_fps_limit > 0 else 0.0
        )
        self._last_image_ts = 0.0

        # Cuboid geometry - can be loaded from process config or passed per-frame
        self.cuboid_dims = params.get("cuboid_dims", {"L": 100.0, "B": 80.0, "H": 60.0})

        # Debug and logging
        self.enable_debug_dumps = bool(params.get("enable_debug_dumps", False))
        self.debug_dump_dir = Path(params.get("debug_dump_dir", "/tmp/cuboid_debug"))
        self.debug_frame_interval = int(params.get("debug_frame_interval", 30))

        # State tracking per object
        self.object_filters: Dict[str, TemporalPoseFilter] = {}
        self.object_geometry: Dict[str, Dict[str, float]] = {}

        # Estimator
        self.pose_estimator = CuboidPoseEstimator(
            self.fx,
            self.fy,
            self.cx,
            self.cy,
            depth_min_m=self.depth_min_m,
            depth_max_m=self.depth_max_m,
            ransac_iterations=self.ransac_iterations,
            ransac_thresh_m=self.ransac_thresh_m,
            min_inliers=self.min_inliers,
        )

        # Frame counter for debug dumps
        self.frame_count = 0

        print(
            f"[{name}] Initialized with intrinsics: fx={self.fx}, fy={self.fy}, cx={self.cx}, cy={self.cy}"
        )
        print(
            f"[{name}] Temporal filter: alpha_trans={self.alpha_trans}, alpha_rot={self.alpha_rot}"
        )

    def run(self, frame_bundle: FrameBundle) -> Dict[str, Any]:
        """
        Process a frame bundle and estimate cuboid 6D poses.

        Expected frame_bundle content:
        - rgb: (H, W, 3) uint8 image
        - depth: (H, W) float32 in meters
        - intrinsics: dict with fx, fy, cx, cy
        - template_results: list of template match results (optional)
        - cuboid_objects: list of object configs (optional)
        """
        self.frame_count += 1
        frame_time = time.time()

        rgb = frame_bundle.rgb
        depth = frame_bundle.depth
        intrinsics = frame_bundle.meta.get("intrinsics", {})
        template_results = frame_bundle.meta.get("template_results", [])
        cuboid_objects = frame_bundle.meta.get("cuboid_objects", [])

        # Validate inputs
        if rgb is None or depth is None:
            return self._error_result("Missing RGB or depth")

        if depth.dtype != np.float32:
            depth = depth.astype(np.float32)

        # If no specific objects defined, use default
        if not cuboid_objects:
            cuboid_objects = [{"object_id": "default", "geometry": self.cuboid_dims}]

        results = []

        for obj_cfg in cuboid_objects:
            obj_id = obj_cfg.get("object_id", "unknown")
            geometry = obj_cfg.get("geometry", self.cuboid_dims)

            # Find template result for this object (if available)
            template_result = self._find_template_result(obj_id, template_results)

            # Process object
            obj_result = self._process_object(
                rgb, depth, obj_id, geometry, template_result, frame_time
            )
            results.append(obj_result)

        # Optional debug dump
        if (
            self.enable_debug_dumps
            and self.frame_count % self.debug_frame_interval == 0
        ):
            self._dump_debug_info(results, depth)

        result_dict = {
            "cuboid_poses": results,
            "frame_index": self.frame_count,
            "timestamp": frame_time,
        }

        # Optional image encoding
        if self.include_image:
            now = time.monotonic()
            if (
                self._image_min_interval_s <= 0
                or (now - self._last_image_ts) >= self._image_min_interval_s
            ):
                encoded = self._encode_image(rgb)
                if encoded.get("ok"):
                    result_dict.update(
                        {
                            "format": encoded["format"],
                            "width": encoded["width"],
                            "height": encoded["height"],
                            "image_b64": encoded["image_b64"],
                        }
                    )
                    self._last_image_ts = now

        return result_dict

    def _process_object(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        obj_id: str,
        geometry: Dict[str, float],
        template_result: Optional[Dict[str, Any]],
        frame_time: float,
    ) -> Dict[str, Any]:
        """Process a single object."""

        # Get or create temporal filter
        if obj_id not in self.object_filters:
            self.object_filters[obj_id] = TemporalPoseFilter(
                alpha_trans=self.alpha_trans,
                alpha_rot=self.alpha_rot,
                max_jump_m=self.max_jump_m,
                max_rot_deg=self.max_rot_deg,
                lost_grace_s=self.lost_grace_s,
            )

        filter_obj = self.object_filters[obj_id]

        # Initialize with template-based results if available
        pose_raw = None
        quality_raw = None
        template_score = 0.0

        if template_result is not None:
            template_score = template_result.get("score", 0.0)

            # Check if template score is sufficient
            if template_score > self.template_score_threshold:
                # Extract ROI from depth
                roi = template_result.get("roi")  # [x, y, w, h]
                if roi is not None:
                    # Get depth ROI
                    depth_roi = self._extract_depth_roi(depth, roi)

                    if depth_roi is not None and depth_roi.size > 0:
                        # Estimate pose from depth
                        est_result = self.pose_estimator.estimate_pose(
                            depth_roi, roi, geometry
                        )
                        pose_raw = est_result.get("pose_cam")
                        quality_raw = est_result.get("quality", {})
            elif self.require_template_match:
                # Template score too low, skip this frame
                pose_raw = None

        # Update temporal filter
        pose_filtered, state, gate_reason = filter_obj.update(
            pose_raw, template_score if pose_raw else 0.0, frame_time
        )

        # Build result
        result = {
            "object_id": obj_id,
            "frame_ts": frame_time,
            "state": state,
            "pose_cam": pose_filtered,
            "pose_cam_raw": pose_raw,
            "confidence": template_score,
            "quality": quality_raw or {},
            "debug": {
                "gate_reason": gate_reason,
                "roi": template_result.get("roi") if template_result else None,
            },
        }

        return result

    def _extract_depth_roi(
        self, depth: np.ndarray, roi: Tuple[int, int, int, int]
    ) -> Optional[np.ndarray]:
        """
        Extract and preprocess depth ROI.

        Args:
            depth: (H, W) float32 depth map
            roi: (x, y, w, h) bounding box

        Returns:
            Extracted depth ROI or None if invalid
        """
        if roi is None:
            return None

        x, y, w, h = roi

        # Add margin
        margin = self.depth_roi_margin_px
        x1 = max(0, x - margin)
        y1 = max(0, y - margin)
        x2 = min(depth.shape[1], x + w + margin)
        y2 = min(depth.shape[0], y + h + margin)

        depth_roi = depth[y1:y2, x1:x2].copy()

        if depth_roi.size == 0:
            return None

        # Downsample if requested
        if self.depth_downsample_stride > 1:
            depth_roi = depth_roi[
                :: self.depth_downsample_stride, :: self.depth_downsample_stride
            ]

        return depth_roi

    def _encode_image(self, image: np.ndarray) -> Dict[str, Any]:
        """Encode image to base64 in requested format."""
        params = []
        ext = ".jpg"
        fmt = self.image_format
        if fmt == "png":
            ext = ".png"
            params = [int(cv2.IMWRITE_PNG_COMPRESSION), 3]
        else:
            ext = ".jpg"
            quality = max(20, min(self.image_quality, 95))
            params = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        ok, buf = cv2.imencode(ext, image, params)
        if not ok:
            return {"ok": False}
        payload = base64.b64encode(buf).decode("ascii")
        h, w = image.shape[:2]
        return {
            "ok": True,
            "image_b64": payload,
            "format": fmt,
            "width": w,
            "height": h,
        }

    def _find_template_result(
        self, obj_id: str, template_results: list
    ) -> Optional[Dict[str, Any]]:
        """Find template matching result for an object ID."""
        for result in template_results:
            if result.get("object_id") == obj_id:
                return result
        return None if self.require_template_match else {}

    def _dump_debug_info(self, results: list, depth: np.ndarray):
        """Dump debug information to files."""
        if not self.enable_debug_dumps:
            return

        self.debug_dump_dir.mkdir(parents=True, exist_ok=True)

        # Dump depth image
        depth_path = self.debug_dump_dir / f"depth_{self.frame_count:06d}.npy"
        np.save(depth_path, depth)

        # Dump results
        results_path = self.debug_dump_dir / f"results_{self.frame_count:06d}.json"
        results_serializable = []
        for r in results:
            r_copy = r.copy()
            r_copy["pose_cam"] = r_copy["pose_cam"] or {}
            results_serializable.append(r_copy)

        with open(results_path, "w") as f:
            json.dump(results_serializable, f, indent=2)

    def _error_result(self, message: str) -> Dict[str, Any]:
        """Return an error result."""
        return {"error": message, "cuboid_poses": [], "timestamp": time.time()}

    def stop(self) -> None:
        """Cleanup on shutdown."""
        self.object_filters.clear()
        print(f"[{self.name}] Stopped")
