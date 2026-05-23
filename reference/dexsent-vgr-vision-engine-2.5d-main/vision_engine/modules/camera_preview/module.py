"""Implementation for `vision_engine.modules.camera_preview.module`."""

import base64
from typing import Any, Dict

import cv2
from vision_engine.core.module_base import VisionModule
from vision_engine.io.data_plane.frame_bundle import FrameBundle


class CameraPreviewModule(VisionModule):
    def __init__(self, name: str, params: Dict[str, Any]):
        super().__init__(name, params)
        self.max_width = int(params.get("max_width", 640))
        self.max_height = int(params.get("max_height", 480))
        self.quality = int(params.get("quality", 75))
        fmt = str(params.get("format", "jpg")).lower().strip(".")
        if fmt not in ("jpg", "jpeg", "png"):
            fmt = "jpg"
        self.format = "jpg" if fmt == "jpeg" else fmt

    def _resize(self, frame):
        height, width = frame.shape[:2]
        if self.max_width <= 0 and self.max_height <= 0:
            return frame
        target_w = self.max_width if self.max_width > 0 else width
        target_h = self.max_height if self.max_height > 0 else height
        scale = min(1.0, target_w / width, target_h / height)
        if scale >= 1.0:
            return frame
        new_w = max(1, int(width * scale))
        new_h = max(1, int(height * scale))
        return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

    def run(self, frame_bundle: FrameBundle) -> Dict[str, Any]:
        frame = frame_bundle.rgb
        if frame is None:
            return {"valid": False, "reason": "missing_rgb"}

        resized = self._resize(frame)
        encode_params = []
        ext = ".jpg"
        if self.format == "png":
            ext = ".png"
            encode_params = [int(cv2.IMWRITE_PNG_COMPRESSION), 3]
        else:
            ext = ".jpg"
            quality = max(10, min(self.quality, 95))
            encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), quality]

        ok, buf = cv2.imencode(ext, resized, encode_params)
        if not ok:
            return {"valid": False, "reason": "encode_failed"}

        payload = base64.b64encode(buf).decode("ascii")
        return {
            "valid": True,
            "format": self.format,
            "width": int(resized.shape[1]),
            "height": int(resized.shape[0]),
            "image_b64": payload,
        }
