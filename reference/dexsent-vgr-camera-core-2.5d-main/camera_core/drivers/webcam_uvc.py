"""Implementation for `camera_core.drivers.webcam_uvc`."""

import time

import cv2
import numpy as np

from .base import CameraDriver


class WebcamUVCDriver(CameraDriver):
    """
    Generic USB / Laptop webcam driver using OpenCV.
    """

    def __init__(self, logger):
        self.log = logger
        self.cap = None
        self.device_index = 0

    def initialize(self, config: dict) -> None:
        self.device_index = int(config.get("device_index", 0))
        width = int(config.get("width", 640))
        height = int(config.get("height", 480))
        fps = int(config.get("fps", 30))

        self.cap = cv2.VideoCapture(self.device_index, cv2.CAP_ANY)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open webcam device {self.device_index}")

        # Best-effort settings
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)

        self.log.info(
            f"Webcam initialized device={self.device_index} "
            f"({width}x{height}@{fps}fps)"
        )

    def start(self) -> None:
        # OpenCV starts capturing immediately
        pass

    def stop(self) -> None:
        if self.cap:
            self.cap.release()

    def capture(self) -> dict:
        if not self.cap:
            raise RuntimeError("Webcam not initialized")

        ok, frame = self.cap.read()
        if not ok:
            raise RuntimeError("Failed to read frame from webcam")

        # OpenCV gives BGR uint8 already
        ts_ns = time.time_ns()

        return {
            "rgb": frame,
            "depth": None,
            "timestamp_ns": ts_ns,
        }
