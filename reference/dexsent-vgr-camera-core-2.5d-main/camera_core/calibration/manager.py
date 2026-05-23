"""Implementation for `camera_core.calibration.manager`."""


class CalibrationManager:
    def __init__(self):
        self._version = 1

    def load(self):
        # load intrinsics/extrinsics/handeye from disk later
        self._version = 1

    def version(self) -> int:
        return self._version

    def apply(self, frame: dict) -> dict:
        # apply rectification/undistort etc later
        return frame

    def bump_version(self):
        self._version += 1
