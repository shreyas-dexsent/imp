"""Implementation for `camera_core.drivers.factory`."""


def _load_driver_class(driver_type: str):
    if driver_type == "webcam_uvc":
        from camera_core.drivers.webcam_uvc import WebcamUVCDriver

        return WebcamUVCDriver

    if driver_type in {"realsense_d435i", "realsense_d405"}:
        try:
            from camera_core.drivers.realsense_d435i import (
                RealSenseD405Driver,
                RealSenseD435iDriver,
            )
        except Exception as e:
            raise RuntimeError("RealSense driver requires pyrealsense2") from e
        if driver_type == "realsense_d405":
            return RealSenseD405Driver
        return RealSenseD435iDriver

    if driver_type == "basler_gige":
        try:
            from camera_core.drivers.basler_gige import BaslerGigEDriver
        except Exception as e:
            raise RuntimeError("Basler driver requires pypylon") from e
        return BaslerGigEDriver

    if driver_type == "flir_blackfly_gige":
        try:
            from camera_core.drivers.flir_blackfly_gige import FlirBlackflyGigEDriver
        except Exception as e:
            raise RuntimeError("FLIR driver requires vendor SDK") from e
        return FlirBlackflyGigEDriver

    return None


def create_camera_driver(driver_type: str, logger):
    driver_cls = _load_driver_class(driver_type)
    if driver_cls is None:
        raise ValueError(
            f"Unknown camera type '{driver_type}'. "
            "Available: ['webcam_uvc', 'realsense_d435i', 'realsense_d405', 'basler_gige', 'flir_blackfly_gige']"
        )
    return driver_cls(logger)
