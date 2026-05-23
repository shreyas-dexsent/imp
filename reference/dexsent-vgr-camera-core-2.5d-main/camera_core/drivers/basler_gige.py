"""Implementation for `camera_core.drivers.basler_gige`."""

import time

import numpy as np
from pypylon import pylon

from .base import CameraDriver


class BaslerGigEDriver(CameraDriver):
    def __init__(self, logger):
        self.log = logger
        self.cam = None
        self.converter = pylon.ImageFormatConverter()

    def initialize(self, config: dict) -> None:
        serial = config.get("serial_number", "") or ""
        tl_factory = pylon.TlFactory.GetInstance()
        devices = tl_factory.EnumerateDevices()
        if not devices:
            raise RuntimeError("No Basler devices found.")

        dev = None
        if serial.strip():
            for d in devices:
                if d.GetSerialNumber() == serial.strip():
                    dev = d
                    break
            if dev is None:
                raise RuntimeError(
                    f"Basler serial {serial} not found. Available: {[d.GetSerialNumber() for d in devices]}"
                )
        else:
            dev = devices[0]
            self.log.warning(
                f"No serial specified. Using first camera: {dev.GetModelName()} / {dev.GetSerialNumber()}"
            )

        self.cam = pylon.InstantCamera(tl_factory.CreateDevice(dev))
        self.cam.Open()

        # GigE tuning (best-effort; some nodes may not exist depending on model/firmware)
        def try_set(node_name, value):
            try:
                node = getattr(self.cam, node_name)
                node.SetValue(value)
                return True
            except Exception:
                return False

        try_set("GevSCPSPacketSize", int(config.get("packet_size", 1500)))
        try_set("GevSCPD", int(config.get("inter_packet_delay", 0)))

        # ROI
        for k in ["OffsetX", "OffsetY", "Width", "Height"]:
            try:
                getattr(self.cam, k).SetValue(
                    int(config.get(k.lower(), config.get(k, 0)))
                )
            except Exception:
                pass

        # Exposure/Gain
        try:
            self.cam.ExposureAuto.SetValue("Off")
        except Exception:
            pass
        try:
            self.cam.ExposureTime.SetValue(float(config.get("exposure_time_us", 5000)))
        except Exception:
            pass
        try:
            self.cam.GainAuto.SetValue("Off")
        except Exception:
            pass
        try:
            self.cam.Gain.SetValue(float(config.get("gain", 0.0)))
        except Exception:
            pass

        # Pixel format
        pixfmt = config.get("pixel_format", "BayerRG8")
        try:
            self.cam.PixelFormat.SetValue(pixfmt)
        except Exception:
            self.log.warning(
                f"Could not set PixelFormat={pixfmt}, using current: {self.cam.PixelFormat.GetValue()}"
            )

        # Convert to BGR8 for consumers
        self.converter.OutputPixelFormat = pylon.PixelType_BGR8packed
        self.converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

        self.log.info(
            f"Basler opened: {self.cam.GetDeviceInfo().GetModelName()} "
            f"SN={self.cam.GetDeviceInfo().GetSerialNumber()}"
        )

    def start(self) -> None:
        if not self.cam:
            raise RuntimeError("Camera not initialized")
        # LatestImageOnly avoids backlog if consumer is slow
        self.cam.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

    def stop(self) -> None:
        if self.cam:
            try:
                if self.cam.IsGrabbing():
                    self.cam.StopGrabbing()
            finally:
                try:
                    self.cam.Close()
                except Exception:
                    pass

    def capture(self) -> dict:
        if not self.cam or not self.cam.IsGrabbing():
            raise RuntimeError("Camera is not grabbing")

        res = self.cam.RetrieveResult(2000, pylon.TimeoutHandling_ThrowException)
        try:
            if not res.GrabSucceeded():
                raise RuntimeError(
                    f"Grab failed: {res.ErrorCode} {res.ErrorDescription}"
                )

            img = self.converter.Convert(res)
            rgb = img.GetArray()  # HxWx3 uint8 BGR
            ts = time.time_ns()  # replace with hardware ts if you map chunk data
            return {"rgb": rgb, "depth": None, "timestamp_ns": ts}
        finally:
            res.Release()
