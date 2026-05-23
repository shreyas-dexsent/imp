"""Implementation for `camera_core.drivers.flir_blackfly_gige`."""

# import time
# import numpy as np
# # import PySpin

# try:
#     import PySpin
# except Exception as e:
#     PySpin = None
#     _PYSPIN_IMPORT_ERROR = e

# if PySpin is None:
#     raise RuntimeError("PySpin/Spinnaker SDK not installed or ABI mismatch; install the SDK wheel. Original: {}".format(_PYSPIN_IMPORT_ERROR))


# from .base import CameraDriver


# class FlirBlackflyGigEDriver(CameraDriver):
#     def __init__(self, logger):
#         self.log = logger
#         self.system = None
#         self.cam = None
#         self.nodemap = None

#     def initialize(self, config: dict) -> None:
#         self.system = PySpin.System.GetInstance()
#         cam_list = self.system.GetCameras()

#         if cam_list.GetSize() == 0:
#             raise RuntimeError("No FLIR cameras detected")

#         serial = config.get("serial_number", "").strip()

#         self.cam = None
#         for cam in cam_list:
#             nodemap_tl = cam.GetTLDeviceNodeMap()
#             sn = PySpin.CStringPtr(nodemap_tl.GetNode("DeviceSerialNumber")).GetValue()
#             if not serial or sn == serial:
#                 self.cam = cam
#                 break

#         if self.cam is None:
#             raise RuntimeError(f"FLIR camera with serial {serial} not found")

#         self.cam.Init()
#         self.nodemap = self.cam.GetNodeMap()

#         self._configure_camera(config)

#         self.log.info(f"FLIR Blackfly initialized SN={sn}")

#     def _configure_camera(self, config: dict):
#         nm = self.nodemap

#         def set_enum(name, value):
#             node = PySpin.CEnumerationPtr(nm.GetNode(name))
#             if PySpin.IsAvailable(node) and PySpin.IsWritable(node):
#                 entry = node.GetEntryByName(value)
#                 node.SetIntValue(entry.GetValue())

#         def set_float(name, value):
#             node = PySpin.CFloatPtr(nm.GetNode(name))
#             if PySpin.IsAvailable(node) and PySpin.IsWritable(node):
#                 node.SetValue(float(value))

#         # Pixel format
#         set_enum("PixelFormat", config.get("pixel_format", "BayerRG8"))

#         # Disable auto features
#         set_enum("ExposureAuto", "Off")
#         set_enum("GainAuto", "Off")

#         set_float("ExposureTime", config.get("exposure_time_us", 5000))
#         set_float("Gain", config.get("gain", 0.0))

#         # GigE tuning (best-effort)
#         try:
#             set_float("GevSCPSPacketSize", config.get("packet_size", 1500))
#             set_float("GevSCPD", config.get("inter_packet_delay", 0))
#         except Exception:
#             pass

#     def start(self) -> None:
#         self.cam.BeginAcquisition()

#     def stop(self) -> None:
#         try:
#             self.cam.EndAcquisition()
#         except Exception:
#             pass
#         try:
#             self.cam.DeInit()
#         except Exception:
#             pass
#         if self.system:
#             self.system.ReleaseInstance()

#     def capture(self) -> dict:
#         image = self.cam.GetNextImage(2000)
#         try:
#             if image.IsIncomplete():
#                 raise RuntimeError(f"Incomplete image {image.GetImageStatus()}")

#             # Convert to BGR8 (consistent with Basler path)
#             image_converted = image.Convert(
#                 PySpin.PixelFormat_BGR8, PySpin.HQ_LINEAR
#             )

#             rgb = image_converted.GetNDArray()
#             ts_ns = time.time_ns()  # map chunk timestamp later if needed

#             return {
#                 "rgb": rgb,
#                 "depth": None,
#                 "timestamp_ns": ts_ns,
#             }
#         finally:
#             image.Release()
