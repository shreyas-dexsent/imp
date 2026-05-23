"""Implementation for `camera_core.config`."""

from typing import Any, Dict, List, Optional

import yaml
from pydantic import BaseModel, Field


class CameraSDKConfig(BaseModel):
    serial_number: str = ""
    packet_size: int = 1500
    inter_packet_delay: int = 0
    exposure_time_us: int = 5000
    gain: float = 0.0
    pixel_format: str = "BayerRG8"
    width: int = 2040
    height: int = 2040
    color_width: Optional[int] = None
    color_height: Optional[int] = None
    color_fps: Optional[int] = None
    color_format: str = "bgr8"
    depth_width: Optional[int] = None
    depth_height: Optional[int] = None
    depth_fps: Optional[int] = None
    depth_format: str = "z16"
    align_to: str = "color"
    options: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    roi: Dict[str, Any] = Field(default_factory=dict)
    post_processing: Dict[str, Any] = Field(default_factory=dict)
    offset_x: int = 0
    offset_y: int = 0


class CameraConfig(BaseModel):
    camera_id: str
    type: str
    mode: str = "rgb"
    fps: int = 15
    sdk: CameraSDKConfig


class ShmRGBConfig(BaseModel):
    name_prefix: str
    width: int
    height: int
    channels: int = 3
    dtype: str = "uint8"
    triple_buffers: List[str] = ["A", "B", "C"]


class ShmDepthConfig(BaseModel):
    name_prefix: str
    width: int
    height: int
    dtype: str = "float32"
    triple_buffers: List[str] = ["A", "B", "C"]


class SharedMemoryConfig(BaseModel):
    rgb: ShmRGBConfig
    depth: Optional[ShmDepthConfig] = None


class IPCConfig(BaseModel):
    zmq_pub: str
    topic: str = "camera"


class HealthConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8091


class RuntimeConfig(BaseModel):
    log_level: str = "INFO"


class CalibrationConfig(BaseModel):
    enable: bool = True
    control_host: str = "127.0.0.1"
    control_port: int = 8210
    server_host: str = "127.0.0.1"
    server_port: int = 8211
    cors_origins: List[str] = Field(default_factory=lambda: ["*"])


class RobotControlConfig(BaseModel):
    command_endpoint: str = "tcp://127.0.0.1:5571"
    timeout_ms: int = 20000


class AppConfig(BaseModel):
    camera: CameraConfig
    shared_memory: SharedMemoryConfig
    ipc: IPCConfig
    health: HealthConfig
    runtime: RuntimeConfig
    calibration: CalibrationConfig = Field(default_factory=CalibrationConfig)
    robot: RobotControlConfig = Field(default_factory=RobotControlConfig)


def load_config(path: str) -> AppConfig:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return AppConfig(**data)
