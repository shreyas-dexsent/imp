"""Implementation for `camera_core.drivers.base`."""

from abc import ABC, abstractmethod


class CameraDriver(ABC):
    @abstractmethod
    def initialize(self, config: dict) -> None: ...
    @abstractmethod
    def start(self) -> None: ...
    @abstractmethod
    def stop(self) -> None: ...
    @abstractmethod
    def capture(self) -> dict: ...
