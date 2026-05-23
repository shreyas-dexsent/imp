"""Implementation for `vision_engine.core.module_base`."""

from abc import ABC, abstractmethod
from typing import Any, Dict

from vision_engine.io.data_plane.frame_bundle import FrameBundle


class VisionModule(ABC):
    """
    Base class for all vision modules.
    """

    def __init__(self, name: str, params: Dict[str, Any]):
        self.name = name
        self.params = params

    @abstractmethod
    def run(self, frame_bundle: FrameBundle) -> Dict[str, Any]:
        """
        Execute module logic.

        Input:
            frame_bundle (FrameBundle)

        Output:
            JSON-serializable dict
        """
        raise NotImplementedError

    def stop(self) -> None:
        """
        Optional cleanup hook.
        """
        pass
