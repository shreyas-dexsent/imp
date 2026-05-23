"""Implementation for `vision_engine.core.errors`."""


class VisionEngineError(Exception):
    pass


class FrameReadError(VisionEngineError):
    pass


class ModuleExecutionError(VisionEngineError):
    pass


class InvalidPipelineError(VisionEngineError):
    pass
