"""motion-path-processor: shortcut-smooth + spline-fit a joint path (spec §9).

Wraps ``algorithms.optimization.shortcut_smooth`` and
``algorithms.optimization.spline_fit`` as the ``plan_path`` post-step (§9).
"""

from .process import PathProcessorModule

__all__ = ["PathProcessorModule"]
