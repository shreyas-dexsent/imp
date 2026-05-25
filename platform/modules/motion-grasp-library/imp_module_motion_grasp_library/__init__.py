"""motion-grasp-library: grasp candidates + (future) feasibility (spec §9).

Ported from the VGR orchestrator's ``robot_engine/planning/`` (the grasp
pieces are *not* in robot-algorithms, which excludes grasping by design --
so they live here as imp's own module). See the README for the migration
source.

Exports the pure ``GraspLibrary`` data layer + ``synthesize_grasps`` op for
direct use; ``SynthesizeGraspsModule`` is lazily imported (needs imp_sdk).
"""

from .library import Grasp, GraspLibrary
from .synthesize import synthesize_grasps

__all__ = ["Grasp", "GraspLibrary", "SynthesizeGraspsModule", "synthesize_grasps"]


def __getattr__(name):
    if name == "SynthesizeGraspsModule":
        from .module import SynthesizeGraspsModule

        return SynthesizeGraspsModule
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
