"""spatial-tf: frame graph over Zenoh (spec §9).

Exports ``TfGraph`` (pure library, used in-process by any module that needs
frame composition) and ``TfModule`` (the bus-resident frame graph node that
subscribes ``imp/<station>/tf`` and maintains a live graph).

``TfModule`` is lazily imported so the pure ``TfGraph`` library is usable
without ``imp_sdk`` / ``zenoh`` installed (e.g. inside unit tests that only
exercise the graph).
"""

from .graph import TfGraph, TfLookupError

__all__ = ["TfGraph", "TfLookupError", "TfModule"]


def __getattr__(name):
    if name == "TfModule":
        from .module import TfModule  # local import: needs imp_sdk + zenoh

        return TfModule
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
