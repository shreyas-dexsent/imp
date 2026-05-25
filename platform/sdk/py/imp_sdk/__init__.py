"""imp Python SDK: the public surface for Python HAL drivers, modules,
services, and jobs (spec §17).

Symbols are exported lazily (PEP 562). ``keyexpr`` and ``discover`` are
stdlib-only and load eagerly; the bus / hal / module layers (which pull in
``zenoh``) and ``schemas`` (which pulls in the generated protobuf bindings)
are imported on first attribute access. This lets pure-Python pieces like
``imp_sdk.discover.discover_plugins`` and ``imp_sdk.keyexpr.hal(...)`` work
on hosts without the full runtime env -- handy for tooling, dev tests, and
the doc-rule / packaging checks that don't talk to Zenoh.
"""

from __future__ import annotations

from . import discover, keyexpr  # stdlib-only; safe to load eagerly

__all__ = [
    # eager
    "discover", "keyexpr",
    # lazy (resolved by __getattr__ below)
    "schemas",
    "Bus", "Publisher", "QosClass", "TypedSub",
    "HalDevice", "HalNode", "Lifecycle", "Pub", "Sub", "run_device",
    "Input", "Module", "ModuleNode", "Output", "run_module",
]


def __getattr__(name):  # PEP 562
    if name == "schemas":
        from . import schemas as _schemas
        return _schemas
    if name in {"Bus", "Publisher", "QosClass", "TypedSub"}:
        from . import bus as _bus
        return getattr(_bus, name)
    if name in {"HalDevice", "HalNode", "Lifecycle", "Pub", "Sub", "run_device"}:
        from . import hal as _hal
        return getattr(_hal, name)
    if name in {"Input", "Module", "ModuleNode", "Output", "run_module"}:
        from . import module as _module
        return getattr(_module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
