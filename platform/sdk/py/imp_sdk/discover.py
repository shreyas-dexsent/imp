"""Entry-point plugin discovery (spec §17, §19).

Plugins (HAL drivers, modules, services, jobs) register under stable
entry-point groups so ``imp`` finds them at runtime without ``PYTHONPATH``
gymnastics:

    [project.entry-points."imp.modules"]
    spatial-tf = "imp_module_spatial_tf:TfModule"

This module exposes the discovery layer the runtime + CLI sit on top of:

    >>> from imp_sdk.discover import discover_plugins, GROUPS
    >>> plugins = discover_plugins()
    >>> sorted(plugins["imp.modules"])         # plugin names
    ['motion-cartesian', 'motion-coal', 'motion-pinocchio', ...]
    >>> plugins["imp.modules"]["spatial-tf"]   # entry point object
    EntryPoint(name='spatial-tf', value='imp_module_spatial_tf:TfModule', ...)

Use :func:`load_plugin` to import the target class lazily when you need
the implementation, not just the registration:

    >>> cls = load_plugin("imp.modules", "spatial-tf")
    >>> instance = cls(station="devstation")

The four groups match the spec's plugin axes (HAL, functional modules,
services, jobs); ``GROUPS`` is the canonical list -- ``imp doctor`` and
the CLI's ``imp node list`` iterate it.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import metadata
from typing import Any, Dict, Iterable, List, Optional

GROUPS = ("imp.hal", "imp.modules", "imp.services", "imp.jobs")


@dataclass(frozen=True)
class PluginRef:
    """A plugin's stable identity: which group it joined and its name."""

    group: str
    name: str
    target: str  # the entry-point value, e.g. "imp_module_spatial_tf:TfModule"


def discover_plugins(
    groups: Iterable[str] = GROUPS,
) -> Dict[str, Dict[str, metadata.EntryPoint]]:
    """Return ``{group: {name: EntryPoint}}`` across the requested groups.

    Resolves through :func:`importlib.metadata.entry_points`, which reads
    the installed distributions' metadata -- no module is actually imported
    here, so discovery is cheap and crash-free even if a plugin's runtime
    deps are missing. Call :func:`load_plugin` to import the target.
    """
    out: Dict[str, Dict[str, metadata.EntryPoint]] = {g: {} for g in groups}
    for group in groups:
        try:
            eps = metadata.entry_points(group=group)
        except TypeError:
            # Python <3.10 fallback: entry_points() returns a dict.
            eps = metadata.entry_points().get(group, [])  # type: ignore[union-attr]
        for ep in eps:
            out[group][ep.name] = ep
    return out


def list_plugins(groups: Iterable[str] = GROUPS) -> List[PluginRef]:
    """Flatten :func:`discover_plugins` into a list of ``PluginRef``."""
    refs: List[PluginRef] = []
    for group, by_name in discover_plugins(groups).items():
        for name, ep in by_name.items():
            refs.append(PluginRef(group=group, name=name, target=ep.value))
    return refs


def load_plugin(group: str, name: str) -> Any:
    """Import and return the target of ``group``'s ``name`` plugin.

    Raises :class:`KeyError` if the plugin is not registered;
    :class:`ImportError` if its target module is missing; the target's own
    exceptions propagate from :meth:`EntryPoint.load`.
    """
    plugins = discover_plugins([group]).get(group, {})
    if name not in plugins:
        raise KeyError(f"no plugin {name!r} in group {group!r} "
                       f"(known: {sorted(plugins)})")
    return plugins[name].load()


def plugin_target(group: str, name: str) -> Optional[str]:
    """Return the entry-point value (``module:attr``) without importing."""
    plugins = discover_plugins([group]).get(group, {})
    ep = plugins.get(name)
    return ep.value if ep is not None else None
