"""Unit tests for imp_sdk.discover (entry-point plugin discovery, spec §17).

Stdlib-only on the test path: the discovery layer reads installed metadata
without importing any plugin's code, and ``imp_sdk.__init__`` resolves the
heavy submodules (bus/hal/module/schemas) lazily so this test runs without
zenoh / protobuf installed.
"""

from __future__ import annotations

from importlib import metadata

import pytest

from imp_sdk.discover import (
    GROUPS,
    PluginRef,
    discover_plugins,
    list_plugins,
    load_plugin,
    plugin_target,
)  # noqa: E402


def test_groups_match_spec():
    """The canonical groups are the four §17 plugin axes -- nothing else."""
    assert GROUPS == ("imp.hal", "imp.modules", "imp.services", "imp.jobs")


def test_discover_plugins_returns_dict_keyed_by_group():
    """Always returns every requested group, even with zero plugins installed."""
    out = discover_plugins()
    assert set(out.keys()) == set(GROUPS)
    for group, plugins in out.items():
        assert isinstance(plugins, dict)
        for ep in plugins.values():
            assert isinstance(ep, metadata.EntryPoint)


def test_list_plugins_flattens(monkeypatch):
    """list_plugins is a flat view of discover_plugins."""
    monkeypatch.setattr(
        "imp_sdk.discover.discover_plugins",
        lambda groups=GROUPS: {
            "imp.modules": {
                "spatial-tf": _ep("spatial-tf", "imp_module_spatial_tf:TfModule"),
                "motion-pinocchio": _ep("motion-pinocchio", "imp_module_motion_pinocchio:FkModule"),
            },
            "imp.hal": {},
            "imp.services": {},
            "imp.jobs": {},
        },
    )
    refs = list_plugins()
    names = sorted((r.group, r.name) for r in refs)
    assert names == [
        ("imp.modules", "motion-pinocchio"),
        ("imp.modules", "spatial-tf"),
    ]
    assert all(isinstance(r, PluginRef) for r in refs)


def test_load_plugin_missing_raises_keyerror(monkeypatch):
    monkeypatch.setattr(
        "imp_sdk.discover.discover_plugins",
        lambda groups=GROUPS: {"imp.modules": {}, "imp.hal": {}, "imp.services": {}, "imp.jobs": {}},
    )
    with pytest.raises(KeyError) as excinfo:
        load_plugin("imp.modules", "nope")
    assert "no plugin 'nope'" in str(excinfo.value)


def test_plugin_target_returns_none_when_missing(monkeypatch):
    monkeypatch.setattr(
        "imp_sdk.discover.discover_plugins",
        lambda groups=GROUPS: {"imp.modules": {}, "imp.hal": {}, "imp.services": {}, "imp.jobs": {}},
    )
    assert plugin_target("imp.modules", "nope") is None


def test_plugin_target_returns_value_when_present(monkeypatch):
    monkeypatch.setattr(
        "imp_sdk.discover.discover_plugins",
        lambda groups=GROUPS: {
            "imp.modules": {"spatial-tf": _ep("spatial-tf", "imp_module_spatial_tf:TfModule")},
            "imp.hal": {},
            "imp.services": {},
            "imp.jobs": {},
        },
    )
    assert plugin_target("imp.modules", "spatial-tf") == "imp_module_spatial_tf:TfModule"


def _ep(name: str, value: str) -> metadata.EntryPoint:
    """Build an EntryPoint with the minimum fields tests need."""
    return metadata.EntryPoint(name=name, value=value, group="imp.modules")
