"""Tests for the Graph Compiler (imp_tasks.compiler).

Uses an in-test fake plugin registered with monkeypatched
``imp_sdk.discover`` -- the compiler walks the same path as in production
(plugin lookup -> class load -> kwargs splat -> port collection -> edge
validation) without needing any real motion-core plugin installed.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from imp_tasks.compiler import CompileError, compile_task
from imp_tasks.spec import EdgeSpec, NodeSpec, SequenceStage, TaskSpec, TaskGraph


# ---------------------------------------------------------------------------
# Fake module shape -- matches imp_sdk.module's expectations (.inputs() /
# .outputs() return objects with .name / .key / .msg_type fields).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Port:
    name: str
    key: str
    msg_type: type


class _FakeProto:
    """Drop-in for an imp_pb2 protobuf class -- only DESCRIPTOR.name matters."""

    def __init__(self, name: str):
        class _Descriptor:
            pass
        self.DESCRIPTOR = _Descriptor()
        self.DESCRIPTOR.name = name


class _FakeUpstream:
    """Publishes one Pose6D on imp/devstation/perc/s1/pose."""

    def __init__(self, station: str = "devstation"):
        self.station = station

    def inputs(self): return []
    def outputs(self):
        return [_Port("pose", f"imp/{self.station}/perc/s1/pose",
                     _FakeProto("Pose6D"))]


class _FakeDownstream:
    """Subscribes the same Pose6D + publishes a JointSolution."""

    def __init__(self, station: str = "devstation"):
        self.station = station

    def inputs(self):
        return [_Port("pose", f"imp/{self.station}/perc/s1/pose",
                      _FakeProto("Pose6D"))]
    def outputs(self):
        return [_Port("solution", f"imp/{self.station}/motion/ik/solution",
                      _FakeProto("JointSolution"))]


class _RequiresParam:
    """Constructor takes a required ``required_param``; used to test kwargs splat."""

    def __init__(self, required_param: str):
        self.required_param = required_param

    def inputs(self): return []
    def outputs(self): return []


def _spec_for(nodes, edges=None, sequence=None):
    return TaskSpec(
        schema="imp.task",
        version=1,
        id="test_task",
        station="devstation",
        graph=TaskGraph(nodes=nodes, edges=edges or []),
        sequence=sequence or [],
    )


def _install_fake_registry(monkeypatch, registry):
    """Patch imp_sdk.discover so the compiler sees ``registry`` instead of
    actual installed entry points."""
    def _plugin_target(group, name):
        return registry.get(name, {}).get("target")

    def _load_plugin(group, name):
        if name not in registry:
            raise KeyError(name)
        return registry[name]["cls"]

    monkeypatch.setattr("imp_sdk.discover.plugin_target", _plugin_target)
    monkeypatch.setattr("imp_sdk.discover.load_plugin", _load_plugin)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_compiles_single_node(monkeypatch):
    _install_fake_registry(monkeypatch, {
        "upstream": {"target": "fake_mod:_FakeUpstream", "cls": _FakeUpstream},
    })
    spec = _spec_for([NodeSpec(id="u", plugin="upstream", params={"station": "devstation"})])
    compiled = compile_task(spec)
    assert len(compiled.nodes) == 1
    node = compiled.nodes[0]
    assert isinstance(node.module, _FakeUpstream)
    assert node.outputs == (("pose", "imp/devstation/perc/s1/pose", "imp.Pose6D/1"),)


def test_compiles_two_nodes_with_matching_edge(monkeypatch):
    _install_fake_registry(monkeypatch, {
        "upstream": {"target": "f:_U", "cls": _FakeUpstream},
        "downstream": {"target": "f:_D", "cls": _FakeDownstream},
    })
    spec = _spec_for(
        [
            NodeSpec(id="u", plugin="upstream"),
            NodeSpec(id="d", plugin="downstream"),
        ],
        edges=[EdgeSpec(src="u.pose", dst="d.pose")],
    )
    compiled = compile_task(spec)
    assert len(compiled.nodes) == 2
    assert compiled.all_output_keys == [
        "imp/devstation/perc/s1/pose",
        "imp/devstation/motion/ik/solution",
    ]


def test_edge_keyexpr_mismatch_fails_compile(monkeypatch):
    # Build an upstream that publishes on a different station so the keys diverge.
    _install_fake_registry(monkeypatch, {
        "upstream": {"target": "f:_U", "cls": _FakeUpstream},
        "downstream": {"target": "f:_D", "cls": _FakeDownstream},
    })
    spec = _spec_for(
        [
            NodeSpec(id="u", plugin="upstream", params={"station": "otherstation"}),
            NodeSpec(id="d", plugin="downstream", params={"station": "devstation"}),
        ],
        edges=[EdgeSpec(src="u.pose", dst="d.pose")],
    )
    with pytest.raises(CompileError) as e:
        compile_task(spec)
    msg = str(e.value)
    assert "keyexpr mismatch" in msg
    assert "otherstation" in msg


def test_missing_plugin_fails_compile(monkeypatch):
    _install_fake_registry(monkeypatch, {})
    spec = _spec_for([NodeSpec(id="u", plugin="ghost")])
    with pytest.raises(CompileError) as e:
        compile_task(spec)
    assert "no plugin 'ghost'" in str(e.value)


def test_missing_required_param_fails_compile(monkeypatch):
    _install_fake_registry(monkeypatch, {
        "needy": {"target": "f:_R", "cls": _RequiresParam},
    })
    spec = _spec_for([NodeSpec(id="x", plugin="needy", params={})])
    with pytest.raises(CompileError) as e:
        compile_task(spec)
    assert "_RequiresParam" in str(e.value)
    # The TypeError about the missing argument should be in the wrapped message.
    assert "required" in str(e.value).lower()


def test_duplicate_node_ids_fail_compile(monkeypatch):
    _install_fake_registry(monkeypatch, {
        "upstream": {"target": "f:_U", "cls": _FakeUpstream},
    })
    spec = _spec_for([
        NodeSpec(id="u", plugin="upstream"),
        NodeSpec(id="u", plugin="upstream"),
    ])
    with pytest.raises(CompileError) as e:
        compile_task(spec)
    assert "duplicate node id" in str(e.value)


def test_edge_unknown_node_fails(monkeypatch):
    _install_fake_registry(monkeypatch, {
        "upstream": {"target": "f:_U", "cls": _FakeUpstream},
    })
    spec = _spec_for(
        [NodeSpec(id="u", plugin="upstream")],
        edges=[EdgeSpec(src="u.pose", dst="ghost.pose")],
    )
    with pytest.raises(CompileError) as e:
        compile_task(spec)
    assert "ghost" in str(e.value)


def test_edge_unknown_port_fails(monkeypatch):
    _install_fake_registry(monkeypatch, {
        "upstream": {"target": "f:_U", "cls": _FakeUpstream},
    })
    spec = _spec_for(
        [NodeSpec(id="u", plugin="upstream")],
        edges=[EdgeSpec(src="u.nopelol", dst="u.pose")],
    )
    with pytest.raises(CompileError) as e:
        compile_task(spec)
    assert "nopelol" in str(e.value)


def test_non_strict_edges_returns_warnings(monkeypatch):
    _install_fake_registry(monkeypatch, {
        "upstream": {"target": "f:_U", "cls": _FakeUpstream},
        "downstream": {"target": "f:_D", "cls": _FakeDownstream},
    })
    spec = _spec_for(
        [
            NodeSpec(id="u", plugin="upstream", params={"station": "otherstation"}),
            NodeSpec(id="d", plugin="downstream"),
        ],
        edges=[EdgeSpec(src="u.pose", dst="d.pose")],
    )
    compiled = compile_task(spec, strict_edges=False)
    warnings = compiled.__dict__.get("_warnings", [])
    assert any("keyexpr mismatch" in w for w in warnings)
