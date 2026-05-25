"""Graph Compiler: TaskSpec -> CompiledTask (spec §5, §11).

The compiler:

1. Resolves every node's plugin class via :func:`imp_sdk.discover.load_plugin`
   (or :mod:`importlib` for explicit ``cls`` overrides).
2. Instantiates each module with ``params`` splatted as ``**kwargs``.
3. Walks every module's declared inputs and outputs, collects
   ``(keyexpr, schema)`` for each, and checks every advisory ``EdgeSpec``
   against them so a mistyped key surfaces at compile time.
4. Returns a :class:`CompiledTask` -- a frozen container the runtime can
   spin without any further yaml lookup.

It does **not** start any module, open any bus session, or touch the
network. Instantiation only -- the runtime owns I/O.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

from imp_sdk import discover

from .spec import EdgeSpec, NodeSpec, TaskSpec


class CompileError(Exception):
    """Raised when a TaskSpec cannot be turned into a CompiledTask."""


@dataclass(frozen=True)
class CompiledNode:
    """One compiled node: the live module instance plus its source spec."""

    spec: NodeSpec
    module: Any  # imp_sdk.module.Module instance (avoiding the heavy import here)
    inputs: Tuple[Tuple[str, str, str], ...]  # (port_name, key, schema_tag)
    outputs: Tuple[Tuple[str, str, str], ...]


@dataclass(frozen=True)
class CompiledTask:
    """A ready-to-run task: every module instantiated and wiring validated."""

    spec: TaskSpec
    nodes: List[CompiledNode]
    edges: List[EdgeSpec] = field(default_factory=list)

    def node(self, node_id: str) -> CompiledNode:
        for n in self.nodes:
            if n.spec.id == node_id:
                return n
        raise KeyError(f"no compiled node {node_id!r}")

    # All output topics across the graph -- handy for the runtime to subscribe
    # to log every produced topic into the run bag (P7+).
    @property
    def all_output_keys(self) -> List[str]:
        return [
            key
            for n in self.nodes
            for (_, key, _) in n.outputs
        ]


# ---------------------------------------------------------------------------
# Plugin resolution
# ---------------------------------------------------------------------------


def _resolve_class(node: NodeSpec) -> Any:
    """Return the class to instantiate for ``node``.

    Without ``class`` override: load the plugin's default class via the
    registered entry point. With override: import the module the entry
    point points at and getattr the requested class name (so different
    classes inside the same plugin -- e.g. ``FkModule`` vs ``IkModule``
    in motion-pinocchio -- are addressable).
    """
    ep_value = discover.plugin_target(node.group, node.plugin)
    if ep_value is None:
        raise CompileError(
            f"node {node.id!r}: no plugin {node.plugin!r} registered in "
            f"group {node.group!r}. Is its wheel installed?"
        )

    if node.cls is None:
        try:
            return discover.load_plugin(node.group, node.plugin)
        except Exception as e:
            raise CompileError(
                f"node {node.id!r}: failed to load {node.group}:{node.plugin}: {e}"
            ) from e

    # ``ep_value`` looks like "imp_module_motion_pinocchio:FkModule";
    # take the module half and getattr the override.
    module_path, _, _ = ep_value.partition(":")
    try:
        mod = importlib.import_module(module_path)
    except Exception as e:
        raise CompileError(
            f"node {node.id!r}: cannot import {module_path!r}: {e}"
        ) from e
    try:
        return getattr(mod, node.cls)
    except AttributeError as e:
        raise CompileError(
            f"node {node.id!r}: class {node.cls!r} not found in {module_path!r}"
        ) from e


def _instantiate(node: NodeSpec, cls: Any) -> Any:
    try:
        return cls(**node.params)
    except TypeError as e:
        raise CompileError(
            f"node {node.id!r}: {cls.__name__}({_pretty_params(node.params)}): {e}"
        ) from e
    except Exception as e:
        raise CompileError(
            f"node {node.id!r}: instantiating {cls.__name__}: {e}"
        ) from e


def _pretty_params(params: Dict[str, Any]) -> str:
    return ", ".join(f"{k}={v!r}" for k, v in params.items())


# ---------------------------------------------------------------------------
# Module port extraction
# ---------------------------------------------------------------------------


def _ports_of(module: Any) -> Tuple[Tuple[Tuple[str, str, str], ...],
                                    Tuple[Tuple[str, str, str], ...]]:
    """Return ``(inputs, outputs)`` as tuples of ``(port_name, key, schema_tag)``.

    Reaches into the module's :meth:`inputs` / :meth:`outputs` methods (the
    Compute-Runtime contract from ``imp_sdk.module``). ``schema_tag`` is
    derived from the protobuf message class name; the bus already uses
    the same convention.
    """
    inputs = tuple(
        (i.name, i.key, _schema_tag(i.msg_type)) for i in module.inputs()
    )
    outputs = tuple(
        (o.name, o.key, _schema_tag(o.msg_type)) for o in module.outputs()
    )
    return inputs, outputs


def _schema_tag(msg_type: Any) -> str:
    name = getattr(msg_type, "DESCRIPTOR", None)
    if name is not None:
        return f"imp.{name.name}/1"
    # Fall back to the class name; the runtime checks the schema= attachment
    # on the wire, not this string, so it's just for the validator's error.
    return f"imp.{msg_type.__name__}/1"


# ---------------------------------------------------------------------------
# Edge validation
# ---------------------------------------------------------------------------


def _check_edges(
    nodes: List[CompiledNode], edges: List[EdgeSpec]
) -> List[str]:
    """Walk advisory edges; return a list of human-readable problem strings."""
    by_id = {n.spec.id: n for n in nodes}

    def _port(side: str) -> Tuple[CompiledNode, str, str, str]:
        node_id, _, port_name = side.partition(".")
        node = by_id.get(node_id)
        if node is None:
            raise CompileError(f"edge mentions unknown node {node_id!r}")
        # Look for the port on either side; the caller decides direction.
        for kind in (node.outputs, node.inputs):
            for pname, key, schema in kind:
                if pname == port_name:
                    return node, pname, key, schema
        raise CompileError(
            f"edge mentions unknown port {node_id}.{port_name!r}; "
            f"known on this node: "
            f"in={[p for p,_,_ in node.inputs]} "
            f"out={[p for p,_,_ in node.outputs]}"
        )

    problems: List[str] = []
    for edge in edges:
        src_node, _, src_key, src_schema = _port(edge.src)
        dst_node, _, dst_key, dst_schema = _port(edge.dst)

        # Both sides must agree on the keyexpr (that's the actual wire).
        if src_key != dst_key:
            problems.append(
                f"edge {edge.src} -> {edge.dst}: keyexpr mismatch "
                f"({src_key!r} vs {dst_key!r}); the modules will not see "
                f"each other's messages."
            )
        if src_schema != dst_schema:
            problems.append(
                f"edge {edge.src} -> {edge.dst}: schema mismatch "
                f"({src_schema!r} vs {dst_schema!r})."
            )
    return problems


# ---------------------------------------------------------------------------
# Public entry
# ---------------------------------------------------------------------------


def compile_task(spec: TaskSpec, *, strict_edges: bool = True) -> CompiledTask:
    """Turn a :class:`TaskSpec` into a :class:`CompiledTask`.

    Raises :class:`CompileError` on any failure: missing plugin, bad
    constructor kwargs, unresolved edge endpoint, key/schema mismatch on
    a declared edge. When ``strict_edges`` is ``True`` (default), any
    edge problem aborts compilation; with ``strict_edges=False`` the
    problems are still raised but only for *unresolved* endpoints --
    keyexpr/schema mismatches become warnings (returned through the
    raised exception's ``args``).
    """
    nodes: List[CompiledNode] = []
    for node_spec in spec.graph.nodes:
        cls = _resolve_class(node_spec)
        module = _instantiate(node_spec, cls)
        inputs, outputs = _ports_of(module)
        nodes.append(
            CompiledNode(
                spec=node_spec,
                module=module,
                inputs=inputs,
                outputs=outputs,
            )
        )

    # Node id uniqueness check.
    ids = [n.spec.id for n in nodes]
    if len(set(ids)) != len(ids):
        dups = sorted({i for i in ids if ids.count(i) > 1})
        raise CompileError(f"duplicate node id(s): {dups}")

    problems = _check_edges(nodes, spec.graph.edges)
    if problems:
        if strict_edges:
            raise CompileError("; ".join(problems))
        # Non-strict: keep going but surface the problems on the returned
        # object via a sentinel attribute the runtime can log.
        compiled = CompiledTask(spec=spec, nodes=nodes, edges=spec.graph.edges)
        compiled.__dict__["_warnings"] = problems  # frozen dataclass workaround
        return compiled

    return CompiledTask(spec=spec, nodes=nodes, edges=spec.graph.edges)
