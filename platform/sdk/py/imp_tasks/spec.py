"""Pydantic schema for ``task.yaml`` (spec §11, §14).

A deployed task is workspace data at
``processes/<process>/tasks/<task>.yaml``; the runtime knows no specific
task ahead of time. This schema is the validation contract.

A task is a **logical graph** + an optional **sequence**:

* Each ``node`` names a registered plugin (an entry-point name under
  ``imp.modules``, ``imp.hal``, etc.), an optional ``class`` override
  (when the plugin's default class isn't the one this node wants), and a
  ``params`` dict that is splatted into the class's constructor.
* ``edges`` are advisory: the modules wire themselves to specific
  keyexprs via their constructor params, so the Graph Compiler validates
  that those keyexprs line up across producers and consumers (schemas
  must match). The ``edges`` list lets the YAML author *document* the
  intent and lets the validator flag missing wires.
* ``sequence`` is a list of stages; v1 honours a single ``until``
  topic per stage as the terminator. Richer FSM (state machine with
  branch/loop) is a P7+ concern.

The ``assets`` and ``placement`` slots match spec §11 / §14 -- they're
held verbatim and surfaced to the runtime (assets) or the future
supervisor (placement).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

# The expected schema tag for task.yaml (spec §14 versioning).
TASK_SCHEMA = "imp.task"
TASK_SCHEMA_VERSION = 1


class NodeSpec(BaseModel):
    """One node in the task graph."""

    model_config = ConfigDict(extra="forbid")

    id: str
    """Unique within the task; used as the human label in events + logs."""

    plugin: str
    """Entry-point name (e.g. 'spatial-transform') under ``group``."""

    group: str = "imp.modules"
    """Which entry-point group to look up the plugin in. Almost always
    ``imp.modules``; ``imp.hal`` for embedded device nodes inside a task."""

    cls: Optional[str] = Field(default=None, alias="class")
    """Override of the plugin's default class. The default class is the
    one the plugin's entry-point names (e.g. ``motion-pinocchio`` points
    at ``FkModule``; set ``class: IkModule`` to use the IK module from
    the same plugin)."""

    params: Dict[str, Any] = Field(default_factory=dict)
    """Splatted as ``**kwargs`` into the chosen class's constructor."""

    @field_validator("id", "plugin", "group")
    @classmethod
    def _non_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("must be non-empty")
        return v


class EdgeSpec(BaseModel):
    """An advisory wiring edge from ``src.port`` to ``dst.port``.

    Modules subscribe to keyexprs configured in their ``params``, so the
    Graph Compiler infers the actual wires from those. ``edges`` lets the
    author *declare* intent so the compiler can flag mismatches (an edge
    whose endpoints don't agree on a keyexpr, or whose schemas differ).
    """

    model_config = ConfigDict(extra="forbid")

    src: str  # "<node_id>.<port_name>"
    dst: str  # "<node_id>.<port_name>"

    @field_validator("src", "dst")
    @classmethod
    def _has_dot(cls, v: str) -> str:
        if "." not in v:
            raise ValueError("must be '<node_id>.<port_name>'")
        return v


class SequenceStage(BaseModel):
    """One step in the high-level sequence (spec §11)."""

    model_config = ConfigDict(extra="forbid")

    stage: str
    """Human-readable stage name (e.g. 'acquire', 'detect_pose', 'place')."""

    until: Optional[str] = None
    """Terminator keyexpr: the runtime advances past this stage once a
    message lands on this topic. For v1 the only meaningful semantics are
    'until = the task's final output topic' -- the run ends when the
    chain emits its last value. Richer transitions land in P7."""


class TaskGraph(BaseModel):
    """The node/edge container."""

    model_config = ConfigDict(extra="forbid")

    nodes: List[NodeSpec]
    edges: List[EdgeSpec] = Field(default_factory=list)


class TaskSpec(BaseModel):
    """A complete deployed task (workspace ``task.yaml``)."""

    model_config = ConfigDict(extra="forbid")

    schema_: str = Field(alias="schema")
    version: int

    id: str
    station: str
    process: Optional[str] = None
    description: Optional[str] = None

    graph: TaskGraph
    sequence: List[SequenceStage] = Field(default_factory=list)

    assets: Dict[str, Any] = Field(default_factory=dict)
    placement: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("schema_")
    @classmethod
    def _schema_matches(cls, v: str) -> str:
        if v != TASK_SCHEMA:
            raise ValueError(f"schema must be {TASK_SCHEMA!r}; got {v!r}")
        return v

    @field_validator("version")
    @classmethod
    def _version_matches(cls, v: int) -> int:
        if v != TASK_SCHEMA_VERSION:
            raise ValueError(
                f"version must be {TASK_SCHEMA_VERSION}; got {v}"
            )
        return v

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TaskSpec":
        import yaml

        text = Path(path).read_text()
        payload = yaml.safe_load(text)
        if not isinstance(payload, dict):
            raise ValueError(f"{path}: task.yaml must be a YAML mapping")
        return cls.model_validate(payload)

    # ------------------------------------------------------------------
    # Convenience: node id -> NodeSpec lookup
    # ------------------------------------------------------------------

    def node(self, node_id: str) -> NodeSpec:
        for n in self.graph.nodes:
            if n.id == node_id:
                return n
        raise KeyError(f"no node {node_id!r} in task {self.id!r}")
