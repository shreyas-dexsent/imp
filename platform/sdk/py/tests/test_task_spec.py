"""Tests for the task.yaml schema + loader (imp_tasks.spec).

Pure-stdlib: doesn't touch the bus or plugin instantiation.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from imp_tasks.spec import EdgeSpec, NodeSpec, TaskSpec


GOOD_TASK = """
schema: imp.task
version: 1
id: pick_place_demo
station: devstation
process: bin-picking-a

graph:
  nodes:
    - id: transform
      plugin: spatial-transform
      params:
        pose_key: imp/devstation/perc/s1/pose
        base_frame: base
    - id: ik
      plugin: motion-pinocchio
      class: IkModule
      params:
        station: devstation
        robot: fr3
        robot_system_path: catalog/robots/franka_fr3_with_franka_hand.yaml
  edges:
    - { src: transform.target, dst: ik.target }

sequence:
  - { stage: acquire }
  - { stage: solve, until: imp/devstation/motion/ik/solution }
"""


def test_loads_a_valid_task(tmp_path: Path):
    p = tmp_path / "task.yaml"
    p.write_text(GOOD_TASK)
    spec = TaskSpec.from_yaml(p)
    assert spec.id == "pick_place_demo"
    assert spec.schema_ == "imp.task"
    assert spec.version == 1
    assert spec.station == "devstation"
    assert len(spec.graph.nodes) == 2
    assert len(spec.graph.edges) == 1
    assert spec.graph.edges[0].src == "transform.target"
    assert spec.graph.edges[0].dst == "ik.target"
    assert [s.stage for s in spec.sequence] == ["acquire", "solve"]
    assert spec.sequence[1].until == "imp/devstation/motion/ik/solution"


def test_node_class_override_round_trips():
    n = NodeSpec(
        id="ik",
        plugin="motion-pinocchio",
        **{"class": "IkModule"},  # alias for `cls`
        params={"x": 1},
    )
    assert n.cls == "IkModule"


def test_node_class_defaults_to_none():
    n = NodeSpec(id="fk", plugin="motion-pinocchio")
    assert n.cls is None
    assert n.group == "imp.modules"


def test_edge_requires_dot():
    with pytest.raises(ValueError):
        EdgeSpec(src="transform", dst="ik.target")
    with pytest.raises(ValueError):
        EdgeSpec(src="transform.target", dst="ik")


def test_wrong_schema_rejected():
    bad = GOOD_TASK.replace("schema: imp.task", "schema: imp.totally-wrong")
    with pytest.raises(Exception) as e:
        TaskSpec.model_validate(_load_yaml(bad))
    assert "schema must be 'imp.task'" in str(e.value)


def test_wrong_version_rejected():
    bad = GOOD_TASK.replace("version: 1", "version: 99")
    with pytest.raises(Exception) as e:
        TaskSpec.model_validate(_load_yaml(bad))
    assert "version must be 1" in str(e.value)


def test_extra_top_level_field_rejected():
    bad = GOOD_TASK + "\nunknown_field: oops\n"
    with pytest.raises(Exception):
        TaskSpec.model_validate(_load_yaml(bad))


def test_node_lookup_helper():
    spec = TaskSpec.model_validate(_load_yaml(GOOD_TASK))
    assert spec.node("ik").plugin == "motion-pinocchio"
    with pytest.raises(KeyError):
        spec.node("missing")


def test_sequence_optional():
    minimal = """
schema: imp.task
version: 1
id: just_one_node
station: devstation
graph:
  nodes:
    - id: fk
      plugin: motion-pinocchio
      params: {station: s, robot: r, world_path: w}
"""
    spec = TaskSpec.model_validate(_load_yaml(minimal))
    assert spec.sequence == []
    assert spec.graph.edges == []


def _load_yaml(text: str) -> dict:
    import yaml
    return yaml.safe_load(text)
