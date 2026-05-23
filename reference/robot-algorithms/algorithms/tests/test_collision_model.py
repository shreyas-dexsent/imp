"""Tests for resolved.CollisionModel.

CollisionModel stores Coal geometry catalogues and static allowed-collision
rules. Runtime collision queries are implemented by the collision operation
layer that consumes this resolved model.
"""
from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("pinocchio")
pytest.importorskip("coal")

from algorithms.descriptions import WorldDescription
from algorithms.resolved import CollisionModel


REPO_ROOT = Path(__file__).resolve().parents[1]


def franka_table_world_path() -> Path:
    return REPO_ROOT / "configs" / "worlds" / "franka_table_world.yaml"


def two_franka_world_path() -> Path:
    return REPO_ROOT / "configs" / "worlds" / "two_franka_table_world.yaml"


def robot_only_world_path() -> Path:
    return REPO_ROOT / "configs" / "worlds" / "franka_robot_only_world.yaml"


# ---------------------------------------------------------------------------
# Single-robot world
# ---------------------------------------------------------------------------


def test_collision_model_builds_geometry_for_single_robot_world():
    world = WorldDescription.from_yaml(franka_table_world_path())
    cm = CollisionModel.from_world(world)

    # The FR3 + hand has 9 collision links combined; the exact count depends
    # on the URDF (some link bodies have multiple primitives). Sanity-check
    # the names rather than the count.
    robot_names = {go.name for go in cm.robot_geom.geometryObjects}
    assert any("fr3_link0" in n for n in robot_names)
    assert any("fr3_link7" in n for n in robot_names)


def test_collision_model_registers_world_objects():
    world = WorldDescription.from_yaml(franka_table_world_path())
    cm = CollisionModel.from_world(world)

    assert cm.has_object("matka")
    assert cm.object_owner["matka"] == "world"


def test_robot_only_world_has_no_world_geometry():
    world = WorldDescription.from_yaml(robot_only_world_path())
    cm = CollisionModel.from_world(world)

    assert len(cm.world_geom.geometryObjects) == 0
    assert len(cm.robot_geom.geometryObjects) > 0


# ---------------------------------------------------------------------------
# Multi-robot namespacing
# ---------------------------------------------------------------------------


def test_multi_robot_world_namespaces_geometry():
    world = WorldDescription.from_yaml(two_franka_world_path())
    cm = CollisionModel.from_world(world)

    names = {go.name for go in cm.robot_geom.geometryObjects}
    left = {n for n in names if n.startswith("left/")}
    right = {n for n in names if n.startswith("right/")}

    assert len(left) > 0
    assert len(right) > 0
    assert left.isdisjoint(right)
    # Two identical FR3 systems: equal geometry counts on each side.
    assert len(left) == len(right)


# ---------------------------------------------------------------------------
# Static ACM
# ---------------------------------------------------------------------------


def test_static_acm_records_robot_allowed_pairs(tmp_path: Path):
    """allowed_pairs on a robot CollisionSpec should land in static_allowed_pairs."""
    world_yaml = (
        "schema: dexsent.algorithms.world\n"
        "version: 2\n"
        "id: w\n"
        "world_frame: world\n"
        "robots:\n"
        f"  - id: arm\n"
        f"    robot_system: " + str((REPO_ROOT / "configs" / "robots" / "franka_fr3_with_franka_hand.yaml").resolve()) + "\n"
        "    namespace: null\n"
        "objects: []\n"
        "collision_matrix:\n"
        "  default_action: check\n"
        "  rules: []\n"
    )
    # This world re-uses the franka YAML which declares adjacent-link
    # pairs (link_i, link_{i+1}) as allowed by mechanical design. Verify
    # those land in the static ACM exactly as written.
    p = tmp_path / "world.yaml"
    p.write_text(world_yaml)
    world = WorldDescription.from_yaml(p)
    cm = CollisionModel.from_world(world)

    expected = frozenset(
        tuple(sorted((f"fr3_link{i}_0", f"fr3_link{i + 1}_0")))
        for i in range(7)
    )
    assert cm.static_allowed_pairs == expected


def test_world_acm_allow_rule_lands_in_static_allowed_pairs(tmp_path: Path):
    yaml_text = (
        "schema: dexsent.algorithms.world\n"
        "version: 2\n"
        "id: w\n"
        "world_frame: world\n"
        "robots:\n"
        "  - id: arm\n"
        "    robot_system: " + str((REPO_ROOT / "configs" / "robots" / "franka_fr3_with_franka_hand.yaml").resolve()) + "\n"
        "    namespace: null\n"
        "objects: []\n"
        "collision_matrix:\n"
        "  default_action: check\n"
        "  rules:\n"
        "    - { a: floor, b: wall, action: allow, reason: \"corner\" }\n"
        "    - { a: ceiling, b: pipe, action: check }\n"
    )
    p = tmp_path / "world.yaml"
    p.write_text(yaml_text)
    world = WorldDescription.from_yaml(p)
    cm = CollisionModel.from_world(world)

    assert cm.is_statically_allowed("floor", "wall")
    assert cm.is_statically_allowed("wall", "floor")  # direction-agnostic
    assert not cm.is_statically_allowed("ceiling", "pipe")
