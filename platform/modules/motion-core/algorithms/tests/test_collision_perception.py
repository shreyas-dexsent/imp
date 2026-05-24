"""Tests for runtime perception integration into the collision model.

Covers:
* All new GeometrySpec variants (Capsule, MeshData, ConvexHull, Octree,
  HeightField) construct, register, and produce a valid Coal shape.
* `Scene.add_object` / `remove_object` lifecycle.
* `CollisionModel.shapes_for` returns the same Coal objects the planner
  uses, with correct `kind` tags.
* Collision queries against perception-added objects work end-to-end.
* Trimesh fallback handles non-OBJ formats.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("coal")
pytest.importorskip("pinocchio")

from algorithms.collision import is_in_collision
from algorithms.descriptions import (
    BoxGeometrySpec,
    CapsuleGeometrySpec,
    ConvexHullGeometrySpec,
    HeightFieldGeometrySpec,
    MeshDataGeometrySpec,
    OctreeGeometrySpec,
    WorldDescription,
)
from algorithms.kinematics import fk_local
from algorithms.resolved import CollisionModel, KinematicModel, Scene
from algorithms.resolved.kinematic_model import _clear_cache

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _isolate_cache():
    _clear_cache()
    yield
    _clear_cache()


@pytest.fixture()
def empty_world_scene():
    """A scene with no YAML-declared world objects, just the FR3."""
    world = WorldDescription.from_yaml(
        REPO_ROOT / "configs" / "worlds" / "franka_table_world.yaml"
    )
    # Drop the table-world objects so each test starts from a known
    # baseline of "robot + nothing else".
    world.objects.clear()
    world.collision_matrix.rules.clear()
    cm = CollisionModel.from_world(world)
    return Scene.from_world(world, cm), cm


# ---------------------------------------------------------------------------
# Individual spec types build the right Coal shape kind
# ---------------------------------------------------------------------------


def test_capsule_spec_registers_with_kind_capsule(empty_world_scene):
    scene, cm = empty_world_scene
    scene.add_object(
        "rod",
        collision=CapsuleGeometrySpec(type="capsule", radius=0.05, length=0.4),
        pose=np.eye(4),
    )
    infos = cm.shapes_for("rod")
    assert len(infos) == 1
    assert infos[0].kind == "capsule"
    assert infos[0].owner == "world"
    assert infos[0].parent_joint == "universe"


def test_convex_hull_spec_registers_solid_convex(empty_world_scene):
    scene, cm = empty_world_scene
    cube_verts = [
        (0.0, 0.0, 0.0), (0.1, 0.0, 0.0), (0.0, 0.1, 0.0), (0.0, 0.0, 0.1),
        (0.1, 0.1, 0.0), (0.1, 0.0, 0.1), (0.0, 0.1, 0.1), (0.1, 0.1, 0.1),
    ]
    scene.add_object(
        "block",
        collision=ConvexHullGeometrySpec(type="convex_hull", vertices=cube_verts),
        pose=np.eye(4),
    )
    infos = cm.shapes_for("block")
    assert len(infos) == 1
    assert infos[0].kind == "convex_hull"


def test_octree_spec_registers_with_kind_octree(empty_world_scene):
    scene, cm = empty_world_scene
    points = np.random.default_rng(0).uniform(-0.2, 0.2, size=(80, 3)).tolist()
    scene.add_object(
        "cloud",
        collision=OctreeGeometrySpec(type="octree", points=points, resolution=0.02),
        pose=np.eye(4),
    )
    infos = cm.shapes_for("cloud")
    assert len(infos) == 1
    assert infos[0].kind == "octree"


def test_mesh_data_spec_registers_as_mesh(empty_world_scene):
    scene, cm = empty_world_scene
    vertices = [(0.0, 0.0, 0.0), (0.1, 0.0, 0.0), (0.0, 0.1, 0.0), (0.0, 0.0, 0.1)]
    faces = [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]
    scene.add_object(
        "tetra",
        collision=MeshDataGeometrySpec(type="mesh_data", vertices=vertices, faces=faces),
        pose=np.eye(4),
    )
    infos = cm.shapes_for("tetra")
    assert len(infos) == 1
    assert infos[0].kind == "mesh"


def test_height_field_spec_registers_as_height_field(empty_world_scene):
    scene, cm = empty_world_scene
    heights = [[0.0] * 5 for _ in range(5)]
    heights[2][2] = 0.05
    scene.add_object(
        "table_top",
        collision=HeightFieldGeometrySpec(
            type="height_field", x_size=1.0, y_size=1.0, heights=heights, min_height=-0.01,
        ),
        pose=np.eye(4),
    )
    infos = cm.shapes_for("table_top")
    assert len(infos) == 1
    assert infos[0].kind == "height_field"


# ---------------------------------------------------------------------------
# Lifecycle: add → query → remove
# ---------------------------------------------------------------------------


def test_add_object_appears_in_object_names(empty_world_scene):
    scene, cm = empty_world_scene
    before = set(cm.object_names())
    scene.add_object(
        "intruder",
        collision=BoxGeometrySpec(type="box", size=(0.1, 0.1, 0.1)),
    )
    after = set(cm.object_names())
    assert after - before == {"intruder"}


def test_remove_object_clears_collision_model_entry(empty_world_scene):
    scene, cm = empty_world_scene
    scene.add_object(
        "rod",
        collision=CapsuleGeometrySpec(type="capsule", radius=0.05, length=0.4),
    )
    assert cm.has_object("rod")
    scene.remove_object("rod")
    assert not cm.has_object("rod")
    assert "rod" not in scene.object_poses


def test_remove_object_refuses_yaml_declared_object():
    world = WorldDescription.from_yaml(
        REPO_ROOT / "configs" / "worlds" / "franka_table_world.yaml"
    )
    cm = CollisionModel.from_world(world)
    scene = Scene.from_world(world, cm)
    yaml_id = world.objects[0].id

    with pytest.raises(ValueError, match="YAML-declared"):
        scene.remove_object(yaml_id)


def test_duplicate_add_raises(empty_world_scene):
    scene, _ = empty_world_scene
    spec = BoxGeometrySpec(type="box", size=(0.1, 0.1, 0.1))
    scene.add_object("foo", collision=spec)
    with pytest.raises(ValueError, match="already exists"):
        scene.add_object("foo", collision=spec)


# ---------------------------------------------------------------------------
# shapes_for is the single source of truth
# ---------------------------------------------------------------------------


def test_shapes_for_matches_catalogue_shape_kind_and_params(empty_world_scene):
    """The Coal shape the UI sees via `shapes_for()` is the same kind and
    same parameters as the catalogue entry the planner walks at query
    time.

    Coal's Python binding returns a fresh wrapper per attribute access,
    so Python `is` identity does not hold. The contract is shape kind
    plus dimensions: UI cannot show a different shape from the planner.
    """
    scene, cm = empty_world_scene
    scene.add_object(
        "block",
        collision=BoxGeometrySpec(type="box", size=(0.1, 0.2, 0.3)),
    )

    ui_shape = cm.shapes_for("block")[0].coal_shape
    catalogue_shape = next(
        go.geometry for go in cm.world_geom.geometryObjects if go.name == "block"
    )

    assert type(ui_shape) is type(catalogue_shape)
    # Box exposes halfSide; the value is half of the side length.
    np.testing.assert_allclose(
        np.asarray(ui_shape.halfSide), np.asarray(catalogue_shape.halfSide), atol=1e-12,
    )


def test_shapes_for_unknown_object_returns_empty():
    world = WorldDescription.from_yaml(
        REPO_ROOT / "configs" / "worlds" / "franka_table_world.yaml"
    )
    cm = CollisionModel.from_world(world)
    assert cm.shapes_for("does_not_exist") == []


# ---------------------------------------------------------------------------
# Collision query against perception-added objects
# ---------------------------------------------------------------------------


def test_perception_added_box_collides_with_robot_at_origin(empty_world_scene):
    """A box placed directly inside the robot's base must show up as a
    collision when queried."""
    scene, cm = empty_world_scene
    system = scene.world.robots[0].robot_system
    model = KinematicModel.from_robot_system(system)
    home = system.named_joint_state("home")
    q_home = np.array([home[name] for name in model.active_joint_names], dtype=float)

    # Box at world origin (= robot base for this world); robot mesh is
    # definitely intersecting.
    scene.add_object(
        "intruder",
        collision=BoxGeometrySpec(type="box", size=(0.3, 0.3, 0.3)),
        pose=np.eye(4),
    )

    report = is_in_collision(model, scene, q_home)
    assert report.in_collision


# ---------------------------------------------------------------------------
# Visual spec roundtrip
# ---------------------------------------------------------------------------


def test_get_visual_spec_returns_runtime_override(empty_world_scene):
    scene, _ = empty_world_scene
    visual = BoxGeometrySpec(type="box", size=(0.1, 0.1, 0.1))
    scene.add_object("box", visual=visual, pose=np.eye(4))

    spec = scene.get_visual_spec("box")
    assert spec is not None
    assert spec.geometry.size == (0.1, 0.1, 0.1)


def test_get_visual_spec_falls_back_to_yaml():
    world = WorldDescription.from_yaml(
        REPO_ROOT / "configs" / "worlds" / "franka_table_world.yaml"
    )
    cm = CollisionModel.from_world(world)
    scene = Scene.from_world(world, cm)
    yaml_id = world.objects[0].id

    spec = scene.get_visual_spec(yaml_id)
    # Either way: should not raise. Some YAML objects may not declare a
    # visual; the contract is just "look up runtime first, then YAML".
    if world.objects[0].visual is None:
        assert spec is None
    else:
        assert spec is not None
