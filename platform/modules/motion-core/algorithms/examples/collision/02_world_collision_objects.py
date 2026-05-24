"""
Inspect world-object collision geometry declared in YAML.

Run from the algorithms directory:

    python examples/collision/02_world_collision_objects.py
"""
from __future__ import annotations

from pathlib import Path

from algorithms.descriptions import MeshGeometrySpec, WorldDescription
from algorithms.resolved import CollisionModel


REPO_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    # 1. Load a world with one mesh workpiece.
    world_path = REPO_ROOT / "configs" / "worlds" / "franka_table_world.yaml"
    world = WorldDescription.from_yaml(world_path)

    # 2. Build the CollisionModel.
    # Mesh collision geometry is loaded eagerly during this call.
    collision_model = CollisionModel.from_world(world)

    # 3. Print YAML-side geometry settings.
    for world_object in world.objects:
        if world_object.collision is None:
            continue

        geometry = world_object.collision.geometry
        processing = world_object.collision.processing

        print("object:", world_object.id)
        print("  type:", geometry.type)
        if isinstance(geometry, MeshGeometrySpec):
            print("  mesh path:", world.resolve_path(geometry.path))
            print("  scale:", geometry.scale)
        print("  processing:", processing.model_dump() if processing else None)

    # 4. Print resolved Coal geometry object types.
    for geometry_object in collision_model.world_geom.geometryObjects:
        print("resolved:", geometry_object.name, type(geometry_object.geometry).__name__)


if __name__ == "__main__":
    main()
