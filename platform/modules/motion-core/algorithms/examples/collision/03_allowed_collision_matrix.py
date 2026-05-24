"""
Inspect static allowed-collision pairs.

Run from the algorithms directory:

    python examples/collision/03_allowed_collision_matrix.py
"""
from __future__ import annotations

from pathlib import Path

from algorithms.descriptions import WorldDescription
from algorithms.resolved import CollisionModel


REPO_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    # 1. Load a world description.
    world_path = REPO_ROOT / "configs" / "worlds" / "franka_table_world.yaml"
    world = WorldDescription.from_yaml(world_path)

    # 2. Build the static collision catalogue.
    collision_model = CollisionModel.from_world(world)

    # 3. Print static allowed pairs.
    if not collision_model.static_allowed_pairs:
        print("No static allowed pairs are declared in this world.")
        return

    for a, b in sorted(collision_model.static_allowed_pairs):
        owner_a = collision_model.object_owner.get(a, "unknown")
        owner_b = collision_model.object_owner.get(b, "unknown")
        origin = "within-robot" if owner_a == owner_b == "robot" else "world-level"
        print(f"{a} <-> {b} ({origin})")


if __name__ == "__main__":
    main()
