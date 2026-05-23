# Assets

Assets are loaded with `trimesh` and converted into collision-ready geometry.

- `asset_loader.py`: detects and validates mesh/scene/point-cloud inputs.
- `mesh_converter.py`: repairs meshes, removes duplicate/degenerate data, and can build convex-hull proxies.
- `collision_geometry.py`: creates Coal geometry, with AABB metadata for broadphase checks.

Unsupported formats, empty meshes, non-finite vertices, and unsafe point-cloud inputs return structured errors instead of being accepted silently.
