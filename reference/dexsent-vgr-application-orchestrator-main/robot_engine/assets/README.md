# assets

Mesh loading, geometry conversion, and collision shape construction.

---

## `asset_loader.py`

### `load_trimesh_asset(config: ObjectAssetConfig, simplify_faces: Optional[int]) → trimesh.Trimesh`

Entry point for loading any mesh asset. Steps:
1. Resolve path from `config.mesh_path`; raise `AssetLoadError(UNSUPPORTED_FORMAT)` if missing.
2. Call `trimesh.load(path)` then dispatch via `_as_mesh`.
3. Apply uniform scaling: `mesh.apply_scale(float(config.scale))`
4. Call `validate_mesh` to assert finite, non-empty geometry.
5. If `simplify_faces` is set and `len(mesh.faces) > simplify_faces`, run `simplify_mesh` then re-validate.

### `_as_mesh(loaded, point_cloud_mode) → trimesh.Trimesh`

Normalises whatever `trimesh.load` returns:

| Input type | Behaviour |
|---|---|
| `trimesh.Scene` | Concatenates all `Trimesh` sub-geometries |
| `trimesh.PointCloud` | Requires `point_cloud_mode == "convex_hull"`; calls `loaded.convex_hull`. If hull construction is unavailable, creates a bounding-box proxy centred at `center = (verts.max(0) + verts.min(0)) / 2` |
| `trimesh.Trimesh` | Pass-through |
| Other | Raise `AssetLoadError(UNSUPPORTED_FORMAT)` |

### `validate_mesh(mesh, frame_id) → None`

Asserts:
- `frame_id` non-empty
- `mesh.vertices` and `mesh.faces` non-empty
- `np.isfinite(mesh.vertices).all()` — no NaN/Inf coordinates
- `extents.max() > 0` — non-degenerate bounding box

### `simplify_mesh(mesh, target_faces) → trimesh.Trimesh`

Reduces face count to `target_faces`:
- Primary: `mesh.simplify_quadratic_decimation(target_faces)` — Garland-Heckbert quadric error minimisation
- If decimation is unavailable: uniform face sampling via `face_ids = np.linspace(0, len(faces)-1, target_faces, dtype=int)`

---

## `collision_geometry.py`

### `CollisionGeometry` (dataclass)

| Field | Type | Description |
|---|---|---|
| `geometry_id` | `str` | Unique name |
| `frame_id` | `str` | Coordinate frame the geometry lives in |
| `mesh` | `Optional[trimesh.Trimesh]` | Triangle mesh |
| `size_xyz` | `Optional[np.ndarray]` | Half-extents for box shapes |
| `coal_geometry` | `object` | Coal BVH, convex, primitive, or octree geometry |

**`aabb_bounds` property**

Returns a `(2, 3)` array `[[x_min, y_min, z_min], [x_max, y_max, z_max]]`:
- Mesh: `np.asarray(mesh.bounds)`
- Box: `±size_xyz / 2`

### `geometry_from_asset(config, simplify_faces) → CollisionGeometry`

Calls `load_trimesh_asset`, wraps in `CollisionGeometry`, and populates `coal_geometry` via Coal mesh conversion.

### `box_geometry(geometry_id, frame_id, size_xyz) → CollisionGeometry`

Validates `size_xyz` is finite and positive, creates `trimesh.creation.box(extents=size)`, and builds a Coal `Box` primitive.

---

## `mesh_converter.py`

Low-level bridge to Coal. All functions raise `RuntimeError` if Coal is not installed.

### `trimesh_to_coal_geometry(mesh) → coal.BVHModelOBBRSS`

Builds a Coal Bounding Volume Hierarchy from mesh data:
```
model = coal.BVHModelOBBRSS()
model.beginModel(num_vertices, num_faces)
model.addSubModel(vertices: float64, faces: int32)
model.endModel()
```
The BVH accelerates narrow-phase GJK/EPA collision queries.

### `trimesh_to_solid_coal_geometry(mesh)`

Uses a Coal convex hull for watertight convex meshes so boxes, walls, and other
closed convex obstacles behave as solid volumes instead of hollow triangle
shells. Non-convex meshes remain BVH geometry and should be decomposed or
approximated with primitives if solid occupancy is required.

### `box_to_coal_geometry(size_xyz) → coal.Box`

Creates an axis-aligned box primitive: `coal.Box(x, y, z)`.

### `matrix_to_coal_transform(matrix: np.ndarray) → coal.Transform3s`

Decomposes a 4x4 homogeneous matrix into rotation `R = matrix[:3,:3]` and translation `t = matrix[:3,3]` for Coal pose attachment.
