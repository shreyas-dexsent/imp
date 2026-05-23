# frames

Coordinate frame management: a directed graph of SE(3) transforms with BFS chain resolution, versioned storage, and cycle detection.

---

## `frame_graph.py`

### `FrameEdge` (dataclass)

Stores one directed edge: `(parent, child, transform: Transform3D)`.

### `FrameGraph`

A directed acyclic graph where nodes are frame IDs and edges are rigid-body transforms `T_parent_child`.

#### `add_frame(frame_id, parent_frame_id, transform)`

Registers a frame and, if a parent is provided, inserts the edge via `update_transform`. The edge is immediately tested for cycles; it is removed and a `ValueError` is raised if a cycle is detected.

#### `update_transform(parent, child, transform: Transform3D)`

Validates:
1. `transform.parent_frame == parent` and `transform.child_frame == child`.
2. `validate_transform(transform.matrix)` — 4×4, finite, last row `[0,0,0,1]`, valid rotation.
3. No cycle introduced (DFS cycle check, see below).

#### `get_transform(parent, child) → Transform3D`

1. Identity if `parent == child`.
2. Finds the undirected path via BFS (`get_chain`).
3. Composes the chain:

```
T_result = I
for each edge (a, b, forward) in chain:
    if forward:  T_result = T_result @ T_ab
    else:        T_result = T_result @ T_ab⁻¹
```

Inversion uses `invert_transform` (efficient block form, not `np.linalg.inv`).

#### `get_chain(parent, child) → List[edge]`

BFS on the **undirected** adjacency (each stored edge `(a,b)` creates adjacency in both directions). Returns a list of `(edge_parent, edge_child, forward: bool)` tuples.

#### `detect_cycles() → bool`

DFS with a recursion stack (`visited` + `stack` sets). Returns `True` if any back-edge is found (i.e. a cycle exists).

#### `remove_frame(frame_id)`

Removes the frame node and all incident edges.

---

## `frame_registry.py`

### `FrameRegistry(FrameGraph)`

Thin subclass of `FrameGraph` providing a named registry interface. Currently no additional logic — serves as a semantic alias for the top-level frame store in the engine.

---

## `transform_store.py`

### `TransformRecord` (dataclass)

Immutable record of one transform update:

| Field | Type | Description |
|---|---|---|
| `transform` | `Transform3D` | The stored transform |
| `version` | `int` | Monotonically increasing per `(parent, child)` key |
| `timestamp` | `datetime` | UTC wall clock at insertion |
| `source` | `str` | String tag identifying who wrote this transform |

### `TransformStore`

Append-only versioned store keyed by `(parent_frame, child_frame)` tuples.

#### `put(transform, source) → TransformRecord`

Appends a new `TransformRecord`. Version = `len(existing_history) + 1`.

#### `latest(parent, child) → TransformRecord`

Returns the last record in the history list. Raises `KeyError` if the key does not exist.

#### `history(parent, child) → List[TransformRecord]`

Returns the full append history for a `(parent, child)` pair in insertion order.

**Note:** `TransformStore` is a pure log — it does not perform chain resolution. Use `FrameGraph.get_transform` for chained lookups.
