"""In-process transform graph: add directed edges, look up composed transforms.

The graph is the abstraction the rest of imp uses to ask "where is frame X
expressed in frame Y?" without caring how the edges got there. Edges arrive
from many sources -- calibration (hand-eye, TCP), the workspace
(`base_pose`/`world` from `world.yaml`), and dynamic publishers -- and every
edge is stored both forward (parent->child) and inverse (child->parent), so
``lookup`` is a single BFS over a directed multigraph that always has a path
when one exists topologically.

Pure: no Zenoh, no threads, no I/O. ``TfModule`` (a Compute-Runtime wrapper)
subscribes ``imp/<station>/tf`` and feeds edges in via :meth:`add_edge`; any
other module can subscribe the same key and keep its own ``TfGraph`` instance.
"""

from __future__ import annotations

from collections import deque
from typing import Dict, FrozenSet, Iterable

import numpy as np


class TfLookupError(KeyError):
    """Raised when a lookup target frame is unreachable from the source."""


def _validate_matrix(matrix) -> np.ndarray:
    m = np.asarray(matrix, dtype=float)
    if m.shape != (4, 4):
        raise ValueError(f"transform must be (4, 4); got {m.shape}")
    if not np.all(np.isfinite(m)):
        raise ValueError("transform contains non-finite values")
    if not np.allclose(m[3], (0.0, 0.0, 0.0, 1.0)):
        raise ValueError(f"transform bottom row must be [0,0,0,1]; got {m[3].tolist()}")
    return m


class TfGraph:
    """Directed multigraph of 4x4 transforms keyed by (parent, child)."""

    def __init__(self) -> None:
        # _edges[u][v] = T_u_v (transform that maps points expressed in v into u).
        self._edges: Dict[str, Dict[str, np.ndarray]] = {}

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add_edge(self, parent: str, child: str, matrix) -> None:
        """Add or replace the edge ``parent -> child``.

        Convention: ``matrix`` is ``T_parent_child`` -- it places points
        expressed in the child frame into the parent frame:
        ``p_parent = T_parent_child @ p_child``. The inverse edge
        ``child -> parent`` is registered automatically.
        """
        if not parent or not child:
            raise ValueError("parent and child frame names must be non-empty")
        if parent == child:
            raise ValueError(f"parent and child must differ; got {parent!r}")
        m = _validate_matrix(matrix)
        self._edges.setdefault(parent, {})[child] = m
        self._edges.setdefault(child, {})[parent] = _inverse_se3(m)

    def remove_edge(self, parent: str, child: str) -> None:
        """Drop the edge ``parent -> child`` (and its inverse). No-op if absent."""
        self._edges.get(parent, {}).pop(child, None)
        self._edges.get(child, {}).pop(parent, None)

    def clear(self) -> None:
        self._edges.clear()

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def frames(self) -> FrozenSet[str]:
        """All frames known to the graph."""
        return frozenset(self._edges.keys())

    def has_edge(self, parent: str, child: str) -> bool:
        return child in self._edges.get(parent, {})

    def neighbours(self, frame: str) -> Iterable[str]:
        return iter(self._edges.get(frame, {}))

    def lookup(self, parent: str, child: str) -> np.ndarray:
        """Return ``T_parent_child``: the transform that maps points
        from ``child`` into ``parent`` (``p_parent = T_parent_child @ p_child``).

        Raises ``TfLookupError`` if no path exists.
        """
        if parent == child:
            return np.eye(4, dtype=float)
        if parent not in self._edges:
            raise TfLookupError(f"unknown frame {parent!r}")
        if child not in self._edges:
            raise TfLookupError(f"unknown frame {child!r}")

        # BFS from `parent`; carry the accumulated T_parent_<node> with each
        # frontier entry. First time we pop `child`, that accumulator is the answer.
        seen = {parent}
        queue = deque([(parent, np.eye(4, dtype=float))])
        while queue:
            node, T_root_node = queue.popleft()
            for neighbour, T_node_neighbour in self._edges.get(node, {}).items():
                if neighbour in seen:
                    continue
                T_root_neighbour = T_root_node @ T_node_neighbour
                if neighbour == child:
                    return T_root_neighbour
                seen.add(neighbour)
                queue.append((neighbour, T_root_neighbour))
        raise TfLookupError(f"no tf path from {parent!r} to {child!r}")


def _inverse_se3(T: np.ndarray) -> np.ndarray:
    """Closed-form inverse of a rigid-body 4x4 transform.

    Faster and more numerically stable than ``np.linalg.inv`` for SE(3):
    ``T = [[R, t], [0, 1]] -> T^{-1} = [[R^T, -R^T t], [0, 1]]``.
    """
    R = T[:3, :3]
    t = T[:3, 3]
    inv = np.eye(4, dtype=float)
    inv[:3, :3] = R.T
    inv[:3, 3] = -R.T @ t
    return inv
