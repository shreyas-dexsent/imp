from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from robot_engine.core_math.transforms import compose_transform, invert_transform, validate_transform
from robot_engine.interfaces.schemas import AlgorithmError, Transform3D


@dataclass
class FrameEdge:
    parent: str
    child: str
    transform: Transform3D


class FrameGraph:
    def __init__(self, root_frame: str | None = None):
        self._edges: Dict[Tuple[str, str], Transform3D] = {}
        self._frames: set[str] = set()
        if root_frame:
            self._frames.add(root_frame)

    def add_frame(self, frame_id: str, parent_frame_id: str | None = None, transform: Transform3D | None = None):
        self._frames.add(frame_id)
        if parent_frame_id is not None:
            if transform is None:
                raise ValueError("transform is required when parent_frame_id is provided")
            self.update_transform(parent_frame_id, frame_id, transform)

    def update_transform(self, parent_frame_id: str, child_frame_id: str, transform: Transform3D):
        if transform.parent_frame != parent_frame_id or transform.child_frame != child_frame_id:
            raise ValueError("transform frame labels must match parent/child ids")
        validate_transform(transform.matrix)
        self._frames.update([parent_frame_id, child_frame_id])
        self._edges[(parent_frame_id, child_frame_id)] = transform
        if self.detect_cycles():
            del self._edges[(parent_frame_id, child_frame_id)]
            raise ValueError("frame graph cycle detected")

    def get_transform(self, parent_frame_id: str, child_frame_id: str) -> Transform3D:
        if parent_frame_id == child_frame_id:
            return Transform3D(parent_frame=parent_frame_id, child_frame=child_frame_id, matrix=np.eye(4).tolist())
        chain = self.get_chain(parent_frame_id, child_frame_id)
        if not chain:
            raise KeyError(f"No transform chain from {parent_frame_id} to {child_frame_id}")
        T = np.eye(4)
        for edge_parent, edge_child, forward in chain:
            edge = self._edges[(edge_parent, edge_child)]
            mat = np.asarray(edge.matrix, dtype=float)
            T = compose_transform(T, mat if forward else invert_transform(mat))
        return Transform3D(parent_frame=parent_frame_id, child_frame=child_frame_id, matrix=T.tolist())

    def has_frame(self, frame_id: str) -> bool:
        return frame_id in self._frames

    def remove_frame(self, frame_id: str):
        self._frames.discard(frame_id)
        self._edges = {k: v for k, v in self._edges.items() if frame_id not in k}

    def validate_tree(self):
        return not self.detect_cycles()

    def detect_cycles(self):
        visited = set()
        stack = set()
        children = {}
        for parent, child in self._edges:
            children.setdefault(parent, []).append(child)

        def visit(node):
            if node in stack:
                return True
            if node in visited:
                return False
            visited.add(node)
            stack.add(node)
            for child in children.get(node, []):
                if visit(child):
                    return True
            stack.remove(node)
            return False

        return any(visit(frame) for frame in list(self._frames))

    def list_frames(self) -> List[str]:
        return sorted(self._frames)

    def get_chain(self, parent_frame_id: str, child_frame_id: str):
        if parent_frame_id not in self._frames or child_frame_id not in self._frames:
            return []
        adjacency = {}
        for parent, child in self._edges:
            adjacency.setdefault(parent, []).append((parent, child, True))
            adjacency.setdefault(child, []).append((parent, child, False))
        queue = deque([(parent_frame_id, [])])
        seen = {parent_frame_id}
        while queue:
            node, path = queue.popleft()
            if node == child_frame_id:
                return path
            for edge in adjacency.get(node, []):
                nxt = edge[1] if edge[2] else edge[0]
                if nxt not in seen:
                    seen.add(nxt)
                    queue.append((nxt, path + [edge]))
        return []
