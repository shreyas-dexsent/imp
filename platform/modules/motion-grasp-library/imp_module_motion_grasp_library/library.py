"""In-memory grasp catalogue: ``Grasp`` records keyed by id, ranked by score.

Mirrors the VGR orchestrator's ``robot_engine/planning/grasp_library.py``
shape (see the README for the migration source) and the spec's ``Grasp`` /
``Grasps`` wire schemas. A grasp's ``t_obj_gripper`` is
the gripper pose in the *object* frame at the moment of grasping -- the world
pose is computed on the fly by ``synthesize_grasps`` using the live
``T_world_object``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np


def _to_4x4(matrix) -> np.ndarray:
    """Accept either a nested 4x4 list or a flat 16-float row-major sequence."""
    arr = np.asarray(matrix, dtype=float)
    if arr.shape == (16,):
        arr = arr.reshape(4, 4)
    if arr.shape != (4, 4):
        raise ValueError(f"t_obj_gripper must be 4x4 or len-16; got shape {arr.shape}")
    if not np.allclose(arr[3], (0.0, 0.0, 0.0, 1.0)):
        raise ValueError(f"t_obj_gripper bottom row must be [0,0,0,1]; got {arr[3].tolist()}")
    return arr


@dataclass
class Grasp:
    """A single grasp candidate.

    ``t_obj_gripper`` is a 4x4 homogeneous transform: gripper pose expressed
    in the object frame. ``score`` is in [0, 1]; higher = better.
    """

    grasp_id: str
    score: float
    t_obj_gripper: np.ndarray = field(repr=False)

    def __post_init__(self):
        self.t_obj_gripper = _to_4x4(self.t_obj_gripper)
        if not (0.0 <= self.score <= 1.0):
            raise ValueError(f"score must be in [0, 1]; got {self.score}")


@dataclass
class GraspLibrary:
    """Keyed store of grasps for a single object id.

    Construct from an iterable of ``Grasp`` or load from a JSON file:

        {
            "object_id": "matka",
            "grasps": [
                {"grasp_id": "g1", "score": 0.9, "t_obj_gripper": [[..4..], ...]}
            ]
        }
    """

    object_id: str
    _by_id: dict = field(default_factory=dict, repr=False)

    @classmethod
    def from_iterable(cls, object_id: str, candidates: Iterable[Grasp]) -> "GraspLibrary":
        lib = cls(object_id=object_id)
        for g in candidates:
            lib.add(g)
        return lib

    @classmethod
    def from_json(cls, path: str | Path) -> "GraspLibrary":
        payload = json.loads(Path(path).read_text())
        object_id = payload["object_id"]
        grasps = [
            Grasp(
                grasp_id=str(item["grasp_id"]),
                score=float(item["score"]),
                t_obj_gripper=item["t_obj_gripper"],
            )
            for item in payload.get("grasps", [])
        ]
        return cls.from_iterable(object_id, grasps)

    def add(self, grasp: Grasp) -> None:
        self._by_id[grasp.grasp_id] = grasp

    def get(self, grasp_id: str) -> Grasp:
        return self._by_id[grasp_id]

    def list(self) -> List[Grasp]:
        """Candidates sorted by score descending (highest-quality first)."""
        return sorted(self._by_id.values(), key=lambda g: g.score, reverse=True)

    def __len__(self) -> int:
        return len(self._by_id)
