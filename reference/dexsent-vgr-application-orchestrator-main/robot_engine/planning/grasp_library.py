from __future__ import annotations

from typing import Iterable, List

from robot_engine.interfaces.schemas import GraspCandidate


class GraspLibrary:
    def __init__(self, candidates: Iterable[GraspCandidate] = ()):
        self.candidates = {candidate.grasp_id: candidate for candidate in candidates}

    def add(self, candidate: GraspCandidate) -> None:
        self.candidates[candidate.grasp_id] = candidate

    def list(self) -> List[GraspCandidate]:
        return sorted(self.candidates.values(), key=lambda item: item.score, reverse=True)
