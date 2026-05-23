from __future__ import annotations

from abc import ABC, abstractmethod

from robot_engine.interfaces.schemas import IKRequest, IKResult


class IKBackend(ABC):
    backend_name = "BASE"

    def supports(self, robot_model) -> bool:
        return True

    @abstractmethod
    def solve(self, request: IKRequest) -> IKResult:
        raise NotImplementedError

