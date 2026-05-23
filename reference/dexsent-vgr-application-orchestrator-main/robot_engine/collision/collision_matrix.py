from __future__ import annotations

from itertools import combinations
from typing import Iterable, List, Tuple

from robot_engine.interfaces.schemas import CollisionMatrix as CollisionMatrixSchema, CollisionPairRule


class CollisionMatrix:
    def __init__(self, schema: CollisionMatrixSchema | None = None):
        self.schema = schema or CollisionMatrixSchema()
        self._rules = {}
        for rule in self.schema.rules:
            self._rules[self._key(rule.object_a, rule.object_b)] = rule

    def action_for(self, object_a: str, object_b: str) -> str:
        rule = self._rules.get(self._key(object_a, object_b))
        return rule.action if rule else self.schema.default_action

    def should_check(self, object_a: str, object_b: str) -> bool:
        return self.action_for(object_a, object_b) == "check"

    def active_pairs(self, object_ids: Iterable[str]) -> List[Tuple[str, str]]:
        return [(a, b) for a, b in combinations(sorted(object_ids), 2) if self.should_check(a, b)]

    def set_rule(self, rule: CollisionPairRule) -> None:
        self._rules[self._key(rule.object_a, rule.object_b)] = rule

    def add_rule(self, object_a_pattern: str, object_b_pattern: str, rule: str) -> None:
        self.set_rule(CollisionPairRule(object_a=object_a_pattern, object_b=object_b_pattern, action=rule.lower()))

    def is_pair_active(self, object_a: str, object_b: str) -> bool:
        return self.should_check(object_a, object_b)

    def get_active_pairs(self, objects: Iterable[str]) -> List[Tuple[str, str]]:
        return self.active_pairs(objects)

    def get_allowed_pairs(self) -> List[Tuple[str, str]]:
        return [key for key, rule in self._rules.items() if rule.action == "allow"]

    def get_ignored_pairs(self) -> List[Tuple[str, str]]:
        return [key for key, rule in self._rules.items() if rule.action == "ignore"]

    @classmethod
    def load_from_dict(cls, data: dict) -> "CollisionMatrix":
        return cls(CollisionMatrixSchema.model_validate(data))

    def export_to_dict(self) -> dict:
        return self.schema.model_dump()

    @staticmethod
    def _key(a: str, b: str):
        return tuple(sorted((a, b)))
