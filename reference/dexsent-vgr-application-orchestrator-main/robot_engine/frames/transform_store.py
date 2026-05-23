from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Tuple

from robot_engine.interfaces.schemas import Transform3D


@dataclass
class TransformRecord:
    transform: Transform3D
    version: int
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    source: str = "unknown"


class TransformStore:
    def __init__(self):
        self._records: Dict[Tuple[str, str], List[TransformRecord]] = {}

    def put(self, transform: Transform3D, source: str = "unknown") -> TransformRecord:
        key = (transform.parent_frame, transform.child_frame)
        version = len(self._records.get(key, [])) + 1
        record = TransformRecord(transform=transform, version=version, source=source)
        self._records.setdefault(key, []).append(record)
        return record

    def latest(self, parent_frame: str, child_frame: str) -> TransformRecord:
        records = self._records.get((parent_frame, child_frame), [])
        if not records:
            raise KeyError(f"Missing transform {parent_frame}->{child_frame}")
        return records[-1]

    def history(self, parent_frame: str, child_frame: str) -> List[TransformRecord]:
        return list(self._records.get((parent_frame, child_frame), []))
