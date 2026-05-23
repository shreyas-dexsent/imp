from __future__ import annotations

import asyncio
import time
import uuid
from collections import defaultdict, deque
from typing import Any, Deque, Dict, Set

from pydantic import BaseModel, ConfigDict, Field


class BusMessage(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    schema_: str = Field(default="imp.message", alias="schema")
    version: int = 1
    message_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    correlation_id: str | None = None
    cell_id: str
    source: str = "orchestrator"
    type: str
    timestamp_ns: int = Field(default_factory=lambda: time.time_ns())
    payload: Dict[str, Any] = Field(default_factory=dict)


class ImpBus:
    """Small in-process event bus used by the FastAPI orchestrator.

    This is intentionally not ROS. It gives the browser topic-like WebSocket
    streams while keeping robot_engine and UI code decoupled.
    """

    def __init__(self, history_size: int = 100) -> None:
        self._subscribers: Dict[str, Set[asyncio.Queue[BusMessage]]] = defaultdict(set)
        self._history: Dict[str, Deque[BusMessage]] = defaultdict(lambda: deque(maxlen=history_size))
        self._lock = asyncio.Lock()

    async def publish(self, message: BusMessage) -> None:
        async with self._lock:
            subscribers = list(self._subscribers.get(message.cell_id, set()))
            self._history[message.cell_id].append(message)
        for queue in subscribers:
            try:
                queue.put_nowait(message)
            except asyncio.QueueFull:
                # Drop the oldest pending message for slow clients; live state
                # should stay current rather than back-pressure planning calls.
                try:
                    queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
                queue.put_nowait(message)

    async def subscribe(self, cell_id: str, replay: bool = True) -> asyncio.Queue[BusMessage]:
        queue: asyncio.Queue[BusMessage] = asyncio.Queue(maxsize=100)
        async with self._lock:
            self._subscribers[cell_id].add(queue)
            if replay:
                for message in self._history.get(cell_id, []):
                    queue.put_nowait(message)
        return queue

    async def unsubscribe(self, cell_id: str, queue: asyncio.Queue[BusMessage]) -> None:
        async with self._lock:
            self._subscribers.get(cell_id, set()).discard(queue)


_bus = ImpBus()


def get_bus() -> ImpBus:
    return _bus
