"""Functional-module framework for Python modules (spec §9).

A module subclasses :class:`Module`, declares typed input/output ports, and
implements ``configure`` + ``compute``. :class:`ModuleNode` is the Compute
Runtime: it subscribes the inputs, keeps the latest of each, calls ``compute``
when all inputs are present, validates schemas, and publishes the outputs.

For motion modules the resolved ``Scene`` is built in ``configure`` and filled
from the latest inputs inside ``compute`` before the stateless op is called
(spec §9).

Mirrors crates/module-contract so Rust and Python modules expose the same shape.
"""

from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional

from .bus import Bus, QosClass


@dataclass
class Input:
    """A subscribed port: a name + the full key + the expected message type."""
    name: str
    key: str
    msg_type: type


@dataclass
class Output:
    """A published port: a name + the full key + message type + QoS."""
    name: str
    key: str
    msg_type: type
    qos: QosClass


class Module(ABC):
    name: str = "module"

    def inputs(self) -> List[Input]:
        return []

    def outputs(self) -> List[Output]:
        return []

    def configure(self) -> None:
        """Build resolved models / load assets once before running."""

    @abstractmethod
    def compute(self, latest: Dict[str, object]) -> Dict[str, object]:
        """Given the latest message per input name, return a message per output
        name (omit an output to skip publishing it this tick)."""


class ModuleNode:
    """Compute Runtime: subscribe -> validate -> call compute -> publish."""

    def __init__(self, module: Module):
        self.module = module
        self._stop = threading.Event()
        self._latest: Dict[str, object] = {}
        self._lock = threading.Lock()
        self._required = {i.name for i in module.inputs()}

    def run(self) -> None:
        bus = Bus.open()
        try:
            self.module.configure()
            publishers = {o.name: (bus.publisher(o.key, o.qos), o) for o in self.module.outputs()}
            threads = []
            for spec in self.module.inputs():
                t = threading.Thread(target=self._sub_loop, args=(bus, spec, publishers), daemon=True)
                t.start()
                threads.append(t)
            print(f"[{self.module.name}] running; in={[i.name for i in self.module.inputs()]} "
                  f"out={[o.name for o in self.module.outputs()]}", flush=True)
            self._stop.wait()
        finally:
            self._stop.set()
            bus.close()

    def stop(self) -> None:
        self._stop.set()

    def _sub_loop(self, bus: Bus, spec: Input, publishers) -> None:
        sub = bus.subscribe(spec.key, spec.msg_type)
        while not self._stop.is_set():
            try:
                msg = sub.recv()
            except Exception:
                if self._stop.is_set():
                    return
                raise
            with self._lock:
                self._latest[spec.name] = msg
                if not self._required.issubset(self._latest.keys()):
                    continue
                snapshot = dict(self._latest)
            outputs = self.module.compute(snapshot)
            for out_name, msg in (outputs or {}).items():
                if out_name in publishers and msg is not None:
                    publishers[out_name][0].put(msg)


def run_module(module: Module) -> None:
    node = ModuleNode(module)
    try:
        node.run()
    except KeyboardInterrupt:
        node.stop()
