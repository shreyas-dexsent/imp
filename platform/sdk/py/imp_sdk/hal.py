"""HAL device framework for Python drivers (spec §8).

A device subclasses :class:`HalDevice`, declares its publish/subscribe topics,
and implements lifecycle + ``read``/``on_command``. :class:`HalNode` runs it:
opens the bus, declares the topics, rate-schedules publications, dispatches
commands on background threads, and emits a heartbeat on the ctrl plane.

Mirrors crates/hal-contract so Rust and Python HAL nodes expose the same surface.
"""

from __future__ import annotations

import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

from . import keyexpr
from .bus import Bus, QosClass
from .schemas import imp_pb2


class Lifecycle(Enum):
    UNCONFIGURED = "unconfigured"
    INACTIVE = "inactive"
    ACTIVE = "active"
    FAULTED = "faulted"


@dataclass
class Pub:
    """A published topic: ``hal/<device>/<signal>`` at ``rate_hz``."""
    signal: str
    msg_type: type
    qos: QosClass
    rate_hz: float


@dataclass
class Sub:
    """A subscribed topic: ``hal/<device>/<signal>``."""
    signal: str
    msg_type: type
    qos: QosClass


class HalDevice(ABC):
    """A hardware device exposed as a node (spec §8)."""

    kind: str = "device"

    def publishes(self) -> List[Pub]:
        return []

    def subscribes(self) -> List[Sub]:
        return []

    # Lifecycle (driven by HalNode / the Supervisor).
    def configure(self) -> None:
        """Acquire the vendor SDK / open the device. Unconfigured -> Inactive."""

    def activate(self) -> None:
        """Begin the device loop. Inactive -> Active."""

    def deactivate(self) -> None:
        """Stop and fail safe. Active -> Inactive."""

    @abstractmethod
    def read(self, signal: str):
        """Return the next protobuf message for a published signal (or None)."""

    def on_command(self, signal: str, msg) -> None:
        """Handle a message on a subscribed signal."""


class HalNode:
    """Runs a :class:`HalDevice`: bus wiring + lifecycle + rate scheduling."""

    def __init__(self, device: HalDevice, station: str, device_id: str):
        self.device = device
        self.station = station
        self.device_id = device_id
        self.state = Lifecycle.UNCONFIGURED
        self._stop = threading.Event()
        self._threads: List[threading.Thread] = []

    def run(self) -> None:
        bus = Bus.open()
        try:
            self.device.configure()
            self.state = Lifecycle.INACTIVE

            publishers = {}
            for p in self.device.publishes():
                key = keyexpr.hal(self.station, self.device_id, p.signal)
                publishers[p.signal] = (bus.publisher(key, p.qos), p)

            for s in self.device.subscribes():
                key = keyexpr.hal(self.station, self.device_id, s.signal)
                t = threading.Thread(target=self._sub_loop, args=(bus, key, s), daemon=True)
                t.start()
                self._threads.append(t)

            self.device.activate()
            self.state = Lifecycle.ACTIVE
            hb_key = keyexpr.ctrl(self.station, self.device_id, "heartbeat")
            print(f"[{self.device_id}] active ({self.device.kind}); publishing "
                  f"{[p.signal for p in self.device.publishes()]}", flush=True)

            # Rate scheduler: publish each signal when due; heartbeat at 1 Hz.
            next_due = {sig: 0.0 for sig in publishers}
            next_hb = 0.0
            seq = 0
            while not self._stop.is_set():
                now = time.monotonic()
                for sig, (pub, spec) in publishers.items():
                    if now >= next_due[sig]:
                        msg = self.device.read(sig)
                        if msg is not None:
                            pub.put(msg)
                        next_due[sig] = now + 1.0 / spec.rate_hz
                if now >= next_hb:
                    bus.put(hb_key, imp_pb2.Scalar(value=float(seq)), QosClass.TELEMETRY)
                    seq += 1
                    next_hb = now + 1.0
                time.sleep(0.001)
        finally:
            self._stop.set()
            try:
                self.device.deactivate()
            finally:
                self.state = Lifecycle.INACTIVE
                bus.close()

    def stop(self) -> None:
        self._stop.set()

    def _sub_loop(self, bus: Bus, key: str, spec: Sub) -> None:
        sub = bus.subscribe(key, spec.msg_type)
        while not self._stop.is_set():
            try:
                msg = sub.recv()
            except Exception:
                if self._stop.is_set():
                    return
                raise
            self.device.on_command(spec.signal, msg)


def run_device(device: HalDevice, station: str, device_id: str) -> None:
    """Run a device until interrupted (Ctrl-C)."""
    node = HalNode(device, station, device_id)
    try:
        node.run()
    except KeyboardInterrupt:
        node.stop()
