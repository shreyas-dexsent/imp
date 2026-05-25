"""Test helpers for bus-attached imp modules.

Promotes the ``verify_*.py`` shell-script pattern (start a module in a
shell, feed inputs from another shell, check the output) into a single
in-process Python helper that pytest can drive:

    with module_under_test(MyModule(...)) as h:
        h.publish("imp/devstation/perc/s1/pose", a_pose_msg)
        got = h.recv("imp/devstation/motion/transform/target", PoseTarget)

Bring-up + teardown of the bus session and the module thread are owned
here so tests stay focused on the message round-trip.
"""

from __future__ import annotations

import contextlib
import threading
import time
from typing import Any, Type

from .bus import Bus, QosClass
from .module import Module, ModuleNode


class _Harness:
    def __init__(self, bus: Bus, node: ModuleNode):
        self._bus = bus
        self._node = node

    def publish(self, key: str, msg: Any, qos: QosClass = QosClass.STATE) -> None:
        """Publish a message on the bus."""
        self._bus.put(key, msg, qos)

    def subscribe(self, key: str, msg_type: Type):
        """Subscribe and return a TypedSub. Use ``.recv()`` to block for one."""
        return self._bus.subscribe(key, msg_type)

    def recv(self, key: str, msg_type: Type, *, timeout_s: float = 5.0):
        """Subscribe and block for the next message on ``key``.

        Convenience wrapper for the common pattern. Raises ``TimeoutError``
        if no message arrives within ``timeout_s``.
        """
        sub = self.subscribe(key, msg_type)
        end = time.monotonic() + timeout_s
        while time.monotonic() < end:
            try:
                return sub.recv()  # blocks until a message lands
            except Exception:
                if not self._node._stop.is_set():
                    raise
                break
        raise TimeoutError(f"no message on {key!r} within {timeout_s}s")


@contextlib.contextmanager
def module_under_test(module: Module, *, settle_s: float = 0.4):
    """Spin ``module`` in a daemon ``ModuleNode`` thread for the test body.

    Yields a ``_Harness`` exposing ``publish`` / ``subscribe`` / ``recv``.
    ``settle_s`` gives Zenoh's discovery a moment to wire subscribers up
    before the test starts publishing (without this, the first publish
    can race the module's subscribe declaration).
    """
    node = ModuleNode(module)
    bus = Bus.open()
    thread = threading.Thread(target=node.run, daemon=True)
    thread.start()
    try:
        time.sleep(settle_s)
        yield _Harness(bus, node)
    finally:
        node.stop()
        bus.close()
        # Give the node a moment to unwind cleanly; don't block forever.
        thread.join(timeout=2.0)
