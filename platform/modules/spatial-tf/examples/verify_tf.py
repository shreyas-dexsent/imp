"""Round-trip check: publish a couple of TfEdge messages, run the module
in-process for a moment, then verify it composed them correctly.

Run:

    python examples/verify_tf.py
"""

from __future__ import annotations

import sys
import threading
import time

import numpy as np

from imp_sdk import Bus, QosClass, keyexpr
from imp_sdk.schemas import imp_pb2

from imp_module_spatial_tf import TfGraph, TfModule
from imp_sdk.module import ModuleNode

STATION = "devstation"


def _edge(parent: str, child: str, T: np.ndarray) -> imp_pb2.TfEdge:
    return imp_pb2.TfEdge(
        header=imp_pb2.Header(schema="imp.TfEdge/1"),
        parent_frame=parent,
        child_frame=child,
        matrix=T.flatten().tolist(),
    )


def main() -> int:
    T_wb = np.eye(4)
    T_wb[:3, 3] = (0.5, 0.0, 1.0)
    T_bt = np.eye(4)
    T_bt[:3, 3] = (0.1, 0.0, 0.3)
    want = T_wb @ T_bt

    module = TfModule(station=STATION)
    node = ModuleNode(module)
    t = threading.Thread(target=node.run, daemon=True)
    t.start()
    time.sleep(0.4)  # let the subscribe declarations settle

    bus = Bus.open()
    try:
        tf_key = keyexpr.tf(STATION)
        bus.put(tf_key, _edge("world", "base", T_wb), QosClass.STATE)
        bus.put(tf_key, _edge("base", "tcp", T_bt), QosClass.STATE)
        time.sleep(0.4)
    finally:
        bus.close()

    got = module.graph.lookup("world", "tcp")
    err = float(np.linalg.norm(got - want))
    print(f"want=\n{want}\ngot=\n{got}\nerr={err:.2e}")
    node.stop()
    ok = err < 1e-9
    print("RESULT:", "OK" if ok else "MISMATCH")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
