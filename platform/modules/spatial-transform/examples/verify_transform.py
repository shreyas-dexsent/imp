"""Round-trip check: publish a tf edge and a perception Pose6D, observe the
PoseTarget the module emits on imp/<station>/motion/transform/target.

Run (the module is launched in-process):

    python examples/verify_transform.py
"""

from __future__ import annotations

import sys
import threading
import time

import numpy as np
from scipy.spatial.transform import Rotation

from imp_sdk import Bus, QosClass, keyexpr
from imp_sdk.module import ModuleNode
from imp_sdk.schemas import imp_pb2

from imp_module_spatial_transform import TransformModule

STATION = "devstation"
POSE_KEY = f"imp/{STATION}/perc/s1/pose"


def _tf_edge(parent: str, child: str, T: np.ndarray) -> imp_pb2.TfEdge:
    return imp_pb2.TfEdge(
        header=imp_pb2.Header(schema="imp.TfEdge/1"),
        parent_frame=parent,
        child_frame=child,
        matrix=T.flatten().tolist(),
    )


def main() -> int:
    T_base_cam = np.eye(4)
    T_base_cam[:3, :3] = Rotation.from_euler("z", np.pi / 2).as_matrix()
    T_base_cam[:3, 3] = (0.2, 0.0, 0.5)

    module = TransformModule(
        station=STATION,
        pose_key=POSE_KEY,
        base_frame="base",
    )
    node = ModuleNode(module)
    threading.Thread(target=node.run, daemon=True).start()
    time.sleep(0.4)

    bus = Bus.open()
    sub = bus.subscribe(keyexpr.motion(STATION, "transform", "target"), imp_pb2.PoseTarget)
    try:
        bus.put(keyexpr.tf(STATION), _tf_edge("base", "camera", T_base_cam), QosClass.STATE)
        time.sleep(0.2)
        bus.put(
            POSE_KEY,
            imp_pb2.Pose6D(
                header=imp_pb2.Header(schema="imp.Pose6D/1", frame_id="camera"),
                position_m=[0.4, 0.0, 0.0],
                quat_xyzw=[0, 0, 0, 1],
                confidence=1.0,
                valid=True,
            ),
            QosClass.STATE,
        )
        got = sub.recv()
    finally:
        node.stop()
        bus.close()

    want = (0.2, 0.4, 0.5)
    err = float(np.linalg.norm(np.array(got.position_m) - np.array(want)))
    print(f"want={want}  got={tuple(got.position_m)}  err={err:.2e}  frame={got.target_frame}")
    ok = err < 1e-9 and got.target_frame == "base"
    print("RESULT:", "OK" if ok else "MISMATCH")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
