"""Cross-check the FK module (over the bus) against motion-core's direct
world-frame FK.

The FK module composes ``T_world_base @ T_base_tcp``; this script computes
the same chain through ``algorithms.kinematics.fk`` and diffs the resulting
``Pose6D``. Mismatch => the Scene seam or the world<-base composition has
drifted.

Run the module in one shell and this in another:

    python -m imp_module_motion_pinocchio --module fk \
        --world <world.yaml> --world-robot arm \
        --station devstation --robot fr3
    python examples/verify_fk.py <world.yaml>
"""

import sys
import threading
import time

import numpy as np
from scipy.spatial.transform import Rotation

from imp_sdk import Bus, QosClass, keyexpr
from imp_sdk.schemas import imp_pb2

from algorithms.descriptions import WorldDescription
from algorithms.kinematics.fk import fk
from algorithms.resolved import KinematicModel, Scene

STATION, ROBOT, WORLD_ROBOT = "devstation", "fr3", "arm"


def main() -> int:
    world_path = sys.argv[1]
    world = WorldDescription.from_yaml(world_path)
    scene = Scene.from_world(world)
    model = KinematicModel.from_robot_system(world.robot(WORLD_ROBOT).robot_system)

    chain = model.chain("arm")
    tcp = chain.tcp_frame or chain.tip_frame

    home = world.robot(WORLD_ROBOT).robot_system.named_joint_state("home")
    q = np.array([home[n] for n in model.active_joint_names], dtype=float)

    T_world_tcp = fk(scene, WORLD_ROBOT, q, tcp)
    want_pos = T_world_tcp[:3, 3]
    want_quat = Rotation.from_matrix(T_world_tcp[:3, :3]).as_quat()

    bus = Bus.open()
    state_key = keyexpr.hal(STATION, ROBOT, "state")
    out_key = keyexpr.motion(STATION, "fk", "tcp")
    sub = bus.subscribe(out_key, imp_pb2.Pose6D)

    stop = threading.Event()

    def feeder():
        while not stop.is_set():
            bus.put(
                state_key,
                imp_pb2.RobotState(
                    header=imp_pb2.Header(schema="imp.RobotState/1"),
                    q=q.tolist(),
                    mode="idle",
                ),
                QosClass.STATE,
            )
            time.sleep(0.1)

    threading.Thread(target=feeder, daemon=True).start()
    got = sub.recv()
    stop.set()

    got_pos = np.array(got.position_m)
    got_quat = np.array(got.quat_xyzw)
    pos_err = float(np.linalg.norm(got_pos - want_pos))
    quat_err = float(min(np.linalg.norm(got_quat - want_quat), np.linalg.norm(got_quat + want_quat)))

    print(f"world_frame={world.world_frame}  tcp_frame={tcp}")
    print(f"direct  pos={want_pos.round(4).tolist()} quat={want_quat.round(4).tolist()}")
    print(f"module  pos={got_pos.round(4).tolist()} quat={got_quat.round(4).tolist()}")
    print(f"frame_id={got.header.frame_id}  pos_err={pos_err:.2e}  quat_err={quat_err:.2e}")
    ok = pos_err < 1e-6 and quat_err < 1e-6 and got.header.frame_id == world.world_frame
    print("RESULT:", "OK" if ok else "MISMATCH")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
