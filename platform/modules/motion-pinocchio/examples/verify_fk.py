"""Cross-check the FK module (over the bus) against motion-core's direct FK.

Run the module in one shell and this in another:

    python -m imp_module_motion_pinocchio --robot-system <robot_only.yaml> --station devstation --robot fr3
    python examples/verify_fk.py <robot_only.yaml>
"""

import sys
import threading
import time

import numpy as np
import yaml
from scipy.spatial.transform import Rotation

from imp_sdk import Bus, QosClass, keyexpr
from imp_sdk.schemas import imp_pb2

from algorithms.descriptions import RobotSystemDescription
from algorithms.kinematics.fk import fk_local
from algorithms.resolved import KinematicModel

STATION, ROBOT = "devstation", "fr3"


def main() -> int:
    rs_path = sys.argv[1]
    system = RobotSystemDescription.from_yaml(rs_path)
    model = KinematicModel.from_robot_system(system)
    chain = model.chain("arm")
    tcp = chain.tcp_frame or chain.tip_frame

    home = yaml.safe_load(open(rs_path))["named_joint_states"]["home"]["joints"]
    q = np.array([home[n] for n in model.active_joint_names], dtype=float)

    t = fk_local(model, q, tcp)
    want_pos = t[:3, 3]
    want_quat = Rotation.from_matrix(t[:3, :3]).as_quat()

    bus = Bus.open()
    state_key = keyexpr.hal(STATION, ROBOT, "state")
    out_key = keyexpr.motion(STATION, "fk", "tcp")
    sub = bus.subscribe(out_key, imp_pb2.Pose6D)

    stop = threading.Event()

    def feeder():
        while not stop.is_set():
            bus.put(state_key, imp_pb2.RobotState(header=imp_pb2.Header(schema="imp.RobotState/1"),
                                                   q=q.tolist(), mode="idle"), QosClass.STATE)
            time.sleep(0.1)

    threading.Thread(target=feeder, daemon=True).start()
    got = sub.recv()  # blocks until the module publishes an FK pose
    stop.set()

    got_pos = np.array(got.position_m)
    got_quat = np.array(got.quat_xyzw)
    pos_err = float(np.linalg.norm(got_pos - want_pos))
    # quaternion sign-insensitive distance
    quat_err = float(min(np.linalg.norm(got_quat - want_quat), np.linalg.norm(got_quat + want_quat)))

    print(f"tcp_frame={tcp}")
    print(f"direct  pos={want_pos.round(4).tolist()} quat={want_quat.round(4).tolist()}")
    print(f"module  pos={got_pos.round(4).tolist()} quat={got_quat.round(4).tolist()}")
    print(f"pos_err={pos_err:.2e}  quat_err={quat_err:.2e}")
    ok = pos_err < 1e-6 and quat_err < 1e-6
    print("RESULT:", "OK" if ok else "MISMATCH")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
