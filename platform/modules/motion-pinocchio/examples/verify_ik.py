"""Cross-check the IK module over the bus: target = FK(home); solve IK from a
perturbed seed; assert FK(solution) reproduces the target pose.

    python -m imp_module_motion_pinocchio --module ik --robot-system <robot_only.yaml>
    python examples/verify_ik.py <robot_only.yaml>
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
    q_home = np.array([home[n] for n in model.active_joint_names], dtype=float)
    t_target = fk_local(model, q_home, tcp)
    target_pos = t_target[:3, 3]
    target_quat = Rotation.from_matrix(t_target[:3, :3]).as_quat()

    # Seed the IK away from home so the solve is non-trivial.
    q_seed = q_home + 0.2

    bus = Bus.open()
    target_key = keyexpr.motion(STATION, "ik", "target")
    seed_key = keyexpr.hal(STATION, ROBOT, "state")
    out_key = keyexpr.motion(STATION, "ik", "solution")
    sub = bus.subscribe(out_key, imp_pb2.JointSolution)

    stop = threading.Event()

    def feeder():
        while not stop.is_set():
            bus.put(seed_key, imp_pb2.RobotState(header=imp_pb2.Header(schema="imp.RobotState/1"),
                                                  q=q_seed.tolist(), mode="idle"), QosClass.STATE)
            bus.put(target_key, imp_pb2.PoseTarget(header=imp_pb2.Header(schema="imp.PoseTarget/1"),
                                                   target_frame=tcp, position_m=target_pos.tolist(),
                                                   quat_xyzw=target_quat.tolist()), QosClass.STATE)
            time.sleep(0.1)

    threading.Thread(target=feeder, daemon=True).start()
    sol = sub.recv()
    stop.set()

    if not sol.valid:
        print(f"IK failed: {sol.reject_reason}")
        print("RESULT: MISMATCH")
        return 1

    q_sol = np.array(sol.q)
    t_sol = fk_local(model, q_sol, tcp)
    pos_err = float(np.linalg.norm(t_sol[:3, 3] - target_pos))
    sol_quat = Rotation.from_matrix(t_sol[:3, :3]).as_quat()
    quat_err = float(min(np.linalg.norm(sol_quat - target_quat), np.linalg.norm(sol_quat + target_quat)))

    print(f"tcp_frame={tcp}  q_solution={q_sol.round(3).tolist()}")
    print(f"pose_err of FK(solution) vs target: pos={pos_err:.2e} quat={quat_err:.2e}")
    ok = pos_err < 1e-3 and quat_err < 1e-3
    print("RESULT:", "OK" if ok else "MISMATCH")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
