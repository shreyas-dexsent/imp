"""Verify the trajectory module over the bus: feed a 2-waypoint Path, receive a
Trajectory, and check it starts/ends at the waypoints, has monotone time, and
respects the model's joint velocity limits.

    python -m imp_module_motion_ruckig --robot-system <franka_fr3_robot_only.yaml>
    python examples/verify_traj.py <franka_fr3_robot_only.yaml>
"""

import sys
import threading
import time

import numpy as np
import yaml

from imp_sdk import Bus, QosClass, keyexpr
from imp_sdk.schemas import imp_pb2

from algorithms.descriptions import RobotSystemDescription
from algorithms.resolved import KinematicModel

STATION, ROBOT = "devstation", "fr3"


def main() -> int:
    rs_path = sys.argv[1]
    model = KinematicModel.from_robot_system(RobotSystemDescription.from_yaml(rs_path))
    names = model.active_joint_names
    home = yaml.safe_load(open(rs_path))["named_joint_states"]["home"]["joints"]
    q_home = np.array([home[n] for n in names], dtype=float)
    q_goal = q_home.copy()
    q_goal[0] += 0.5
    n_dof = len(names)
    path_flat = np.concatenate([q_home, q_goal]).tolist()

    bus = Bus.open()
    path_key = keyexpr.motion(STATION, "plan", "path")
    out_key = keyexpr.motion(STATION, "plan", "trajectory")
    sub = bus.subscribe(out_key, imp_pb2.Trajectory)

    stop = threading.Event()

    def feeder():
        while not stop.is_set():
            bus.put(path_key, imp_pb2.Path(header=imp_pb2.Header(schema="imp.Path/1"),
                                           q_wp=path_flat, n_dof=n_dof), QosClass.STATE)
            time.sleep(0.2)

    threading.Thread(target=feeder, daemon=True).start()
    traj = sub.recv()
    stop.set()

    t = np.array(traj.t_s)
    pos = np.array(traj.q_wp).reshape(-1, traj.n_dof)
    m = pos.shape[0]

    dt = np.diff(t)
    vmax = np.max(np.abs(np.diff(pos, axis=0) / dt[:, None]), axis=0) if m > 1 else np.zeros(n_dof)
    vlim = model.active_velocity_limits()

    print(f"trajectory: M={m} duration={t[-1]:.3f}s")
    print(f"  start_err={np.linalg.norm(pos[0]-q_home):.2e} goal_err={np.linalg.norm(pos[-1]-q_goal):.2e}")
    print(f"  max|vel|={vmax.round(3).tolist()}")
    print(f"  vel_limit={np.asarray(vlim).round(3).tolist()}")
    ok = (
        m >= 2
        and t[0] == 0.0
        and np.all(dt >= 0.0)
        and t[-1] > 0.0
        and np.allclose(pos[0], q_home, atol=1e-3)
        and np.allclose(pos[-1], q_goal, atol=1e-3)
        and np.all(vmax <= np.asarray(vlim) * 1.05 + 1e-6)
    )
    print("RESULT:", "OK" if ok else "MISMATCH")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
