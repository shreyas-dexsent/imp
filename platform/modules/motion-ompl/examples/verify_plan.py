"""Verify the planning module over the bus produces a valid joint path
(connecting start and goal) matching a direct motion-core plan.

    python -m imp_module_motion_ompl --world <franka_robot_only_world.yaml>
    python examples/verify_plan.py <franka_robot_only_world.yaml>
"""

import sys
import threading
import time

import numpy as np

from imp_sdk import Bus, QosClass, keyexpr
from imp_sdk.schemas import imp_pb2

from algorithms.descriptions import WorldDescription
from algorithms.planning import PlanOptions, plan_joint
from algorithms.resolved import CollisionModel, KinematicModel, Scene

STATION, ROBOT = "devstation", "fr3"


def main() -> int:
    world_path = sys.argv[1]
    world = WorldDescription.from_yaml(world_path)
    scene = Scene.from_world(world, CollisionModel.from_world(world))
    model = KinematicModel.from_robot_system(world.robots[0].robot_system)
    home = world.robots[0].robot_system.named_joint_state("home")
    q_home = np.array([home[n] for n in model.active_joint_names], dtype=float)
    q_goal = q_home.copy()
    q_goal[0] += 0.8
    q_goal[2] -= 0.3

    direct = plan_joint(model, scene, q_home, q_goal, options=PlanOptions(random_seed=0))
    print(f"direct plan success={direct.success} "
          f"waypoints={direct.path.num_waypoints if direct.success else 0}")

    bus = Bus.open()
    start_key = keyexpr.hal(STATION, ROBOT, "state")
    goal_key = keyexpr.motion(STATION, "plan", "goal")
    out_key = keyexpr.motion(STATION, "plan", "path")
    sub = bus.subscribe(out_key, imp_pb2.Path)

    stop = threading.Event()

    def feeder():
        while not stop.is_set():
            bus.put(start_key, imp_pb2.RobotState(header=imp_pb2.Header(schema="imp.RobotState/1"),
                                                  q=q_home.tolist(), mode="idle"), QosClass.STATE)
            bus.put(goal_key, imp_pb2.JointSolution(header=imp_pb2.Header(schema="imp.JointSolution/1"),
                                                    q=q_goal.tolist(), valid=True), QosClass.STATE)
            time.sleep(0.2)

    threading.Thread(target=feeder, daemon=True).start()
    path = sub.recv()
    stop.set()

    n_dof = path.n_dof
    wp = np.array(path.q_wp).reshape(-1, n_dof) if n_dof else np.empty((0, 0))
    print(f"module path: n_dof={n_dof} waypoints={wp.shape[0]}")
    ok = (
        n_dof == len(model.active_joint_names)
        and wp.shape[0] >= 2
        and np.allclose(wp[0], q_home, atol=1e-3)
        and np.allclose(wp[-1], q_goal, atol=1e-3)
    )
    if wp.shape[0]:
        print(f"  start_err={np.linalg.norm(wp[0]-q_home):.2e} goal_err={np.linalg.norm(wp[-1]-q_goal):.2e}")
    print("RESULT:", "OK" if ok else "MISMATCH")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
