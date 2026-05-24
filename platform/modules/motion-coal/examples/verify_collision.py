"""Cross-check the collision module (over the bus) against motion-core's direct
`is_in_collision`. Run the module first, then this:

    python -m imp_module_motion_coal --world <franka_table_world.yaml>
    python examples/verify_collision.py <franka_table_world.yaml>
"""

import sys
import threading
import time

import numpy as np

from imp_sdk import Bus, QosClass, keyexpr
from imp_sdk.schemas import imp_pb2

from algorithms.collision import is_in_collision
from algorithms.descriptions import WorldDescription
from algorithms.resolved import CollisionModel, KinematicModel, Scene

STATION, ROBOT = "devstation", "fr3"


def main() -> int:
    world_path = sys.argv[1]
    world = WorldDescription.from_yaml(world_path)
    cm = CollisionModel.from_world(world)
    scene = Scene.from_world(world, cm)
    model = KinematicModel.from_robot_system(world.robot("arm").robot_system)
    home = world.robot("arm").robot_system.named_joint_state("home")
    q = np.array([home[n] for n in model.active_joint_names], dtype=float)

    direct = is_in_collision(model, scene, q)
    want = float(len(direct.contacts)) if direct.in_collision else 0.0

    bus = Bus.open()
    state_key = keyexpr.hal(STATION, ROBOT, "state")
    out_key = keyexpr.motion(STATION, "collision", "collision")
    sub = bus.subscribe(out_key, imp_pb2.Scalar)

    stop = threading.Event()

    def feeder():
        while not stop.is_set():
            bus.put(state_key, imp_pb2.RobotState(header=imp_pb2.Header(schema="imp.RobotState/1"),
                                                   q=q.tolist(), mode="idle"), QosClass.STATE)
            time.sleep(0.1)

    threading.Thread(target=feeder, daemon=True).start()
    got = sub.recv()
    stop.set()

    print(f"checked_pairs={direct.checked_pairs} in_collision={direct.in_collision}")
    print(f"direct value={want}  module value={got.value}")
    ok = abs(got.value - want) < 1e-9
    print("RESULT:", "OK" if ok else "MISMATCH")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
