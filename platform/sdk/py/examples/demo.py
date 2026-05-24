"""Cross-language bus demo. Interoperates with the Rust `imp` CLI and the
`demo_pub` example.

  # Python publishes, Rust echoes:
  python examples/demo.py pub
  imp topic echo 'imp/devstation/hal/sim/state'

  # Rust publishes, Python echoes:
  cargo run -p imp-bus --example demo_pub
  python examples/demo.py echo
"""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from imp_sdk import Bus, QosClass, keyexpr
from imp_sdk.schemas import imp_pb2


def key() -> str:
    station = os.environ.get("IMP_STATION", "devstation")
    device = os.environ.get("IMP_DEVICE", "sim")
    return keyexpr.hal(station, device, "state")


def do_pub() -> None:
    with Bus.open() as bus:
        pub = bus.publisher(key(), QosClass.STATE)
        print(f"publishing RobotState on {key()} at 10 Hz", flush=True)
        seq = 0
        while True:
            msg = imp_pb2.RobotState(
                header=imp_pb2.Header(seq=seq, stamp_ns=time.time_ns(), schema="imp.RobotState/1"),
                q=[0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785],
                mode="idle",
            )
            pub.put(msg)
            seq += 1
            time.sleep(0.1)


def do_echo() -> None:
    with Bus.open() as bus:
        sub = bus.subscribe(key(), imp_pb2.RobotState)
        print(f"echo {key()} (Ctrl-C to stop)", flush=True)
        while True:
            msg = sub.recv()
            print(f"seq={msg.header.seq} mode={msg.mode} q={list(msg.q)}", flush=True)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "echo"
    if mode == "pub":
        do_pub()
    else:
        do_echo()
