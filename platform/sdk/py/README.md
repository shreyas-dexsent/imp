# imp-sdk (Python)

The public surface for Python HAL drivers, modules, services and jobs. Phase 1
ships the bus + key conventions + generated wire schemas, matching the Rust core
exactly on the wire (same Protobuf, same key namespace, same schema tags).

## Install

```bash
pip install -e .            # runtime deps: eclipse-zenoh, protobuf
pip install -e '.[codegen]' # + grpcio-tools, to regenerate schemas
```

## Usage

```python
from imp_sdk import Bus, QosClass, keyexpr
from imp_sdk.schemas import imp_pb2

with Bus.open() as bus:
    key = keyexpr.hal("st1", "cam_d405", "state")
    sub = bus.subscribe(key, imp_pb2.RobotState)   # rejects schema mismatches
    msg = sub.recv()
```

`Bus.open()` forces IPv4 TCP listening by default (the zenoh default binds
`tcp/[::]:0`, which fails on IPv4-only hosts). Set `IMP_ZENOH_CONFIG` to a json5
file to override.

## Regenerating schemas

`imp_sdk/schemas/imp_pb2.py` is generated from `../../crates/schemas/proto/imp.proto`
(the same proto that generates the Rust types):

```bash
python -m grpc_tools.protoc -I ../../crates/schemas/proto \
    --python_out=imp_sdk/schemas ../../crates/schemas/proto/imp.proto
```
