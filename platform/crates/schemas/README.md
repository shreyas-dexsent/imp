# schemas

**Kind:** Crate &nbsp;|&nbsp; **Status:** phase 1 — Rust + Python implemented

The single Protobuf IDL (`proto/imp.proto`, spec §6) and its generated bindings.
Rust is generated at build time via `prost` (vendored `protoc`, serde-derived so
any message renders to JSON). The same proto generates the Python bindings in
`sdk/py/imp_sdk/schemas/imp_pb2.py`, so Rust and Python share one wire format.

- `ImpMessage` — ties a generated type to its versioned schema id (`imp.Pose6D/1`).
- `decode_to_json(schema, bytes)` — render any known message from wire bytes (used by `imp topic echo`).
- `KNOWN_SCHEMAS` — every schema name imp knows.

C++ / TS bindings: later phases. See the root `README.md` §6/§7.
