# bus

**Kind:** Crate &nbsp;|&nbsp; **Status:** phase 1 — implemented

Zenoh transport: applies imp's key conventions, maps `QosClass` onto Zenoh QoS,
tags every publication with its schema, and rejects mismatched schemas on
receive (spec §6). Verified interoperating with the Python SDK on the wire.

- `Bus::open_default()` — IPv4-friendly default config (`IMP_ZENOH_CONFIG` to override).
- `put` / `publisher` — typed, schema-tagged publication.
- `subscribe` (typed, drops mismatches) and `subscribe_raw` (untyped, for the CLI).

Run the demo + round-trip:

```bash
cargo run -p imp-bus --example demo_pub        # publishes RobotState at 10 Hz
cargo test -p imp-bus --test roundtrip          # typed round-trip + schema reject
```

See `examples/demo_pub.rs` and the root `README.md` §6.
