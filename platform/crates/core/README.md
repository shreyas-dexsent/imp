# core

**Kind:** Crate &nbsp;|&nbsp; **Status:** phase 1 — implemented

Ids, errors, the key-expression conventions (`imp/<station>/...`, spec §6), the
`QosClass` enum, the `SchemaTag` format (`imp.Pose6D/1`), and the shared
`Interface` descriptor (spec §7). No transport here — `imp-bus` maps these onto
Zenoh.

- `keyexpr` — builders for hal / perc / motion / tf / svc / ctrl keys.
- `SchemaTag` — parse/format + `accepts()` (rejects newer-than-known versions).
- `Interface` / `Direction` — the one descriptor shape for every I/O surface.

See the root `README.md` §6/§7 for the contract.
