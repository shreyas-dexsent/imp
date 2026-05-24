# imp

`imp` — a Zenoh-based, topic-driven robotics middleware and no-code VGR platform.

This `platform/` directory is the `imp` source root. The **canonical specification** is the
top-level [`../README.md`](../README.md); this tree implements it section by section. When the frozen
`reference/` codebase is removed, `platform/` is promoted to the repository root.

## Layout (see spec §19)

| Folder | Role |
|---|---|
| `crates/` | Sealed Rust core — contracts + runtimes only (no drivers). |
| `hal/` | HAL device drivers — one short folder per device. |
| `modules/` | Functional modules — perception / motion / spatial. |
| `services/` | Synchronous queryables. |
| `jobs/` | Long-running cancelable ops. |
| `catalog/` | Pre-configured asset catalog (seeded from `reference/robot-algorithms/assets/`). |
| `sdk/` | Public plugin / task author surface (Python + Rust). |
| `ui/` | Single TypeScript app (`imp-ui`), served by `crates/ui-host`. |
| `docs/` | User / developer / architecture / reference docs. |
| `examples/` | Runnable example workspaces (tutorials). |
| `tools/` | Build, training, and dev tooling. |

## Status

Scaffold. Each `crates/*`, `hal/*`, `modules/*`, `services/*`, and `jobs/*` carries its own
`README.md` + `examples/` (enforced by `tools/dev/check_docs.py`). Implementation proceeds in the
phases listed in the spec §22.
