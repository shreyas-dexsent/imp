# tasks

**Kind:** Crate &nbsp;|&nbsp; **Status:** Rust scaffold (Python engine is the working implementation)

The task layer (spec §11) — Graph Compiler + Task Runtime + sequence FSM.

**The working engine is the Python implementation** at
[`platform/sdk/py/imp_tasks/`](../../sdk/py/imp_tasks/). This Rust crate
exists to:

1. Keep `crates/` the canonical home of sealed-core engines, so a Rust
   replacement can land here in a later phase without churning the rest
   of the tree (same pragma as `crates/module-contract` vs `imp_sdk.module`).
2. Reserve the published artifact name `imp-tasks` on the Rust side.

See PLAN.md §2 (D2 / D11) for the polyglot-runtime trade-off being made
here.

## Use

For now, drive tasks from Python or via the `imp task` CLI subcommand
(which shells out to the Python engine):

```bash
imp task validate path/to/task.yaml
imp task run      path/to/task.yaml --stage-timeout-s 30
```

Or directly:

```bash
python -m imp_job_run_task --task path/to/task.yaml
```

## What lives where

| Concept | Lives in | Status |
|---|---|---|
| `task.yaml` schema + loader (Pydantic) | `imp_tasks.spec` | P5 |
| Graph Compiler (validate wiring + instantiate modules) | `imp_tasks.compiler` | P5 |
| Task Runtime (spin modules, drive sequence, emit events) | `imp_tasks.runtime` | P5 |
| Sealed Rust engine | `crates/tasks` (this) | P7+ |
| Bus-side queryable (`task.validate` / `task.run` services) | `services/task` | P8 |
