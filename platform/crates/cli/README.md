# cli

**Kind:** Crate &nbsp;|&nbsp; **Status:** phase 1 — `topic` + `version`

The `imp` binary: a thin client over the same Zenoh planes the UI uses (spec §16).
Phase 1 implements topic introspection; the remaining verbs land with their
subsystems.

```bash
imp topic echo 'imp/devstation/hal/sim/state'   # decode + print messages (any schema)
imp topic hz   'imp/devstation/hal/sim/state'   # measure message rate
imp version
```

`echo` decodes known schemas to JSON via `imp-schemas`; unknown schemas print as
byte length. Wildcards work (`imp/st1/perc/**`). See the root `README.md` §16.
