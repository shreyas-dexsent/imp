# cli

**Kind:** Crate &nbsp;|&nbsp; **Status:** phase 2 — `topic`, `station`, `process`, `version`

The `imp` binary: a thin client over the same Zenoh planes the UI uses (spec §16).
The remaining verbs land with their subsystems.

```bash
imp topic echo 'imp/devstation/hal/ur5e/state'   # decode + print messages (any schema)
imp topic hz   'imp/devstation/hal/ur5e/state'   # measure message rate
imp --workspace <path> station list              # stations in a workspace
imp --workspace <path> station info devstation   # details, devices, processes
imp --workspace <path> process list devstation
imp version
```

`echo` decodes known schemas to JSON via `imp-schemas`; unknown schemas print as
byte length. Wildcards work (`imp/st1/perc/**`). Workspace defaults to
`$IMP_WORKSPACE` or `~/.imp/workspace`. See the root `README.md` §16.
