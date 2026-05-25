# run-store

**Kind:** Service &nbsp;|&nbsp; **Status:** phase 5 — `meta.json` implemented; timeline / bag / artifacts in P7

Workspace-backed persistence of task runs (spec §10, §14):

```
workspace/runs/<run_id>/
├── meta.json        ← phase 5: RunResult summary
├── timeline.json    ← phase 7: structured event log
├── log.txt          ← phase 7: human log
├── bag/             ← phase 7: Zenoh bag of subscribed topics
└── artifacts/       ← phase 7: debug images / plots
```

## Python API

```python
from imp_service_run_store import write_run, read_run, list_runs, RunStore

# write
write_run(my_run_result, workspace_root="/path/to/ws")

# read
rec = read_run("synthetic_pick-abcd1234")
print(rec.status, rec.elapsed_s)

# list (optionally filtered by task_id)
for rec in list_runs(task_id="pick_place"):
    print(rec.run_id, rec.status)

# OO wrapper
store = RunStore(workspace_root="/path/to/ws")
store.write(my_run_result)
```

Workspace root resolution: explicit arg > `IMP_WORKSPACE` env >
`~/.imp/workspace`.

## Tests

```bash
cd platform/services/run-store
PYTHONPATH=. pytest tests -q   # 5 passed
```

## Migrates from reference

`orchestrator/runs/` — same per-run folder shape, but the runtime layer
on top is now `imp_tasks.TaskRun` (P5) instead of the bespoke executor.
The bus-side queryable (`run.list` / `run.show` / `run.timeline` over
Zenoh) lands with the rest of the services schemas in P8.
