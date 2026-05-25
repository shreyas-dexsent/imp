# run-task

**Kind:** Job &nbsp;|&nbsp; **Status:** phase 5 — implemented (Python; bus-side queryable in P8)

Execute a workspace `task.yaml` as a cancelable job (spec §10, §11).
Thin wrapper around [`imp_tasks.TaskRun`](../../sdk/py/imp_tasks/runtime.py):

1. Load + validate the `task.yaml` (compile-only path via
   `--validate-only`).
2. Compile the graph through
   [`imp_tasks.compile_task`](../../sdk/py/imp_tasks/compiler.py).
3. Drive the runtime; emit `run.started` / `run.stage_changed` /
   `run.succeeded` / `run.failed` JSON events on
   `imp/<station>/ctrl/runs/<run_id>/events`.
4. Persist the resulting `RunResult` to `workspace/runs/<run_id>/meta.json`
   via the [`run-store`](../../services/run-store/) service.

## Run

```bash
# validate only -- exit 0 if the task would compile + run
python -m imp_job_run_task --task path/to/task.yaml --validate-only

# run end-to-end
python -m imp_job_run_task --task path/to/task.yaml --stage-timeout-s 30

# same thing through the imp CLI (shells out to the Python module)
imp task validate path/to/task.yaml
imp task run      path/to/task.yaml
```

## Migrates from reference

`orchestrator/core/executor.py` + the per-task `orchestrator/tasks/*.py`
scripts. The replacement is task-agnostic — a single engine driven by
the `task.yaml` graph.
