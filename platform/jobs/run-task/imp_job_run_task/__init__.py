"""run-task: execute a task.yaml as a cancelable job (spec §10, §11).

Thin wrapper around :class:`imp_tasks.TaskRun` that:

1. Loads + validates the task.yaml at ``request.task_path`` (the
   workspace task file under ``processes/<pid>/tasks/<tid>.yaml``).
2. Compiles the graph via :func:`imp_tasks.compile_task`.
3. Drives :class:`imp_tasks.TaskRun` to terminal state.
4. Persists the resulting :class:`imp_tasks.RunResult` to
   ``workspace/runs/<run_id>/meta.json`` via :func:`imp_service_run_store.write_run`.

In v1 the job exposes a Python entry point only; the bus-side
queryable wrapper (so ``imp job submit run-task ...`` calls it over
Zenoh) lands with the services/jobs schemas in P8.

Migrates the ``orchestrator/core/executor.py`` task driver from the VGR
reference -- replaces its bespoke ``_pick_runtime.py`` per-task scripts
with a single task-agnostic engine.
"""

from .job import RunTaskRequest, RunTaskResult, run_task

__all__ = ["RunTaskRequest", "RunTaskResult", "run_task"]
