"""run-store: workspace-backed persistence of task runs (spec §10).

A run is one execution instance of a task. The run store owns:

* ``workspace/runs/<run_id>/meta.json``       -- ``RunResult`` summary.
* ``workspace/runs/<run_id>/timeline.json``   -- structured event log (P7+).
* ``workspace/runs/<run_id>/log.txt``         -- human log (P7+).
* ``workspace/runs/<run_id>/bag/``            -- Zenoh bag of subscribed topics (P7+).
* ``workspace/runs/<run_id>/artifacts/``      -- debug images, plots, etc. (P7+).

V1 only implements ``meta.json`` -- the timeline/bag/artifacts wiring
lands when supervisor + ``imp bag record`` arrive in P7.

The bus-side queryable (``run.list`` / ``run.show`` / ``run.timeline``
services) lands in P8 with the rest of the services schemas; for now
the Python API here is the way callers (the ``run-task`` job, the CLI,
tests) read and write run metadata.
"""

from .store import RunStore, list_runs, read_run, write_run

__all__ = ["RunStore", "list_runs", "read_run", "write_run"]
