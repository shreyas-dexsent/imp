"""imp_tasks: the Python Task Layer (spec §11).

Three pieces:

* :mod:`imp_tasks.spec`     -- ``TaskSpec`` pydantic model for ``task.yaml``.
* :mod:`imp_tasks.compiler` -- ``compile_task(spec)`` -> ``CompiledTask`` of
  module instances + a wiring check across producer/consumer keyexprs.
* :mod:`imp_tasks.runtime`  -- ``TaskRun(compiled).run()`` spins every
  module in a daemon ``ModuleNode`` thread, emits ``run.started`` /
  ``run.succeeded`` / ``run.failed`` events on
  ``imp/<station>/ctrl/runs/<run_id>/events``, and stops on a configured
  terminator topic.

Replaces the VGR orchestrator's bespoke ``core/executor.py`` plus the
per-task ``tasks/<name>.py`` runtimes (see the per-package READMEs for
the migration source). The engine is task-agnostic -- deployed tasks
are workspace YAML at ``processes/<process>/tasks/<task>.yaml`` (spec §11).

The bus-touching pieces (``runtime.TaskRun``, ``RunStatus``,
``RunResult``) are lazily imported so the pure ``spec`` + ``compiler``
modules remain usable without ``imp_sdk`` / ``zenoh`` installed.
"""

from .compiler import CompiledNode, CompiledTask, CompileError, compile_task
from .spec import EdgeSpec, NodeSpec, SequenceStage, TaskSpec

__all__ = [
    # eager
    "CompiledNode", "CompiledTask", "CompileError", "compile_task",
    "EdgeSpec", "NodeSpec", "SequenceStage", "TaskSpec",
    # lazy (needs imp_sdk + zenoh)
    "RunResult", "RunStatus", "TaskRun",
]


def __getattr__(name):  # PEP 562
    if name in {"RunResult", "RunStatus", "TaskRun"}:
        from . import runtime
        return getattr(runtime, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
