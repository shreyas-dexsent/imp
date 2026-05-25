"""Run or validate a task.yaml from the CLI:

    python -m imp_job_run_task --task path/to/task.yaml --validate-only
    python -m imp_job_run_task --task path/to/task.yaml
"""

import argparse
import json
import sys

from .job import RunTaskRequest, run_task


def main() -> int:
    ap = argparse.ArgumentParser(prog="imp-job-run-task")
    ap.add_argument("--task", required=True, help="path to a task.yaml")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--stage-timeout-s", type=float, default=30.0)
    ap.add_argument("--workspace-root", default=None)
    ap.add_argument(
        "--validate-only",
        action="store_true",
        help="load + compile the task without running; exit 0 if it would run.",
    )
    args = ap.parse_args()

    if args.validate_only:
        # Compile-only path -- no bus session, no module threads.
        from imp_tasks import TaskSpec, compile_task

        try:
            spec = TaskSpec.from_yaml(args.task)
            compiled = compile_task(spec)
        except Exception as e:
            print(f"INVALID: {e}", file=sys.stderr)
            return 2
        print(f"OK: {spec.id} ({len(compiled.nodes)} nodes, "
              f"{len(spec.graph.edges)} edges, {len(spec.sequence)} stages)")
        return 0

    result = run_task(RunTaskRequest(
        task_path=args.task,
        workspace_root=args.workspace_root,
        run_id=args.run_id,
        stage_timeout_s=args.stage_timeout_s,
    ))
    json.dump(result.__dict__, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0 if result.status == "succeeded" else 1


if __name__ == "__main__":
    sys.exit(main())
