"""Implementation for `orchestrator.logging`."""

import logging
import sys


class _SkipNoisyAccessFilter(logging.Filter):
    _SUPPRESSED_PATH_PREFIXES = (
        "/ui",
        "/ui/",
        "/task_types",
        "/processes/",
        "/health",
        "/ready",
        "/stations",
        "/runs/",
        "/camera/cameras",
        "/camera/fps",
        "/camera/frame",
        "/robot/state",
        "/tasks/",
        "/vision/cameras",
        "/vision/latest",
        "/vision/stream",
    )

    def filter(self, record: logging.LogRecord) -> bool:
        if record.name != "uvicorn.access":
            return True
        args = record.args
        if isinstance(args, tuple) and len(args) >= 5:
            method = str(args[1]).upper()
            path = str(args[2])
            status = int(args[4])
            if method in ("GET", "HEAD") and path.startswith("/vision/frame"):
                return False
            if method in ("GET", "HEAD") and path.startswith("/camera/frame"):
                return False
            if (
                method in ("GET", "HEAD")
                and status < 400
                and any(
                    path.startswith(prefix) for prefix in self._SUPPRESSED_PATH_PREFIXES
                )
            ):
                return False
        return True


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logging.getLogger("uvicorn.access").addFilter(_SkipNoisyAccessFilter())


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
