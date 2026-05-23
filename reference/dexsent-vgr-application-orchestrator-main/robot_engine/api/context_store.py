from __future__ import annotations

import shutil
import tempfile
import threading
import uuid
from pathlib import Path
from typing import Dict, Optional

from robot_engine.interfaces.ui_api import RobotEngineContext


class ContextNotFound(KeyError):
    pass


class RobotEngineContextStore:
    def __init__(self) -> None:
        self._contexts: Dict[str, RobotEngineContext] = {}
        self._tmp_dirs: Dict[str, Path] = {}
        self._lock = threading.Lock()

    def create(self) -> str:
        context_id = str(uuid.uuid4())
        tmp = Path(tempfile.mkdtemp(prefix=f"re_ctx_{context_id[:8]}_"))
        with self._lock:
            self._contexts[context_id] = RobotEngineContext()
            self._tmp_dirs[context_id] = tmp
        return context_id

    def get(self, context_id: str) -> RobotEngineContext:
        with self._lock:
            ctx = self._contexts.get(context_id)
        if ctx is None:
            raise ContextNotFound(context_id)
        return ctx

    def delete(self, context_id: str) -> None:
        with self._lock:
            self._contexts.pop(context_id, None)
            tmp = self._tmp_dirs.pop(context_id, None)
        if tmp and tmp.exists():
            shutil.rmtree(tmp, ignore_errors=True)

    def tmp_dir(self, context_id: str) -> Path:
        with self._lock:
            tmp = self._tmp_dirs.get(context_id)
        if tmp is None:
            raise ContextNotFound(context_id)
        return tmp

    def list_ids(self):
        with self._lock:
            return list(self._contexts)


_store = RobotEngineContextStore()


def get_store() -> RobotEngineContextStore:
    return _store
