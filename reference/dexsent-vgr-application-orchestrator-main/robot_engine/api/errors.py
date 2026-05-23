from __future__ import annotations

from fastapi import HTTPException

from robot_engine.api.context_store import ContextNotFound


def context_not_found(context_id: str) -> HTTPException:
    return HTTPException(status_code=404, detail=f"Context not found: {context_id}")


def handle_context_not_found(exc: ContextNotFound) -> HTTPException:
    return context_not_found(str(exc))
