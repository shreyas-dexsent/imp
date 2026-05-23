from __future__ import annotations

from typing import Any, Dict, Optional

from pydantic import BaseModel, Field

from robot_engine.interfaces.error_codes import ErrorCode


class ErrorInfo(BaseModel):
    error_code: ErrorCode = ErrorCode.OK
    error_message: str = ""
    details: Dict[str, Any] = Field(default_factory=dict)


class APIResult(BaseModel):
    success: bool
    error_code: ErrorCode = ErrorCode.OK
    error_message: str = ""
    failed_stage: Optional[str] = None
    failed_waypoint_index: Optional[int] = None
    failed_segment_index: Optional[int] = None
    debug_info: Dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def ok(cls, **kwargs):
        return cls(success=True, **kwargs)

    @classmethod
    def fail(cls, code: ErrorCode, message: str, **kwargs):
        return cls(success=False, error_code=code, error_message=message, **kwargs)
