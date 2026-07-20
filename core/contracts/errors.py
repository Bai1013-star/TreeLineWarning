"""Unified error contract for pipeline and IO stages."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

from core.contracts._json import json_safe_mapping, reject_unknown_fields, require_fields

ErrorSeverity = Literal["info", "warning", "error", "fatal"]


@dataclass(frozen=True)
class UnifiedError:
    """JSON-safe error payload shared across pipeline stages."""

    code: str
    message: str
    severity: ErrorSeverity
    module: str
    recoverable: bool
    details: Mapping[str, Any] = field(default_factory=dict)
    cause: str | None = None

    def __post_init__(self) -> None:
        if not self.code:
            raise ValueError("UnifiedError.code must be non-empty")
        if not self.message:
            raise ValueError("UnifiedError.message must be non-empty")
        if self.severity not in {"info", "warning", "error", "fatal"}:
            raise ValueError("UnifiedError.severity must be one of: info, warning, error, fatal")
        if not self.module:
            raise ValueError("UnifiedError.module must be non-empty")
        object.__setattr__(self, "details", json_safe_mapping(self.details, "details"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "severity": self.severity,
            "module": self.module,
            "recoverable": self.recoverable,
            "details": dict(self.details),
            "cause": self.cause,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> UnifiedError:
        required = {"code", "message", "severity", "module", "recoverable"}
        allowed = required | {"details", "cause"}
        require_fields(data, required, cls.__name__)
        reject_unknown_fields(data, allowed, cls.__name__)
        return cls(
            code=data["code"],
            message=data["message"],
            severity=data["severity"],
            module=data["module"],
            recoverable=data["recoverable"],
            details=data.get("details", {}),
            cause=data.get("cause"),
        )
