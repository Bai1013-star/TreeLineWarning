"""Minimal progress and cancellation contracts for pipeline orchestration."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from core.contracts._json import json_safe_mapping, reject_unknown_fields, require_fields
from core.contracts.errors import UnifiedError

PipelineStatus = Literal["pending", "running", "cancelled", "failed", "completed"]


@dataclass(frozen=True)
class PipelineProgress:
    """JSON-safe progress event for long-running pipeline stages."""

    task_id: str
    stage: str
    status: PipelineStatus
    processed_points: int | None = None
    total_points: int | None = None
    percent: float | None = None
    message: str | None = None
    error: UnifiedError | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.task_id:
            raise ValueError("PipelineProgress.task_id must be non-empty")
        if not self.stage:
            raise ValueError("PipelineProgress.stage must be non-empty")
        if self.status not in {"pending", "running", "cancelled", "failed", "completed"}:
            raise ValueError("PipelineProgress.status is invalid")
        if self.percent is not None:
            if not isinstance(self.percent, int | float) or not math.isfinite(self.percent):
                raise ValueError("PipelineProgress.percent must be a finite number")
            if self.percent < 0 or self.percent > 100:
                raise ValueError("PipelineProgress.percent must be between 0 and 100")
            object.__setattr__(self, "percent", float(self.percent))
        for field_name in ("processed_points", "total_points"):
            value = getattr(self, field_name)
            if value is not None and (
                not isinstance(value, int) or isinstance(value, bool) or value < 0
            ):
                raise ValueError(f"PipelineProgress.{field_name} must be a non-negative integer")
        if (
            self.processed_points is not None
            and self.total_points is not None
            and self.processed_points > self.total_points
        ):
            raise ValueError("PipelineProgress.processed_points cannot exceed total_points")
        if self.error is not None and not isinstance(self.error, UnifiedError):
            raise ValueError("PipelineProgress.error must be a UnifiedError")
        object.__setattr__(self, "metadata", json_safe_mapping(self.metadata, "metadata"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "stage": self.stage,
            "status": self.status,
            "processed_points": self.processed_points,
            "total_points": self.total_points,
            "percent": self.percent,
            "message": self.message,
            "error": self.error.to_dict() if self.error is not None else None,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> PipelineProgress:
        required = {"task_id", "stage", "status"}
        allowed = required | {
            "processed_points",
            "total_points",
            "percent",
            "message",
            "error",
            "metadata",
        }
        require_fields(data, required, cls.__name__)
        reject_unknown_fields(data, allowed, cls.__name__)
        error = data.get("error")
        return cls(
            task_id=data["task_id"],
            stage=data["stage"],
            status=data["status"],
            processed_points=data.get("processed_points"),
            total_points=data.get("total_points"),
            percent=data.get("percent"),
            message=data.get("message"),
            error=UnifiedError.from_dict(error) if error is not None else None,
            metadata=data.get("metadata", {}),
        )


class ProgressReporter(Protocol):
    """Smallest pipeline progress reporter interface."""

    def update(self, progress: PipelineProgress) -> None:
        """Receive a progress event."""


class CancellationToken:
    """Small mutable cancellation flag shared by long-running pipeline stages."""

    def __init__(self) -> None:
        self._is_cancelled = False

    def is_cancelled(self) -> bool:
        return self._is_cancelled

    def cancel(self) -> None:
        self._is_cancelled = True
