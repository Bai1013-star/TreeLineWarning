from __future__ import annotations

import json

import pytest
from core.contracts import CancellationToken, PipelineProgress, ProgressReporter, UnifiedError


class InMemoryReporter:
    def __init__(self) -> None:
        self.events: list[PipelineProgress] = []

    def update(self, progress: PipelineProgress) -> None:
        self.events.append(progress)


def test_unified_error_accepts_four_severity_levels() -> None:
    for severity in ("info", "warning", "error", "fatal"):
        error = UnifiedError(
            code=f"{severity.upper()}_CODE",
            message="message",
            severity=severity,
            module="contracts",
            recoverable=severity != "fatal",
            details={"severity": severity},
        )
        payload = error.to_dict()
        json.dumps(payload)
        assert UnifiedError.from_dict(payload).severity == severity

    with pytest.raises(ValueError, match="severity must be one of"):
        UnifiedError(
            code="BAD",
            message="message",
            severity="debug",  # type: ignore[arg-type]
            module="contracts",
            recoverable=True,
        )


def test_pipeline_progress_status_percent_and_point_validation() -> None:
    progress = PipelineProgress(
        task_id="task-1",
        stage="las-header",
        status="running",
        processed_points=5,
        total_points=20,
        percent=25,
    )
    payload = progress.to_dict()
    json.dumps(payload)

    restored = PipelineProgress.from_dict(payload)
    assert restored.status == "running"
    assert restored.percent == 25.0
    assert restored.processed_points == 5

    with pytest.raises(ValueError, match="between 0 and 100"):
        PipelineProgress(task_id="task-1", stage="las-header", status="running", percent=101)

    with pytest.raises(ValueError, match="status is invalid"):
        PipelineProgress(task_id="task-1", stage="las-header", status="done")  # type: ignore[arg-type]


def test_pipeline_progress_rejects_processed_points_above_total_points() -> None:
    with pytest.raises(ValueError, match="processed_points cannot exceed total_points"):
        PipelineProgress(
            task_id="task-1",
            stage="chunk-read",
            status="running",
            processed_points=11,
            total_points=10,
        )


def test_pipeline_progress_serializes_unified_error() -> None:
    progress = PipelineProgress(
        task_id="task-1",
        stage="las-header",
        status="failed",
        percent=40,
        error=UnifiedError(
            code="LAS_HEADER_FAILED",
            message="Unable to read LAS header",
            severity="error",
            module="algorithms.io",
            recoverable=True,
            details={"stage": "header"},
            cause="mock failure",
        ),
    )

    payload = progress.to_dict()
    json.dumps(payload)
    restored = PipelineProgress.from_dict(payload)

    assert restored.error is not None
    assert restored.error.code == "LAS_HEADER_FAILED"
    assert restored.error.details["stage"] == "header"


def test_cancellation_token_initial_and_cancelled_state() -> None:
    token = CancellationToken()
    assert token.is_cancelled() is False

    token.cancel()
    assert token.is_cancelled() is True


def test_progress_reporter_minimal_fake_implementation() -> None:
    reporter: ProgressReporter = InMemoryReporter()
    progress = PipelineProgress(
        task_id="task-1",
        stage="las-header",
        status="completed",
        percent=100,
    )

    reporter.update(progress)

    assert isinstance(reporter, InMemoryReporter)
    assert reporter.events == [progress]
