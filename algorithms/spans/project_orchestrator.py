"""Sequential project-level multi-Span orchestration."""

from __future__ import annotations

import hashlib
import html
import json
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.contracts.project import AnalysisStatus, ArtifactReference
from core.project.run_index import (
    attach_span_artifact_index,
    complete_run,
    fail_run,
    interrupt_run,
    reconcile_run_staleness,
    register_run,
    start_run,
)
from core.project.session import WritableProjectSession
from core.project.span_sync import SpanSynchronizationResult, synchronize_group_spans

from algorithms.spans.canonical_executor import run_tracked_canonical_span_science
from algorithms.spans.project_adapter import (
    ScienceAdapterError,
    ScienceExecutor,
    SpanAnalysisInput,
    prepare_source_shared,
    publish_terminal_artifact_index,
    run_canonical_span_science,
)

_SPAN_LOCKS: dict[tuple[str, str], threading.Lock] = {}
_SPAN_LOCKS_GUARD = threading.Lock()


@dataclass(frozen=True, slots=True)
class SpanRunOutcome:
    span_id: str
    status: str
    run_id: str | None
    reused: bool = False
    error: str | None = None
    summary: dict[str, Any] | None = None


def _span_lock(project_root: Path, span_id: str) -> threading.Lock:
    key = (str(project_root.resolve()), span_id)
    with _SPAN_LOCKS_GUARD:
        return _SPAN_LOCKS.setdefault(key, threading.Lock())


def _artifact_reference_valid(project_root: Path, reference: ArtifactReference | None) -> bool:
    if reference is None:
        return False
    path = project_root / Path(reference.relative_path)
    try:
        resolved = path.resolve()
        resolved.relative_to(project_root.resolve())
        data = resolved.read_bytes()
    except (OSError, ValueError):
        return False
    return len(data) == reference.byte_size and hashlib.sha256(data).hexdigest() == reference.sha256


def _run_id(span_id: str) -> str:
    return f"{span_id.lower()}-{uuid.uuid4().hex}"


def _safe_failure_path(project_root: Path, group_id: str, span_id: str, run_id: str) -> Path:
    return (
        project_root
        / "artifacts"
        / "project-runs"
        / "groups"
        / group_id
        / span_id
        / run_id
        / "failure.json"
    )


class ProjectSpanOrchestrator:
    """One controlled sequential worker boundary shared by all Owner actions."""

    def __init__(self, *, science_executor: ScienceExecutor | None = None) -> None:
        self.science_executor = science_executor or run_tracked_canonical_span_science

    def synchronize(self, session: WritableProjectSession) -> SpanSynchronizationResult:
        result = synchronize_group_spans(session)
        self.rebuild_project_result_index(session)
        return result

    def _span(self, session: WritableProjectSession, span_id: str):
        try:
            return next(span for span in session.span_registry.records if span.span_id == span_id)
        except StopIteration as exc:
            raise ScienceAdapterError(f"UNKNOWN_SPAN:{span_id}") from exc

    def analyze_span(
        self,
        session: WritableProjectSession,
        span_id: str,
        *,
        force_new_run: bool = False,
    ) -> SpanRunOutcome:
        lock = _span_lock(session.project_root, span_id)
        if not lock.acquire(blocking=False):
            return SpanRunOutcome(span_id, "RUNNING", None, error="DUPLICATE_ANALYSIS_BLOCKED")
        try:
            self.synchronize(session)
            reconcile_run_staleness(session)
            span = self._span(session, span_id)
            if (
                not force_new_run
                and span.analysis.state is AnalysisStatus.COMPLETE
                and _artifact_reference_valid(
                    session.project_root, span.artifact_index_reference
                )
            ):
                return SpanRunOutcome(span_id, "COMPLETE", span.analysis.active_run_id, reused=True)
            if span.group_id is None:
                raise ScienceAdapterError("SPAN_GROUP_METADATA_MISSING")
            run_id: str
            relative_root: Path
            if span.analysis.state is AnalysisStatus.READY and span.analysis.active_run_id:
                # A READY run is a durable pre-execution reservation: resume it
                # instead of allocating a second run identity.
                run_id = span.analysis.active_run_id
                entry = next(item for item in session.run_index.entries if item.run_id == run_id)
                relative_root = Path(entry.artifact_root_reference or "")
                if not relative_root.parts:
                    raise ScienceAdapterError("READY_RUN_ARTIFACT_ROOT_MISSING")
                start_run(session, run_id)
            else:
                if span.analysis.state is AnalysisStatus.RUNNING and span.analysis.active_run_id:
                    interrupt_run(session, span.analysis.active_run_id)
                run_id = _run_id(span_id)
            relative_root = (
                Path("artifacts")
                / "project-runs"
                / "groups"
                / span.group_id
                / span_id
                / run_id
            )
            output_root = session.project_root / relative_root
            output_root.mkdir(parents=True, exist_ok=True)
            if not any(item.run_id == run_id for item in session.run_index.entries):
                register_run(
                    session,
                    run_id,
                    span_id,
                    artifact_root_reference=relative_root.as_posix(),
                )
                start_run(session, run_id)
            try:
                span = self._span(session, span_id)
                towers = {
                    entry.tower_id: entry.current
                    for entry in session.tower_registry.records
                }
                input_data = SpanAnalysisInput.from_project(
                    session.project.project_id, span, towers
                )
                input_data = SpanAnalysisInput(
                    **{
                        **input_data.to_dict(),
                        "source_point_count": (
                            session.project.source_reference.source_identity.source_point_count
                        ),
                    }
                )
                preparation = prepare_source_shared(session.project_root, input_data)
                input_data = SpanAnalysisInput(
                    **{
                        **input_data.to_dict(),
                        "validated_source_preparation_references": (
                            preparation.manifest_reference,
                        ),
                    }
                )
                result = run_canonical_span_science(
                    input_data, preparation, output_root, executor=self.science_executor
                )
                artifact_reference = publish_terminal_artifact_index(
                    session.project_root, input_data, preparation, result, output_root
                )
                attach_span_artifact_index(session, span_id, artifact_reference)
                complete_run(session, run_id)
                self.rebuild_project_result_index(session)
                return SpanRunOutcome(
                    span_id,
                    "COMPLETE",
                    run_id,
                    summary={
                        "tree_count": result.tree_count,
                        "minimum_clearance_m": result.minimum_clearance_m,
                        "risk_threshold_m": result.risk_threshold_m,
                        "risk_counts": dict(result.risk_counts),
                    },
                )
            except Exception as exc:  # terminal state must truthfully remain FAILED
                _write_failure(
                    _safe_failure_path(session.project_root, span.group_id, span_id, run_id),
                    exc,
                )
                fail_run(session, run_id)
                self.rebuild_project_result_index(session)
                return SpanRunOutcome(
                    span_id, "FAILED", run_id, error=f"{type(exc).__name__}: {exc}"
                )
        finally:
            lock.release()

    def analyze_pending(self, session: WritableProjectSession) -> tuple[SpanRunOutcome, ...]:
        self.synchronize(session)
        reconcile_run_staleness(session)
        pending = [
            span.span_id
            for span in session.span_registry.records
            if span.topology_state.value == "CONFIRMED"
            and span.analysis.state
            in {AnalysisStatus.NOT_STARTED, AnalysisStatus.READY, AnalysisStatus.STALE}
        ]
        return tuple(self.analyze_span(session, span_id) for span_id in pending)

    def retry_failed(self, session: WritableProjectSession) -> tuple[SpanRunOutcome, ...]:
        failed = [
            span.span_id
            for span in session.span_registry.records
            if span.analysis.state is AnalysisStatus.FAILED
        ]
        return tuple(self.analyze_span(session, span_id) for span_id in failed)

    def rebuild_project_result_index(self, session: WritableProjectSession) -> dict[str, Any]:
        root = session.project_root / "artifacts" / "project-runs"
        root.mkdir(parents=True, exist_ok=True)
        groups: dict[str, list[dict[str, Any]]] = {}
        runs = {entry.run_id: entry for entry in session.run_index.entries}
        for span in sorted(
            session.span_registry.records, key=lambda item: int(item.span_id[1:])
        ):
            group_id = span.group_id or "UNGROUPED"
            summary: dict[str, Any] = {
                "span_id": span.span_id,
                "within_group_order": span.within_group_order,
                "from_tower_id": span.from_tower_id,
                "to_tower_id": span.to_tower_id,
                "status": _display_status(span.analysis.state),
                "active_run_id": span.analysis.active_run_id,
                "artifact_index_reference": (
                    span.artifact_index_reference.to_dict()
                    if span.artifact_index_reference
                    else None
                ),
            }
            if span.artifact_index_reference and _artifact_reference_valid(
                session.project_root, span.artifact_index_reference
            ):
                try:
                    artifact = json.loads(
                        (
                            session.project_root
                            / span.artifact_index_reference.relative_path
                        ).read_text(encoding="utf-8")
                    )
                    summary.update(artifact.get("summary", {}))
                except (OSError, ValueError, json.JSONDecodeError):
                    summary["status"] = "FAILED"
            if (
                span.analysis.active_run_id in runs
                and runs[span.analysis.active_run_id].status is AnalysisStatus.FAILED
            ):
                failure_path = (
                    session.project_root
                    / "artifacts"
                    / "project-runs"
                    / "groups"
                    / (span.group_id or "UNGROUPED")
                    / span.span_id
                    / span.analysis.active_run_id
                    / "failure.json"
                )
                if failure_path.is_file():
                    try:
                        summary["failure_reason"] = json.loads(
                            failure_path.read_text(encoding="utf-8")
                        ).get("reason")
                    except (OSError, ValueError, json.JSONDecodeError):
                        summary["failure_reason"] = "failure details unavailable"
            groups.setdefault(group_id, []).append(summary)
        payload = {
            "schema": "treeline-project-result-index-v1",
            "project_id": session.project.project_id,
            "generation_id": session.generation_id,
            "groups": groups,
        }
        _write_json(root / "project-result-index.json", payload)
        rows = []
        for group_id, spans in groups.items():
            rows.append(f"<h2>{html.escape(group_id)}</h2><ul>")
            for span in spans:
                rows.append(
                    "<li><b>{span_id}</b> {from_tower_id} -&gt; {to_tower_id} "
                    "<span>{status}</span></li>".format(
                        **{
                            key: html.escape(str(value))
                            for key, value in span.items()
                            if key
                            in {"span_id", "from_tower_id", "to_tower_id", "status"}
                        }
                    )
                )
            rows.append("</ul>")
        html_body = (
            "<!doctype html><meta charset='utf-8'>"
            "<title>TreeLineWarning results</title>"
            + "".join(rows)
        )
        _write_text(root / "project-result-index.html", html_body)
        return payload


def _display_status(status: AnalysisStatus) -> str:
    return "READY" if status is AnalysisStatus.NOT_STARTED else status.value


def _write_json(path: Path, payload: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_text(path: Path, value: str) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(value + "\n", encoding="utf-8")
    temporary.replace(path)


def _write_failure(path: Path, error: Exception) -> None:
    _write_json(
        path,
        {"schema": "treeline-span-failure-v1", "reason": f"{type(error).__name__}: {error}"},
    )


__all__ = ["ProjectSpanOrchestrator", "SpanRunOutcome"]
