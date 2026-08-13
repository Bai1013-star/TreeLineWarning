"""Locked authoritative P2C-B2 run-index lifecycle mutations.

This module records project-side run provenance only.  It does not dispatch
computation, create result files, or interpret scientific artifacts.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
from typing import Any

from core.contracts.project import (
    AnalysisStatus,
    SpanAnalysis,
    SpanTopologyState,
)
from core.project import persistence
from core.project.persistence import (
    PostPointerDurabilityError,
    RunIndex,
    RunIndexEntry,
    SpanRegistry,
)
from core.project.registry import WriterAuthorityError
from core.project.session import (
    ProjectSessionClosedError,
    ProjectSessionError,
    WritableProjectSession,
)


class RunIndexMutationError(RuntimeError):
    """Base error for P2C-B2 run-index mutation failures."""


class InvalidRunCandidate(RunIndexMutationError, ValueError):
    """Raised when a requested run mutation violates a frozen run invariant."""


class DuplicateRunError(RunIndexMutationError):
    """Raised when a historical run identity is reused."""


class UnknownRunError(RunIndexMutationError):
    """Raised when a requested run identity is not present in the run index."""


class InvalidRunTransitionError(RunIndexMutationError):
    """Raised when a lifecycle transition is not allowed by the run model."""


class RunIndexPublicationUncertainError(RunIndexMutationError):
    """Raised after pointer replacement when root durability is uncertain."""

    def __init__(self, message: str, *, original: PostPointerDurabilityError) -> None:
        super().__init__(message)
        self.original = original
        self.pointer_replaced = original.pointer_replaced


def _timestamp_at_or_after(value: str) -> str:
    now = datetime.now(UTC)
    try:
        previous = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError:
        previous = now
    if now < previous:
        now = previous
    return now.isoformat(timespec="microseconds").replace("+00:00", "Z")


def _require_writable(session: WritableProjectSession) -> Any:
    if type(session) is not WritableProjectSession:
        raise WriterAuthorityError("P2C-B2 mutation requires an exact open WritableProjectSession")
    try:
        return session._begin_registry_mutation()
    except (ProjectSessionClosedError, ProjectSessionError):
        raise
    except (AttributeError, TypeError) as exc:
        raise WriterAuthorityError("P2C-B2 mutation received a fabricated writable session") from exc


def _updated_project(project: Any) -> Any:
    return replace(project, updated_at=_timestamp_at_or_after(project.updated_at))


def _run_entry(loaded: Any, run_id: str) -> RunIndexEntry:
    try:
        return next(entry for entry in loaded.run_index.entries if entry.run_id == run_id)
    except StopIteration as exc:
        raise UnknownRunError("run_id is not present in the authoritative run index") from exc


def _span(loaded: Any, span_id: str):
    try:
        return next(item for item in loaded.span_registry.records if item.span_id == span_id)
    except StopIteration as exc:
        raise InvalidRunCandidate("run references an unknown Span") from exc


def _history_with_active(analysis: SpanAnalysis) -> tuple[str, ...]:
    history = list(analysis.historical_run_ids)
    if analysis.active_run_id is not None and analysis.active_run_id not in history:
        history.append(analysis.active_run_id)
    return tuple(history)


def _analysis_for_run(
    entry: RunIndexEntry,
    *,
    history: tuple[str, ...],
) -> SpanAnalysis:
    return persistence._span_analysis_projection_for_run(
        entry, historical_run_ids=history
    )


def _require_active_span_for_entry(loaded: Any, entry: RunIndexEntry):
    span = _span(loaded, entry.span_id)
    analysis = span.analysis
    if (
        analysis.active_run_id != entry.run_id
        or analysis.run_dependencies != entry.span_dependencies
        or analysis.state is not entry.status
    ):
        raise InvalidRunCandidate(
            "run-index entry is not the current coherent Span analysis state"
        )
    return span


def _replace_span_analysis(loaded: Any, span_id: str, analysis: SpanAnalysis) -> SpanRegistry:
    records = tuple(
        replace(span, analysis=analysis) if span.span_id == span_id else span
        for span in loaded.span_registry.records
    )
    return SpanRegistry(
        loaded.span_registry.schema_version,
        loaded.span_registry.generation_id,
        loaded.span_registry.project_id,
        reserved_sequences=loaded.span_registry.reserved_sequences,
        records=records,
    )


def _with_entry(loaded: Any, replacement: RunIndexEntry) -> RunIndex:
    entries = tuple(
        replacement if item.run_id == replacement.run_id else item
        for item in loaded.run_index.entries
    )
    return RunIndex(
        loaded.run_index.schema_version,
        loaded.run_index.generation_id,
        loaded.run_index.project_id,
        entries=tuple(sorted(entries, key=lambda item: item.run_id)),
    )


def _with_new_entry(loaded: Any, entry: RunIndexEntry) -> RunIndex:
    return RunIndex(
        loaded.run_index.schema_version,
        loaded.run_index.generation_id,
        loaded.run_index.project_id,
        entries=tuple(sorted((*loaded.run_index.entries, entry), key=lambda item: item.run_id)),
    )


def _publish(
    session: WritableProjectSession,
    loaded: Any,
    *,
    run_index: RunIndex,
    span_registry: SpanRegistry,
) -> None:
    try:
        published = persistence._publish_generation(
            loaded.project_root,
            _updated_project(loaded.project),
            source_sidecar=loaded.source_sidecar,
            tower_registry=loaded.tower_registry,
            span_registry=span_registry,
            run_index=run_index,
        )
    except PostPointerDurabilityError as exc:
        try:
            session._revalidate_authoritative_state()
        except BaseException as revalidation_error:
            raise RunIndexPublicationUncertainError(
                "run-index publication is uncertain and authoritative revalidation failed: "
                f"{revalidation_error}",
                original=exc,
            ) from exc
        raise RunIndexPublicationUncertainError(
            "run-index publication is uncertain; authoritative state was reopened under "
            "the writer lock",
            original=exc,
        ) from exc
    session._accept_published_generation(published)


def register_run(
    session: WritableProjectSession,
    run_id: str,
    span_id: str,
    *,
    artifact_root_reference: str | None = None,
) -> RunIndexEntry:
    """Register one READY run against the current immutable Span dependencies."""

    loaded = _require_writable(session)
    if any(entry.run_id == run_id for entry in loaded.run_index.entries):
        raise DuplicateRunError("run_id has already been used by a historical run")
    span = _span(loaded, span_id)
    if span.topology_state is not SpanTopologyState.CONFIRMED:
        raise InvalidRunCandidate("new runs cannot be registered for an archived Span")
    if span.analysis.state in {AnalysisStatus.READY, AnalysisStatus.RUNNING}:
        raise InvalidRunCandidate("Span already has an active nonterminal run")
    # READY authorizes a later source-dependent execution, so it retains the
    # P2B strong source gate rather than trusting only a fast availability check.
    session.require_strong_source_identity()
    created_at = persistence._utc_timestamp()
    try:
        entry = RunIndexEntry(
            run_id=run_id,
            span_id=span_id,
            status=AnalysisStatus.READY,
            artifact_root_reference=artifact_root_reference,
            source_content_identity_digest=(
                span.current_dependencies.source_binding.source_content_identity_digest
            ),
            source_identity_record_digest=(
                span.current_dependencies.source_binding.source_identity_record_digest
            ),
            span_dependencies=span.current_dependencies,
            created_at=created_at,
            completed_at=None,
        )
    except (TypeError, ValueError) as exc:
        raise InvalidRunCandidate(f"invalid run registration: {exc}") from exc
    candidate_spans = _replace_span_analysis(
        loaded,
        span_id,
        _analysis_for_run(entry, history=_history_with_active(span.analysis)),
    )
    _publish(
        session,
        loaded,
        run_index=_with_new_entry(loaded, entry),
        span_registry=candidate_spans,
    )
    return entry


def start_run(session: WritableProjectSession, run_id: str) -> RunIndexEntry:
    """Transition exactly one READY run to RUNNING."""

    loaded = _require_writable(session)
    entry = _run_entry(loaded, run_id)
    if entry.status is not AnalysisStatus.READY:
        raise InvalidRunTransitionError("only READY runs may transition to RUNNING")
    span = _require_active_span_for_entry(loaded, entry)
    if entry.span_dependencies != span.current_dependencies:
        raise InvalidRunTransitionError("stale run dependencies must be reconciled before start")
    session.require_strong_source_identity()
    replacement = replace(entry, status=AnalysisStatus.RUNNING)
    _publish(
        session,
        loaded,
        run_index=_with_entry(loaded, replacement),
        span_registry=_replace_span_analysis(
            loaded,
            span.span_id,
            _analysis_for_run(replacement, history=span.analysis.historical_run_ids),
        ),
    )
    return replacement


def _finish_run(
    session: WritableProjectSession,
    run_id: str,
    *,
    target_status: AnalysisStatus,
) -> RunIndexEntry:
    loaded = _require_writable(session)
    entry = _run_entry(loaded, run_id)
    if entry.status is not AnalysisStatus.RUNNING:
        raise InvalidRunTransitionError(
            f"only RUNNING runs may transition to {target_status.value}"
        )
    span = _require_active_span_for_entry(loaded, entry)
    if entry.span_dependencies != span.current_dependencies:
        raise InvalidRunTransitionError("stale run dependencies must be reconciled before finish")
    replacement = replace(
        entry,
        status=target_status,
        completed_at=_timestamp_at_or_after(entry.created_at),
    )
    history = _history_with_active(span.analysis)
    _publish(
        session,
        loaded,
        run_index=_with_entry(loaded, replacement),
        span_registry=_replace_span_analysis(
            loaded,
            span.span_id,
            _analysis_for_run(replacement, history=history),
        ),
    )
    return replacement


def complete_run(session: WritableProjectSession, run_id: str) -> RunIndexEntry:
    """Record contract-valid lifecycle completion; no scientific work is performed."""

    return _finish_run(session, run_id, target_status=AnalysisStatus.COMPLETE)


def fail_run(session: WritableProjectSession, run_id: str) -> RunIndexEntry:
    """Persist a FAILED historical run; the frozen schema has no failure payload."""

    return _finish_run(session, run_id, target_status=AnalysisStatus.FAILED)


def interrupt_run(session: WritableProjectSession, run_id: str) -> RunIndexEntry:
    """Persist an INTERRUPTED historical run without deleting its provenance."""

    return _finish_run(session, run_id, target_status=AnalysisStatus.INTERRUPTED)


def attach_run_artifact_root(
    session: WritableProjectSession,
    run_id: str,
    artifact_root_reference: str,
) -> RunIndexEntry:
    """Bind the one frozen safe artifact-root reference before terminalization."""

    loaded = _require_writable(session)
    entry = _run_entry(loaded, run_id)
    if entry.status not in {AnalysisStatus.READY, AnalysisStatus.RUNNING}:
        raise InvalidRunTransitionError(
            "artifact_root_reference cannot be changed on a terminal historical run"
        )
    span = _require_active_span_for_entry(loaded, entry)
    if entry.artifact_root_reference is not None:
        if entry.artifact_root_reference == artifact_root_reference:
            return entry
        raise InvalidRunCandidate("run already has an immutable artifact_root_reference")
    try:
        replacement = replace(entry, artifact_root_reference=artifact_root_reference)
    except (TypeError, ValueError) as exc:
        raise InvalidRunCandidate(f"invalid artifact_root_reference: {exc}") from exc
    _publish(
        session,
        loaded,
        run_index=_with_entry(loaded, replacement),
        span_registry=loaded.span_registry,
    )
    return replacement


def reconcile_run_staleness(session: WritableProjectSession) -> tuple[RunIndexEntry, ...]:
    """Mark dependency-mismatched historical runs STALE without rewriting them."""

    loaded = _require_writable(session)
    spans = {span.span_id: span for span in loaded.span_registry.records}
    replacements: dict[str, RunIndexEntry] = {}
    replacement_analyses: dict[str, SpanAnalysis] = {}
    for entry in loaded.run_index.entries:
        span = spans[entry.span_id]
        if entry.status is AnalysisStatus.STALE or entry.span_dependencies == span.current_dependencies:
            continue
        replacement = replace(entry, status=AnalysisStatus.STALE)
        replacements[entry.run_id] = replacement
        if span.analysis.active_run_id == entry.run_id:
            replacement_analyses[span.span_id] = _analysis_for_run(
                replacement,
                history=_history_with_active(span.analysis),
            )
    if not replacements:
        return ()
    run_index = RunIndex(
        loaded.run_index.schema_version,
        loaded.run_index.generation_id,
        loaded.run_index.project_id,
        entries=tuple(
            sorted(
                (replacements.get(entry.run_id, entry) for entry in loaded.run_index.entries),
                key=lambda item: item.run_id,
            )
        ),
    )
    span_registry = SpanRegistry(
        loaded.span_registry.schema_version,
        loaded.span_registry.generation_id,
        loaded.span_registry.project_id,
        reserved_sequences=loaded.span_registry.reserved_sequences,
        records=tuple(
            replace(span, analysis=replacement_analyses[span.span_id])
            if span.span_id in replacement_analyses
            else span
            for span in loaded.span_registry.records
        ),
    )
    _publish(session, loaded, run_index=run_index, span_registry=span_registry)
    return tuple(replacements[entry.run_id] for entry in loaded.run_index.entries if entry.run_id in replacements)


__all__ = [
    "DuplicateRunError",
    "InvalidRunCandidate",
    "InvalidRunTransitionError",
    "RunIndexMutationError",
    "RunIndexPublicationUncertainError",
    "UnknownRunError",
    "attach_run_artifact_root",
    "complete_run",
    "fail_run",
    "interrupt_run",
    "reconcile_run_staleness",
    "register_run",
    "start_run",
]
