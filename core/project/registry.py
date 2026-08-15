"""Locked P2C-A Tower/Span registry mutations.

This module is deliberately an orchestration boundary.  It owns identity
allocation, immutable registry candidates, cross-reference checks, and the
copy-on-write publication call; it does not perform source relocation or any
scientific processing.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from typing import Any

from core.contracts.project import (
    CoordinateContext,
    ArtifactReference,
    PredictionCapability,
    SourceIdentityBinding,
    SPAN_SCHEMA_VERSION,
    SpanAnalysis,
    SpanContract,
    SpanGeometry,
    SpanTopologyState,
    TOWER_SCHEMA_VERSION,
    TowerContract,
    TowerGeometry,
    TowerLifecycleState,
    TowerRefinement,
    TowerRevisionReference,
    tower_display_id_for_sequence,
    span_display_id_for_sequence,
)
from core.project import persistence
from core.project.persistence import (
    PostPointerDurabilityError,
    RunIndex,
    SourceReferenceSidecar,
    SpanRegistry,
    TowerRegistry,
    TowerRegistryEntry,
)
from core.project.session import WritableProjectSession


class RegistryMutationError(RuntimeError):
    """Base class for P2C-A mutation failures."""


class WriterAuthorityError(RegistryMutationError):
    """Raised when a mutation is attempted without a live writable session."""


class InvalidRegistryCandidate(RegistryMutationError, ValueError):
    """Raised when a candidate is not eligible for a registry mutation."""


class ReservationError(RegistryMutationError):
    """Raised when a caller supplies an unknown or already-consumed reservation."""


class DuplicateSpanError(RegistryMutationError):
    """Raised when an active physical span topology already exists."""


class DependentSpanError(RegistryMutationError):
    """Raised when archiving a Tower would strand active dependent Spans."""


class MutationPublicationUncertainError(RegistryMutationError):
    """Raised after P2A replaced the active pointer but durability is ambiguous."""

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


def _tower_contract_from_fields(
    *,
    tower_id: str,
    revision: int,
    schema_version: str,
    display_name: str,
    state: TowerLifecycleState,
    source_binding: SourceIdentityBinding,
    coordinate_context: CoordinateContext,
    approximate_click_xyz_absolute_m: tuple[float, float, float],
    refined_geometry: TowerGeometry,
    refinement: TowerRefinement,
    notes: str | None,
    created_at: str,
    updated_at: str,
) -> TowerContract:
    return TowerContract(
        schema_version=schema_version,
        tower_id=tower_id,
        display_name=display_name,
        state=state,
        source_binding=source_binding,
        coordinate_context=coordinate_context,
        approximate_click_xyz_absolute_m=approximate_click_xyz_absolute_m,
        refined_geometry=refined_geometry,
        refinement=refinement,
        revision=revision,
        notes=notes,
        created_at=created_at,
        updated_at=updated_at,
    )


@dataclass(frozen=True, slots=True)
class TowerDraft:
    """Validated Tower payload with identity and revision assigned by the service."""

    schema_version: str
    display_name: str
    state: TowerLifecycleState
    source_binding: SourceIdentityBinding
    coordinate_context: CoordinateContext
    approximate_click_xyz_absolute_m: tuple[float, float, float]
    refined_geometry: TowerGeometry
    refinement: TowerRefinement
    notes: str | None
    created_at: str
    updated_at: str

    def __post_init__(self) -> None:
        # Validate the complete P1 contract shape without trusting a caller ID.
        _tower_contract_from_fields(
            tower_id="T001",
            revision=1,
            schema_version=self.schema_version,
            display_name=self.display_name,
            state=self.state,
            source_binding=self.source_binding,
            coordinate_context=self.coordinate_context,
            approximate_click_xyz_absolute_m=self.approximate_click_xyz_absolute_m,
            refined_geometry=self.refined_geometry,
            refinement=self.refinement,
            notes=self.notes,
            created_at=self.created_at,
            updated_at=self.updated_at,
        )

    @classmethod
    def from_contract(cls, tower: TowerContract) -> "TowerDraft":
        if type(tower) is not TowerContract:
            raise InvalidRegistryCandidate("tower candidate must be a TowerContract")
        return cls(
            schema_version=tower.schema_version,
            display_name=tower.display_name,
            state=tower.state,
            source_binding=tower.source_binding,
            coordinate_context=tower.coordinate_context,
            approximate_click_xyz_absolute_m=tower.approximate_click_xyz_absolute_m,
            refined_geometry=tower.refined_geometry,
            refinement=tower.refinement,
            notes=tower.notes,
            created_at=tower.created_at,
            updated_at=tower.updated_at,
        )

    def to_contract(self, tower_id: str, revision: int) -> TowerContract:
        return _tower_contract_from_fields(
            tower_id=tower_id,
            revision=revision,
            schema_version=self.schema_version,
            display_name=self.display_name,
            state=self.state,
            source_binding=self.source_binding,
            coordinate_context=self.coordinate_context,
            approximate_click_xyz_absolute_m=self.approximate_click_xyz_absolute_m,
            refined_geometry=self.refined_geometry,
            refinement=self.refinement,
            notes=self.notes,
            created_at=self.created_at,
            updated_at=self.updated_at,
        )


@dataclass(frozen=True, slots=True)
class SpanDraft:
    """Validated Span payload with the project-local span ID assigned by the service."""

    schema_version: str
    display_name: str
    from_tower_id: str
    to_tower_id: str
    from_tower_revision: int
    to_tower_revision: int
    source_binding: SourceIdentityBinding
    topology_state: SpanTopologyState
    geometry: SpanGeometry
    analysis: SpanAnalysis
    artifact_index_reference: ArtifactReference | None = None
    prediction: PredictionCapability = field(default_factory=PredictionCapability.not_connected)

    def __post_init__(self) -> None:
        SpanContract(
            schema_version=self.schema_version,
            span_id="S001",
            display_name=self.display_name,
            from_tower_id=self.from_tower_id,
            to_tower_id=self.to_tower_id,
            from_tower_revision=self.from_tower_revision,
            to_tower_revision=self.to_tower_revision,
            source_binding=self.source_binding,
            topology_state=self.topology_state,
            geometry=self.geometry,
            analysis=self.analysis,
            artifact_index_reference=self.artifact_index_reference,
            prediction=self.prediction,
        )

    @classmethod
    def from_contract(cls, span: SpanContract) -> "SpanDraft":
        if type(span) is not SpanContract:
            raise InvalidRegistryCandidate("span candidate must be a SpanContract")
        return cls(
            schema_version=span.schema_version,
            display_name=span.display_name,
            from_tower_id=span.from_tower_id,
            to_tower_id=span.to_tower_id,
            from_tower_revision=span.from_tower_revision,
            to_tower_revision=span.to_tower_revision,
            source_binding=span.source_binding,
            topology_state=span.topology_state,
            geometry=span.geometry,
            analysis=span.analysis,
            artifact_index_reference=span.artifact_index_reference,
            prediction=span.prediction,
        )

    def to_contract(self, span_id: str) -> SpanContract:
        return SpanContract(
            schema_version=self.schema_version,
            span_id=span_id,
            display_name=self.display_name,
            from_tower_id=self.from_tower_id,
            to_tower_id=self.to_tower_id,
            from_tower_revision=self.from_tower_revision,
            to_tower_revision=self.to_tower_revision,
            source_binding=self.source_binding,
            topology_state=self.topology_state,
            geometry=self.geometry,
            analysis=self.analysis,
            artifact_index_reference=self.artifact_index_reference,
            prediction=self.prediction,
        )


def _tower_draft(candidate: TowerDraft | TowerContract) -> TowerDraft:
    if type(candidate) is TowerDraft:
        return candidate
    if type(candidate) is TowerContract:
        return TowerDraft.from_contract(candidate)
    raise InvalidRegistryCandidate("tower candidate must be a TowerDraft or TowerContract")


def _span_draft(candidate: SpanDraft | SpanContract) -> SpanDraft:
    if type(candidate) is SpanDraft:
        return candidate
    if type(candidate) is SpanContract:
        return SpanDraft.from_contract(candidate)
    raise InvalidRegistryCandidate("span candidate must be a SpanDraft or SpanContract")


def _require_writable(session: WritableProjectSession) -> Any:
    if type(session) is not WritableProjectSession:
        raise WriterAuthorityError("P2C-A mutation requires an exact open WritableProjectSession")
    return session._begin_registry_mutation()


def _reservation(sequence: int, display_id: str) -> dict[str, Any]:
    return {"sequence": sequence, "display_id": display_id}


def _reserved_sequences(registry: TowerRegistry | SpanRegistry) -> list[dict[str, Any]]:
    return [dict(item) for item in registry.reserved_sequences]


def _with_reservation(
    registry: TowerRegistry | SpanRegistry, *, sequence: int, display_id: str
) -> tuple[dict[str, Any], ...]:
    current = _reserved_sequences(registry)
    if not any(item["sequence"] == sequence for item in current):
        current.append(_reservation(sequence, display_id))
    current.sort(key=lambda item: item["sequence"])
    return tuple(current)


def _tower_sequences(registry: TowerRegistry) -> set[int]:
    return {
        *(int(item.tower_id[1:]) for entry in registry.records for item in entry.revisions),
        *(int(item["sequence"]) for item in registry.reserved_sequences),
    }


def _span_sequences(registry: SpanRegistry) -> set[int]:
    return {
        *(int(item.span_id[1:]) for item in registry.records),
        *(int(item["sequence"]) for item in registry.reserved_sequences),
    }


def _require_counter_ahead(value: int, represented: set[int], name: str) -> None:
    if value <= max(represented, default=0):
        raise InvalidRegistryCandidate(f"{name} is not above every represented/reserved identity")


def _parse_reservation(value: str, *, prefix: str, display_id_for_sequence) -> int:
    if type(value) is not str or not value.startswith(prefix):
        raise ReservationError(f"{prefix} reservation must be a canonical project display ID")
    try:
        sequence = int(value[1:])
    except (TypeError, ValueError) as exc:
        raise ReservationError("reservation is not a canonical project display ID") from exc
    try:
        expected = display_id_for_sequence(sequence)
    except ValueError as exc:
        raise ReservationError("reservation is not canonical") from exc
    if value != expected:
        raise ReservationError("reservation is not canonical")
    return sequence


def _same_current_source(session: WritableProjectSession, binding: SourceIdentityBinding) -> None:
    current = session.project.source_reference.binding
    if binding != current:
        raise InvalidRegistryCandidate(
            "new Tower/Span records must bind to the current project SourceIdentity"
        )


def _publish(
    session: WritableProjectSession,
    loaded: Any,
    *,
    project: Any,
    tower_registry: TowerRegistry,
    span_registry: SpanRegistry,
) -> Any:
    try:
        published = persistence._publish_generation(
            loaded.project_root,
            project,
            source_sidecar=loaded.source_sidecar,
            tower_registry=tower_registry,
            span_registry=span_registry,
            run_index=loaded.run_index,
            candidate_store=loaded.candidate_store,
        )
    except PostPointerDurabilityError as exc:
        # The lock is still held.  Reopen the pointer before reporting the
        # uncertain result or allowing a later mutation.
        try:
            session._revalidate_authoritative_state()
        except BaseException as revalidation_error:
            raise MutationPublicationUncertainError(
                f"mutation publication is uncertain and authoritative revalidation failed: {revalidation_error}",
                original=exc,
            ) from exc
        raise MutationPublicationUncertainError(
            "mutation publication is uncertain; authoritative state was reopened under the writer lock",
            original=exc,
        ) from exc
    session._accept_published_generation(published)
    return published


def _updated_project(project: Any, **changes: Any) -> Any:
    return replace(project, updated_at=_timestamp_at_or_after(project.updated_at), **changes)


def reserve_tower_id(session: WritableProjectSession) -> str:
    """Durably reserve the next never-before-used Tower identity."""

    loaded = _require_writable(session)
    registry = loaded.tower_registry
    sequence = loaded.project.next_tower_sequence
    _require_counter_ahead(sequence, _tower_sequences(registry), "next_tower_sequence")
    tower_id = tower_display_id_for_sequence(sequence)
    candidate_registry = TowerRegistry(
        registry.schema_version,
        registry.generation_id,
        registry.project_id,
        reserved_sequences=_with_reservation(registry, sequence=sequence, display_id=tower_id),
        records=registry.records,
    )
    candidate_project = _updated_project(
        loaded.project, next_tower_sequence=sequence + 1
    )
    _publish(
        session,
        loaded,
        project=candidate_project,
        tower_registry=candidate_registry,
        span_registry=loaded.span_registry,
    )
    return tower_id


def _allocated_tower(
    loaded: Any,
    draft: TowerDraft,
    *,
    reservation: str | None = None,
) -> tuple[TowerContract, TowerRegistry, Any]:
    """Pure in-memory next-Tower allocation; never publishes or touches the session.

    Both ``create_tower`` and the V5A1-B ``confirm_reviewed_tower_candidate``
    adapter prepare their canonical ``T###`` through this single allocation path.
    On any later publication failure the prepared identity lives only in an
    unreferenced staging generation and never becomes authoritative.
    """

    if type(draft) is not TowerDraft:
        raise InvalidRegistryCandidate("tower candidate must be a TowerDraft")
    registry = loaded.tower_registry
    represented = _tower_sequences(registry)
    if reservation is None:
        sequence = loaded.project.next_tower_sequence
        _require_counter_ahead(sequence, represented, "next_tower_sequence")
        new_counter = sequence + 1
    else:
        sequence = _parse_reservation(
            reservation, prefix="T", display_id_for_sequence=tower_display_id_for_sequence
        )
        if not any(item["display_id"] == reservation for item in registry.reserved_sequences):
            raise ReservationError("tower reservation is not durably present")
        if sequence in {int(entry.current.tower_id[1:]) for entry in registry.records}:
            raise ReservationError("tower reservation has already been confirmed")
        new_counter = loaded.project.next_tower_sequence
    tower_id = tower_display_id_for_sequence(sequence)
    tower = draft.to_contract(tower_id, 1)
    entry = TowerRegistryEntry(tower_id=tower_id, current_revision=1, revisions=(tower,))
    records = tuple((*registry.records, entry))
    records = tuple(sorted(records, key=lambda item: int(item.tower_id[1:])))
    candidate_registry = TowerRegistry(
        registry.schema_version,
        registry.generation_id,
        registry.project_id,
        reserved_sequences=_with_reservation(registry, sequence=sequence, display_id=tower_id),
        records=records,
    )
    candidate_project = _updated_project(loaded.project, next_tower_sequence=new_counter)
    return tower, candidate_registry, candidate_project


def create_tower(
    session: WritableProjectSession,
    candidate: TowerDraft | TowerContract,
    *,
    reservation: str | None = None,
) -> TowerContract:
    """Validate and publish one confirmed Tower, assigning its project-local ID."""

    draft = _tower_draft(candidate)
    if draft.state is not TowerLifecycleState.CONFIRMED:
        raise InvalidRegistryCandidate("new Tower candidates must be CONFIRMED")
    loaded = _require_writable(session)
    session.require_strong_source_identity()
    _same_current_source(session, draft.source_binding)
    tower, candidate_registry, candidate_project = _allocated_tower(
        loaded, draft, reservation=reservation
    )
    _publish(
        session,
        loaded,
        project=candidate_project,
        tower_registry=candidate_registry,
        span_registry=loaded.span_registry,
    )
    return tower


def revise_tower(
    session: WritableProjectSession,
    tower_id: str,
    candidate: TowerDraft | TowerContract,
) -> TowerContract:
    """Append a new immutable domain revision to one existing Tower identity."""

    draft = _tower_draft(candidate)
    if draft.state is not TowerLifecycleState.CONFIRMED:
        raise InvalidRegistryCandidate("Tower revisions must remain CONFIRMED")
    loaded = _require_writable(session)
    session.require_strong_source_identity()
    _same_current_source(session, draft.source_binding)
    try:
        entry = loaded.tower_registry.get(tower_id)
    except KeyError as exc:
        raise InvalidRegistryCandidate("cannot revise an unknown Tower") from exc
    if entry.current.state is TowerLifecycleState.ARCHIVED:
        raise InvalidRegistryCandidate("an archived Tower cannot receive a new confirmed revision")
    revision = entry.current_revision + 1
    revised = draft.to_contract(tower_id, revision)
    replacement = TowerRegistryEntry(
        tower_id=tower_id,
        current_revision=revision,
        revisions=(*entry.revisions, revised),
    )
    records = tuple(replacement if item.tower_id == tower_id else item for item in loaded.tower_registry.records)
    candidate_registry = TowerRegistry(
        loaded.tower_registry.schema_version,
        loaded.tower_registry.generation_id,
        loaded.tower_registry.project_id,
        reserved_sequences=loaded.tower_registry.reserved_sequences,
        records=records,
    )
    _publish(
        session,
        loaded,
        project=_updated_project(loaded.project),
        tower_registry=candidate_registry,
        span_registry=loaded.span_registry,
    )
    return revised


def archive_tower(
    session: WritableProjectSession,
    tower_id: str,
    *,
    cascade_dependent_spans: bool = False,
) -> TowerContract:
    """Archive a Tower, optionally archiving all active dependent Spans atomically."""

    loaded = _require_writable(session)
    session.require_strong_source_identity()
    try:
        entry = loaded.tower_registry.get(tower_id)
    except KeyError as exc:
        raise InvalidRegistryCandidate("cannot archive an unknown Tower") from exc
    if entry.current.state is TowerLifecycleState.ARCHIVED:
        raise InvalidRegistryCandidate("Tower is already archived")
    dependents = tuple(
        span
        for span in loaded.span_registry.records
        if span.topology_state is SpanTopologyState.CONFIRMED
        and tower_id in {span.from_tower_id, span.to_tower_id}
    )
    if dependents and not cascade_dependent_spans:
        raise DependentSpanError(
            "active dependent Spans exist; pass cascade_dependent_spans=True explicitly"
        )
    archived = replace(
        entry.current,
        state=TowerLifecycleState.ARCHIVED,
        revision=entry.current_revision + 1,
        updated_at=_timestamp_at_or_after(entry.current.updated_at),
    )
    archived_entry = TowerRegistryEntry(
        tower_id=tower_id,
        current_revision=archived.revision,
        revisions=(*entry.revisions, archived),
    )
    tower_records = tuple(
        archived_entry if item.tower_id == tower_id else item
        for item in loaded.tower_registry.records
    )
    span_records = tuple(
        replace(span, topology_state=SpanTopologyState.ARCHIVED)
        if cascade_dependent_spans and span in dependents
        else span
        for span in loaded.span_registry.records
    )
    candidate_towers = TowerRegistry(
        loaded.tower_registry.schema_version,
        loaded.tower_registry.generation_id,
        loaded.tower_registry.project_id,
        reserved_sequences=loaded.tower_registry.reserved_sequences,
        records=tower_records,
    )
    candidate_spans = SpanRegistry(
        loaded.span_registry.schema_version,
        loaded.span_registry.generation_id,
        loaded.span_registry.project_id,
        reserved_sequences=loaded.span_registry.reserved_sequences,
        records=span_records,
    )
    _publish(
        session,
        loaded,
        project=_updated_project(loaded.project),
        tower_registry=candidate_towers,
        span_registry=candidate_spans,
    )
    return archived


def reserve_span_id(session: WritableProjectSession) -> str:
    """Durably reserve the next never-before-used Span identity."""

    loaded = _require_writable(session)
    registry = loaded.span_registry
    sequence = loaded.project.next_span_sequence
    _require_counter_ahead(sequence, _span_sequences(registry), "next_span_sequence")
    span_id = span_display_id_for_sequence(sequence)
    candidate_registry = SpanRegistry(
        registry.schema_version,
        registry.generation_id,
        registry.project_id,
        reserved_sequences=_with_reservation(registry, sequence=sequence, display_id=span_id),
        records=registry.records,
    )
    candidate_project = _updated_project(loaded.project, next_span_sequence=sequence + 1)
    _publish(
        session,
        loaded,
        project=candidate_project,
        tower_registry=loaded.tower_registry,
        span_registry=candidate_registry,
    )
    return span_id


def create_span(
    session: WritableProjectSession,
    candidate: SpanDraft | SpanContract,
    *,
    reservation: str | None = None,
) -> SpanContract:
    """Validate endpoints and publish one Span with an allocator-assigned ID."""

    draft = _span_draft(candidate)
    if draft.topology_state is not SpanTopologyState.CONFIRMED:
        raise InvalidRegistryCandidate("new Span candidates must be CONFIRMED")
    loaded = _require_writable(session)
    session.require_strong_source_identity()
    _same_current_source(session, draft.source_binding)
    try:
        from_entry = loaded.tower_registry.get(draft.from_tower_id)
        to_entry = loaded.tower_registry.get(draft.to_tower_id)
        from_entry.revision(draft.from_tower_revision)
        to_entry.revision(draft.to_tower_revision)
    except KeyError as exc:
        raise InvalidRegistryCandidate("Span endpoints and endpoint revisions must exist") from exc
    if (
        from_entry.current.state is TowerLifecycleState.ARCHIVED
        or to_entry.current.state is TowerLifecycleState.ARCHIVED
    ):
        raise InvalidRegistryCandidate("active Span cannot use an archived Tower")
    physical_pair = frozenset((draft.from_tower_id, draft.to_tower_id))
    if any(
        span.topology_state is SpanTopologyState.CONFIRMED
        and frozenset((span.from_tower_id, span.to_tower_id)) == physical_pair
        for span in loaded.span_registry.records
    ):
        raise DuplicateSpanError("an active physical Span with these endpoints already exists")
    registry = loaded.span_registry
    represented = _span_sequences(registry)
    if reservation is None:
        sequence = loaded.project.next_span_sequence
        _require_counter_ahead(sequence, represented, "next_span_sequence")
        new_counter = sequence + 1
    else:
        sequence = _parse_reservation(
            reservation, prefix="S", display_id_for_sequence=span_display_id_for_sequence
        )
        if not any(item["display_id"] == reservation for item in registry.reserved_sequences):
            raise ReservationError("span reservation is not durably present")
        if sequence in {int(span.span_id[1:]) for span in registry.records}:
            raise ReservationError("span reservation has already been confirmed")
        new_counter = loaded.project.next_span_sequence
    span_id = span_display_id_for_sequence(sequence)
    span = draft.to_contract(span_id)
    span_records = tuple(sorted((*registry.records, span), key=lambda item: int(item.span_id[1:])))
    candidate_registry = SpanRegistry(
        registry.schema_version,
        registry.generation_id,
        registry.project_id,
        reserved_sequences=_with_reservation(registry, sequence=sequence, display_id=span_id),
        records=span_records,
    )
    candidate_project = _updated_project(loaded.project, next_span_sequence=new_counter)
    _publish(
        session,
        loaded,
        project=candidate_project,
        tower_registry=loaded.tower_registry,
        span_registry=candidate_registry,
    )
    return span


def archive_span(session: WritableProjectSession, span_id: str) -> SpanContract:
    """Archive a Span without deleting its topology or identity reservation."""

    loaded = _require_writable(session)
    try:
        current = next(span for span in loaded.span_registry.records if span.span_id == span_id)
    except StopIteration as exc:
        raise InvalidRegistryCandidate("cannot archive an unknown Span") from exc
    if current.topology_state is SpanTopologyState.ARCHIVED:
        raise InvalidRegistryCandidate("Span is already archived")
    archived = replace(current, topology_state=SpanTopologyState.ARCHIVED)
    records = tuple(archived if span.span_id == span_id else span for span in loaded.span_registry.records)
    candidate_registry = SpanRegistry(
        loaded.span_registry.schema_version,
        loaded.span_registry.generation_id,
        loaded.span_registry.project_id,
        reserved_sequences=loaded.span_registry.reserved_sequences,
        records=records,
    )
    _publish(
        session,
        loaded,
        project=_updated_project(loaded.project),
        tower_registry=loaded.tower_registry,
        span_registry=candidate_registry,
    )
    return archived


__all__ = [
    "DependentSpanError",
    "DuplicateSpanError",
    "InvalidRegistryCandidate",
    "MutationPublicationUncertainError",
    "RegistryMutationError",
    "ReservationError",
    "SpanDraft",
    "TowerDraft",
    "WriterAuthorityError",
    "archive_span",
    "archive_tower",
    "create_span",
    "create_tower",
    "reserve_span_id",
    "reserve_tower_id",
    "revise_tower",
]
