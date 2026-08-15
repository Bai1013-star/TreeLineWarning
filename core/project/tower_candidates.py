"""Locked V5A1-B tower-candidate persistence and review-to-TowerRegistry adapter.

This module is an orchestration boundary.  It persists immutable V5A1-A
candidate artifacts through project generations, appends immutable review
events through a writable session, and confirms one effective ACCEPTED +
RAW_REFINED candidate review into the existing ``TowerRegistry`` with one
canonical ``T###`` allocation and one atomic publication.

It performs no automatic detection, no RAW refinement, no geometry/wire
topology analysis, no Giro3D interaction, no span ordering, and no span
creation.  ``T###`` allocation reuses the existing registry allocation
semantics; ``P###`` candidate identities are never tower identities.
"""

from __future__ import annotations

import secrets
from dataclasses import replace
from datetime import UTC, datetime
from typing import Any

from core.contracts.project import (
    TOWER_REFINEMENT_SCHEMA_VERSION,
    TOWER_SCHEMA_VERSION,
    TowerLifecycleState,
    TowerRefinement,
    TowerRefinementQuality,
    TowerRefinementStatus,
)
from core.contracts.tower_candidates import (
    TOWER_CANDIDATE_CONFIRMATION_SCHEMA_VERSION,
    TOWER_CANDIDATE_REVIEW_EVENT_SCHEMA_VERSION,
    TOWER_CANDIDATE_REVIEW_LEDGER_SCHEMA_VERSION,
    CandidateSetReference,
    RefinementState,
    ReviewDecision,
    TowerCandidate,
    TowerCandidateConfirmation,
    TowerCandidateRawRefinementEvidence,
    TowerCandidateReviewEvent,
    TowerCandidateReviewLedger,
    TowerCandidateSet,
    validate_tower_candidate_confirmation,
    validate_tower_candidate_review_ledger,
)
from core.project import persistence
from core.project.persistence import (
    CANDIDATE_STORE_INDEX_SCHEMA_VERSION,
    CandidateStore,
    PostPointerDurabilityError,
)
from core.project.registry import (
    TowerDraft,
    WriterAuthorityError,
    _allocated_tower,
)
from core.project.session import (
    ProjectSessionClosedError,
    ProjectSessionError,
    WritableProjectSession,
)


class TowerCandidateMutationError(RuntimeError):
    """Base error for V5A1-B tower-candidate persistence mutations."""


class InvalidTowerCandidateState(TowerCandidateMutationError, ValueError):
    """Raised when a candidate/review/confirmation request violates a frozen invariant."""


class AlreadyConfirmedError(TowerCandidateMutationError):
    """Raised when a candidate review has already produced a Tower confirmation."""


class CandidateConfirmationPublicationUncertainError(TowerCandidateMutationError):
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


def _updated_project(project: Any, **changes: Any) -> Any:
    return replace(project, updated_at=_timestamp_at_or_after(project.updated_at), **changes)


def _require_writable(session: WritableProjectSession) -> Any:
    if type(session) is not WritableProjectSession:
        raise WriterAuthorityError(
            "V5A1-B mutation requires an exact open WritableProjectSession"
        )
    try:
        return session._begin_registry_mutation()
    except (ProjectSessionClosedError, ProjectSessionError):
        raise
    except (AttributeError, TypeError) as exc:
        raise WriterAuthorityError(
            "V5A1-B mutation received a fabricated writable session"
        ) from exc


def _same_current_source(session: WritableProjectSession, binding: Any) -> None:
    current = session.project.source_reference.binding
    if binding != current:
        raise InvalidTowerCandidateState(
            "candidate Tower records must bind to the current project SourceIdentity"
        )


def _candidate_by_id(candidate_set: TowerCandidateSet, candidate_id: str) -> TowerCandidate:
    for candidate in candidate_set.candidates:
        if candidate.candidate_id == candidate_id:
            return candidate
    raise InvalidTowerCandidateState(
        f"candidate {candidate_id} is absent from the candidate set"
    )


def _resolve_candidate_set(
    loaded: Any, reference: CandidateSetReference
) -> TowerCandidateSet:
    if type(reference) is not CandidateSetReference:
        raise InvalidTowerCandidateState("candidate_set_reference must be a CandidateSetReference")
    for candidate_set in loaded.candidate_store.candidate_sets:
        if candidate_set.candidate_set_id == reference.candidate_set_id:
            if candidate_set.reference != reference:
                raise InvalidTowerCandidateState(
                    "candidate-set reference digest does not match the actual persisted set"
                )
            return candidate_set
    raise InvalidTowerCandidateState(
        "candidate-set reference does not resolve to a persisted candidate set"
    )


def _resolve_ledger(
    store: CandidateStore, candidate_set: TowerCandidateSet
) -> TowerCandidateReviewLedger | None:
    for ledger in store.review_ledgers:
        if ledger.candidate_sets and (
            ledger.candidate_sets[0].candidate_set_id == candidate_set.candidate_set_id
        ):
            return ledger
    return None


def _confirmation_for(
    store: CandidateStore, candidate_set_id: str, candidate_id: str
) -> TowerCandidateConfirmation | None:
    for confirmation in store.confirmations:
        if (
            confirmation.candidate_set_reference.candidate_set_id == candidate_set_id
            and confirmation.candidate_id == candidate_id
        ):
            return confirmation
    return None


def _replace_ledger(
    ledgers: tuple[TowerCandidateReviewLedger, ...],
    candidate_set_id: str,
    replacement: TowerCandidateReviewLedger,
) -> tuple[TowerCandidateReviewLedger, ...]:
    return tuple(
        replacement if item.review_ledger_id == replacement.review_ledger_id else item
        for item in ledgers
    )


def _new_review_ledger_id() -> str:
    return f"rl-{secrets.token_hex(8)}"


def _publish(
    session: WritableProjectSession,
    loaded: Any,
    *,
    candidate_store: CandidateStore,
    project: Any | None = None,
    tower_registry: Any | None = None,
) -> Any:
    if project is None:
        project = _updated_project(loaded.project)
    if tower_registry is None:
        tower_registry = loaded.tower_registry
    try:
        published = persistence._publish_generation(
            loaded.project_root,
            project,
            source_sidecar=loaded.source_sidecar,
            tower_registry=tower_registry,
            span_registry=loaded.span_registry,
            run_index=loaded.run_index,
            candidate_store=candidate_store,
        )
    except PostPointerDurabilityError as exc:
        try:
            session._revalidate_authoritative_state()
        except BaseException as revalidation_error:
            raise CandidateConfirmationPublicationUncertainError(
                "candidate-store publication is uncertain and authoritative "
                f"revalidation failed: {revalidation_error}",
                original=exc,
            ) from exc
        raise CandidateConfirmationPublicationUncertainError(
            "candidate-store publication is uncertain; authoritative state was "
            "reopened under the writer lock",
            original=exc,
        ) from exc
    session._accept_published_generation(published)
    return published


def persist_tower_candidate_set(
    session: WritableProjectSession,
    candidate_set: TowerCandidateSet,
) -> TowerCandidateSet:
    """Persist one immutable candidate-set artifact through a new generation."""
    if type(candidate_set) is not TowerCandidateSet:
        raise InvalidTowerCandidateState("candidate_set must be a TowerCandidateSet")
    loaded = _require_writable(session)
    if candidate_set.project_id != loaded.project.project_id:
        raise InvalidTowerCandidateState("candidate-set project_id does not match the project")
    _same_current_source(session, candidate_set.source_binding)
    if any(
        item.candidate_set_id == candidate_set.candidate_set_id
        for item in loaded.candidate_store.candidate_sets
    ):
        raise InvalidTowerCandidateState(
            "a candidate set with this id is already persisted in the project"
        )
    candidate_store = CandidateStore(
        schema_version=CANDIDATE_STORE_INDEX_SCHEMA_VERSION,
        candidate_sets=(*loaded.candidate_store.candidate_sets, candidate_set),
        review_ledgers=loaded.candidate_store.review_ledgers,
        confirmations=loaded.candidate_store.confirmations,
    )
    _publish(session, loaded, candidate_store=candidate_store)
    return candidate_set


def append_tower_candidate_review(
    session: WritableProjectSession,
    *,
    candidate_set_reference: CandidateSetReference,
    candidate_id: str,
    decision: ReviewDecision,
    occurred_at: str,
    reason: str,
    refinement_state: RefinementState,
    refined_raw_geometry: Any | None = None,
    refinement_evidence: TowerCandidateRawRefinementEvidence | None = None,
    operator_identity: str | None = None,
    review_adjusted_seed_xyz_absolute_m: tuple[float, float, float] | None = None,
    notes: str | None = None,
) -> TowerCandidateReviewEvent:
    """Append one immutable review event through a new project generation.

    The candidate-set reference must resolve to the actual persisted set, the
    review proposal is taken from the immutable candidate proposal, and the
    next review revision/prior-event linkage is derived from the current
    authoritative ledger so no gap or wrong-linkage can be introduced.
    """
    loaded = _require_writable(session)
    candidate_set = _resolve_candidate_set(loaded, candidate_set_reference)
    candidate = _candidate_by_id(candidate_set, candidate_id)
    if _confirmation_for(
        loaded.candidate_store, candidate_set.candidate_set_id, candidate_id
    ) is not None:
        raise AlreadyConfirmedError(
            "candidate is already confirmed; review events cannot be appended "
            "after a confirmation"
        )
    ledger = _resolve_ledger(loaded.candidate_store, candidate_set)
    next_revision = (
        ledger.next_review_revision(candidate_set.reference, candidate_id)
        if ledger is not None
        else 1
    )
    prior_revision = next_revision - 1 if next_revision > 1 else None
    try:
        event = TowerCandidateReviewEvent(
            schema_version=TOWER_CANDIDATE_REVIEW_EVENT_SCHEMA_VERSION,
            candidate_set_reference=candidate_set.reference,
            candidate_id=candidate_id,
            review_revision=next_revision,
            decision=decision,
            occurred_at=occurred_at,
            reason=reason,
            proposal_xyz_absolute_m=candidate.candidate_xyz_absolute_m,
            proposal_bounds_absolute_m=candidate.candidate_bounds_absolute_m,
            refinement_state=refinement_state,
            refined_raw_geometry=refined_raw_geometry,
            refinement_evidence=refinement_evidence,
            operator_identity=operator_identity,
            review_adjusted_seed_xyz_absolute_m=review_adjusted_seed_xyz_absolute_m,
            notes=notes,
            prior_event_revision=prior_revision,
        )
    except (TypeError, ValueError) as exc:
        raise InvalidTowerCandidateState(f"invalid review event: {exc}") from exc
    if ledger is None:
        new_ledger = TowerCandidateReviewLedger(
            schema_version=TOWER_CANDIDATE_REVIEW_LEDGER_SCHEMA_VERSION,
            review_ledger_id=_new_review_ledger_id(),
            candidate_sets=(candidate_set.reference,),
            events=(event,),
        )
    else:
        new_ledger = ledger.with_event(event)
    try:
        validate_tower_candidate_review_ledger(candidate_set, new_ledger)
    except ValueError as exc:
        raise InvalidTowerCandidateState(
            f"review ledger is invalid after appending the event: {exc}"
        ) from exc
    if ledger is None:
        updated_ledgers = (*loaded.candidate_store.review_ledgers, new_ledger)
    else:
        updated_ledgers = _replace_ledger(
            loaded.candidate_store.review_ledgers,
            candidate_set.candidate_set_id,
            new_ledger,
        )
    candidate_store = CandidateStore(
        schema_version=CANDIDATE_STORE_INDEX_SCHEMA_VERSION,
        candidate_sets=loaded.candidate_store.candidate_sets,
        review_ledgers=updated_ledgers,
        confirmations=loaded.candidate_store.confirmations,
    )
    _publish(session, loaded, candidate_store=candidate_store)
    return event


def _tower_draft_from_accepted(
    candidate_set: TowerCandidateSet,
    candidate: TowerCandidate,
    event: TowerCandidateReviewEvent,
    accepted_review_revision: int,
) -> TowerDraft:
    """Map one effective ACCEPTED + RAW_REFINED review onto the existing TowerDraft."""
    assert event.refined_raw_geometry is not None
    assert event.refinement_evidence is not None
    seed = event.review_adjusted_seed_xyz_absolute_m or candidate.candidate_xyz_absolute_m
    evidence = {
        "candidate_set_id": candidate_set.candidate_set_id,
        "candidate_set_digest": candidate_set.digest,
        "candidate_id": candidate.candidate_id,
        "accepted_review_revision": accepted_review_revision,
        "raw_refinement_evidence": event.refinement_evidence.to_dict(),
    }
    return TowerDraft(
        schema_version=TOWER_SCHEMA_VERSION,
        display_name=candidate.candidate_id,
        state=TowerLifecycleState.CONFIRMED,
        source_binding=candidate_set.source_binding,
        coordinate_context=candidate_set.coordinate_context,
        approximate_click_xyz_absolute_m=seed,
        refined_geometry=event.refined_raw_geometry,
        refinement=TowerRefinement(
            schema_version=TOWER_REFINEMENT_SCHEMA_VERSION,
            status=TowerRefinementStatus.DEFENSIBLE,
            quality=TowerRefinementQuality.MANUAL_CONFIRMED,
            evidence=evidence,
            algorithm_version=event.refinement_evidence.algorithm_version,
            confirmed_at=event.occurred_at,
            local_region_reference=None,
        ),
        notes=event.notes,
        created_at=event.occurred_at,
        updated_at=event.occurred_at,
    )


def confirm_reviewed_tower_candidate(
    session: WritableProjectSession,
    *,
    candidate_set_reference: CandidateSetReference,
    candidate_id: str,
    accepted_review_revision: int,
    reviewer_identity: str | None = None,
    notes: str | None = None,
    confirmed_at: str | None = None,
) -> TowerCandidateConfirmation:
    """Confirm one effective ACCEPTED review into the existing TowerRegistry.

    Validates the actual persisted candidate set and review ledger, refuses to
    allocate a second tower for an already-confirmed candidate, prepares the
    next canonical ``T###`` purely in memory, and publishes the Tower registry
    plus the matching candidate confirmation atomically in one generation.
    """
    loaded = _require_writable(session)
    candidate_set = _resolve_candidate_set(loaded, candidate_set_reference)
    candidate = _candidate_by_id(candidate_set, candidate_id)
    ledger = _resolve_ledger(loaded.candidate_store, candidate_set)
    if ledger is None:
        raise InvalidTowerCandidateState("candidate set has no review ledger")
    try:
        validate_tower_candidate_review_ledger(candidate_set, ledger)
    except ValueError as exc:
        raise InvalidTowerCandidateState(
            f"review ledger is invalid for the candidate set: {exc}"
        ) from exc
    source_events = [
        event
        for event in ledger.events
        if event.candidate_id == candidate_id
        and event.review_revision == accepted_review_revision
    ]
    if len(source_events) != 1:
        raise InvalidTowerCandidateState(
            "accepted_review_revision must resolve to exactly one review event"
        )
    source = source_events[0]
    if source.decision is not ReviewDecision.ACCEPTED:
        raise InvalidTowerCandidateState(
            "only an effective ACCEPTED review may confirm a tower"
        )
    if (
        source.refinement_state is not RefinementState.RAW_REFINED
        or source.refined_raw_geometry is None
        or source.refinement_evidence is None
    ):
        raise InvalidTowerCandidateState(
            "ACCEPTED confirmation requires RAW_REFINED geometry and refinement evidence"
        )
    if any(
        event.review_revision > source.review_revision
        for event in ledger.events
        if event.candidate_id == candidate_id
    ):
        raise InvalidTowerCandidateState("accepted review is not the effective decision")
    if _confirmation_for(
        loaded.candidate_store, candidate_set.candidate_set_id, candidate_id
    ) is not None:
        raise AlreadyConfirmedError(
            "candidate has already been confirmed; refusing to allocate another tower"
        )
    session.require_strong_source_identity()
    draft = _tower_draft_from_accepted(
        candidate_set, candidate, source, accepted_review_revision
    )
    _same_current_source(session, draft.source_binding)
    tower, candidate_tower_registry, candidate_project = _allocated_tower(loaded, draft)
    if confirmed_at is None:
        confirmed_at = _timestamp_at_or_after(source.occurred_at)
    try:
        confirmation = TowerCandidateConfirmation(
            schema_version=TOWER_CANDIDATE_CONFIRMATION_SCHEMA_VERSION,
            candidate_set_reference=candidate_set.reference,
            candidate_id=candidate_id,
            accepted_review_revision=accepted_review_revision,
            confirmed_tower=tower.revision_reference,
            confirmed_at=confirmed_at,
            detector_provenance=candidate.detector_provenance,
            reviewer_identity=reviewer_identity or source.operator_identity,
            notes=notes if notes is not None else source.notes,
        )
    except (TypeError, ValueError) as exc:
        raise InvalidTowerCandidateState(f"invalid candidate confirmation: {exc}") from exc
    try:
        validate_tower_candidate_confirmation(candidate_set, ledger, confirmation)
    except ValueError as exc:
        raise InvalidTowerCandidateState(
            f"invalid candidate confirmation: {exc}"
        ) from exc
    candidate_store = CandidateStore(
        schema_version=CANDIDATE_STORE_INDEX_SCHEMA_VERSION,
        candidate_sets=loaded.candidate_store.candidate_sets,
        review_ledgers=loaded.candidate_store.review_ledgers,
        confirmations=(*loaded.candidate_store.confirmations, confirmation),
    )
    _publish(
        session,
        loaded,
        project=candidate_project,
        tower_registry=candidate_tower_registry,
        candidate_store=candidate_store,
    )
    return confirmation


__all__ = [
    "AlreadyConfirmedError",
    "CandidateConfirmationPublicationUncertainError",
    "InvalidTowerCandidateState",
    "TowerCandidateMutationError",
    "append_tower_candidate_review",
    "confirm_reviewed_tower_candidate",
    "persist_tower_candidate_set",
]
