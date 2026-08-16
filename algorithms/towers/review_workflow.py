"""V5A2/V5A3 tower-review orchestration (thin layer over V5A1-B services).

This module connects the automatic/manual candidate sets, the bounded RAW tile
cache, and the RAW refinement algorithm to the existing locked V5A1-B project
services.  It never allocates a ``T###`` directly and never writes around the
project/session layer; ``confirm_reviewed_tower_candidate`` remains the only
T###-allocating path.

One-click ACCEPT::

    candidate -> optional review-adjusted seed -> bounded RAW refinement
        -> success: append ACCEPTED + RAW_REFINED review
        -> confirm_reviewed_tower_candidate() -> persist T### -> updated state
        -> failure: append PENDING_REVIEW + RAW_REFINEMENT_FAILED, no T###,
           return a clear failure reason for the UI

REJECT::

    candidate -> append REJECTED review event -> publish -> state
    (no RAW refinement required merely to reject; no T###)

MANUAL ADD::

    viewer click -> approximate SOURCE_LAS_ABSOLUTE XYZ -> immutable
    supplemental TowerCandidateSet (origin MANUAL_ADD) -> persist -> pending P###
    marker.  The Owner then ACCEPTs it like any automatic candidate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from core.config.fast_baseline import canonical_digest
from core.contracts.project import CoordinateContext, SourceIdentityBinding, TowerGeometry
from core.contracts.tower_candidates import (
    CandidateOrigin,
    CandidateSetReference,
    DetectorKind,
    DetectorProvenance,
    RefinementState,
    ReviewDecision,
    TowerCandidate,
    TowerCandidateManualAddProvenance,
    TowerCandidateQualityFlag,
    TowerCandidateReviewEvent,
    TowerCandidateReviewLedger,
    TowerCandidateSet,
)
from core.project.session import ProjectSession, WritableProjectSession
from core.project.tower_candidates import AlreadyConfirmedError, InvalidTowerCandidateState

from algorithms.towers.raw_access import RawTileCache, RawTileCacheError
from algorithms.towers.refinement import (
    TowerRawRefinementConfig,
    refine_tower_from_raw,
)

MANUAL_ADD_DETECTOR_NAME = "manual-add-operator"
MANUAL_ADD_DETECTOR_VERSION = "v5a2-manual-add-1"
MANUAL_ADD_DETECTOR_SCHEMA = "treeline-tower-manual-add-v1"


class TowerReviewWorkflowError(RuntimeError):
    """Closed failure raised by the tower-review orchestration boundary."""


class CandidateNotReviewableError(TowerReviewWorkflowError):
    """Raised when a candidate's effective review state forbids the requested action."""


def _utc_timestamp() -> str:
    return datetime.now(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _candidate_by_id(candidate_set: TowerCandidateSet, candidate_id: str) -> TowerCandidate:
    for candidate in candidate_set.candidates:
        if candidate.candidate_id == candidate_id:
            return candidate
    raise InvalidTowerCandidateState(f"candidate {candidate_id} is absent from the candidate set")


def resolve_candidate_set(
    session: ProjectSession, reference: CandidateSetReference
) -> TowerCandidateSet:
    """Resolve a digest-bound candidate-set reference against the authoritative project."""
    if type(reference) is not CandidateSetReference:
        raise ValueError("reference must be a CandidateSetReference")
    for candidate_set in session.candidate_store.candidate_sets:
        if candidate_set.candidate_set_id == reference.candidate_set_id:
            if candidate_set.reference != reference:
                raise InvalidTowerCandidateState(
                    "candidate-set reference digest does not match the actual persisted set"
                )
            return candidate_set
    raise InvalidTowerCandidateState(
        "candidate-set reference does not resolve to a persisted candidate set"
    )


def resolve_ledger(
    session: ProjectSession, candidate_set: TowerCandidateSet
) -> TowerCandidateReviewLedger | None:
    for ledger in session.candidate_store.review_ledgers:
        if ledger.candidate_sets and (
            ledger.candidate_sets[0].candidate_set_id == candidate_set.candidate_set_id
        ):
            return ledger
    return None


def _effective_decision(
    ledger: TowerCandidateReviewLedger | None,
    candidate_set: TowerCandidateSet,
    candidate_id: str,
) -> ReviewDecision | None:
    if ledger is None:
        return None
    return ledger.effective_decision(candidate_set.reference, candidate_id)


def _is_confirmed(
    session: ProjectSession, candidate_set_id: str, candidate_id: str
) -> bool:
    return any(
        confirmation.candidate_set_reference.candidate_set_id == candidate_set_id
        and confirmation.candidate_id == candidate_id
        for confirmation in session.candidate_store.confirmations
    )


def reject_candidate(
    session: WritableProjectSession,
    candidate_set: TowerCandidateSet,
    candidate_id: str,
    *,
    reason: str = "rejected by reviewer",
    operator_identity: str | None = None,
    occurred_at: str | None = None,
) -> TowerCandidateReviewEvent:
    """Reject one real persisted candidate (no RAW refinement, no T###)."""
    if type(candidate_set) is not TowerCandidateSet:
        raise ValueError("candidate_set must be a TowerCandidateSet")
    _candidate_by_id(candidate_set, candidate_id)
    if _is_confirmed(session, candidate_set.candidate_set_id, candidate_id):
        raise AlreadyConfirmedError(
            "candidate is already confirmed; it cannot be rejected"
        )
    return session.append_tower_candidate_review(
        candidate_set_reference=candidate_set.reference,
        candidate_id=candidate_id,
        decision=ReviewDecision.REJECTED,
        occurred_at=occurred_at or _utc_timestamp(),
        reason=reason,
        refinement_state=RefinementState.UNREFINED,
        operator_identity=operator_identity,
    )


@dataclass(frozen=True, slots=True)
class ReviewAcceptResult:
    """Outcome of one one-click ACCEPT (may be a closed refinement failure)."""

    success: bool
    candidate_set: TowerCandidateSet
    candidate_id: str
    event: TowerCandidateReviewEvent | None = None
    confirmation: Any | None = None
    tower_id: str | None = None
    geometry: TowerGeometry | None = None
    failure_reason: str | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "candidate_set_id": self.candidate_set.candidate_set_id,
            "candidate_set_digest": self.candidate_set.digest,
            "candidate_id": self.candidate_id,
            "review_revision": self.event.review_revision if self.event is not None else None,
            "tower_id": self.tower_id,
            "refinement_state": (
                self.event.refinement_state.value if self.event is not None else None
            ),
            "geometry": self.geometry.to_dict() if self.geometry is not None else None,
            "failure_reason": self.failure_reason,
            "diagnostics": dict(self.diagnostics),
        }


def accept_candidate(
    session: WritableProjectSession,
    candidate_set: TowerCandidateSet,
    candidate_id: str,
    raw_cache: RawTileCache,
    *,
    refinement_config: TowerRawRefinementConfig | None = None,
    review_adjusted_seed_xyz_absolute_m: tuple[float, float, float] | None = None,
    operator_identity: str | None = None,
    notes: str | None = None,
    occurred_at: str | None = None,
) -> ReviewAcceptResult:
    """One-click ACCEPT: bounded RAW refine, then confirm through V5A1-B."""
    if type(candidate_set) is not TowerCandidateSet:
        raise ValueError("candidate_set must be a TowerCandidateSet")
    if type(raw_cache) is not RawTileCache:
        raise ValueError("raw_cache must be a RawTileCache")
    if refinement_config is None:
        refinement_config = TowerRawRefinementConfig()
    if type(refinement_config) is not TowerRawRefinementConfig:
        raise ValueError("refinement_config must be a TowerRawRefinementConfig")
    candidate = _candidate_by_id(candidate_set, candidate_id)
    if _is_confirmed(session, candidate_set.candidate_set_id, candidate_id):
        raise AlreadyConfirmedError(
            "candidate is already confirmed; refusing a duplicate tower"
        )
    ledger = resolve_ledger(session, candidate_set)
    effective = _effective_decision(ledger, candidate_set, candidate_id)
    if effective is ReviewDecision.REJECTED:
        raise CandidateNotReviewableError(
            "candidate is rejected; it cannot be accepted without a new proposal"
        )

    seed = review_adjusted_seed_xyz_absolute_m or candidate.candidate_xyz_absolute_m
    if len(seed) != 3:
        raise ValueError("review_adjusted_seed_xyz_absolute_m must contain exactly 3 values")
    if not all(isinstance(value, int | float) and value == value for value in seed):
        raise ValueError("review_adjusted_seed_xyz_absolute_m must be finite")
    try:
        query = raw_cache.query(
            tuple(float(value) for value in seed),
            xy_radius_m=float(refinement_config.search_radius_m),
            z_context_below_m=float(refinement_config.z_context_below_m),
            z_context_above_m=float(refinement_config.z_context_above_m),
        )
        result = refine_tower_from_raw(query, candidate_set.source_binding, refinement_config)
    except (RawTileCacheError, ValueError) as exc:
        # The cache is bound to the set source; any mismatch is a closed failure.
        return ReviewAcceptResult(
            success=False,
            candidate_set=candidate_set,
            candidate_id=candidate_id,
            failure_reason=f"RAW_CACHE_ERROR: {exc}",
            diagnostics={},
        )

    if not result.success:
        event = session.append_tower_candidate_review(
            candidate_set_reference=candidate_set.reference,
            candidate_id=candidate_id,
            decision=ReviewDecision.PENDING_REVIEW,
            occurred_at=occurred_at or _utc_timestamp(),
            reason=result.failure_reason or "RAW refinement failed",
            refinement_state=RefinementState.RAW_REFINEMENT_FAILED,
            operator_identity=operator_identity,
            review_adjusted_seed_xyz_absolute_m=tuple(float(value) for value in seed)
            if review_adjusted_seed_xyz_absolute_m is not None
            else None,
            notes=notes,
        )
        return ReviewAcceptResult(
            success=False,
            candidate_set=candidate_set,
            candidate_id=candidate_id,
            event=event,
            failure_reason=result.failure_reason,
            diagnostics=result.diagnostics,
        )

    assert result.geometry is not None
    assert result.evidence is not None
    event = session.append_tower_candidate_review(
        candidate_set_reference=candidate_set.reference,
        candidate_id=candidate_id,
        decision=ReviewDecision.ACCEPTED,
        occurred_at=occurred_at or _utc_timestamp(),
        reason="validated against full-resolution RAW",
        refinement_state=RefinementState.RAW_REFINED,
        refined_raw_geometry=result.geometry,
        refinement_evidence=result.evidence,
        operator_identity=operator_identity,
        review_adjusted_seed_xyz_absolute_m=tuple(float(value) for value in seed)
        if review_adjusted_seed_xyz_absolute_m is not None
        else None,
        notes=notes,
    )
    confirmation = session.confirm_reviewed_tower_candidate(
        candidate_set_reference=candidate_set.reference,
        candidate_id=candidate_id,
        accepted_review_revision=event.review_revision,
        reviewer_identity=operator_identity,
        notes=notes,
    )
    return ReviewAcceptResult(
        success=True,
        candidate_set=candidate_set,
        candidate_id=candidate_id,
        event=event,
        confirmation=confirmation,
        tower_id=confirmation.confirmed_tower.tower_id,
        geometry=result.geometry,
        diagnostics=result.diagnostics,
    )


def manual_add_candidate_set(
    session: WritableProjectSession,
    *,
    project_id: str,
    source_binding: SourceIdentityBinding,
    coordinate_context: CoordinateContext,
    candidate_xyz_absolute_m: tuple[float, float, float],
    candidate_set_id: str,
    generated_at: str,
    reason: str = "owner manual add",
    operator_identity: str | None = None,
    note: str | None = None,
) -> TowerCandidateSet:
    """Persist an immutable supplemental MANUAL_ADD candidate set (pending P###).

    The click is transformed/validated to a SOURCE_LAS_ABSOLUTE XYZ seed; the
    set is persisted through V5A1-B.  No T### is allocated here — the Owner must
    ACCEPT the pending candidate to refine and confirm it.
    """
    if type(source_binding) is not SourceIdentityBinding:
        raise ValueError("source_binding must be a SourceIdentityBinding")
    if type(coordinate_context) is not CoordinateContext:
        raise ValueError("coordinate_context must be a CoordinateContext")
    if coordinate_context != source_binding.coordinate_context:
        raise ValueError("coordinate_context must match source_binding")
    if len(candidate_xyz_absolute_m) != 3:
        raise ValueError("candidate_xyz_absolute_m must contain exactly 3 values")
    if not all(
        isinstance(value, int | float) and value == value for value in candidate_xyz_absolute_m
    ):
        raise ValueError("candidate_xyz_absolute_m must be finite")
    candidate = TowerCandidate(
        candidate_id="P001",
        origin=CandidateOrigin.MANUAL_ADD,
        coordinate_frame=coordinate_context.frame,
        coordinate_unit=coordinate_context.units,
        candidate_xyz_absolute_m=tuple(float(value) for value in candidate_xyz_absolute_m),
        candidate_bounds_absolute_m=None,
        confidence=1.0,
        quality_flags=TowerCandidateQualityFlag.PROPOSAL_NEEDS_REFINEMENT,
        detector_provenance=None,
        manual_add_provenance=TowerCandidateManualAddProvenance(
            added_at=generated_at,
            reason=reason,
            operator_identity=operator_identity,
            note=note,
        ),
        evidence={},
    )
    detector_provenance = DetectorProvenance(
        detector_kind=DetectorKind.GEOMETRY,
        detector_name=MANUAL_ADD_DETECTOR_NAME,
        detector_version=MANUAL_ADD_DETECTOR_VERSION,
        configuration_digest=canonical_digest(
            {
                "schema": MANUAL_ADD_DETECTOR_SCHEMA,
                "reason": reason,
            }
        ),
        detector_evidence={
            "schema": MANUAL_ADD_DETECTOR_SCHEMA,
            "origin": CandidateOrigin.MANUAL_ADD.value,
        },
    )
    candidate_set = TowerCandidateSet.create(
        candidate_set_id=candidate_set_id,
        project_id=project_id,
        generated_at=generated_at,
        source_binding=source_binding,
        coordinate_context=coordinate_context,
        detector_provenance=detector_provenance,
        candidates=[candidate],
        corridor_bounds_absolute_m=None,
    )
    session.persist_tower_candidate_set(candidate_set)
    return candidate_set


# ---------------------------------------------------------------------------
# Review-facing state projection (JSON-safe, no RAW arrays)
# ---------------------------------------------------------------------------


def _review_state_projection(
    session: ProjectSession,
    candidate_set: TowerCandidateSet,
    candidate: TowerCandidate,
) -> dict[str, Any]:
    ledger = resolve_ledger(session, candidate_set)
    if ledger is None:
        decision = ReviewDecision.PENDING_REVIEW
        refinement_state = RefinementState.UNREFINED
        geometry = None
        failure_reason = None
    else:
        decision = ledger.effective_decision(candidate_set.reference, candidate.candidate_id)
        if decision is None:
            decision = ReviewDecision.PENDING_REVIEW
        chain = tuple(
            event
            for event in ledger.events
            if event.candidate_id == candidate.candidate_id
        )
        latest = chain[-1] if chain else None
        if latest is None:
            refinement_state = RefinementState.UNREFINED
            geometry = None
            failure_reason = None
        else:
            refinement_state = latest.refinement_state
            geometry = latest.refined_raw_geometry
            failure_reason = (
                latest.reason
                if refinement_state is RefinementState.RAW_REFINEMENT_FAILED
                and latest.decision is ReviewDecision.PENDING_REVIEW
                else None
            )
    confirmed = _is_confirmed(session, candidate_set.candidate_set_id, candidate.candidate_id)
    return {
        "candidate_set_id": candidate_set.candidate_set_id,
        "candidate_set_digest": candidate_set.digest,
        "candidate_id": candidate.candidate_id,
        "origin": candidate.origin.value,
        "proposal_xyz_absolute_m": list(candidate.candidate_xyz_absolute_m),
        "candidate_bounds_absolute_m": (
            candidate.candidate_bounds_absolute_m.to_dict()
            if candidate.candidate_bounds_absolute_m is not None
            else None
        ),
        "confidence": candidate.confidence,
        "quality_flags": [
            flag.name for flag in TowerCandidateQualityFlag if candidate.quality_flags & flag
        ],
        "review_state": decision.value,
        "refinement_state": refinement_state.value,
        "refined_geometry": geometry.to_dict() if geometry is not None else None,
        "failure_reason": failure_reason,
        "confirmed_tower_id": (
            next(
                (
                    confirmation.confirmed_tower.tower_id
                    for confirmation in session.candidate_store.confirmations
                    if confirmation.candidate_set_reference.candidate_set_id
                    == candidate_set.candidate_set_id
                    and confirmation.candidate_id == candidate.candidate_id
                ),
                None,
            )
            if confirmed
            else None
        ),
    }


def project_tower_review_state(session: ProjectSession) -> dict[str, Any]:
    """Build the review-facing state projection needed by the Viewer (no RAW arrays)."""
    project = session.project
    source_reference = project.source_reference
    candidate_sets: list[dict[str, Any]] = []
    for candidate_set in session.candidate_store.candidate_sets:
        candidates = [
            _review_state_projection(session, candidate_set, candidate)
            for candidate in candidate_set.candidates
        ]
        candidate_sets.append(
            {
                "candidate_set_id": candidate_set.candidate_set_id,
                "candidate_set_digest": candidate_set.digest,
                "generated_at": candidate_set.generated_at,
                "detector_provenance": candidate_set.detector_provenance.to_dict(),
                "origin": "AUTOMATIC"
                if any(c.origin is CandidateOrigin.AUTOMATIC for c in candidate_set.candidates)
                else "MANUAL_ADD",
                "candidates": candidates,
            }
        )
    confirmed_towers: list[dict[str, Any]] = []
    for confirmation in session.candidate_store.confirmations:
        try:
            entry = session.tower_registry.get(confirmation.confirmed_tower.tower_id)
            tower = entry.revision(confirmation.confirmed_tower.revision)
        except KeyError:
            continue
        confirmed_towers.append(
            {
                "tower_id": tower.tower_id,
                "revision": tower.revision,
                "display_xyz_absolute_m": list(tower.refined_geometry.base_xyz_absolute_m),
                "top_xyz_absolute_m": list(tower.refined_geometry.top_xyz_absolute_m),
                "height_m": tower.refined_geometry.height_m,
                "source_candidate": {
                    "candidate_set_id": confirmation.candidate_set_reference.candidate_set_id,
                    "candidate_set_digest": (
                        confirmation.candidate_set_reference.candidate_set_digest
                    ),
                    "candidate_id": confirmation.candidate_id,
                },
            }
        )
    return {
        "project_id": project.project_id,
        "generation_id": session.generation_id,
        "source": {
            "source_content_identity_digest": (
                source_reference.source_identity.source_content_identity_digest
            ),
            "source_identity_record_digest": (
                source_reference.source_identity.source_identity_record_digest
            ),
            "source_point_count": source_reference.source_identity.source_point_count,
            "coordinate_frame": source_reference.coordinate_context.frame.value,
            "coordinate_unit": source_reference.coordinate_context.units.value,
        },
        "candidate_sets": candidate_sets,
        "confirmed_towers": confirmed_towers,
    }


__all__ = [
    "CandidateNotReviewableError",
    "MANUAL_ADD_DETECTOR_NAME",
    "MANUAL_ADD_DETECTOR_SCHEMA",
    "MANUAL_ADD_DETECTOR_VERSION",
    "ReviewAcceptResult",
    "TowerReviewWorkflowError",
    "accept_candidate",
    "manual_add_candidate_set",
    "project_tower_review_state",
    "reject_candidate",
    "resolve_candidate_set",
    "resolve_ledger",
]
