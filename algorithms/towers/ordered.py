"""V5A3 ordered manual tower selection workflow (SELECT / PREVIEW / CONFIRM).

This is the PRIMARY workflow: the Owner clicks a real tower, a bounded
FULL_RESOLUTION_RAW local membership segmentation runs, the Owner inspects the
actual highlighted member-point preview, and only an explicit CONFIRM allocates
the next canonical ``T###`` in confirmation order (T001, T002, T003, ...).

The ordered workflow is a thin orchestration over the locked V5A1-B services
and the V5A2/V5A3 RAW access + segmentation:

- ``SELECT``       persists a MANUAL_ADD candidate, runs the local seed-driven
                   membership segmentation, and creates a pending preview token.
                   NO ``T###`` is allocated.
- ``PREVIEW``      returns a deterministic VIEWER-ONLY capped display sample of
                   the exact RAW membership plus preview metadata.
- ``CONFIRM``      persists the exact membership artifact, appends an ACCEPTED +
                   RAW_REFINED review carrying the membership provenance, and
                   allocates ``T###`` through ``confirm_reviewed_tower_candidate``
                   with the artifact linked via ``TowerRefinement.local_region_reference``.
- ``RESELECT``     supersedes the current pending candidate (existing REJECTED
                   review semantics) and clears the preview.  NO ``T###``.
- ``CANCEL``       clears the preview without allocating a ``T###``.

Token safety: every preview token is a high-entropy opaque secret bound in
memory to the project identity, generation, candidate-set id + digest,
candidate id, source identity, and membership digest.  Confirming validates the
record against the *current* authoritative project before any mutation; a stale
or incompatible token fails closed.

Atomicity: the exact membership artifact is written before the confirming
generation is published; if publication fails there is NO active ``T###``, NO
active confirmation, and NO active membership reference (the artifact is an
orphan with no immutable generation referencing it).  Retry remains valid.
"""

from __future__ import annotations

import secrets
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import numpy as np
from core.config.fast_baseline import canonical_digest
from core.contracts.project import (
    TowerGeometry,
)
from core.contracts.tower_candidates import (
    DetectorKind,
    DetectorProvenance,
    RefinementState,
    ReviewDecision,
    TowerCandidateSet,
)
from core.project.session import ProjectSession, WritableProjectSession

from algorithms.towers.membership_artifact import (
    write_membership_artifact,
)
from algorithms.towers.raw_access import RawTileCache
from algorithms.towers.refinement import build_tower_raw_refinement_evidence
from algorithms.towers.review_workflow import (
    manual_add_candidate_set,
    project_tower_review_state,
    resolve_candidate_set,
    resolve_ledger,
)
from algorithms.towers.segmentation import (
    TOWER_LOCAL_SEGMENTATION_VERSION,
    TowerLocalMembership,
    TowerLocalSegmentationConfig,
    segment_tower_membership,
)

ORDERED_WORKFLOW_VERSION = "ordered-manual-tower-selection-v1"
ORDERED_MANUAL_ADD_DETECTOR_NAME = "ordered-manual-select"
ORDERED_MANUAL_ADD_DETECTOR_VERSION = "v5a3-ordered-manual-1"
ORDERED_MANUAL_ADD_DETECTOR_SCHEMA = "treeline-tower-ordered-manual-select-v1"
ORDERED_CANDIDATE_SET_PREFIX = "ordered-"
PREVIEW_REFINEMENT_READY = "PREVIEW_READY"
DEFAULT_PREVIEW_SAMPLE_MAX = 20_000


class OrderedWorkflowError(RuntimeError):
    """Closed failure raised by the ordered tower-selection boundary."""


class OrderedPreviewInvalidError(OrderedWorkflowError):
    """Raised when a preview token is stale, missing, or incompatible."""


def _utc_timestamp() -> str:
    return datetime.now(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


# ---------------------------------------------------------------------------
# Preview store (in-memory; single loopback server process)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class OrderedPreviewRecord:
    """All facts a preview token is bound to, held until CONFIRM/RESELECT/CANCEL."""

    token: str
    created_at: str
    project_id: str
    generation_id: str
    candidate_set_id: str
    candidate_set_digest: str
    candidate_id: str
    source_content_identity_digest: str
    source_identity_record_digest: str
    group_id: str
    group_display_name: str
    membership: TowerLocalMembership
    membership_digest: str
    raw_cache_identity: dict[str, Any]
    operator_identity: str | None = None
    notes: str | None = None

    def __post_init__(self) -> None:
        if not self.token:
            raise OrderedWorkflowError("preview token must be non-empty")
        if type(self.membership) is not TowerLocalMembership:
            raise ValueError("membership must be a TowerLocalMembership")
        if self.membership.membership_digest() != self.membership_digest:
            raise ValueError("membership_digest must match the membership content")


class OrderedPreviewStore:
    """In-memory pending preview registry (one token per pending preview)."""

    def __init__(self) -> None:
        self._records: dict[str, OrderedPreviewRecord] = {}

    def put(self, record: OrderedPreviewRecord) -> None:
        if type(record) is not OrderedPreviewRecord:
            raise OrderedWorkflowError("record must be an OrderedPreviewRecord")
        self._records[record.token] = record

    def get(self, token: str) -> OrderedPreviewRecord:
        if type(token) is not str or not token:
            raise OrderedPreviewInvalidError("preview token is missing")
        record = self._records.get(token)
        if record is None:
            raise OrderedPreviewInvalidError("preview token is stale or unknown")
        return record

    def remove(self, token: str) -> bool:
        return self._records.pop(token, None) is not None

    def current(self) -> OrderedPreviewRecord | None:
        """Return the single active preview, if any."""
        records = list(self._records.values())
        if len(records) > 1:
            # Every SELECT replaces the previous preview; more than one pending
            # preview means an earlier one was not cleared.  The newest wins.
            records.sort(key=lambda record: record.created_at)
            return records[-1]
        return records[0] if records else None

    def clear(self) -> None:
        self._records.clear()


def new_preview_token() -> str:
    return secrets.token_urlsafe(32)


# ---------------------------------------------------------------------------
# Deterministic VIEWER-ONLY display sample
# ---------------------------------------------------------------------------


def preview_sample_points(
    membership: TowerLocalMembership, *, max_points: int = DEFAULT_PREVIEW_SAMPLE_MAX
) -> list[list[float]]:
    """Deterministic capped display sample of the exact RAW membership.

    The sample is VIEWER-ONLY: it is a deterministic (source_index-ordered,
    stride) subsample that never replaces the authoritative exact membership
    held server-side.  ``max_points`` is a hard display cap.
    """
    if type(membership) is not TowerLocalMembership:
        raise OrderedWorkflowError("membership must be a TowerLocalMembership")
    if type(max_points) is not int or max_points <= 0:
        raise ValueError("max_points must be a positive integer")
    order = np.argsort(membership.source_index, kind="stable")
    count = membership.member_count
    if count <= max_points:
        sample = membership.xyz_absolute_m[order]
    else:
        stride = count / max_points
        selected = np.floor(np.arange(max_points) * stride).astype(np.int64)
        selected = np.minimum(selected, count - 1)
        sample = membership.xyz_absolute_m[order[selected]]
    return [[float(x), float(y), float(z)] for x, y, z in sample]


# ---------------------------------------------------------------------------
# State projection
# ---------------------------------------------------------------------------


def ordered_review_state(session: ProjectSession) -> dict[str, Any]:
    """Project the review state plus the ordered-selection primacy metadata."""
    from core.project.tower_groups import tower_group_state

    state = project_tower_review_state(session)
    next_tower_sequence = session.project.next_tower_sequence
    # Confirmation order IS the intended corridor order; display confirmed tower
    # IDs in canonical T### ascending order so the Owner always reads T001..T00N.
    confirmed_ids = [tower["tower_id"] for tower in state["confirmed_towers"]]
    confirmed_ids.sort(key=lambda tower_id: int(tower_id[1:]))
    groups = tower_group_state(session)
    state["ordered"] = {
        "mode": "ORDERED_MANUAL",
        "workflow_version": ORDERED_WORKFLOW_VERSION,
        "next_tower_id": f"T{next_tower_sequence:03d}",
        "confirmed_tower_ids": confirmed_ids,
        "confirmed_tower_count": len(state["confirmed_towers"]),
        "tower_groups": groups["groups"],
        "next_group_sequence": groups["next_group_sequence"],
        "group_bootstrap_pending": groups["bootstrap_pending"],
    }
    return state


# ---------------------------------------------------------------------------
# SELECT / PREVIEW / CONFIRM / RESELECT / CANCEL
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class OrderedSelectResult:
    """Outcome of one ordered SELECT (success or closed segmentation failure)."""

    success: bool
    preview: dict[str, Any] | None = None
    candidate_set_id: str | None = None
    candidate_id: str | None = None
    failure_reason: str | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "preview": self.preview,
            "candidate_set_id": self.candidate_set_id,
            "candidate_id": self.candidate_id,
            "failure_reason": self.failure_reason,
            "diagnostics": dict(self.diagnostics),
        }


def _detector_provenance_for_ordered() -> DetectorProvenance:
    return DetectorProvenance(
        detector_kind=DetectorKind.GEOMETRY,
        detector_name=ORDERED_MANUAL_ADD_DETECTOR_NAME,
        detector_version=ORDERED_MANUAL_ADD_DETECTOR_VERSION,
        configuration_digest=canonical_digest(
            {"schema": ORDERED_MANUAL_ADD_DETECTOR_SCHEMA, "workflow": ORDERED_WORKFLOW_VERSION}
        ),
        detector_evidence={"schema": ORDERED_MANUAL_ADD_DETECTOR_SCHEMA},
    )


def ordered_select(
    session: WritableProjectSession,
    raw_cache: RawTileCache,
    store: OrderedPreviewStore,
    *,
    candidate_xyz_absolute_m: tuple[float, float, float],
    group_id: str,
    operator_identity: str | None = None,
    notes: str | None = None,
    segmentation_config: TowerLocalSegmentationConfig | None = None,
) -> OrderedSelectResult:
    """Step 1 — SELECT: persist a MANUAL_ADD candidate and run local segmentation.

    Creates a pending preview token (NO ``T###``) when segmentation succeeds;
    the preview is bound to the explicit ``group_id`` and can only confirm into
    that group.  On a closed segmentation failure the candidate stays pending
    and a PENDING_REVIEW + RAW_REFINEMENT_FAILED review records the failure.
    """
    if type(session) is not WritableProjectSession:
        raise ValueError("ordered SELECT requires a writable project session")
    if type(raw_cache) is not RawTileCache:
        raise ValueError("raw_cache must be a RawTileCache")
    if type(store) is not OrderedPreviewStore:
        raise ValueError("store must be an OrderedPreviewStore")
    if len(candidate_xyz_absolute_m) != 3 or not all(
        isinstance(value, int | float) and value == value
        for value in candidate_xyz_absolute_m
    ):
        raise ValueError("candidate_xyz_absolute_m must contain 3 finite values")
    if type(group_id) is not str or not group_id:
        raise ValueError("group_id must be a non-empty string")
    if segmentation_config is None:
        segmentation_config = TowerLocalSegmentationConfig()
    if type(segmentation_config) is not TowerLocalSegmentationConfig:
        raise ValueError("segmentation_config must be a TowerLocalSegmentationConfig")

    # A legacy group-less project is bootstrapped to G001 before any SELECT.
    if session.tower_group_registry.bootstrap_pending:
        session.ensure_ordered_group()
    try:
        group = session.tower_group_registry.get(group_id)
    except KeyError as exc:
        raise OrderedWorkflowError(f"group {group_id} is absent from the project") from exc

    binding = session.project.source_reference.binding
    coordinate_context = session.project.source_reference.coordinate_context
    candidate_set_id = f"{ORDERED_CANDIDATE_SET_PREFIX}{secrets.token_hex(8)}"
    generated_at = _utc_timestamp()
    candidate_set = manual_add_candidate_set(
        session,
        project_id=session.project.project_id,
        source_binding=binding,
        coordinate_context=coordinate_context,
        candidate_xyz_absolute_m=tuple(float(value) for value in candidate_xyz_absolute_m),
        candidate_set_id=candidate_set_id,
        generated_at=generated_at,
        reason="ordered manual tower selection",
        operator_identity=operator_identity,
        note=notes,
    )

    seed = tuple(float(value) for value in candidate_xyz_absolute_m)
    query = raw_cache.query(
        seed,
        xy_radius_m=float(segmentation_config.search_radius_m),
        z_context_below_m=float(segmentation_config.z_context_below_m),
        z_context_above_m=float(segmentation_config.z_context_above_m),
    )
    result = segment_tower_membership(query, binding, segmentation_config)
    if not result.success:
        _record_failed_segmentation(session, candidate_set, operator_identity)
        return OrderedSelectResult(
            success=False,
            candidate_set_id=candidate_set.candidate_set_id,
            candidate_id=candidate_set.candidates[0].candidate_id,
            failure_reason=result.failure_reason,
            diagnostics=result.diagnostics,
        )

    assert result.membership is not None
    membership = result.membership
    record = OrderedPreviewRecord(
        token=new_preview_token(),
        created_at=_utc_timestamp(),
        project_id=session.project.project_id,
        generation_id=session.generation_id,
        candidate_set_id=candidate_set.candidate_set_id,
        candidate_set_digest=candidate_set.digest,
        candidate_id=candidate_set.candidates[0].candidate_id,
        source_content_identity_digest=binding.source_content_identity_digest,
        source_identity_record_digest=binding.source_identity_record_digest,
        group_id=group.group_id,
        group_display_name=group.display_name,
        membership=membership,
        membership_digest=membership.membership_digest(),
        raw_cache_identity=dict(membership.raw_tile_cache_identity),
        operator_identity=operator_identity,
        notes=notes,
    )
    store.put(record)
    preview = _preview_payload(record, sample=True)
    return OrderedSelectResult(
        success=True,
        preview=preview,
        candidate_set_id=candidate_set.candidate_set_id,
        candidate_id=record.candidate_id,
    )


def _record_failed_segmentation(
    session: WritableProjectSession,
    candidate_set: TowerCandidateSet,
    operator_identity: str | None,
) -> None:
    from core.project import append_tower_candidate_review

    append_tower_candidate_review(
        session,
        candidate_set_reference=candidate_set.reference,
        candidate_id=candidate_set.candidates[0].candidate_id,
        decision=ReviewDecision.PENDING_REVIEW,
        occurred_at=_utc_timestamp(),
        reason="LOCAL_SEGMENTATION_FAILED: no defensible local tower structure",
        refinement_state=RefinementState.RAW_REFINEMENT_FAILED,
        operator_identity=operator_identity,
    )


def ordered_preview(
    store: OrderedPreviewStore,
    token: str,
    *,
    max_sample_points: int = DEFAULT_PREVIEW_SAMPLE_MAX,
) -> dict[str, Any]:
    """Step 1b — PREVIEW: return the display sample + metadata for a pending token."""
    if type(store) is not OrderedPreviewStore:
        raise ValueError("store must be an OrderedPreviewStore")
    record = store.get(token)
    return _preview_payload(record, sample=True, max_sample_points=max_sample_points)


def _preview_payload(
    record: OrderedPreviewRecord,
    *,
    sample: bool,
    max_sample_points: int = DEFAULT_PREVIEW_SAMPLE_MAX,
) -> dict[str, Any]:
    membership = record.membership
    minimum, maximum = membership.member_bounds_absolute_m
    payload: dict[str, Any] = {
        "token": record.token,
        "created_at": record.created_at,
        "workflow_version": ORDERED_WORKFLOW_VERSION,
        "project_id": record.project_id,
        "generation_id": record.generation_id,
        "candidate_set_id": record.candidate_set_id,
        "candidate_set_digest": record.candidate_set_digest,
        "candidate_id": record.candidate_id,
        "group_id": record.group_id,
        "group_display_name": record.group_display_name,
        "source": {
            "source_content_identity_digest": record.source_content_identity_digest,
            "source_identity_record_digest": record.source_identity_record_digest,
        },
        "member_count": membership.member_count,
        "height_m": membership.height_m,
        "member_bounds_absolute_m": {
            "min_xyz_absolute_m": list(minimum),
            "max_xyz_absolute_m": list(maximum),
        },
        "refined_geometry": {
            "base_xyz_absolute_m": list(membership.base_xyz_absolute_m),
            "top_xyz_absolute_m": list(membership.top_xyz_absolute_m),
            "height_m": membership.height_m,
        },
        "quality": {
            "algorithm_version": membership.algorithm_version,
            "config_digest": membership.config_digest,
            "membership_digest": record.membership_digest,
            "core_point_count": membership.quality_diagnostics.get("core_point_count"),
            "median_verticality": membership.quality_diagnostics.get("median_verticality"),
            "median_linearity": membership.quality_diagnostics.get("median_linearity"),
            "ground_support_points": membership.quality_diagnostics.get("ground_support_points"),
            "crossarm_above_core_points": membership.quality_diagnostics.get(
                "crossarm_above_core_points"
            ),
        },
        "raw_cache": dict(membership.raw_tile_cache_identity),
        "refinement_ready": PREVIEW_REFINEMENT_READY,
        "preview_sample_cap": max_sample_points,
        "preview_sample_authority": "VIEWER_ONLY",
        "automatic_suggestions_remain": True,
    }
    if sample:
        points = preview_sample_points(membership, max_points=max_sample_points)
        payload["preview_sample_size"] = len(points)
        payload["preview_sample"] = points
    return payload


def _validate_preview_against_session(
    record: OrderedPreviewRecord, session: WritableProjectSession
) -> TowerCandidateSet:
    if type(record) is not OrderedPreviewRecord:
        raise OrderedPreviewInvalidError("preview record is invalid")
    if record.project_id != session.project.project_id:
        raise OrderedPreviewInvalidError("preview token is bound to a different project")
    current_binding = session.project.source_reference.binding
    if (
        record.source_content_identity_digest
        != current_binding.source_content_identity_digest
        or record.source_identity_record_digest
        != current_binding.source_identity_record_digest
    ):
        raise OrderedPreviewInvalidError(
            "preview token is bound to a different source identity"
        )
    from core.contracts.tower_candidates import CandidateSetReference

    reference = CandidateSetReference(
        candidate_set_id=record.candidate_set_id,
        candidate_set_digest=record.candidate_set_digest,
    )
    candidate_set = resolve_candidate_set(session, reference)
    candidate = next(
        (
            candidate
            for candidate in candidate_set.candidates
            if candidate.candidate_id == record.candidate_id
        ),
        None,
    )
    if candidate is None:
        raise OrderedPreviewInvalidError("preview candidate is absent from the candidate set")
    if any(
        confirmation.candidate_set_reference.candidate_set_id == record.candidate_set_id
        and confirmation.candidate_id == record.candidate_id
        for confirmation in session.candidate_store.confirmations
    ):
        raise OrderedPreviewInvalidError("preview candidate is already confirmed")
    ledger = resolve_ledger(session, candidate_set)
    if ledger is not None:
        effective = ledger.effective_decision(candidate_set.reference, record.candidate_id)
        if effective in {ReviewDecision.REJECTED, ReviewDecision.SUPERSEDED}:
            raise OrderedPreviewInvalidError("preview candidate is no longer confirmable")
    if record.membership.membership_digest() != record.membership_digest:
        raise OrderedPreviewInvalidError("preview membership digest is inconsistent")
    # The token-bound group must still exist; a preview can only confirm into it.
    try:
        session.tower_group_registry.get(record.group_id)
    except KeyError as exc:
        raise OrderedPreviewInvalidError(
            f"preview group {record.group_id} is absent from the project"
        ) from exc
    return candidate_set


def ordered_confirm(
    session: WritableProjectSession,
    store: OrderedPreviewStore,
    token: str,
    *,
    operator_identity: str | None = None,
    notes: str | None = None,
) -> dict[str, Any]:
    """Step 2 — CONFIRM: persist the exact membership and allocate the next T###.

    Validates the token against the current authoritative project, writes the
    content-addressed membership artifact, appends the ACCEPTED + RAW_REFINED
    review (with membership provenance), and confirms through V5A1-B so the
    tower's ``TowerRefinement.local_region_reference`` links the artifact.
    """
    if type(session) is not WritableProjectSession:
        raise ValueError("ordered CONFIRM requires a writable project session")
    if type(store) is not OrderedPreviewStore:
        raise ValueError("store must be an OrderedPreviewStore")
    record = store.get(token)
    candidate_set = _validate_preview_against_session(record, session)

    membership = record.membership
    artifact_record = write_membership_artifact(session.project_root, membership)
    assert artifact_record.relative_path
    artifact_reference = artifact_record.to_artifact_reference()

    geometry = TowerGeometry(
        base_xyz_absolute_m=membership.base_xyz_absolute_m,
        top_xyz_absolute_m=membership.top_xyz_absolute_m,
        height_m=membership.height_m,
    )
    minimum, maximum = membership.member_bounds_absolute_m
    evidence = build_tower_raw_refinement_evidence(
        membership.source_binding,
        details={
            "method": "tower-local-membership",
            "algorithm_version": TOWER_LOCAL_SEGMENTATION_VERSION,
            "membership_digest": artifact_record.membership_digest,
            "artifact_relative_path": artifact_record.relative_path,
            "artifact_byte_size": artifact_record.byte_size,
            "member_count": membership.member_count,
            "member_bounds_min_xyz": list(minimum),
            "member_bounds_max_xyz": list(maximum),
            "source_content_identity_digest": (
                membership.source_binding.source_content_identity_digest
            ),
            "source_identity_record_digest": (
                membership.source_binding.source_identity_record_digest
            ),
            "segmentation_config_digest": membership.config_digest,
        },
    )
    event = session.append_tower_candidate_review(
        candidate_set_reference=candidate_set.reference,
        candidate_id=record.candidate_id,
        decision=ReviewDecision.ACCEPTED,
        occurred_at=_utc_timestamp(),
        reason="validated against full-resolution RAW membership",
        refinement_state=RefinementState.RAW_REFINED,
        refined_raw_geometry=geometry,
        refinement_evidence=evidence,
        operator_identity=operator_identity or record.operator_identity,
        review_adjusted_seed_xyz_absolute_m=None,
        notes=notes if notes is not None else record.notes,
    )
    # The T### allocation, candidate confirmation, Tower registry update and the
    # token-bound group membership append all publish in ONE successor
    # generation (atomic; a failed publication leaves NO active tower missing its
    # intended group membership).
    from core.project.tower_groups import build_registry_with_tower

    def _append_to_group(tower, loaded):
        return build_registry_with_tower(
            loaded.tower_group_registry,
            group_id=record.group_id,
            tower_id=tower.tower_id,
        )

    confirmation = session.confirm_reviewed_tower_candidate(
        candidate_set_reference=candidate_set.reference,
        candidate_id=record.candidate_id,
        accepted_review_revision=event.review_revision,
        reviewer_identity=operator_identity or record.operator_identity,
        notes=notes if notes is not None else record.notes,
        local_region_reference=artifact_reference,
        tower_group_registry_updater=_append_to_group,
    )
    store.remove(token)
    return {
        "tower_id": confirmation.confirmed_tower.tower_id,
        "confirmed_tower": confirmation.confirmed_tower.to_dict(),
        "membership": artifact_record.to_dict(),
        "artifact_reference": artifact_reference.to_dict(),
        "accepted_review_revision": event.review_revision,
        "candidate_set_id": record.candidate_set_id,
        "candidate_id": record.candidate_id,
        "group_id": record.group_id,
        "group_display_name": record.group_display_name,
    }


def ordered_reselect(
    session: WritableProjectSession,
    store: OrderedPreviewStore,
    token: str,
    *,
    operator_identity: str | None = None,
    reason: str = "superseded by owner reselect",
) -> dict[str, Any]:
    """RESELECT: supersede the current pending candidate (existing review semantics).

    Appends a REJECTED review (no ``T###``), clears the preview, and returns the
    current ordered review state so the Owner can click the next tower.
    """
    if type(session) is not WritableProjectSession:
        raise ValueError("ordered RESELECT requires a writable project session")
    if type(store) is not OrderedPreviewStore:
        raise ValueError("store must be an OrderedPreviewStore")
    record = store.get(token)
    candidate_set = _validate_preview_against_session(record, session)
    from core.project import append_tower_candidate_review

    append_tower_candidate_review(
        session,
        candidate_set_reference=candidate_set.reference,
        candidate_id=record.candidate_id,
        decision=ReviewDecision.REJECTED,
        occurred_at=_utc_timestamp(),
        reason=reason,
        refinement_state=RefinementState.UNREFINED,
        operator_identity=operator_identity or record.operator_identity,
    )
    store.remove(token)
    return {"state": ordered_review_state(session)}


def ordered_cancel(
    session: WritableProjectSession,
    store: OrderedPreviewStore,
    token: str,
    *,
    operator_identity: str | None = None,
    reason: str = "cancelled by owner",
) -> dict[str, Any]:
    """CANCEL: clear the preview and leave no ``T###`` allocation.

    Marks the pending candidate cancelled through existing review semantics
    (REJECTED review) so it can never be confirmed later, and returns the
    current ordered review state.
    """
    if type(session) is not WritableProjectSession:
        raise ValueError("ordered CANCEL requires a writable project session")
    if type(store) is not OrderedPreviewStore:
        raise ValueError("store must be an OrderedPreviewStore")
    record = store.get(token)
    candidate_set = _validate_preview_against_session(record, session)
    from core.project import append_tower_candidate_review

    append_tower_candidate_review(
        session,
        candidate_set_reference=candidate_set.reference,
        candidate_id=record.candidate_id,
        decision=ReviewDecision.REJECTED,
        occurred_at=_utc_timestamp(),
        reason=reason,
        refinement_state=RefinementState.UNREFINED,
        operator_identity=operator_identity or record.operator_identity,
    )
    store.remove(token)
    return {"state": ordered_review_state(session)}


__all__ = [
    "DEFAULT_PREVIEW_SAMPLE_MAX",
    "ORDERED_CANDIDATE_SET_PREFIX",
    "ORDERED_MANUAL_ADD_DETECTOR_NAME",
    "ORDERED_MANUAL_ADD_DETECTOR_SCHEMA",
    "ORDERED_MANUAL_ADD_DETECTOR_VERSION",
    "ORDERED_WORKFLOW_VERSION",
    "PREVIEW_REFINEMENT_READY",
    "OrderedPreviewInvalidError",
    "OrderedPreviewRecord",
    "OrderedPreviewStore",
    "OrderedSelectResult",
    "OrderedWorkflowError",
    "new_preview_token",
    "ordered_cancel",
    "ordered_confirm",
    "ordered_preview",
    "ordered_reselect",
    "ordered_review_state",
    "ordered_select",
    "preview_sample_points",
]
