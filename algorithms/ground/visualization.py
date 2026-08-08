"""Bundle D: bounded, review-only Ground Human Gate diagnostics.

Authority: docs/full-resolution-ground-classification-design.md sections 6, 12
and docs/full-resolution-ground-classification-implementation-plan.md section 6
(PLAN-FROZEN).  Diagnostics consume only a fully validated FGC publication
rooted at ``fgc_completion_manifest.json``.  They never recompute Ground
authority, never change decisions, never re-run FGC numerics, never edit Human
Gate results, and never invoke downstream A1/A2A/A2B/A3/A4/W1/V1/V2.

The only formal discovery root is ``fgc_completion_manifest.json``.  Without a
valid completion manifest the directory is INCOMPLETE_NONAUTHORITATIVE and the
viewer REFUSES.  Classified LAS alone is NONAUTHORITATIVE; filenames are never
trusted.

Sampling is deterministic and source-index bound: the same publication and
configuration always produce the same displayed review points.  No wall-clock
or random state is used.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from core.contracts.frv import (
    frv_digest,
    validate_source_content_identity_digest,
    validate_source_identity_record_digest,
)
from core.contracts.ground_classification import (
    FGC_ALGORITHM_ID,
    FGC_COMPLETION_MANIFEST_PATH,
    FGC_DECISIONS_DIR,
    FGC_DIAGNOSTICS_DIR,
    FgcCompletionManifest,
    FgcDecisionManifest,
    FgcPublicationProof,
    GroundClassificationPolicyConfig,
    fgc_decision_digest,
)

# ---------------------------------------------------------------------------
# Stable refusal and failure codes (review-only, non-authoritative).
# ---------------------------------------------------------------------------

FGC_VIS_COMPLETION_MISSING = "COMPLETION_MANIFEST_MISSING"
FGC_VIS_COMPLETION_INVALID = "COMPLETION_MANIFEST_INVALID"
FGC_VIS_HASH_MISMATCH = "PUBLICATION_HASH_MISMATCH"
FGC_VIS_DECISION_INVALID = "DECISION_MANIFEST_INVALID"
FGC_VIS_PROOF_INVALID = "PUBLICATION_PROOF_INVALID"
FGC_VIS_IDENTITY_INVALID = "CLASSIFIED_IDENTITY_INVALID"
FGC_VIS_CROSS_BINDING_INVALID = "PUBLICATION_CROSS_BINDING_INVALID"
FGC_VIS_CONFIG_INVALID = "VISUALIZATION_CONFIG_INVALID"
FGC_VIS_OUTPUT_FAILED = "REVIEW_OUTPUT_FAILED"

_REVIEW_SUMMARY_SCHEMA_VERSION = "fgc-ground-review-summary-v1"
_REVIEW_HTML_BASENAME = "fgc_ground_review.html"
_REVIEW_SUMMARY_BASENAME = "fgc_ground_review_summary.json"

# State display keys (must match decision manifest total_state_counts).
_STATE_KEYS = ("confirmed_ground", "review_ground", "non_ground")
_STATE_DISPLAY = {
    "confirmed_ground": "CONFIRMED_GROUND",
    "review_ground": "REVIEW_GROUND",
    "non_ground": "NON_GROUND",
}


class FgcVisualizationError(Exception):
    """Stable review-only refusal/failure with a machine-readable code."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FgcReviewSamplingConfig:
    """Deterministic source-index sampling bounds for the review viewer.

    Selection is always ``indices[::stride]`` over ascending original indices
    per state, where ``stride = ceil(count / cap)``.  The same publication and
    configuration therefore always select the same review points.
    """

    sample_cap_per_state: int = 20_000
    non_ground_sample_cap: int = 15_000
    max_displayed_points: int = 60_000
    sampling_rule: str = "stride"

    def __post_init__(self) -> None:
        for field_name in (
            "sample_cap_per_state",
            "non_ground_sample_cap",
            "max_displayed_points",
        ):
            value = int(getattr(self, field_name))
            if value <= 0:
                raise ValueError(f"{field_name} must be positive")
            object.__setattr__(self, field_name, value)
        if self.sampling_rule != "stride":
            raise ValueError("only the deterministic 'stride' sampling rule is supported")
        # Confirmed and review share the per-state cap; non-ground is capped
        # separately.  The displayed total is therefore bounded by
        # 2 * sample_cap_per_state + non_ground_sample_cap.
        bound = 2 * self.sample_cap_per_state + self.non_ground_sample_cap
        if bound > self.max_displayed_points:
            raise ValueError(
                "per-state caps exceed max_displayed_points (2*cap + non_ground_cap)"
            )


# ---------------------------------------------------------------------------
# Validated publication view
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ValidatedFgcPublication:
    """A formally validated FGC publication with resolved component paths.

    Constructed only by :func:`validate_fgc_publication`; the completion
    manifest, decision manifest, proof, and classified identity have all been
    exact-schema validated and cross-digest bound.
    """

    root: Path
    completion_manifest: FgcCompletionManifest
    decision_manifest: FgcDecisionManifest
    publication_proof: FgcPublicationProof
    classified_las_path: Path
    decision_chunk_paths: tuple[Path, ...]
    policy_is_frozen_reviewed: bool

    @property
    def source_point_count(self) -> int:
        return self.completion_manifest.classified_las_point_count


def validate_fgc_publication(publication_root: Path | str) -> ValidatedFgcPublication:
    """Exact-schema validate a complete FGC publication rooted at *root*.

    Refuses (raises :class:`FgcVisualizationError`) on a missing or invalid
    completion manifest, any component hash/size mismatch, an invalid decision
    manifest or proof, an invalid classified identity, or any cross-digest
    binding mismatch.  The viewer never trusts filenames: all paths are bound
    by the validated completion manifest and resolved safely inside *root*.
    """
    root = Path(publication_root).resolve()
    if not root.is_dir():
        raise FgcVisualizationError(
            FGC_VIS_COMPLETION_MISSING,
            f"publication root is not a directory: {root}",
        )

    completion_path = root / FGC_COMPLETION_MANIFEST_PATH
    if not completion_path.is_file():
        raise FgcVisualizationError(
            FGC_VIS_COMPLETION_MISSING,
            f"no {FGC_COMPLETION_MANIFEST_PATH}; the directory is "
            "INCOMPLETE_NONAUTHORITATIVE",
        )
    try:
        completion_payload = json.loads(completion_path.read_text(encoding="utf-8"))
        completion = FgcCompletionManifest.from_dict(completion_payload)
    except (json.JSONDecodeError, TypeError, ValueError, KeyError, OSError) as exc:
        raise FgcVisualizationError(
            FGC_VIS_COMPLETION_INVALID,
            f"{FGC_COMPLETION_MANIFEST_PATH} is not a valid "
            "fgc-completion-manifest-v1",
        ) from exc

    try:
        completion.verify_bindings(root)
    except (ValueError, OSError) as exc:
        raise FgcVisualizationError(
            FGC_VIS_HASH_MISMATCH,
            f"published component SHA/size does not match the completion "
            f"manifest: {exc}",
        ) from exc

    decision_manifest = _load_decision_manifest(root, completion)
    publication_proof = _load_publication_proof(root, completion)
    _check_cross_bindings(completion, decision_manifest, publication_proof)

    classified_path = root / completion.classified_las_path
    if not classified_path.is_file():
        raise FgcVisualizationError(
            FGC_VIS_HASH_MISMATCH,
            f"classified LAS is missing at {completion.classified_las_path}",
        )

    chunk_paths = tuple(
        (root / FGC_DECISIONS_DIR / chunk.relative_path)
        for chunk in completion.decision_chunks
    )

    frozen_policy = GroundClassificationPolicyConfig()
    policy_is_frozen_reviewed = (
        completion.policy_digest == frozen_policy.policy_digest
    )

    return ValidatedFgcPublication(
        root=root,
        completion_manifest=completion,
        decision_manifest=decision_manifest,
        publication_proof=publication_proof,
        classified_las_path=classified_path,
        decision_chunk_paths=chunk_paths,
        policy_is_frozen_reviewed=policy_is_frozen_reviewed,
    )


def _load_decision_manifest(
    root: Path, completion: FgcCompletionManifest
) -> FgcDecisionManifest:
    decision_path = root / completion.decision_manifest_path
    try:
        payload = json.loads(decision_path.read_text(encoding="utf-8"))
        manifest = FgcDecisionManifest.from_dict(payload)
    except (json.JSONDecodeError, TypeError, ValueError, KeyError, OSError) as exc:
        raise FgcVisualizationError(
            FGC_VIS_DECISION_INVALID,
            f"decision manifest is not a valid fgc-decision-manifest-v1: {exc}",
        ) from exc
    expected_digest = fgc_decision_digest(
        source_identity=manifest.source_identity,
        policy_digest=manifest.policy_digest,
        source_point_count=manifest.source_point_count,
        chunks=manifest.chunks,
    )
    if manifest.decision_digest != expected_digest:
        raise FgcVisualizationError(
            FGC_VIS_CROSS_BINDING_INVALID,
            "decision_digest does not match identity/policy/N/ordered chunks",
        )
    if manifest.source_point_count != completion.classified_las_point_count:
        raise FgcVisualizationError(
            FGC_VIS_CROSS_BINDING_INVALID,
            "decision manifest point count does not match the completion manifest",
        )
    return manifest


def _load_publication_proof(
    root: Path, completion: FgcCompletionManifest
) -> FgcPublicationProof:
    proof_path = root / completion.publication_proof_path
    try:
        payload = json.loads(proof_path.read_text(encoding="utf-8"))
        proof = FgcPublicationProof.from_dict(payload)
    except (json.JSONDecodeError, TypeError, ValueError, KeyError, OSError) as exc:
        raise FgcVisualizationError(
            FGC_VIS_PROOF_INVALID,
            f"publication proof is not a valid fgc-publication-proof-v1: {exc}",
        ) from exc
    return proof


def _check_cross_bindings(
    completion: FgcCompletionManifest,
    decision_manifest: FgcDecisionManifest,
    proof: FgcPublicationProof,
) -> None:
    mismatches: list[str] = []
    if proof.policy_digest != completion.policy_digest:
        mismatches.append("proof policy_digest != completion policy_digest")
    if decision_manifest.policy_digest != completion.policy_digest:
        mismatches.append("decision manifest policy_digest != completion policy_digest")
    if proof.decision_digest != decision_manifest.decision_digest:
        mismatches.append("proof decision_digest != decision manifest decision_digest")
    if proof.decision_manifest_digest != completion.decision_manifest_digest:
        mismatches.append(
            "proof decision_manifest_digest != completion decision_manifest_digest"
        )
    if frv_digest(decision_manifest.to_dict()) != completion.decision_manifest_digest:
        mismatches.append("decision manifest digest != completion decision_manifest_digest")
    if proof.classified_source_identity != completion.classified_source_identity:
        mismatches.append("proof classified identity != completion classified identity")
    if proof.classified_source_sha256 != completion.classified_las_sha256:
        mismatches.append("proof classified SHA != completion classified SHA")
    if proof.raw_source_identity != completion.raw_source_identity:
        mismatches.append("proof raw identity != completion raw identity")
    if proof.source_point_count != completion.classified_las_point_count:
        mismatches.append("proof point count != completion point count")
    if mismatches:
        raise FgcVisualizationError(
            FGC_VIS_CROSS_BINDING_INVALID,
            "FGC publication cross-digest bindings do not hold: "
            + "; ".join(mismatches),
        )
    try:
        validate_source_content_identity_digest(completion.classified_source_identity)
        validate_source_identity_record_digest(completion.classified_source_identity)
    except (TypeError, ValueError) as exc:
        raise FgcVisualizationError(
            FGC_VIS_IDENTITY_INVALID,
            f"classified SourceIdentity digests are invalid: {exc}",
        ) from exc


# ---------------------------------------------------------------------------
# Deterministic source-index sampling
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FgcReviewPoint:
    """One deterministic displayed review point (original-index bound)."""

    original_index: int
    state: str
    x: float
    y: float
    z: float
    classification: int


@dataclass(frozen=True)
class FgcReviewSample:
    """Bounded deterministic review sample over a validated publication."""

    source_point_count: int
    displayed_total: int
    source_state_counts: dict[str, int] = field(default_factory=dict)
    displayed_state_counts: dict[str, int] = field(default_factory=dict)
    strides: dict[str, int] = field(default_factory=dict)
    sample_cap_per_state: int = 0
    non_ground_sample_cap: int = 0
    sampling_rule: str = "stride"
    bounds_min_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)
    bounds_max_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)
    points: tuple[FgcReviewPoint, ...] = ()

    def points_for(self, state: str) -> list[FgcReviewPoint]:
        return [point for point in self.points if point.state == state]


def sample_review_points(
    publication: ValidatedFgcPublication,
    config: FgcReviewSamplingConfig | None = None,
) -> FgcReviewSample:
    """Select the deterministic review points from a validated publication.

    State bytes come only from the validated decision chunks; X/Y/Z and base
    classification come only from the hash-verified classified LAS.  No FGC
    numeric is recomputed and no decision is modified.
    """
    if config is None:
        config = FgcReviewSamplingConfig()
    if type(config) is not FgcReviewSamplingConfig:
        raise FgcVisualizationError(
            FGC_VIS_CONFIG_INVALID,
            "config must be a FgcReviewSamplingConfig",
        )

    indices_by_state: dict[str, list[int]] = {key: [] for key in _STATE_KEYS}
    for chunk_path in publication.decision_chunk_paths:
        try:
            chunk_bytes = chunk_path.read_bytes()
        except OSError as exc:
            raise FgcVisualizationError(
                FGC_VIS_HASH_MISMATCH,
                f"decision chunk is unreadable: {chunk_path.name}",
            ) from exc
        states = np.frombuffer(chunk_bytes, dtype=np.uint8)
        for state_key in _STATE_KEYS:
            state_value = _state_value(state_key)
            positions = np.flatnonzero(states == state_value)
            if positions.size:
                indices_by_state[state_key].extend(int(position) for position in positions)

    displayed: dict[str, list[int]] = {}
    strides: dict[str, int] = {}
    for state_key in _STATE_KEYS:
        indices = indices_by_state[state_key]
        cap = (
            config.non_ground_sample_cap
            if state_key == "non_ground"
            else config.sample_cap_per_state
        )
        stride = max(1, math.ceil(len(indices) / cap)) if indices else 1
        strides[state_key] = stride
        displayed[state_key] = indices[::stride]

    source_counts = _source_state_counts(publication)
    displayed_counts = {
        key: len(displayed[key]) for key in _STATE_KEYS
    }
    selected = sorted(
        index for indices in displayed.values() for index in indices
    )

    xyz, classification = _read_sample_coordinates(
        publication.classified_las_path,
        selected,
        point_count=publication.source_point_count,
    )
    point_index = 0
    points: list[FgcReviewPoint] = []
    for state_key in _STATE_KEYS:
        for index in displayed[state_key]:
            points.append(
                FgcReviewPoint(
                    original_index=index,
                    state=state_key,
                    x=float(xyz[point_index, 0]),
                    y=float(xyz[point_index, 1]),
                    z=float(xyz[point_index, 2]),
                    classification=int(classification[point_index]),
                )
            )
            point_index += 1

    return FgcReviewSample(
        source_point_count=publication.source_point_count,
        displayed_total=len(points),
        source_state_counts=dict(source_counts),
        displayed_state_counts=displayed_counts,
        strides=strides,
        sample_cap_per_state=config.sample_cap_per_state,
        non_ground_sample_cap=config.non_ground_sample_cap,
        sampling_rule=config.sampling_rule,
        bounds_min_xyz=_read_bounds(publication.classified_las_path)[0],
        bounds_max_xyz=_read_bounds(publication.classified_las_path)[1],
        points=tuple(points),
    )


def _state_value(state_key: str) -> int:
    from core.contracts.ground_classification import (
        CONFIRMED_GROUND,
        NON_GROUND,
        REVIEW_GROUND,
    )

    if state_key == "confirmed_ground":
        return CONFIRMED_GROUND
    if state_key == "review_ground":
        return REVIEW_GROUND
    if state_key == "non_ground":
        return NON_GROUND
    raise FgcVisualizationError(FGC_VIS_CONFIG_INVALID, f"unknown state: {state_key}")


def _source_state_counts(
    publication: ValidatedFgcPublication,
) -> dict[str, int]:
    total = publication.decision_manifest.total_state_counts
    return {
        "confirmed_ground": int(total.get("confirmed_ground", 0)),
        "review_ground": int(total.get("review_ground", 0)),
        "non_ground": int(total.get("non_ground", 0)),
    }


def _read_bounds(
    classified_path: Path,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    import laspy

    with laspy.open(classified_path, mode="r") as reader:
        header = reader.header
        return (
            (float(header.mins[0]), float(header.mins[1]), float(header.mins[2])),
            (float(header.maxs[0]), float(header.maxs[1]), float(header.maxs[2])),
        )


def _read_sample_coordinates(
    classified_path: Path,
    selected_indices: list[int],
    *,
    point_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Read X/Y/Z and base classification at *selected_indices*.

    Grouped contiguous runs keep the LAS random access bounded: only the
    displayed points are decoded, never the whole scene.
    """
    import laspy

    xyz = np.empty((len(selected_indices), 3), dtype=np.float64)
    classification = np.empty(len(selected_indices), dtype=np.uint8)
    if not selected_indices:
        return xyz, classification

    with laspy.open(classified_path, mode="r") as reader:
        if int(reader.header.point_count) != point_count:
            raise FgcVisualizationError(
                FGC_VIS_CROSS_BINDING_INVALID,
                "classified LAS point count does not match the publication",
            )
        out_pos = 0
        for start, count in _contiguous_runs(selected_indices):
            reader.seek(start)
            record = reader.read_points(count)
            xs = np.asarray(record.x)
            ys = np.asarray(record.y)
            zs = np.asarray(record.z)
            cls = np.asarray(record.classification, dtype=np.uint8)
            end = out_pos + count
            xyz[out_pos:end, 0] = xs
            xyz[out_pos:end, 1] = ys
            xyz[out_pos:end, 2] = zs
            classification[out_pos:end] = cls
            out_pos = end
    return xyz, classification


def _contiguous_runs(sorted_indices: list[int]) -> list[tuple[int, int]]:
    """Group ascending indices into (start, length) contiguous runs."""
    runs: list[tuple[int, int]] = []
    for index in sorted_indices:
        if runs and index == runs[-1][0] + runs[-1][1]:
            runs[-1] = (runs[-1][0], runs[-1][1] + 1)
        else:
            runs.append((index, 1))
    return runs


# ---------------------------------------------------------------------------
# Review summary and HTML artifacts (non-authoritative, bounded)
# ---------------------------------------------------------------------------


def build_ground_review_summary(
    publication: ValidatedFgcPublication,
    sample: FgcReviewSample,
) -> dict[str, Any]:
    """Structured JSON-safe machine summary over validated artifacts only."""
    completion = publication.completion_manifest
    proof = publication.publication_proof
    decision = publication.decision_manifest
    n = sample.source_point_count

    def fraction(count: int) -> float:
        if n == 0:
            return 0.0
        return round(count / n, 6)

    source_counts = sample.source_state_counts
    coverage = decision.coverage
    coverage_summary: dict[str, Any] | None = None
    if coverage is not None:
        components = tuple(coverage.unsupported_components)
        interior = tuple(component for component in components if not component.is_edge)
        max_interior_width = 0.0
        for component in interior:
            max_interior_width = max(
                max_interior_width,
                component.width_m(coverage_cell_m=1.0),
            )
        coverage_summary = {
            "observed_cell_count": int(coverage.observed_cell_count),
            "confirmed_occupied_cell_count": int(coverage.confirmed_occupied_cell_count),
            "occupancy_fraction": round(float(coverage.occupancy_fraction), 6),
            "unsupported_component_count": len(components),
            "interior_component_count": len(interior),
            "edge_component_count": len(components) - len(interior),
            "max_interior_width_m": round(max_interior_width, 3),
        }

    mapping = dict(proof.classification_mapping_counts)
    class_2_published = int(mapping.get("to_2", 0))

    frozen_policy = GroundClassificationPolicyConfig()
    resource_block: dict[str, Any] | None = None
    if publication.policy_is_frozen_reviewed:
        resource_block = {
            "tile_side_m": frozen_policy.tile_side_m,
            "halo_m": frozen_policy.halo_m,
            "fine_cell_m": frozen_policy.fine_cell_m,
            "coverage_cell_m": frozen_policy.coverage_cell_m,
            "max_source_points": frozen_policy.max_source_points,
            "max_core_tile_points": frozen_policy.max_core_tile_points,
            "max_halo_tile_points": frozen_policy.max_halo_tile_points,
            "max_halo_grid_cells": frozen_policy.max_halo_grid_cells,
            "max_plane_batch": frozen_policy.max_plane_batch,
            "decision_chunk_rows": frozen_policy.decision_chunk_rows,
            "working_set_bytes": frozen_policy.max_halo_tile_points * 256
            + frozen_policy.max_halo_grid_cells * 128
            + frozen_policy.max_plane_batch * 128
            + 96 * 1024 * 1024
            + 256 * 1024 * 1024,
        }

    decision_chunk_bytes = 0
    for chunk_path in publication.decision_chunk_paths:
        try:
            decision_chunk_bytes += chunk_path.stat().st_size
        except OSError:
            pass

    return {
        "schema_version": _REVIEW_SUMMARY_SCHEMA_VERSION,
        "status": "READY_FOR_OWNER_HUMAN_GATE_REVIEW",
        "machine_validity": "PASS",
        "owner_decision": "PENDING",
        "algorithm_id": FGC_ALGORITHM_ID,
        "publication": {
            "root": str(publication.root),
            "completion_manifest": FGC_COMPLETION_MANIFEST_PATH,
            "completion_digest": completion.completion_digest,
            "decision_manifest_digest": completion.decision_manifest_digest,
            "publication_proof_digest": proof.proof_digest,
            "policy_digest": completion.policy_digest,
            "policy_is_frozen_reviewed": publication.policy_is_frozen_reviewed,
            "p_source_replay_status": str(proof.p_source_replay.get("status", "")),
        },
        "counts": {
            "total_points": n,
            "confirmed_ground_count": int(source_counts["confirmed_ground"]),
            "confirmed_fraction": fraction(source_counts["confirmed_ground"]),
            "review_ground_count": int(source_counts["review_ground"]),
            "review_fraction": fraction(source_counts["review_ground"]),
            "non_ground_count": int(source_counts["non_ground"]),
            "non_ground_fraction": fraction(source_counts["non_ground"]),
            "class_2_published_count": class_2_published,
        },
        "coverage": coverage_summary,
        "tiles": {
            "tile_count": len(decision.tile_summaries),
            "tile_ids": [
                [int(tile.tile_id_ty), int(tile.tile_id_tx)]
                for tile in decision.tile_summaries
            ],
        },
        "resource_statistics": resource_block,
        "p_hag_compatibility": dict(proof.p_hag_compatibility),
        "identity": {
            "raw_source_id": completion.raw_source_identity.source_id,
            "raw_source_sha256": completion.raw_source_identity.source_file_sha256,
            "raw_source_identity_record_digest": (
                completion.raw_source_identity.source_identity_record_digest
            ),
            "classified_source_id": completion.classified_source_identity.source_id,
            "classified_source_sha256": completion.classified_las_sha256,
            "classified_source_identity_record_digest": (
                completion.classified_source_identity.source_identity_record_digest
            ),
            "classified_las_point_count": completion.classified_las_point_count,
        },
        "sampling": {
            "rule": sample.sampling_rule,
            "source_count": sample.source_point_count,
            "displayed_total": sample.displayed_total,
            "displayed_state_counts": dict(sample.displayed_state_counts),
            "source_state_counts": dict(sample.source_state_counts),
            "strides": dict(sample.strides),
            "sample_cap_per_state": sample.sample_cap_per_state,
            "non_ground_sample_cap": sample.non_ground_sample_cap,
            "max_displayed_points": (
                sample.sample_cap_per_state * 2 + sample.non_ground_sample_cap
            ),
        },
        "review_artifacts": {
            "html_basename": _REVIEW_HTML_BASENAME,
            "summary_basename": _REVIEW_SUMMARY_BASENAME,
        },
    }


def build_ground_review_html(
    publication: ValidatedFgcPublication,
    sample: FgcReviewSample,
    summary: dict[str, Any],
) -> str:
    """Return a self-contained interactive HTML report (review only).

    The report is non-authoritative and never creates a Human Gate result.
    Problem/unsupported cells and tile boundaries are drawn only when the
    frozen reviewed policy is proven by digest, because the frozen policy
    defines the coverage-cell and tile geometry.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    completion = publication.completion_manifest
    frozen = publication.policy_is_frozen_reviewed

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Top XY - CONFIRMED / REVIEW / NON-GROUND (sampled)",
            "State distribution (source counts)",
            "Side profile Z vs X (sampled)",
            "Side profile Z vs Y (sampled)",
        ),
        specs=[
            [{"type": "scattergl"}, {"type": "bar"}],
            [{"type": "scattergl"}, {"type": "scattergl"}],
        ],
        vertical_spacing=0.16,
        horizontal_spacing=0.10,
    )

    _add_state_trace(fig, sample, "confirmed_ground", "#1a7f37", 2, 1.0, row=1, col=1)
    _add_state_trace(fig, sample, "review_ground", "#d97706", 2, 1.0, row=1, col=1)
    _add_state_trace(fig, sample, "non_ground", "#9ca3af", 1, 0.5, row=1, col=1)

    if frozen:
        _add_tile_boundaries(fig, publication, sample)
        _add_problem_cells(fig, publication)

    fig.add_trace(
        go.Bar(
            x=[_STATE_DISPLAY[key] for key in _STATE_KEYS],
            y=[sample.source_state_counts[key] for key in _STATE_KEYS],
            marker_color=["#1a7f37", "#d97706", "#9ca3af"],
            name="source counts",
        ),
        row=1,
        col=2,
    )

    _add_profile_trace(fig, sample, "x", row=2, col=1)
    _add_profile_trace(fig, sample, "y", row=2, col=2)

    fig.update_layout(
        title_text=(
            f"FGC Ground Human Gate review - {completion.raw_source_identity.source_id}"
        ),
        height=1000,
        showlegend=True,
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        hovermode="closest",
    )
    fig.update_xaxes(title_text="X (m)", row=1, col=1)
    fig.update_yaxes(title_text="Y (m)", row=1, col=1)
    fig.update_xaxes(title_text="state", row=1, col=2)
    fig.update_yaxes(title_text="count", row=1, col=2)
    fig.update_xaxes(title_text="X (m)", row=2, col=1)
    fig.update_yaxes(title_text="Z (m)", row=2, col=1)
    fig.update_xaxes(title_text="Y (m)", row=2, col=2)
    fig.update_yaxes(title_text="Z (m)", row=2, col=2)

    header = _html_header(summary)
    footer = _html_footer(summary)
    body = fig.to_html(
        full_html=False,
        include_plotlyjs=True,
        config={"scrollZoom": True, "displaylogo": False},
    )
    return header + body + footer


def _add_state_trace(
    fig: Any,
    sample: FgcReviewSample,
    state: str,
    color: str,
    size: int,
    opacity: float,
    *,
    row: int,
    col: int,
) -> None:
    import plotly.graph_objects as go

    points = sample.points_for(state)
    if not points:
        fig.add_trace(
            go.Scattergl(
                x=[],
                y=[],
                mode="markers",
                name=_STATE_DISPLAY[state],
                marker=dict(size=size, color=color),
            ),
            row=row,
            col=col,
        )
        return
    fig.add_trace(
        go.Scattergl(
            x=[point.x for point in points],
            y=[point.y for point in points],
            mode="markers",
            name=_STATE_DISPLAY[state],
            opacity=opacity,
            customdata=[
                [point.original_index, round(point.z, 2), point.classification]
                for point in points
            ],
            hovertemplate=(
                "index=%{customdata[0]}<br>z=%{customdata[1]} m<br>class=%{customdata[2]}"
            ),
            marker=dict(size=size, color=color),
        ),
        row=row,
        col=col,
    )


def _add_profile_trace(fig: Any, sample: FgcReviewSample, axis: str, *, row: int, col: int) -> None:
    import plotly.graph_objects as go

    for state, color, size in (
        ("confirmed_ground", "#1a7f37", 2),
        ("review_ground", "#d97706", 2),
        ("non_ground", "#9ca3af", 1),
    ):
        points = sample.points_for(state)
        if not points:
            continue
        if axis == "x":
            horizontal = [point.x for point in points]
        else:
            horizontal = [point.y for point in points]
        fig.add_trace(
            go.Scattergl(
                x=horizontal,
                y=[point.z for point in points],
                mode="markers",
                name=_STATE_DISPLAY[state],
                marker=dict(size=size, color=color),
                showlegend=False,
            ),
            row=row,
            col=col,
        )


def _add_tile_boundaries(
    fig: Any,
    publication: ValidatedFgcPublication,
    sample: FgcReviewSample,
) -> None:
    x0, y0, _ = sample.bounds_min_xyz
    x_max, y_max, _ = sample.bounds_max_xyz
    tile_side = 256.0
    for tile in publication.decision_manifest.tile_summaries:
        tx_lo = x0 + tile.tile_id_tx * tile_side
        tx_hi = min(x0 + (tile.tile_id_tx + 1) * tile_side, x_max)
        ty_lo = y0 + tile.tile_id_ty * tile_side
        ty_hi = min(y0 + (tile.tile_id_ty + 1) * tile_side, y_max)
        fig.add_shape(
            type="rect",
            x0=tx_lo,
            y0=ty_lo,
            x1=tx_hi,
            y1=ty_hi,
            line=dict(color="rgba(100,100,100,0.45)", width=1),
            fillcolor="rgba(0,0,0,0)",
            row=1,
            col=1,
        )


def _add_problem_cells(fig: Any, publication: ValidatedFgcPublication) -> None:
    coverage = publication.decision_manifest.coverage
    if coverage is None:
        return
    x0 = None
    y0 = None
    for component in coverage.unsupported_components:
        # Coverage cells are 1.0 m source-min-anchored (frozen policy).  The
        # decoded origin equals the classified LAS minimum.
        if x0 is None:
            x0, y0, _ = _read_bounds(publication.classified_las_path)[0]
        cell = 1.0
        x_lo = x0 + component.min_ix * cell
        x_hi = x0 + (component.max_ix + 1) * cell
        y_lo = y0 + component.min_iy * cell
        y_hi = y0 + (component.max_iy + 1) * cell
        fig.add_shape(
            type="rect",
            x0=x_lo,
            y0=y_lo,
            x1=x_hi,
            y1=y_hi,
            line=dict(color="rgba(220,38,38,0.8)", width=1),
            fillcolor="rgba(220,38,38,0.12)",
            name="unsupported/problem cell",
            row=1,
            col=1,
        )


def _html_header(summary: dict[str, Any]) -> str:
    identity = summary["identity"]
    counts = summary["counts"]
    sampling = summary["sampling"]
    title = identity["raw_source_id"]
    sha_raw = identity["raw_source_sha256"]
    sha_cls = identity["classified_source_sha256"]
    total = counts["total_points"]
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>FGC Ground Human Gate review - {title}</title>
<style>
  body {{
    font-family: -apple-system, "Segoe UI", Roboto, sans-serif;
    margin: 24px; color: #1f2937;
  }}
  h1 {{ font-size: 20px; }}
  table {{ border-collapse: collapse; margin: 12px 0; }}
  th, td {{
    border: 1px solid #d1d5db; padding: 4px 10px;
    font-size: 13px; text-align: left;
  }}
  th {{ background: #f3f4f6; }}
  .status-pill {{
    display: inline-block; padding: 2px 10px; border-radius: 9999px;
    background: #dcfce7; color: #166534;
    font-size: 12px; font-weight: 600;
  }}
  .pending-pill {{
    display: inline-block; padding: 2px 10px; border-radius: 9999px;
    background: #fef9c3; color: #854d0e;
    font-size: 12px; font-weight: 600;
  }}
  details {{ margin-top: 16px; }}
</style>
</head>
<body>
<h1>FGC Ground Human Gate review</h1>
<p>
  <span class="status-pill">MACHINE VALIDITY: PASS</span>&nbsp;
  <span class="pending-pill">OWNER DECISION: PENDING</span>
</p>
<p>Raw source: <code>{title}</code> &middot; {total} points &middot;
   algorithm <code>{summary["algorithm_id"]}</code></p>
<table>
<tr><th>Total</th><td>{total}</td></tr>
<tr><th>Confirmed ground</th>
    <td>{counts["confirmed_ground_count"]} ({counts["confirmed_fraction"]:.1%})</td></tr>
<tr><th>Review ground</th>
    <td>{counts["review_ground_count"]} ({counts["review_fraction"]:.1%})</td></tr>
<tr><th>Non-ground</th>
    <td>{counts["non_ground_count"]} ({counts["non_ground_fraction"]:.1%})</td></tr>
<tr><th>Class 2 published</th>
    <td>{counts["class_2_published_count"]}</td></tr>
<tr><th>Sampled / source</th>
    <td>{sampling["displayed_total"]} / {sampling["source_count"]}</td></tr>
<tr><th>Raw SHA</th><td><code>{sha_raw}</code></td></tr>
<tr><th>Classified SHA</th><td><code>{sha_cls}</code></td></tr>
</table>
"""


def _html_footer(summary: dict[str, Any]) -> str:
    encoded = json.dumps(summary, indent=2)
    encoded = encoded.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return f"""<details>
<summary>Structured review summary (JSON)</summary>
<pre style="background:#f9fafb; padding:12px; overflow-x:auto;">{encoded}</pre>
</details>
</body>
</html>
"""


def generate_ground_human_gate_material(
    publication_root: Path | str,
    review_dir: Path | str,
    config: FgcReviewSamplingConfig | None = None,
) -> dict[str, Any]:
    """Validate, sample, and write the bounded Ground Human Gate review material.

    Writes ``fgc_ground_review.html`` and ``fgc_ground_review_summary.json``
    into *review_dir* and returns the structured machine summary.  This never
    creates a Human Gate result and never auto-approves.
    """
    if config is None:
        config = FgcReviewSamplingConfig()
    if type(config) is not FgcReviewSamplingConfig:
        raise FgcVisualizationError(
            FGC_VIS_CONFIG_INVALID,
            "config must be a FgcReviewSamplingConfig",
        )
    publication = validate_fgc_publication(publication_root)
    sample = sample_review_points(publication, config)
    summary = build_ground_review_summary(publication, sample)
    try:
        out_dir = Path(review_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        html = build_ground_review_html(publication, sample, summary)
        html_path = out_dir / _REVIEW_HTML_BASENAME
        html_path.write_text(html, encoding="utf-8")
        summary_path = out_dir / _REVIEW_SUMMARY_BASENAME
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    except OSError as exc:
        raise FgcVisualizationError(
            FGC_VIS_OUTPUT_FAILED,
            f"failed to write review material into {review_dir}: {exc}",
        ) from exc
    summary["review_artifacts"]["html_path"] = str(html_path)
    summary["review_artifacts"]["summary_path"] = str(summary_path)
    return summary


__all__ = [
    "FGC_DIAGNOSTICS_DIR",
    "FgcReviewPoint",
    "FgcReviewSample",
    "FgcReviewSamplingConfig",
    "FgcVisualizationError",
    "ValidatedFgcPublication",
    "build_ground_review_html",
    "build_ground_review_summary",
    "generate_ground_human_gate_material",
    "sample_review_points",
    "validate_fgc_publication",
]
