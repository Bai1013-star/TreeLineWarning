"""V2A deterministic pair-only review associations from verified V2 siblings.

This module intentionally contains no merge, identity, manual-state, or future
module integration path.  It consumes artifacts read-only and emits a flat,
independently reloadable pair table.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from core.contracts import (
    BLOCKING_EVIDENCE_NAMES,
    POSITIVE_EVIDENCE_NAMES,
    CancellationToken,
    LocalScaleState,
    PipelineProgress,
    ProgressReporter,
    VegetationAssociationConfig,
    VegetationAssociationProvenanceFlag,
    VegetationAssociationReasonFlag,
    VegetationAssociationResult,
    VegetationAssociationStatistics,
    VegetationAssociationStatus,
    VegetationCandidateLevel,
    VegetationHagState,
    VegetationInstanceReasonFlag,
    validate_association_arrays,
)
from core.contracts.errors import UnifiedError
from scipy.spatial import cKDTree

from algorithms.vegetation.candidates import load_vegetation_candidates
from algorithms.vegetation.instances import load_vegetation_instances

SCHEMA_VERSION = "vegetation-associations-v1"
ALGORITHM_VERSION = "v2a-review-level-vegetation-association-v1"
_BASE = "{source_id}_vegetation_associations"


class VegetationAssociationError(RuntimeError):
    def __init__(self, error: UnifiedError):
        super().__init__(error.message)
        self.error = error


@dataclass(frozen=True)
class _Summary:
    cluster_id: int
    rows: np.ndarray
    xyz: np.ndarray
    original: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    center: np.ndarray
    direction: np.ndarray
    direction_reliability: float
    scale: float
    scale_valid: bool
    clamped_high_fraction: float
    density: float
    hag: float
    hag_reliable: bool
    profile: float
    morphology: float
    planar: float
    slender: float
    compact: float
    mixed: float
    parent: int
    unresolved: bool


def run_vegetation_associations(
    v2_npz_path: Path | str,
    v2_json_path: Path | str,
    v1_npz_path: Path | str,
    v1_json_path: Path | str,
    a2a_npz_path: Path | str,
    a2a_json_path: Path | str,
    a2b_npz_path: Path | str,
    a2b_json_path: Path | str,
    output_dir: Path | str,
    *,
    config: VegetationAssociationConfig | None = None,
    overwrite: bool = False,
    reporter: ProgressReporter | None = None,
    cancellation_token: CancellationToken | None = None,
    project_root: Path | str | None = None,
) -> VegetationAssociationResult:
    """Create review-only V2A pairs; no raw point, LAS, A3, or V2 write occurs."""
    cfg = config or VegetationAssociationConfig()
    root = (Path(project_root) if project_root is not None else Path.cwd()).resolve()
    outdir = Path(output_dir).resolve()
    _validate_output_dir(outdir, root / "outputs" / "vegetation_associations")
    npz_out = outdir / f"{_BASE.format(source_id=cfg.source_id)}.npz"
    json_out = outdir / f"{_BASE.format(source_id=cfg.source_id)}.json"
    _validate_outputs((npz_out, json_out), overwrite)
    started, process = time.perf_counter(), _current_process() if cfg.record_memory else None
    rss_start = _rss(process)
    phases: dict[str, float] = {}
    _check_cancelled(cancellation_token, "before V2A validation")
    phase = time.perf_counter()
    v2_result, v2 = load_vegetation_instances(v2_npz_path, v2_json_path)
    v1_result, v1 = load_vegetation_candidates(v1_npz_path, v1_json_path)
    a2a, a2a_meta = _load_pair(Path(a2a_npz_path), Path(a2a_json_path))
    a2b, a2b_meta = _load_pair(Path(a2b_npz_path), Path(a2b_json_path))
    paths = {
        "v2_npz": Path(v2_npz_path).resolve(),
        "v2_json": Path(v2_json_path).resolve(),
        "v1_npz": Path(v1_npz_path).resolve(),
        "v1_json": Path(v1_json_path).resolve(),
        "a2a_npz": Path(a2a_npz_path).resolve(),
        "a2a_json": Path(a2a_json_path).resolve(),
        "a2b_npz": Path(a2b_npz_path).resolve(),
        "a2b_json": Path(a2b_json_path).resolve(),
    }
    hashes = {name: _sha(path) for name, path in paths.items()}
    _validate_inputs(v2_result, v2, v1_result, v1, a2a, a2a_meta, a2b, a2b_meta, hashes, cfg)
    phases["validation"] = time.perf_counter() - phase
    _report(reporter, cfg.source_id, "validation", 1, 1)
    _check_cancelled(cancellation_token, "after V2A validation")
    phase = time.perf_counter()
    summaries = _summaries(v2, a2a, a2b)
    phases["summaries"] = time.perf_counter() - phase
    _report(reporter, cfg.source_id, "summaries", len(summaries), len(summaries))
    _check_cancelled(cancellation_token, "after V2A summaries")
    phase = time.perf_counter()
    candidates = _candidate_pairs(summaries, cfg)
    phases["candidates"] = time.perf_counter() - phase
    _report(reporter, cfg.source_id, "candidates", len(candidates), max(len(candidates), 1))
    phase = time.perf_counter()
    rows = [
        _evaluate_pair(summaries[a], summaries[b], v2, v1, a2a, a2b, cfg) for a, b in candidates
    ]
    arrays = _arrays(rows)
    phases["evidence_and_statuses"] = time.perf_counter() - phase
    _check_cancelled(cancellation_token, "before V2A publication")
    values = arrays["pair_status"]
    stats = VegetationAssociationStatistics(
        input_cluster_count=len(summaries),
        candidate_pair_count=len(rows),
        abstained_count=int(np.count_nonzero(values == int(VegetationAssociationStatus.ABSTAINED))),
        possible_review_count=int(
            np.count_nonzero(values == int(VegetationAssociationStatus.POSSIBLE_REVIEW_ASSOCIATION))
        ),
        strong_review_count=int(
            np.count_nonzero(values == int(VegetationAssociationStatus.STRONG_REVIEW_ASSOCIATION))
        ),
        blocked_count=int(
            np.count_nonzero(values == int(VegetationAssociationStatus.BLOCKED_BY_CONFLICT))
        ),
        elapsed_seconds=time.perf_counter() - started,
        rss_start_bytes=rss_start,
        rss_peak_bytes=_rss(process),
        rss_end_bytes=_rss(process),
        phase_timings_seconds=phases,
    )
    outdir.mkdir(parents=True, exist_ok=True)
    result = _publish(arrays, npz_out, json_out, cfg, v2_result, hashes, stats)
    _report(reporter, cfg.source_id, "completed", 1, 1, "completed")
    return result


def load_vegetation_associations(
    npz_path: Path | str | None = None, json_path: Path | str | None = None
) -> tuple[VegetationAssociationResult, dict[str, np.ndarray]]:
    """Reload and independently validate a published V2A pair artifact."""
    if json_path is None:
        if npz_path is None or Path(npz_path).suffix.lower() != ".json":
            raise _error("V2A_LOAD_PATH", "supply a JSON path or sibling NPZ/JSON paths")
        json_file = Path(npz_path)
        raw = _load_json(json_file)
        npz_file = json_file.with_name(str(raw.get("output_npz_filename", "")))
    else:
        npz_file, json_file = Path(npz_path), Path(json_path)
    if not npz_file.is_file() or not json_file.is_file():
        raise _error("V2A_LOAD_MISSING", "V2A NPZ and JSON are both required")
    raw_result = _load_json(json_file)
    result = VegetationAssociationResult.from_dict(
        {key: raw_result[key] for key in VegetationAssociationResult.__dataclass_fields__}
    )
    if npz_file.name != result.output_npz_filename or _sha(npz_file) != result.output_npz_sha256:
        raise _error("V2A_LOAD_HASH", "V2A NPZ does not match JSON manifest")
    arrays = _load_npz(npz_file)
    try:
        validate_association_arrays(arrays)
    except ValueError as exc:
        raise _error("V2A_LOAD_ARRAY", "invalid V2A output arrays", cause=str(exc)) from exc
    if result.statistics.candidate_pair_count != len(arrays["pair_status"]):
        raise _error("V2A_LOAD_STATS", "pair count differs from manifest")
    return result, arrays


def _summaries(
    v2: Mapping[str, np.ndarray], a2a: Mapping[str, np.ndarray], a2b: Mapping[str, np.ndarray]
) -> dict[int, _Summary]:
    cluster_ids = np.asarray(v2["cluster_id"], dtype=np.int32)
    point_cluster = np.asarray(v2["cluster_id_by_point"], dtype=np.int32)
    output: dict[int, _Summary] = {}
    for position, cid in enumerate(sorted(map(int, cluster_ids))):
        rows = np.flatnonzero(point_cluster == cid).astype(np.int64)
        xyz = v2["xyz_local"][rows].astype(np.float64, copy=False)
        core = rows[
            np.isin(
                v2["vegetation_level"][rows],
                [int(VegetationCandidateLevel.MEDIUM), int(VegetationCandidateLevel.HIGH)],
            )
        ]
        scales = v2["local_scale_m"][core]
        states = v2["local_scale_state"][core]
        valid = (
            np.isfinite(scales)
            & (scales > 0)
            & (states != int(LocalScaleState.LOCAL_SCALE_UNAVAILABLE))
        )
        hag_values = a2b["height_agl_m"][core]
        reliable = v2["hag_state"][core] == int(VegetationHagState.RELIABLE)

        def descriptor(name: str, default: float = 0.0, row: int = position) -> float:
            return float(v2[name][row]) if name in v2 else default

        unresolved = bool(
            np.any(
                v2["assignment_reason_flags"][rows]
                & int(VegetationInstanceReasonFlag.UNRESOLVED_TOPOLOGY)
            )
        )
        direction, direction_reliability = _stable_direction(xyz)
        output[cid] = _Summary(
            cid,
            rows,
            xyz,
            v2["original_index"][rows],
            xyz.min(axis=0),
            xyz.max(axis=0),
            xyz.mean(axis=0),
            direction,
            direction_reliability,
            float(np.median(scales[valid])) if valid.any() else 0.0,
            bool(valid.any()),
            float(np.mean(states == int(LocalScaleState.LOCAL_SCALE_CLAMPED_HIGH)))
            if len(states)
            else 0.0,
            float(np.median(a2a["local_density"][core])) if len(core) else 0.0,
            float(np.median(hag_values[reliable])) if reliable.any() else 0.0,
            bool(reliable.any()),
            descriptor("cluster_profile_score"),
            descriptor("cluster_vegetation_morphology_score"),
            descriptor("cluster_planar_ambiguity_score"),
            descriptor("cluster_isolated_slender_ambiguity_score"),
            descriptor("cluster_compact_layered_ambiguity_score"),
            descriptor("cluster_mixed_morphology_conflict_score"),
            int(v2.get("cluster_parent_id", np.full(len(cluster_ids), -1))[position]),
            unresolved,
        )
    return output


def _stable_direction(xyz: np.ndarray) -> tuple[np.ndarray, float]:
    """Return a sign-invariant-ready principal axis and its PCA stability.

    The reliability is the normalized separation between the largest two
    covariance eigenvalues.  Approximately isotropic point sets therefore have
    low reliability and cannot manufacture directional continuation evidence.
    """
    if len(xyz) < 4 or not np.isfinite(xyz).all():
        return np.zeros(3, np.float64), 0.0
    centered = xyz.astype(np.float64, copy=False) - np.mean(xyz, axis=0)
    covariance = centered.T @ centered / max(len(centered) - 1, 1)
    values, vectors = np.linalg.eigh(covariance)
    largest, second = float(values[-1]), float(values[-2])
    if largest <= 1e-12 or not np.isfinite(values).all():
        return np.zeros(3, np.float64), 0.0
    axis = np.asarray(vectors[:, -1], np.float64)
    norm = float(np.linalg.norm(axis))
    if norm <= 1e-12:
        return np.zeros(3, np.float64), 0.0
    return axis / norm, float(np.clip((largest - second) / largest, 0.0, 1.0))


def _candidate_pairs(
    summaries: Mapping[int, _Summary], cfg: VegetationAssociationConfig
) -> list[tuple[int, int]]:
    items = [summaries[c] for c in sorted(summaries)]
    if len(items) < 2:
        return []
    centers = np.asarray([s.center for s in items])
    tree = cKDTree(centers)
    half = np.asarray([np.linalg.norm((s.upper - s.lower) / 2) for s in items])
    largest = float(half.max(initial=0))
    directed: dict[int, list[tuple[float, int]]] = {s.cluster_id: [] for s in items}
    for i, left in enumerate(items):
        search = cfg.maximum_search_radius_m + half[i] + largest
        for j in sorted(tree.query_ball_point(left.center, search)):
            if j <= i:
                continue
            right = items[j]
            radius, diagnostic = _pair_radius(left, right, cfg)
            gap = _aabb_gap(left.lower, left.upper, right.lower, right.upper)
            if (radius is not None and gap <= radius) or (diagnostic and gap <= 0.0):
                score = gap / max(radius or cfg.minimum_search_radius_m, 1e-12)
                directed[left.cluster_id].append((score, right.cluster_id))
                directed[right.cluster_id].append((score, left.cluster_id))
    chosen: set[tuple[int, int]] = set()
    for cid, values in directed.items():
        for _, other in sorted(values, key=lambda x: (x[0], x[1]))[: cfg.per_cluster_candidate_cap]:
            chosen.add((min(cid, other), max(cid, other)))
    return sorted(chosen)


def _pair_radius(
    a: _Summary, b: _Summary, cfg: VegetationAssociationConfig
) -> tuple[float | None, bool]:
    if not a.scale_valid and not b.scale_valid:
        return None, True
    if not a.scale_valid or not b.scale_valid:
        return cfg.maximum_search_radius_m, False
    return float(
        np.clip(
            cfg.search_multiplier * math.sqrt(a.scale * b.scale),
            cfg.minimum_search_radius_m,
            cfg.maximum_search_radius_m,
        )
    ), False


def _evaluate_pair(
    a: _Summary,
    b: _Summary,
    v2: Mapping[str, np.ndarray],
    v1: Mapping[str, np.ndarray],
    a2a: Mapping[str, np.ndarray],
    a2b: Mapping[str, np.ndarray],
    cfg: VegetationAssociationConfig,
) -> dict[str, Any]:
    radius, invalid_diag = _pair_radius(a, b, cfg)
    effective = radius or cfg.minimum_search_radius_m
    dist, vertical, anisotropic, selected = _boundary(a, b, effective, cfg.boundary_sample_cap)
    scale_ratio = _ratio(a.scale, b.scale) if a.scale_valid and b.scale_valid else 0.0
    density_ratio = _ratio(a.density, b.density)
    hag_gap = abs(a.hag - b.hag) if a.hag_reliable and b.hag_reliable else 0.0
    positive = np.zeros(10, np.float64)
    blocker = np.zeros(10, np.float64)
    flags = VegetationAssociationReasonFlag.INPUTS_VERIFIED
    spatial = max(0.0, 1.0 - max(dist, anisotropic) / max(effective, 1e-12))
    positive[0] = spatial
    positive[1] = _compat(density_ratio, cfg.density_ratio_width)
    directional, direction_available = _directional_continuation(a, b, cfg)
    positive[2] = directional
    positive[3] = _compat(scale_ratio, cfg.scale_ratio_width) if scale_ratio else 0.0
    positive[4] = 1.0 - min(1.0, abs(a.morphology - b.morphology) + abs(a.planar - b.planar))
    positive[5] = (
        max(0.0, 1.0 - hag_gap / max(effective * 2.0, 1e-12))
        if a.hag_reliable and b.hag_reliable
        else 0.0
    )
    positive[6] = max(0.0, 1.0 - abs(a.profile - b.profile) / max(cfg.profile_width, 1e-12))
    low_counts, low_value, low_ambiguous = _low_support(a, b, v2, v1, effective, cfg)
    positive[7] = low_value
    positive[8] = 1.0 if a.parent >= 0 and a.parent == b.parent else 0.0
    positive[9] = (
        0.0 if a.unresolved or b.unresolved else (1.0 if a.parent >= 0 and b.parent >= 0 else 0.5)
    )
    blocker[0] = min(1.0, max(dist, anisotropic) / max(effective, 1e-12))
    blocker[1] = min(1.0, vertical / max(effective * 2.0, 1e-12))
    blocker[2] = min(1.0, spatial * abs(a.profile - b.profile) / max(cfg.profile_width, 1e-12))
    blocker[3] = max(a.planar, b.planar)
    blocker[4] = max(a.slender, b.slender)
    blocker[5] = max(a.compact, b.compact)
    blocker[6] = max(a.mixed, b.mixed)
    blocker[7] = (
        min(1.0, hag_gap / max(effective * 1.2, 1e-12))
        if a.hag_reliable and b.hag_reliable
        else 0.0
    )
    blocker[8] = max(1.0 - positive[1], 1.0 - positive[3] if scale_ratio else 0.0)
    # No first-version reliable contradiction formula exists.  Topology/lineage
    # uncertainty is recorded below, but is never represented as a blocker.
    blocker[9] = 0.0
    prov = (
        VegetationAssociationProvenanceFlag.V2_VERIFIED
        | VegetationAssociationProvenanceFlag.V1_VERIFIED
        | VegetationAssociationProvenanceFlag.A2A_VERIFIED
        | VegetationAssociationProvenanceFlag.A2B_VERIFIED
    )
    if a.scale_valid:
        prov |= VegetationAssociationProvenanceFlag.SCALE_A_VALID
    if b.scale_valid:
        prov |= VegetationAssociationProvenanceFlag.SCALE_B_VALID
    if a.hag_reliable:
        prov |= VegetationAssociationProvenanceFlag.HAG_RELIABLE_A
    if b.hag_reliable:
        prov |= VegetationAssociationProvenanceFlag.HAG_RELIABLE_B
    if direction_available[0]:
        prov |= VegetationAssociationProvenanceFlag.DIRECTION_A_STABLE
    if direction_available[1]:
        prov |= VegetationAssociationProvenanceFlag.DIRECTION_B_STABLE
    if not all(direction_available):
        flags |= VegetationAssociationReasonFlag.DIRECTIONAL_EVIDENCE_UNAVAILABLE
    if low_ambiguous:
        flags |= VegetationAssociationReasonFlag.LOW_MULTI_CLUSTER_AMBIGUITY
        prov |= VegetationAssociationProvenanceFlag.LOW_AMBIGUOUS
    if not a.scale_valid or not b.scale_valid:
        flags |= (
            VegetationAssociationReasonFlag.ONE_INVALID_SCALE
            if a.scale_valid != b.scale_valid
            else VegetationAssociationReasonFlag.BOTH_INVALID_SCALES
        )
    if a.clamped_high_fraction > 0.5 or b.clamped_high_fraction > 0.5:
        flags |= VegetationAssociationReasonFlag.CLAMPED_HIGH_HEAVY
    if scale_ratio and scale_ratio > cfg.scale_ratio_width:
        flags |= VegetationAssociationReasonFlag.EXTREME_SCALE_RATIO
    if abs(dist - anisotropic) > effective * 0.25:
        flags |= VegetationAssociationReasonFlag.ANISOTROPIC_ISOTROPIC_CONFLICT
    lineage_available = not (a.unresolved or b.unresolved)
    if not lineage_available:
        flags |= VegetationAssociationReasonFlag.LINEAGE_UNAVAILABLE
        prov |= VegetationAssociationProvenanceFlag.SPLIT_LINEAGE_UNAVAILABLE
    # This is availability of mandatory current decision inputs, not a claim that
    # optional HAG or historical split provenance exists.  In particular,
    # unresolved V2 topology is uncertainty, not an explicit contradiction.
    complete = float(
        a.scale_valid
        and b.scale_valid
        and math.isfinite(dist)
        and math.isfinite(vertical)
        and math.isfinite(anisotropic)
        and math.isfinite(a.density)
        and math.isfinite(b.density)
        and all(math.isfinite(value) for value in blocker[:9])
        and not invalid_diag
    )
    positive_complete = int(np.count_nonzero(positive >= 0.5))
    confidence = float(np.mean(positive) * (1.0 - 0.5 * float(np.max(blocker))))
    decisive = bool(np.max(blocker) >= cfg.decisive_blocker_threshold)
    if decisive:
        status = VegetationAssociationStatus.BLOCKED_BY_CONFLICT
        flags |= VegetationAssociationReasonFlag.DECISIVE_BLOCKER
    elif not complete:
        status = VegetationAssociationStatus.ABSTAINED
        flags |= VegetationAssociationReasonFlag.DECISION_INPUT_UNAVAILABLE
        if invalid_diag:
            flags |= VegetationAssociationReasonFlag.INVALID_SCALE_AABB_DIAGNOSTIC
    elif (
        confidence >= cfg.strong_confidence_threshold
        and positive_complete >= cfg.required_independent_positive_channels
        and max(positive[0], positive[2], positive[4], positive[6]) >= 0.55
        and lineage_available
        and positive[7] < 1.0
    ):
        status = VegetationAssociationStatus.STRONG_REVIEW_ASSOCIATION
        flags |= VegetationAssociationReasonFlag.STRONG_REVIEW
    elif (
        confidence >= cfg.possible_confidence_threshold
        and positive[0] >= cfg.spatial_possible_threshold
    ):
        status = VegetationAssociationStatus.POSSIBLE_REVIEW_ASSOCIATION
        flags |= VegetationAssociationReasonFlag.POSSIBLE_REVIEW
    else:
        status = VegetationAssociationStatus.ABSTAINED
        flags |= VegetationAssociationReasonFlag.DECISION_INPUT_UNAVAILABLE
    return {
        "ids": (a.cluster_id, b.cluster_id),
        "status": int(status),
        "confidence": confidence,
        "positive": positive,
        "blocking": blocker,
        "flags": int(flags),
        "distance": (
            dist,
            vertical,
            anisotropic,
            effective,
            _aabb_gap(a.lower, a.upper, b.lower, b.upper),
        ),
        "scale": (a.scale, b.scale, scale_ratio, density_ratio),
        "hag": (a.hag, b.hag, hag_gap, float(a.hag_reliable and b.hag_reliable)),
        "low": low_counts,
        "provenance": int(prov),
        "decision_input_completeness": complete,
    }


def _directional_continuation(
    a: _Summary, b: _Summary, cfg: VegetationAssociationConfig
) -> tuple[float, tuple[bool, bool]]:
    """Score stable PCA-axis continuation across the cluster-centre boundary.

    Axes are sign invariant.  The score is the minimum of A-to-boundary,
    B-to-boundary, and mutual-axis alignment, multiplied by the weaker PCA
    stability. `directional_width` is a dimensionless tolerance width: a raw
    compatibility at least `1 - width` starts contributing and transitions
    linearly to one at perfect compatibility.
    """
    available = (a.direction_reliability >= 0.20, b.direction_reliability >= 0.20)
    if not all(available):
        return 0.0, available
    boundary = b.center - a.center
    norm = float(np.linalg.norm(boundary))
    if not math.isfinite(norm) or norm <= 1e-12:
        return 0.0, available
    unit = boundary / norm
    align_a = abs(float(np.dot(a.direction, unit)))
    align_b = abs(float(np.dot(b.direction, unit)))
    mutual = abs(float(np.dot(a.direction, b.direction)))
    raw = float(np.clip(min(align_a, align_b, mutual), 0.0, 1.0))
    width = max(cfg.directional_width, 1e-12)
    transition = float(np.clip(1.0 - (1.0 - raw) / width, 0.0, 1.0))
    return transition * min(a.direction_reliability, b.direction_reliability), available


def _boundary(
    a: _Summary, b: _Summary, radius: float, cap: int
) -> tuple[float, float, float, np.ndarray]:
    tree = cKDTree(b.xyz)
    distances, indices = tree.query(a.xyz, k=1, distance_upper_bound=max(radius * 2, 1e-9))
    valid = np.flatnonzero(np.isfinite(distances) & (indices < len(b.xyz)))
    if not len(valid):
        return (
            float(_aabb_gap(a.lower, a.upper, b.lower, b.upper)),
            abs(float(a.center[2] - b.center[2])),
            float(_aabb_gap(a.lower, a.upper, b.lower, b.upper)),
            np.empty(0, np.int64),
        )
    ordering = sorted(
        valid,
        key=lambda i: (
            float(distances[i] / max(radius, 1e-12)),
            int(a.original[i]),
            int(b.original[int(indices[i])]),
        ),
    )[:cap]
    sel = np.asarray(ordering, np.int64)
    delta = a.xyz[sel] - b.xyz[indices[sel]]
    return (
        float(np.median(distances[sel])),
        float(np.median(np.abs(delta[:, 2]))),
        float(np.median(np.linalg.norm(delta, axis=1))),
        sel,
    )


def _low_support(
    a: _Summary,
    b: _Summary,
    v2: Mapping[str, np.ndarray],
    v1: Mapping[str, np.ndarray],
    radius: float,
    cfg: VegetationAssociationConfig,
) -> tuple[tuple[int, int, int, int, int, int], float, bool]:
    low = np.flatnonzero(v1["vegetation_level"] == int(VegetationCandidateLevel.LOW))
    if not len(low):
        return (0, 0, 0, 0, 0, 0), 0.0, False
    xyz = v2["xyz_local"][low]
    da = cKDTree(a.xyz).query(xyz, k=1)[0] <= radius
    db = cKDTree(b.xyz).query(xyz, k=1)[0] <= radius
    near_a, near_b, both = int(da.sum()), int(db.sum()), int((da & db).sum())
    all_centers = np.asarray([a.center, b.center])
    memberships = cKDTree(all_centers).query_ball_point(xyz, radius)
    more = int(sum(len(x) > 2 for x in memberships))
    ambiguity = more > 0
    unassigned = v2["cluster_id_by_point"][low] < 0
    no_support = int(np.count_nonzero(unassigned & ~da & ~db))
    unassigned_ambiguity = int(np.count_nonzero(unassigned & (da ^ db)))
    value = 0.0 if ambiguity else min(1.0, both / max(cfg.low_support_requirement, 1))
    return (near_a, near_b, both, more, no_support, unassigned_ambiguity), value, ambiguity


def _arrays(rows: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    rows = sorted(rows, key=lambda r: r["ids"])
    count = len(rows)

    def array(key: str, dtype: Any, shape: tuple[int, ...] = ()) -> np.ndarray:
        return np.asarray([r[key] for r in rows], dtype=dtype).reshape((count, *shape))

    data = {
        "pair_cluster_ids": array("ids", np.int32, (2,)),
        "pair_status": array("status", np.uint8),
        "pair_confidence": array("confidence", np.float32),
        "pair_positive_evidence": array("positive", np.float32, (10,)),
        "pair_blocking_evidence": array("blocking", np.float32, (10,)),
        "pair_reason_flags": array("flags", np.uint32),
        "pair_distance_summary_m": array("distance", np.float32, (5,)),
        "pair_scale_density_summary": array("scale", np.float32, (4,)),
        "pair_hag_summary": array("hag", np.float32, (4,)),
        "pair_low_support_counts": array("low", np.int32, (6,)),
        "pair_provenance_flags": array("provenance", np.uint32),
        "pair_decision_input_completeness": array("decision_input_completeness", np.float32),
    }
    validate_association_arrays(data)
    return data


def _validate_inputs(
    v2r: Any,
    v2: Mapping[str, np.ndarray],
    v1r: Any,
    v1: Mapping[str, np.ndarray],
    a2a: Mapping[str, np.ndarray],
    a2am: Mapping[str, Any],
    a2b: Mapping[str, np.ndarray],
    a2bm: Mapping[str, Any],
    hashes: Mapping[str, str],
    cfg: VegetationAssociationConfig,
) -> None:
    if (
        v2r.source_id != cfg.source_id
        or v1r.source_id != cfg.source_id
        or str(a2am.get("source_id")) != cfg.source_id
        or str(a2bm.get("source_id")) != cfg.source_id
    ):
        raise _error("V2A_SOURCE", "source IDs are inconsistent")
    if (
        len(
            {
                v2r.manifest_source_id,
                v1r.manifest_source_id,
                str(a2am.get("manifest_source_id")),
                str(a2bm.get("manifest_source_id")),
            }
        )
        != 1
    ):
        raise _error("V2A_MANIFEST", "manifest IDs are inconsistent")
    if v2r.output_npz_sha256 != hashes["v2_npz"]:
        raise _error("V2A_PROVENANCE", "V2 JSON does not match supplied V2 NPZ")
    for name in ("v1_npz", "v1_json", "a2a_npz", "a2a_json", "a2b_npz", "a2b_json"):
        if v2r.input_provenance.get(name, {}).get("sha256") != hashes[name]:
            raise _error(
                "V2A_PROVENANCE", "V2 provenance does not match supplied sibling", input=name
            )
    for name in ("a2a_npz", "a2a_json", "a2b_npz", "a2b_json"):
        if v1r.input_provenance.get(name, {}).get("sha256") != hashes[name]:
            raise _error(
                "V2A_PROVENANCE", "V1 provenance does not match supplied sibling", input=name
            )
    if (
        a2am.get("output_npz_sha256") != hashes["a2a_npz"]
        or a2bm.get("output_npz_sha256") != hashes["a2b_npz"]
    ):
        raise _error("V2A_HASH", "A2 JSON does not match supplied NPZ")
    if not str(a2am.get("algorithm_version", "")).startswith("a2a-local-pca-") or not str(
        a2bm.get("algorithm_version", "")
    ).startswith("a2b-grid-quantile-"):
        raise _error("V2A_VERSION", "unsupported A2 schema or algorithm")
    m = len(v1["original_index"])
    for data, label in ((v2, "V2"), (a2a, "A2A"), (a2b, "A2B")):
        if (
            len(data.get("original_index", ())) != m
            or not np.array_equal(v1["original_index"], data["original_index"])
            or not np.array_equal(v1["sample_index"], data["sample_index"])
        ):
            raise _error("V2A_ALIGNMENT", f"{label} point indexes differ")
    if (
        not np.array_equal(v1["xyz_local"], v2["xyz_local"])
        or not np.array_equal(v1["xyz_local"], a2a["xyz_local"])
        or not np.array_equal(
            np.asarray(v1["origin_xyz"], np.float64), np.asarray(v2["origin_xyz"], np.float64)
        )
        or not np.array_equal(
            np.asarray(v1["origin_xyz"], np.float64),
            np.asarray(a2am.get("origin_xyz"), np.float64),
        )
        or not np.array_equal(
            np.asarray(v1["origin_xyz"], np.float64),
            np.asarray(a2bm.get("origin_xyz"), np.float64),
        )
    ):
        raise _error("V2A_ALIGNMENT", "coordinates or origin differ")
    for name in ("linearity", "planarity", "scattering", "verticality", "local_density"):
        if (
            name not in a2a
            or a2a[name].dtype != np.float32
            or len(a2a[name]) != m
            or not np.isfinite(a2a[name]).all()
        ):
            raise _error("V2A_A2A", "invalid A2A feature", array=name)
    for name in ("height_agl_m", "raw_height_agl_m", "ground_distance_m"):
        if (
            name not in a2b
            or a2b[name].dtype != np.float32
            or len(a2b[name]) != m
            or not np.isfinite(a2b[name]).all()
        ):
            raise _error("V2A_A2B", "invalid A2B feature", array=name)


def _publish(
    arrays: Mapping[str, np.ndarray],
    npz_out: Path,
    json_out: Path,
    cfg: VegetationAssociationConfig,
    v2r: Any,
    hashes: Mapping[str, str],
    stats: VegetationAssociationStatistics,
) -> VegetationAssociationResult:
    temp_npz, temp_json = _temporary(npz_out), _temporary(json_out)
    published = False
    try:
        np.savez_compressed(temp_npz, **arrays)
        validate_association_arrays(_load_npz(temp_npz))
        provenance = {
            name: {"logical_path": name, "sha256": digest} for name, digest in hashes.items()
        }
        provenance["v2_recorded_upstream_hashes"] = {
            name: data["sha256"]
            for name, data in v2r.input_provenance.items()
            if isinstance(data, Mapping) and "sha256" in data
        }
        result = VegetationAssociationResult(
            SCHEMA_VERSION,
            ALGORITHM_VERSION,
            v2r.source_id,
            v2r.manifest_source_id,
            tuple(map(float, v2r.origin_xyz)),
            npz_out.name,
            _sha(temp_npz),
            temp_npz.stat().st_size,
            cfg,
            provenance,
            {
                name: {"shape": list(value.shape), "dtype": str(value.dtype)}
                for name, value in arrays.items()
            },
            stats,
            (),
            (
                "Pair-only review evidence; it never merges clusters or creates "
                "vegetation identity.",
                "C1/R1/P1 integration requires separate future isolation and invariance tests.",
            ),
            datetime.now(timezone.utc).isoformat(),
        )
        payload = result.to_dict()
        payload.update(
            {
                "enum_mappings": {item.name: int(item) for item in VegetationAssociationStatus},
                "positive_evidence_columns": list(POSITIVE_EVIDENCE_NAMES),
                "blocking_evidence_columns": list(BLOCKING_EVIDENCE_NAMES),
                "reason_flag_mappings": {
                    item.name: int(item) for item in VegetationAssociationReasonFlag
                },
                "provenance_flag_mappings": {
                    item.name: int(item) for item in VegetationAssociationProvenanceFlag
                },
            }
        )
        temp_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        # Parse through the contract after removing descriptive JSON extensions.
        raw = _load_json(temp_json)
        VegetationAssociationResult.from_dict(
            {k: raw[k] for k in VegetationAssociationResult.__dataclass_fields__}
        )
        os.replace(temp_npz, npz_out)
        published = True
        os.replace(temp_json, json_out)
        return result
    except Exception:
        temp_npz.unlink(missing_ok=True)
        temp_json.unlink(missing_ok=True)
        if published:
            npz_out.unlink(missing_ok=True)
        raise


def _aabb_gap(lo1: np.ndarray, hi1: np.ndarray, lo2: np.ndarray, hi2: np.ndarray) -> float:
    return float(np.linalg.norm(np.maximum(0.0, np.maximum(lo1 - hi2, lo2 - hi1))))


def _ratio(a: float, b: float) -> float:
    return (
        max(a, b) / max(min(a, b), 1e-12)
        if a > 0 and b > 0 and math.isfinite(a) and math.isfinite(b)
        else 0.0
    )


def _compat(ratio: float, width: float) -> float:
    return (
        max(0.0, 1.0 - math.log(max(ratio, 1.0)) / math.log(max(width, 1.000001))) if ratio else 0.0
    )


def _load_pair(npz: Path, meta: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    return _load_npz(npz), _load_json(meta)


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as f:
            values = {name: f[name] for name in f.files}
    except (OSError, ValueError) as exc:
        raise _error("V2A_NPZ", "unable to load NPZ", cause=str(exc), path=str(path)) from exc
    if any(value.dtype == object for value in values.values()):
        raise _error("V2A_NPZ_OBJECT", "object arrays are forbidden")
    return values


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise _error("V2A_JSON", "unable to load JSON", cause=str(exc), path=str(path)) from exc
    if not isinstance(value, dict):
        raise _error("V2A_JSON", "JSON must be an object")
    return value


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _temporary(path: Path) -> Path:
    return path.with_name(f".{path.stem}.{uuid.uuid4().hex}.tmp{path.suffix}")


def _validate_output_dir(output: Path, allowed: Path) -> None:
    try:
        output.resolve().relative_to(allowed.resolve())
    except ValueError as exc:
        raise _error(
            "V2A_OUTPUT_FORBIDDEN", "output must remain in V2A output directory", output=str(output)
        ) from exc


def _validate_outputs(paths: tuple[Path, Path], overwrite: bool) -> None:
    existing = [str(p) for p in paths if p.exists()]
    if existing and not overwrite:
        raise _error("V2A_OUTPUT_EXISTS", "output exists and overwrite is disabled", paths=existing)


def _current_process() -> Any | None:
    try:
        import psutil

        return psutil.Process()
    except ImportError:
        return None


def _rss(process: Any | None) -> int | None:
    try:
        return int(process.memory_info().rss) if process is not None else None
    except Exception:
        return None


def _check_cancelled(token: CancellationToken | None, stage: str) -> None:
    if token is not None and token.is_cancelled():
        raise _error("V2A_CANCELLED", "V2A operation was cancelled", stage=stage)


def _report(
    reporter: ProgressReporter | None,
    task: str,
    stage: str,
    processed: int,
    total: int,
    status: str = "running",
) -> None:
    if reporter is not None:
        reporter.update(
            PipelineProgress(
                task_id=task,
                stage=stage,
                status=status,
                processed_points=processed,
                total_points=total,
                percent=100 * processed / max(total, 1),
            )
        )


def _error(code: str, message: str, **details: Any) -> VegetationAssociationError:
    cause = details.pop("cause", None)
    return VegetationAssociationError(
        UnifiedError(
            code=code,
            message=message,
            severity="error",
            module="vegetation.associations",
            recoverable=False,
            details=details,
            cause=cause,
        )
    )
