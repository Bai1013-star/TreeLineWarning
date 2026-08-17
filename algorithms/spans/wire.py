"""Per-span FAST mature wire pipeline.

Each confirmed span is processed locally.  Source-native wire-region points are
re-voxelized to fine (0.10 m) representatives, converted into the span-local
(s, u, z) frame and filtered with the span geometry as a strong prior: a true
conductor must extend significantly along ``s``, remain narrow in ``u``, vary
smoothly in ``z``, coexist with neighbouring parallel conductors and connect
plausibly between the two endpoint poles.

Stages: local PCA geometry evidence -> DBSCAN continuity -> strand separation
in (u, z) -> robust low-order curve reconstruction -> W1-compatible
distance-ready output.

Output is explicitly marked ``FAST_MATURE_SPAN_WIRE`` + ``NON_CERTIFIED``; it
never inherits Formal W1 authority.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from core.config.fast_baseline import FastBaselineConfig

from algorithms.poles.evidence import PoleEvidenceError
from algorithms.poles.topology import SpanRecord

SPAN_WIRE_SCHEMA = "fast-baseline-span-wire-v1"
FAST_MATURE_SPAN_WIRE = "FAST_MATURE_SPAN_WIRE"

# W1-compatible enum values (core/contracts/wire_models.py)
_CONFIRMED_MODEL = 0
_SMOOTHED_POLYLINE = 1


@dataclass(frozen=True, slots=True)
class SpanWireStrand:
    """One reconstructed conductor strand inside a span."""

    wire_id: int
    wire_name: str
    representative_count: int
    raw_point_count: int
    s_span_m: float
    coverage_ratio: float
    median_u_m: float
    median_hag_m: float
    curve_sample_count: int
    curve_xyz_local_m: np.ndarray  # float32 (M, 3) scene-local
    residual_stats_m: tuple[float, float]  # (median, p95)
    approach_start: bool
    approach_end: bool
    status: str = "CONFIRMED"
    model_type: str = "ROBUST_QUADRATIC"
    supported_s_domain_m: tuple[float, float] = (0.0, 0.0)
    supported_slice_count: int = 0
    supported_slice_ratio: float = 0.0
    maximum_gap_m: float = 0.0
    inlier_ratio: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "wire_id": self.wire_id,
            "wire_name": self.wire_name,
            "representative_count": self.representative_count,
            "raw_point_count": self.raw_point_count,
            "s_span_m": self.s_span_m,
            "coverage_ratio": self.coverage_ratio,
            "median_u_m": self.median_u_m,
            "median_hag_m": self.median_hag_m,
            "curve_sample_count": self.curve_sample_count,
            "residual_stats_m": list(self.residual_stats_m),
            "approach_start": self.approach_start,
            "approach_end": self.approach_end,
            "status": self.status,
            "model_type": self.model_type,
            "supported_s_domain_m": list(self.supported_s_domain_m),
            "supported_slice_count": self.supported_slice_count,
            "supported_slice_ratio": self.supported_slice_ratio,
            "maximum_gap_m": self.maximum_gap_m,
            "inlier_ratio": self.inlier_ratio,
        }


@dataclass(frozen=True, slots=True)
class SpanWireResult:
    """Per-span conductor reconstruction result with W1-compatible arrays.

    ``confirmed_member_source_index`` is the exact, sorted-unique FULL_RESOLUTION
    source-row identity of every raw point belonging to a FINAL CONFIRMED
    conductor strand (the strand's fine-representative voxels).  This is the
    only source-index set a tree stage may subtract from vegetation — broad wire
    candidate/search points are never excluded from vegetation by role index.
    """

    strands: tuple[SpanWireStrand, ...]
    w1_arrays: dict[str, np.ndarray]
    candidate_mask: np.ndarray  # bool per fine rep
    cluster_id_by_rep: np.ndarray  # int32 per fine rep
    diagnostics: dict[str, Any]
    confirmed_member_source_index: np.ndarray = field(  # sorted unique
        default_factory=lambda: np.empty(0, dtype=np.int64)
    )

    @property
    def strand_count(self) -> int:
        return len(self.strands)

    @property
    def distance_ready_segments(self) -> int:
        return int(self.w1_arrays.get("segment_distance_ready", np.asarray([False])).sum())


# ---------------------------------------------------------------------------
# Local PCA geometry features (deterministic, scipy-only) live in the shared
# coarse-evidence module so pole and wire detection use one implementation.
# ---------------------------------------------------------------------------


def _local_pca_features(
    xyz_local_m: np.ndarray, *, neighbor_count: int, chunk_size: int = 100_000
) -> dict[str, np.ndarray]:
    from algorithms.poles.evidence import local_pca_features as _features

    return _features(xyz_local_m, neighbor_count=neighbor_count, chunk_size=chunk_size)


# ---------------------------------------------------------------------------
# Deterministic DBSCAN (scipy-only)
# ---------------------------------------------------------------------------


def _confirmed_strand_member_source_indices(
    region: Any,
    strand_groups: list[np.ndarray],
    voxel_size_m: float,
    *,
    authoritative_source_point_count: int | None = None,
    chunk_size: int = 1_000_000,
) -> np.ndarray:
    """Exact source-row identity of every raw point in confirmed strand voxels.

    Each confirmed strand is a set of fine-representative (voxel) row indices.
    The voxel assignment is re-derived from the RAW wire region with the exact
    same keys as ``build_fine_representatives``, so every raw point is mapped to
    its representative and the strand's raw member source indices are recovered
    exactly (never approximated from Viewer/nearest-point geometry).
    """
    if not strand_groups:
        return np.empty(0, dtype=np.int64)
    from algorithms.vegetation.coarse_to_fine import _linear_keys

    count = region.count
    bounds_min = np.asarray(region.bounds_min_xyz, dtype=np.float64)
    cell_shape = (
        np.floor((np.asarray(region.bounds_max_xyz) - bounds_min) / voxel_size_m).astype(
            np.int64
        )
        + 2
    )
    # Copy the small one-dimensional identity vector before validation so a
    # fail-closed error does not leave an open memmap handle on Windows.
    source_index = np.asarray(region.original_index(), dtype=np.int64).copy()
    if not np.issubdtype(source_index.dtype, np.integer) or np.any(source_index < 0):
        raise PoleEvidenceError("SPAN_WIRE_SOURCE_INDEX_INVALID")
    if (
        authoritative_source_point_count is not None
        and np.any(source_index >= authoritative_source_point_count)
    ):
        raise PoleEvidenceError("SPAN_WIRE_SOURCE_INDEX_OUT_OF_RANGE")
    if len(np.unique(source_index)) != len(source_index):
        raise PoleEvidenceError("SPAN_WIRE_SOURCE_INDEX_DUPLICATE_OWNERSHIP")
    raw_xyz = region.xyz()
    strand_rep_rows = np.unique(
        np.concatenate([np.asarray(rows, dtype=np.int64) for rows in strand_groups])
    )
    if sum(len(np.asarray(rows, dtype=np.int64)) for rows in strand_groups) != len(strand_rep_rows):
        raise PoleEvidenceError("SPAN_WIRE_REPRESENTATIVE_DUPLICATE_OWNERSHIP")
    # Global voxel keys — the exact same representative indexing that
    # ``build_fine_representatives`` produced — so each raw point maps to its
    # representative and the strand's exact raw member source indices are
    # recovered without any temp file.
    raw_keys = np.empty(count, dtype=np.int64)
    for start in range(0, count, chunk_size):
        stop = min(count, start + chunk_size)
        raw_keys[start:stop] = _linear_keys(
            raw_xyz[start:stop].astype("<f8", copy=False),
            bounds_min,
            voxel_size_m,
            cell_shape,
        )
    unique_keys = np.unique(raw_keys)
    rep_index = np.searchsorted(unique_keys, raw_keys)
    member = np.isin(rep_index, strand_rep_rows)
    if not np.any(member):
        return np.empty(0, dtype=np.int64)
    return np.unique(np.asarray(source_index[member], dtype=np.int64))


def _track_member_source_indices(
    region: Any,
    track_groups: list[np.ndarray],
    voxel_size_m: float,
    representative_count: int,
    *,
    authoritative_source_point_count: int | None = None,
    chunk_size: int = 1_000_000,
) -> tuple[list[np.ndarray], list[dict[str, int]]]:
    """Recover per-track raw source identities from actual V2 representative rows.

    This is an artifact projection only.  It reuses the fine-voxel ownership
    identity already established by the running pipeline and does not use a
    fitted curve, a cluster label, or any Viewer-derived geometry.
    """
    from algorithms.vegetation.coarse_to_fine import _linear_keys

    normalized = [np.unique(np.asarray(rows, dtype=np.int64)) for rows in track_groups]
    for rows in normalized:
        if np.any(rows < 0) or np.any(rows >= representative_count):
            raise PoleEvidenceError("SPAN_WIRE_TRACK_REPRESENTATIVE_OUT_OF_RANGE")
    owner = np.full(representative_count, -1, dtype=np.int64)
    overlap_pairs: dict[tuple[int, int], int] = {}
    for track_id, rows in enumerate(normalized):
        previous = owner[rows]
        for prior_track in np.unique(previous[previous >= 0]):
            pair = (int(prior_track), track_id)
            overlap_pairs[pair] = int(np.count_nonzero(previous == prior_track))
        owner[rows[previous < 0]] = track_id
    overlaps = [
        {"first_track_id": first, "second_track_id": second, "representative_count": count}
        for (first, second), count in sorted(overlap_pairs.items())
    ]
    count = region.count
    source_index = np.asarray(region.original_index(), dtype=np.int64).copy()
    if not np.issubdtype(source_index.dtype, np.integer) or np.any(source_index < 0):
        raise PoleEvidenceError("SPAN_WIRE_SOURCE_INDEX_INVALID")
    if (
        authoritative_source_point_count is not None
        and np.any(source_index >= authoritative_source_point_count)
    ):
        raise PoleEvidenceError("SPAN_WIRE_SOURCE_INDEX_OUT_OF_RANGE")
    if len(np.unique(source_index)) != len(source_index):
        raise PoleEvidenceError("SPAN_WIRE_SOURCE_INDEX_DUPLICATE_OWNERSHIP")
    bounds_min = np.asarray(region.bounds_min_xyz, dtype=np.float64)
    cell_shape = (
        np.floor((np.asarray(region.bounds_max_xyz) - bounds_min) / voxel_size_m).astype(
            np.int64
        )
        + 2
    )
    raw_xyz = region.xyz()
    raw_keys = np.empty(count, dtype=np.int64)
    for start in range(0, count, chunk_size):
        stop = min(count, start + chunk_size)
        raw_keys[start:stop] = _linear_keys(
            raw_xyz[start:stop].astype("<f8", copy=False),
            bounds_min,
            voxel_size_m,
            cell_shape,
        )
    rep_index = np.searchsorted(np.unique(raw_keys), raw_keys)
    if overlaps:
        # A non-disjoint candidate is anomalous but remains inspectable.  Do
        # not silently assign its shared raw source identities to the first
        # owner; materialize each actual group for the report instead.
        result = [
            np.sort(source_index[np.isin(rep_index, rows)]).astype(np.int64, copy=False)
            for rows in normalized
        ]
    else:
        raw_owner = owner[rep_index]
        result = [
            np.sort(source_index[raw_owner == track_id]).astype(np.int64, copy=False)
            for track_id in range(len(normalized))
        ]
    for values in result:
        if len(values) != len(np.unique(values)):
            raise PoleEvidenceError("SPAN_WIRE_TRACK_SOURCE_DUPLICATE_OWNERSHIP")
    return result, overlaps


def _dbscan(
    xyz: np.ndarray, *, eps: float, min_samples: int
) -> np.ndarray:
    """Return integer cluster labels (-1 for noise) using core-based DBSCAN."""

    from scipy.spatial import cKDTree

    if len(xyz) == 0:
        return np.empty(0, dtype=np.int32)
    tree = cKDTree(xyz)
    labels = np.full(len(xyz), -1, dtype=np.int32)
    cluster_id = 0
    core = np.zeros(len(xyz), dtype=bool)
    neighbors = tree.query_ball_point(xyz, eps)
    for position, members in enumerate(neighbors):
        core[position] = len(members) >= min_samples
    for position in range(len(xyz)):
        if labels[position] != -1 or not core[position]:
            continue
        labels[position] = cluster_id
        stack = [position]
        while stack:
            current = stack.pop()
            for member in neighbors[current]:
                if labels[member] == -1:
                    labels[member] = cluster_id
                    if core[member]:
                        stack.append(member)
        cluster_id += 1
    return labels


# ---------------------------------------------------------------------------
# Strand separation in (u, z)
# ---------------------------------------------------------------------------


def _split_strand_by_u_z(
    suz: np.ndarray, labels: np.ndarray, cluster_id: int, config: FastBaselineConfig
) -> list[np.ndarray]:
    """Split a single DBSCAN cluster into strands by (u, z) modes.

    Returns a list of row-index arrays, one per separated strand.  Parallel
    conductors that DBSCAN merged are separated when their (u, z) centres are
    farther apart than the strand separation scale.
    """

    from scipy.spatial import cKDTree

    rows = np.flatnonzero(labels == cluster_id)
    if len(rows) < 2:
        return [rows]
    points = suz[rows]
    scale_u = max(config.wire_strand_u_separation_m, 0.2)
    scale_z = max(config.wire_strand_z_separation_m, 0.2)
    scaled_uz = np.column_stack((points[:, 1] / scale_u, points[:, 2] / scale_z))
    tree = cKDTree(scaled_uz)
    neighbors = tree.query_ball_point(scaled_uz, r=1.0)
    remaining = set(range(len(rows)))
    groups: list[np.ndarray] = []
    while remaining:
        seed = min(remaining)
        stack = [seed]
        component: list[int] = []
        while stack:
            current = stack.pop()
            if current not in remaining:
                continue
            remaining.discard(current)
            component.append(current)
            for member in neighbors[current]:
                if member in remaining:
                    stack.append(member)
        if len(component) >= 2:
            groups.append(rows[np.asarray(sorted(component), dtype=np.int64)])
    if not groups:
        return [rows]
    return groups


# ---------------------------------------------------------------------------
# Robust low-order curve reconstruction
# ---------------------------------------------------------------------------


def _robust_quadratic_fit(
    s_values: np.ndarray, axis_values: np.ndarray, residual_tolerance_m: float
) -> tuple[np.ndarray, np.ndarray, int]:
    """Iteratively fit a quadratic ``axis(s)`` with residual rejection.

    Returns ``(coefficients, residuals, kept_count)``.  The fit is a low-order
    quadratic (sag model) whose residuals are re-evaluated after each
    rejection pass; outliers beyond *residual_tolerance_m* are removed and the
    fit is repeated (bounded iterations) so tree branches, lamp arms and roof
    edges never pull the curve away from the conductor.
    """

    s = np.asarray(s_values, dtype=np.float64)
    values = np.asarray(axis_values, dtype=np.float64)
    mask = np.ones(len(s), dtype=bool)
    design_base = np.column_stack((np.ones(len(s)), s, s * s))
    coefficients: np.ndarray = np.zeros(3, dtype=np.float64)
    residuals = np.full(len(s), np.inf, dtype=np.float64)
    for _iteration in range(6):
        rows = np.flatnonzero(mask)
        if len(rows) < 3:
            break
        design = design_base[rows]
        try:
            coefficients, *_ = np.linalg.lstsq(design, values[rows], rcond=None)
        except np.linalg.LinAlgError:  # pragma: no cover - guarded fit
            break
        fitted = design_base @ coefficients
        residuals = np.abs(fitted - values)
        new_mask = residuals <= residual_tolerance_m
        if np.array_equal(new_mask, mask):
            break
        mask = new_mask
    return coefficients, residuals, int(np.count_nonzero(mask))


# ---------------------------------------------------------------------------
# Main per-span wire pipeline
# ---------------------------------------------------------------------------


def run_span_wire_pipeline(
    span: SpanRecord,
    wire_region_subset: Any,
    config: FastBaselineConfig,
    *,
    origin_xyz: tuple[float, float, float],
    output_dir: Any = None,
    fine_voxel_size_m: float | None = None,
    authoritative_source_point_count: int | None = None,
) -> SpanWireResult:
    """Reconstruct conductor strands inside one confirmed span.

    *wire_region_subset* is a ``RawRegion`` (source-native wire raw rows) whose
    points lie inside the span halo box.  All output coordinates are in the
    scene-local frame shared with the tree pipeline (origin ``origin_xyz``).
    When *output_dir* is given the fine representatives and per-rep wire
    diagnostics are persisted for review and checkpointing.
    """

    from pathlib import Path

    from algorithms.vegetation.coarse_to_fine import build_fine_representatives

    output_root = Path(output_dir) if output_dir is not None else None
    if output_root is not None:
        output_root.mkdir(parents=True, exist_ok=True)
    voxel_size = float(fine_voxel_size_m or config.wire_fine_voxel_size_m)
    started = time.perf_counter()
    if wire_region_subset.count == 0:
        return SpanWireResult(
            strands=(),
            w1_arrays=_empty_w1_arrays(),
            candidate_mask=np.empty(0, dtype=bool),
            cluster_id_by_rep=np.empty(0, dtype=np.int32),
            confirmed_member_source_index=np.empty(0, dtype=np.int64),
            diagnostics={
                "schema": SPAN_WIRE_SCHEMA,
                "reason": "no wire region points",
                "fine_representatives_npz_path": None,
            },
        )

    if output_root is not None:
        fine_path = output_root / "fine_representatives.npz"
        arrays = build_fine_representatives(
            wire_region_subset, fine_path, voxel_size_m=voxel_size
        )
    else:
        import tempfile

        with tempfile.TemporaryDirectory(prefix="span_wire_fine_") as temporary:
            fine_path = Path(temporary) / "wire_fine.npz"
            arrays = build_fine_representatives(
                wire_region_subset, fine_path, voxel_size_m=voxel_size
            )
    output_fine = arrays["xyz_local_m"].astype(np.float64, copy=False)
    raw_count_fine = arrays["raw_count"].astype(np.int64, copy=False)
    if len(output_fine) == 0:
        return SpanWireResult(
            strands=(),
            w1_arrays=_empty_w1_arrays(),
            candidate_mask=np.empty(0, dtype=bool),
            cluster_id_by_rep=np.empty(0, dtype=np.int32),
            confirmed_member_source_index=np.empty(0, dtype=np.int64),
            diagnostics={"schema": SPAN_WIRE_SCHEMA, "reason": "empty fine wire reps"},
        )
    # Scene-local absolute-frame fine wire representatives.
    origin = np.asarray(origin_xyz, dtype=np.float64)
    fine_xyz_scene = output_fine + origin

    features = _local_pca_features(
        fine_xyz_scene, neighbor_count=config.wire_pca_neighbor_count
    )
    # A single thin conductor is locally linear, but a bundle of parallel
    # conductors is locally planar; gating only on Linearity would miss real
    # conductor bundles.  Candidate points are any low-scattering, horizontal
    # structures; DBSCAN continuity, strand separation in (u, z), longitudinal
    # coverage and pole-approach evidence are the authoritative filters.
    candidate = (
        (features["scattering"] <= config.wire_scattering_max)
        & (features["main_direction_z"] <= config.wire_verticality_max)
    )

    from algorithms.poles.topology import span_to_suz

    suz_axes = span_to_suz(span, fine_xyz_scene, origin_xyz)
    s_values = suz_axes[0]
    u_values = suz_axes[1]
    z_values = suz_axes[2]
    # HAG is not available on wire reps directly; use z within the span's
    # conductor band (above ground) — the wire region was recovered for
    # high/line-like structures, so this is an effective prior filter.
    span_length = span.length_m
    in_corridor = (
        (np.abs(u_values) <= config.span_half_width_m)
        & (s_values >= -config.span_halo_m)
        & (s_values <= span_length + config.span_halo_m)
    )
    candidate &= in_corridor

    candidate_rows = np.flatnonzero(candidate)
    candidate_suz = np.column_stack(
        (s_values[candidate_rows], u_values[candidate_rows], z_values[candidate_rows])
    )
    cluster_labels = _dbscan(
        candidate_suz,
        eps=config.wire_dbscan_eps_m,
        min_samples=config.wire_dbscan_min_samples,
    )
    cluster_id_by_rep = np.full(len(s_values), -1, dtype=np.int32)
    cluster_id_by_rep[candidate_rows] = cluster_labels

    # DBSCAN remains a high-recall diagnostic only.  V2 never promotes a
    # global cluster into a conductor: independent longitudinal slices are
    # associated with bounded `(u,z)` continuity, then robustly fitted and
    # fail-closed through explicit confirmation gates.
    from algorithms.spans.wire_v2 import (
        CONFIRMED,
        REVIEW_ONLY,
        evaluate_catenary,
        evaluate_polynomial,
        reconstruct_slice_tracks,
    )

    tracks = reconstruct_slice_tracks(
        candidate_rows,
        s_values,
        u_values,
        z_values,
        raw_count_fine,
        span_length,
        config,
    )
    strands: list[SpanWireStrand] = []
    confirmed_groups: list[np.ndarray] = []
    track_diagnostics: list[dict[str, Any]] = []
    # The exact curve that is passed into the existing W1 clearance interface,
    # keyed by the originating V2 track.  The evidence sidecar reuses this
    # object rather than re-evaluating a mathematically equivalent model.
    confirmed_model_samples: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for track_id, track in enumerate(tracks):
        rows = track.inlier_rows
        approach_start = _strand_approaches_pole(
            s_values[rows], u_values[rows], z_values[rows], span, 0.0, config
        ) if len(rows) else False
        approach_end = _strand_approaches_pole(
            s_values[rows], u_values[rows], z_values[rows], span, span_length, config
        ) if len(rows) else False
        status, reasons = track.status, list(track.reasons)
        if status == CONFIRMED and not (approach_start or approach_end):
            status, reasons = REVIEW_ONLY, reasons + ["missing_endpoint_approach_evidence"]
        if (
            status == CONFIRMED
            and not (approach_start and approach_end)
            and track.coverage_ratio < config.wire_confirmed_strong_interior_coverage_ratio
        ):
            status, reasons = REVIEW_ONLY, reasons + [
                "weak_endpoint_requires_stronger_interior_coverage"
            ]
        diagnostic = track.diagnostic_dict(
            track_id, approach_start=approach_start, approach_end=approach_end
        )
        diagnostic["status"] = status
        diagnostic["reasons"] = reasons
        track_diagnostics.append(diagnostic)
        if status != CONFIRMED:
            continue
        span_run = track.supported_s_max_m - track.supported_s_min_m
        sample_step = max(config.wire_curve_sample_spacing_m, 0.1)
        sample_count = min(
            config.wire_max_curve_samples,
            max(2, int(math.ceil(span_run / sample_step)) + 1),
        )
        # The published curve has no samples outside this raw inlier-supported
        # domain.  Large unsupported gaps have already failed the track gate.
        sample_s = np.linspace(
            track.supported_s_min_m, track.supported_s_max_m, sample_count
        )
        sample_u = evaluate_polynomial(track.model_u_coefficients, sample_s)
        sample_z = (
            evaluate_catenary(track.catenary_parameters, sample_s)
            if track.model_type == "CATENARY"
            else evaluate_polynomial(track.model_z_coefficients, sample_s)
        ) - origin[2]
        axis = np.asarray(span.direction_xy[:2], dtype=np.float64)
        perp = np.asarray((-axis[1], axis[0]), dtype=np.float64)
        start_xy = np.asarray(span.start_xyz_absolute_m[:2], dtype=np.float64) - origin[:2]
        curve_xy = (
            start_xy[None, :]
            + sample_s[:, None] * axis[None, :]
            + sample_u[:, None] * perp[None, :]
        )
        curve_xyz = np.column_stack((curve_xy, sample_z)).astype(np.float32)
        confirmed_model_samples[track_id] = (sample_s, curve_xyz)
        strands.append(
            SpanWireStrand(
                wire_id=len(strands),
                wire_name="",
                representative_count=int(len(rows)),
                raw_point_count=int(np.sum(raw_count_fine[rows], dtype=np.int64)),
                s_span_m=span_run,
                coverage_ratio=track.coverage_ratio,
                median_u_m=track.median_u_m,
                median_hag_m=(
                    float(np.median(z_values[rows])) - span.start_xyz_absolute_m[2]
                ),
                curve_sample_count=sample_count,
                curve_xyz_local_m=curve_xyz,
                residual_stats_m=(track.median_residual_m, track.p95_residual_m),
                approach_start=approach_start,
                approach_end=approach_end,
                status=status,
                model_type=track.model_type,
                supported_s_domain_m=(track.supported_s_min_m, track.supported_s_max_m),
                supported_slice_count=track.supported_slice_count,
                supported_slice_ratio=track.supported_slice_ratio,
                maximum_gap_m=track.maximum_gap_m,
                inlier_ratio=track.inlier_ratio,
            )
        )
        confirmed_groups.append(rows)

    # A read-only Owner artifact may show the fitted diagnostic model for any
    # track, including REVIEW_ONLY and REJECTED tracks.  These samples are made
    # from the already-fitted coefficients over the already-published inlier
    # domain; they never become clearance geometry unless the final status is
    # CONFIRMED.  This intentionally runs after all endpoint policy decisions.
    evidence_model_s: list[np.ndarray] = []
    evidence_model_xyz: list[np.ndarray] = []
    axis = np.asarray(span.direction_xy[:2], dtype=np.float64)
    perp = np.asarray((-axis[1], axis[0]), dtype=np.float64)
    start_xy = np.asarray(span.start_xyz_absolute_m[:2], dtype=np.float64) - origin[:2]
    for track_id, track in enumerate(tracks):
        if track_id in confirmed_model_samples:
            sample_s, curve_xyz = confirmed_model_samples[track_id]
            evidence_model_s.append(sample_s)
            evidence_model_xyz.append(curve_xyz)
            continue
        span_run = track.supported_s_max_m - track.supported_s_min_m
        if (
            len(track.inlier_rows) < 2
            or not np.isfinite(span_run)
            or span_run <= 0.0
        ):
            evidence_model_s.append(np.empty(0, dtype=np.float64))
            evidence_model_xyz.append(np.empty((0, 3), dtype=np.float32))
            continue
        sample_step = max(config.wire_curve_sample_spacing_m, 0.1)
        sample_count = min(
            config.wire_max_curve_samples,
            max(2, int(math.ceil(span_run / sample_step)) + 1),
        )
        sample_s = np.linspace(track.supported_s_min_m, track.supported_s_max_m, sample_count)
        sample_u = evaluate_polynomial(track.model_u_coefficients, sample_s)
        sample_z = (
            evaluate_catenary(track.catenary_parameters, sample_s)
            if track.model_type == "CATENARY"
            else evaluate_polynomial(track.model_z_coefficients, sample_s)
        ) - origin[2]
        curve_xy = (
            start_xy[None, :]
            + sample_s[:, None] * axis[None, :]
            + sample_u[:, None] * perp[None, :]
        )
        evidence_model_s.append(sample_s)
        evidence_model_xyz.append(np.column_stack((curve_xy, sample_z)).astype(np.float32))

    # Deterministic wire ordering by (median u, then wire id).
    strands.sort(key=lambda strand: (strand.median_u_m, strand.wire_id))
    for position, strand in enumerate(strands):
        wire_name = f"{span.span_name}-Wire{position + 1:02d}"
        strands[position] = SpanWireStrand(
            wire_id=position,
            wire_name=wire_name,
            representative_count=strand.representative_count,
            raw_point_count=strand.raw_point_count,
            s_span_m=strand.s_span_m,
            coverage_ratio=strand.coverage_ratio,
            median_u_m=strand.median_u_m,
            median_hag_m=strand.median_hag_m,
            curve_sample_count=strand.curve_sample_count,
            curve_xyz_local_m=strand.curve_xyz_local_m,
            residual_stats_m=strand.residual_stats_m,
            approach_start=strand.approach_start,
            approach_end=strand.approach_end,
            status=strand.status,
            model_type=strand.model_type,
            supported_s_domain_m=strand.supported_s_domain_m,
            supported_slice_count=strand.supported_slice_count,
            supported_slice_ratio=strand.supported_slice_ratio,
            maximum_gap_m=strand.maximum_gap_m,
            inlier_ratio=strand.inlier_ratio,
        )
    w1_arrays = _build_w1_arrays(span, strands, config)
    # Exact FULL_RESOLUTION source-row identity of every raw point in a FINAL
    # CONFIRMED conductor strand.  Only this exact set may be subtracted from
    # vegetation; broad wire candidate/search points are never excluded.
    if output_root is None:
        confirmed_member_source_index = _confirmed_strand_member_source_indices(
            wire_region_subset,
            confirmed_groups,
            voxel_size,
            authoritative_source_point_count=authoritative_source_point_count,
        )
        support_source_indices: list[np.ndarray] = []
        inlier_source_indices: list[np.ndarray] = []
        support_overlap_pairs: list[dict[str, int]] = []
        inlier_overlap_pairs: list[dict[str, int]] = []
    else:
        support_source_indices, support_overlap_pairs = _track_member_source_indices(
            wire_region_subset,
            [track.rows for track in tracks],
            voxel_size,
            len(s_values),
            authoritative_source_point_count=authoritative_source_point_count,
        )
        inlier_source_indices, inlier_overlap_pairs = _track_member_source_indices(
            wire_region_subset,
            [track.inlier_rows for track in tracks],
            voxel_size,
            len(s_values),
            authoritative_source_point_count=authoritative_source_point_count,
        )
        for track, sources in zip(tracks, support_source_indices, strict=True):
            if len(sources) != track.raw_support_count:
                raise PoleEvidenceError("SPAN_WIRE_TRACK_SOURCE_SUPPORT_COUNT_MISMATCH")
        final_statuses = [str(item["status"]) for item in track_diagnostics]
        for overlap in inlier_overlap_pairs:
            if (
                final_statuses[overlap["first_track_id"]] == CONFIRMED
                and final_statuses[overlap["second_track_id"]] == CONFIRMED
            ):
                raise PoleEvidenceError("SPAN_WIRE_CONFIRMED_TRACK_OWNERSHIP_OVERLAP")
        confirmed_source_groups = [
            source
            for status, source in zip(final_statuses, inlier_source_indices, strict=True)
            if status == CONFIRMED
        ]
        confirmed_member_source_index = (
            np.unique(np.concatenate(confirmed_source_groups)).astype(np.int64)
            if confirmed_source_groups
            else np.empty(0, dtype=np.int64)
        )
    if output_root is not None:
        _write_npz(
            output_root / "span_wire_diagnostics.npz",
            fine_xyz_local_m=arrays["xyz_local_m"],
            fine_original_index=arrays["original_index"],
            raw_count=arrays["raw_count"],
            candidate_mask=candidate,
            cluster_id_by_rep=cluster_id_by_rep,
            s_m=s_values,
            u_m=u_values,
            z_m=z_values,
            confirmed_member_source_index=confirmed_member_source_index,
        )
        _write_npz(output_root / "w1_arrays.npz", **w1_arrays)
        from algorithms.spans.wire_v2_evidence import write_track_evidence

        write_track_evidence(
            output_root,
            span_name=span.span_name,
            source_point_count=authoritative_source_point_count,
            tracks=tracks,
            diagnostics=track_diagnostics,
            support_source_indices=support_source_indices,
            inlier_source_indices=inlier_source_indices,
            model_s_m=evidence_model_s,
            model_xyz_local_m=evidence_model_xyz,
            support_overlap_pairs=support_overlap_pairs,
            inlier_overlap_pairs=inlier_overlap_pairs,
        )
        _write_json(
            output_root / "span_wire_result.json",
            {
                "schema": SPAN_WIRE_SCHEMA,
                "marker": FAST_MATURE_SPAN_WIRE,
                "strands": [strand.to_dict() for strand in strands],
                "strand_count": len(strands),
                "diagnostics": {
                    "schema": SPAN_WIRE_SCHEMA,
                    "marker": FAST_MATURE_SPAN_WIRE,
                    "fine_wire_representative_count": int(len(output_fine)),
                    "wire_candidate_point_count": int(np.count_nonzero(candidate)),
                    "wire_cluster_count": int(
                        np.count_nonzero(np.unique(cluster_labels[cluster_labels >= 0]))
                    ),
                    "wire_strand_count": len(strands),
                    "v2_track_count": len(track_diagnostics),
                    "v2_tracks": track_diagnostics,
                    "confirmed_conductor_member_count": int(len(confirmed_member_source_index)),
                    "runtime_seconds": time.perf_counter() - started,
                },
            },
        )
    return SpanWireResult(
        strands=tuple(strands),
        w1_arrays=w1_arrays,
        candidate_mask=candidate,
        cluster_id_by_rep=cluster_id_by_rep,
        confirmed_member_source_index=confirmed_member_source_index,
        diagnostics={
            "schema": SPAN_WIRE_SCHEMA,
            "marker": FAST_MATURE_SPAN_WIRE,
            "fine_wire_representative_count": int(len(output_fine)),
            "wire_candidate_point_count": int(np.count_nonzero(candidate)),
            "wire_cluster_count": int(
                np.count_nonzero(np.unique(cluster_labels[cluster_labels >= 0]))
            ),
            "wire_strand_count": len(strands),
            "v2_track_count": len(track_diagnostics),
            "v2_tracks": track_diagnostics,
            "confirmed_conductor_member_count": int(len(confirmed_member_source_index)),
            "fine_representatives_npz_path": str(output_root / "fine_representatives.npz")
            if output_root is not None
            else None,
            "diagnostics_npz_path": str(output_root / "span_wire_diagnostics.npz")
            if output_root is not None
            else None,
            "runtime_seconds": time.perf_counter() - started,
        },
    )


def _write_npz(path: Any, **arrays: np.ndarray) -> None:
    from pathlib import Path

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.npz")
    np.savez(temporary, **arrays)
    temporary.replace(path)


def _write_json(path: Any, payload: dict[str, Any]) -> None:
    from pathlib import Path

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2), "utf-8"
    )
    temporary.replace(path)


def load_span_wire_result(artifact_root: Any) -> SpanWireResult:
    """Reload a persisted per-span wire result (arrays + strand summaries)."""

    from pathlib import Path

    root = Path(artifact_root)
    w1_path = root / "w1_arrays.npz"
    manifest_path = root / "span_wire_result.json"
    if not w1_path.is_file() or not manifest_path.is_file():
        raise PoleEvidenceError("SPAN_WIRE_ARTIFACT_MISSING")
    with np.load(w1_path, allow_pickle=False) as stream:
        w1_arrays = {name: np.asarray(stream[name]) for name in stream.files}
    payload = json.loads(manifest_path.read_text("utf-8"))
    sampled_xyz = np.asarray(w1_arrays["sampled_curve_xyz"], dtype=np.float32)
    offsets = np.asarray(w1_arrays["sampled_curve_offsets"], dtype=np.int64)
    strands = []
    for position, item in enumerate(payload["strands"]):
        if position + 1 < len(offsets):
            curve = sampled_xyz[int(offsets[position]): int(offsets[position + 1])]
        else:
            curve = np.empty((0, 3), dtype=np.float32)
        strands.append(
            SpanWireStrand(
                wire_id=int(item["wire_id"]),
                wire_name=str(item["wire_name"]),
                representative_count=int(item["representative_count"]),
                raw_point_count=int(item["raw_point_count"]),
                s_span_m=float(item["s_span_m"]),
                coverage_ratio=float(item["coverage_ratio"]),
                median_u_m=float(item["median_u_m"]),
                median_hag_m=float(item["median_hag_m"]),
                curve_sample_count=int(item["curve_sample_count"]),
                curve_xyz_local_m=curve,
                residual_stats_m=(
                    float(item["residual_stats_m"][0]),
                    float(item["residual_stats_m"][1]),
                ),
                approach_start=bool(item["approach_start"]),
                approach_end=bool(item["approach_end"]),
                status=str(item.get("status", "CONFIRMED")),
                model_type=str(item.get("model_type", "ROBUST_QUADRATIC")),
                supported_s_domain_m=tuple(
                    float(value) for value in item.get("supported_s_domain_m", (0.0, 0.0))
                ),
                supported_slice_count=int(item.get("supported_slice_count", 0)),
                supported_slice_ratio=float(item.get("supported_slice_ratio", 0.0)),
                maximum_gap_m=float(item.get("maximum_gap_m", 0.0)),
                inlier_ratio=float(item.get("inlier_ratio", 0.0)),
            )
        )
    strands = tuple(strands)
    diagnostics_path = root / "span_wire_diagnostics.npz"
    candidate_mask = np.empty(0, dtype=bool)
    cluster_id_by_rep = np.empty(0, dtype=np.int32)
    confirmed_member_source_index = np.empty(0, dtype=np.int64)
    if diagnostics_path.is_file():
        with np.load(diagnostics_path, allow_pickle=False) as stream:
            candidate_mask = np.asarray(stream["candidate_mask"], dtype=bool)
            cluster_id_by_rep = np.asarray(stream["cluster_id_by_rep"], dtype=np.int32)
            if "confirmed_member_source_index" in stream.files:
                confirmed_member_source_index = np.asarray(
                    stream["confirmed_member_source_index"], dtype=np.int64
                )
    return SpanWireResult(
        strands=strands,
        w1_arrays=w1_arrays,
        candidate_mask=candidate_mask,
        cluster_id_by_rep=cluster_id_by_rep,
        confirmed_member_source_index=confirmed_member_source_index,
        diagnostics=dict(payload.get("diagnostics", {})),
    )


def _merge_strand_groups(
    groups: list[np.ndarray],
    s_values: np.ndarray,
    u_values: np.ndarray,
    z_values: np.ndarray,
    span_length: float,
    config: FastBaselineConfig,
) -> list[np.ndarray]:
    """Reconnect DBSCAN fragments of the same conductor into one strand.

    Two groups are merged only on strong evidence: they share the same (u, z)
    track (within the strand-separation scale) AND their longitudinal ``s``
    ranges are adjacent (overlap or a bounded gap).  Different parallel
    conductors are never merged.
    """

    if len(groups) < 2:
        return list(groups)
    merge_gap = max(3.0, 0.15 * span_length)
    merged: list[set[int]] = [set(group.tolist()) for group in groups]
    changed = True
    while changed:
        changed = False
        position = 0
        while position < len(merged):
            other = position + 1
            while other < len(merged):
                left = merged[position]
                right = merged[other]
                left_indices = np.asarray(sorted(left), dtype=np.int64)
                right_indices = np.asarray(sorted(right), dtype=np.int64)
                u_left = float(np.median(u_values[left_indices]))
                z_left = float(np.median(z_values[left_indices]))
                u_right = float(np.median(u_values[right_indices]))
                z_right = float(np.median(z_values[right_indices]))
                s_min_left = float(np.min(s_values[left_indices]))
                s_max_left = float(np.max(s_values[left_indices]))
                s_min_right = float(np.min(s_values[right_indices]))
                s_max_right = float(np.max(s_values[right_indices]))
                same_track = (
                    abs(u_left - u_right) <= max(config.wire_strand_u_separation_m, 0.2)
                    and abs(z_left - z_right) <= max(config.wire_strand_z_separation_m, 0.2)
                )
                adjacent_s = (
                    (s_max_left >= s_min_right - merge_gap)
                    and (s_min_left <= s_max_right + merge_gap)
                )
                if same_track and adjacent_s:
                    merged[position] |= right
                    del merged[other]
                    changed = True
                else:
                    other += 1
            position += 1
    return [np.asarray(sorted(group), dtype=np.int64) for group in merged]


def _strand_approaches_pole(
    s_values: np.ndarray,
    u_values: np.ndarray,
    z_values: np.ndarray,
    span: SpanRecord,
    pole_s: float,
    config: FastBaselineConfig,
) -> bool:
    """True when the strand approaches the pole zone at *pole_s*."""

    near = np.abs(s_values - pole_s) <= config.wire_approach_pole_radius_m
    if not np.any(near):
        return False
    pole_top_z = (
        span.start_pole_top_z_m if pole_s == 0.0 else span.end_pole_top_z_m
    )
    return bool(
        np.any(
            (np.abs(u_values[near]) <= config.span_half_width_m)
            & (np.abs(z_values[near] - pole_top_z) <= config.wire_endpoint_hag_tolerance_m)
        )
    )


def _empty_w1_arrays() -> dict[str, np.ndarray]:
    return {
        "wire_id": np.empty(0, dtype=np.int32),
        "wire_status": np.empty(0, dtype=np.uint8),
        "wire_method": np.empty(0, dtype=np.uint8),
        "wire_quality_flags": np.empty(0, dtype=np.uint16),
        "wire_has_distance_ready_segments": np.empty(0, dtype=np.bool_),
        "wire_continuous_distance_ready": np.empty(0, dtype=np.bool_),
        "wire_segment_offsets": np.zeros(1, dtype=np.int64),
        "segment_wire_id": np.empty(0, dtype=np.int32),
        "segment_index": np.empty(0, dtype=np.int32),
        "segment_method": np.empty(0, dtype=np.uint8),
        "segment_quality_flags": np.empty(0, dtype=np.uint16),
        "segment_distance_ready": np.empty(0, dtype=np.bool_),
        "segment_parameter_domain": np.empty((0, 2), dtype=np.float64),
        "sampled_curve_xyz": np.empty((0, 3), dtype=np.float32),
        "sampled_curve_parameter": np.empty(0, dtype=np.float32),
        "sampled_curve_offsets": np.zeros(1, dtype=np.int64),
    }


def _build_w1_arrays(
    span: SpanRecord,
    strands: list[SpanWireStrand],
    config: FastBaselineConfig,
) -> dict[str, np.ndarray]:
    """Assemble W1-compatible distance-ready arrays (NON_CERTIFIED FAST).

    The array keys match the existing W1 schema consumed by
    ``build_wire_edge_bundle`` so the exact Tree-Wire engine can be reused
    without inheriting Formal authority.
    """

    if not strands:
        return _empty_w1_arrays()
    wire_id = np.arange(len(strands), dtype=np.int32)
    wire_status = np.full(len(strands), _CONFIRMED_MODEL, dtype=np.uint8)
    wire_method = np.full(len(strands), _SMOOTHED_POLYLINE, dtype=np.uint8)
    wire_quality_flags = np.zeros(len(strands), dtype=np.uint16)
    wire_has_ready = np.ones(len(strands), dtype=np.bool_)
    wire_continuous = np.ones(len(strands), dtype=np.bool_)
    segment_wire_id: list[int] = []
    segment_index: list[int] = []
    segment_method: list[int] = []
    segment_quality_flags: list[int] = []
    segment_distance_ready: list[bool] = []
    segment_parameter_domain: list[list[float]] = []
    sampled_curve_xyz: list[np.ndarray] = []
    sampled_curve_parameter: list[np.ndarray] = []
    sampled_curve_offsets = [0]
    wire_segment_offsets = [0]
    for position, strand in enumerate(strands):
        curve = np.asarray(strand.curve_xyz_local_m, dtype=np.float32)
        samples = len(curve)
        if samples < 2:
            continue
        segment_parameter = np.asarray(
            [float(position), float(position + 1)], dtype=np.float64
        )
        segment_wire_id.append(position)
        segment_index.append(0)
        segment_method.append(_SMOOTHED_POLYLINE)
        segment_quality_flags.append(0)
        segment_distance_ready.append(True)
        segment_parameter_domain.append([float(segment_parameter[0]), float(segment_parameter[1])])
        sampled_curve_xyz.append(curve)
        sampled_curve_parameter.append(np.linspace(0.0, 1.0, samples, dtype=np.float32))
        sampled_curve_offsets.append(sampled_curve_offsets[-1] + samples)
        wire_segment_offsets.append(wire_segment_offsets[-1] + 1)
    if not segment_wire_id:
        return _empty_w1_arrays()
    return {
        "wire_id": np.asarray(wire_id, dtype=np.int32),
        "wire_status": wire_status,
        "wire_method": wire_method,
        "wire_quality_flags": wire_quality_flags,
        "wire_has_distance_ready_segments": wire_has_ready,
        "wire_continuous_distance_ready": wire_continuous,
        "wire_segment_offsets": np.asarray(wire_segment_offsets, dtype=np.int64),
        "segment_wire_id": np.asarray(segment_wire_id, dtype=np.int32),
        "segment_index": np.asarray(segment_index, dtype=np.int32),
        "segment_method": np.asarray(segment_method, dtype=np.uint8),
        "segment_quality_flags": np.asarray(segment_quality_flags, dtype=np.uint16),
        "segment_distance_ready": np.asarray(segment_distance_ready, dtype=np.bool_),
        "segment_parameter_domain": np.asarray(segment_parameter_domain, dtype=np.float64),
        "sampled_curve_xyz": np.concatenate(sampled_curve_xyz, axis=0),
        "sampled_curve_parameter": np.concatenate(sampled_curve_parameter, axis=0),
        "sampled_curve_offsets": np.asarray(sampled_curve_offsets, dtype=np.int64),
    }


__all__ = [
    "SPAN_WIRE_SCHEMA",
    "FAST_MATURE_SPAN_WIRE",
    "SpanWireStrand",
    "SpanWireResult",
    "run_span_wire_pipeline",
    "_local_pca_features",
    "_dbscan",
]
